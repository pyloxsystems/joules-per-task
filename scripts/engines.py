#!/usr/bin/env python3
"""Two-engine decomposition of SmolVLA inference, tensor-in/tensor-out for ONNX->TRT.

Engine A (prefix/VLM): images+lang+state -> stacked prefix K/V (16 layers) + pad-mask.  Runs once/chunk.
Engine B (denoise step): x_t + timestep + prefix K/V -> velocity v_t.  Runs num_steps times.
The Euler loop lives in plain Python and chains B; A + B together reproduce policy.model.sample_actions.
"""
import os
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


def _patch_sincos_fp32():
    """Force the timestep sinusoidal embedding to float32 (default float64 has no ORT-CPU Cos
    kernel, and float32 is what runs on the edge anyway)."""
    import math as _math
    import lerobot.policies.smolvla.modeling_smolvla as _M

    def _f(time, dimension, min_period, max_period, device="cpu"):
        dt = torch.float32
        frac = torch.linspace(0.0, 1.0, dimension // 2, dtype=dt, device=device)
        period = min_period * (max_period / min_period) ** frac
        sf = 1.0 / period * 2 * _math.pi
        si = sf[None, :] * time[:, None].to(dt)
        return torch.cat([torch.sin(si), torch.cos(si)], dim=1)

    _M.create_sinusoidal_pos_embedding = _f


_patch_sincos_fp32()


class PrefixEngine(nn.Module):
    """embed_prefix + VLM prefill -> (K_stack, V_stack, prefix_pad_masks)."""
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, images, img_masks, lang_tokens, lang_masks, state):
        wdt = self.m.state_proj.weight.dtype
        images = images.to(wdt); state = state.to(wdt)
        imgs = [images[:, i] for i in range(images.shape[1])]
        masks = [img_masks[:, i] for i in range(img_masks.shape[1])]
        pe, ppm, pam = self.m.embed_prefix(imgs, masks, lang_tokens, lang_masks, state=state)
        a2d = make_att_2d_masks(ppm, pam)
        pos = torch.cumsum(ppm, dim=1) - 1
        _, pkv = self.m.vlm_with_expert.forward(
            attention_mask=a2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[pe, None], use_cache=True,
        )
        K = torch.stack([pkv.layers[i].keys for i in range(len(pkv.layers))], dim=0)   # [L,B,kvh,plen,hd]
        V = torch.stack([pkv.layers[i].values for i in range(len(pkv.layers))], dim=0)
        return K, V, ppm.to(torch.float32)


class DenoiseEngine(nn.Module):
    """embed_suffix + expert cross/self-attn over fixed prefix K/V -> velocity v_t."""
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.chunk = model.config.chunk_size

    def forward(self, x_t, timestep, prefix_pad_masks, K, V):
        m = self.m
        ppm = prefix_pad_masks.to(torch.bool)
        wdt = m.action_in_proj.weight.dtype
        x_t = x_t.to(wdt); K = K.to(wdt); V = V.to(wdt)
        # rebuild a DynamicCache pre-loaded with the fixed prefix K/V (self-attn layers append suffix in-place)
        cache = DynamicCache()
        L = K.shape[0]
        for i in range(L):
            cache.update(K[i], V[i], i)

        suffix_embs, spm, sam = m.embed_suffix(x_t, timestep)
        suffix_len = spm.shape[1]
        bsz = ppm.shape[0]
        prefix_len = ppm.shape[1]
        prefix_pad_2d = ppm[:, None, :].expand(bsz, suffix_len, prefix_len)
        suffix_2d = make_att_2d_masks(spm, sam)
        full_2d = torch.cat([prefix_pad_2d, suffix_2d], dim=2)
        prefix_offsets = torch.sum(ppm, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(spm, dim=1) - 1

        outs, _ = m.vlm_with_expert.forward(
            attention_mask=full_2d, position_ids=position_ids, past_key_values=cache,
            inputs_embeds=[None, suffix_embs], use_cache=True,
        )
        suffix_out = outs[1][:, -self.chunk:].to(m.action_out_proj.weight.dtype)
        return m.action_out_proj(suffix_out).to(torch.float32)


def euler_chain(prefix_engine, denoise_engine, images, img_masks, lang_tokens, lang_masks,
                state, noise, num_steps):
    """Python Euler loop chaining A once + B num_steps times (mirrors euler_integrate)."""
    K, V, ppm = prefix_engine(images, img_masks, lang_tokens, lang_masks, state)
    dt = -1.0 / num_steps
    x_t = noise
    bsz = noise.shape[0]
    for step in range(num_steps):
        t = 1.0 + step * dt
        tt = torch.tensor(t, dtype=torch.float32, device=noise.device).expand(bsz)
        v_t = denoise_engine(x_t, tt, ppm, K, V)
        x_t = x_t + dt * v_t
    return x_t


class VisionEngine(nn.Module):
    """SigLIP tower only: (1,ncam,3,H,W) -> per-camera token embeddings (1,ncam,T,D)."""
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, images):
        wdt = self.m.state_proj.weight.dtype
        images = images.to(wdt)
        embs = []
        for i in range(images.shape[1]):
            e = self.m.vlm_with_expert.embed_image(images[:, i])   # (1,T,D)
            embs.append(e)
        return torch.stack(embs, dim=1)                            # (1,ncam,T,D)


class PrefillEngine(nn.Module):
    """embed_prefix assembly (from precomputed image embeddings) + VLM prefill -> K,V,ppm.
    Mirrors SmolVLAPolicy.model.embed_prefix exactly, minus the SigLIP calls."""
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, img_embs, img_masks, lang_tokens, lang_masks, state):
        m = self.m
        wdt = m.state_proj.weight.dtype
        state = state.to(wdt)
        embs, pad_masks, att_masks = [], [], []
        ncam = img_embs.shape[1]
        for i in range(ncam):
            e = img_embs[:, i].to(wdt)
            e = e * torch.tensor(e.shape[-1] ** 0.5, dtype=e.dtype, device=e.device)
            bsize, n_tok = e.shape[:2]
            embs.append(e)
            pad_masks.append(img_masks[:, i][:, None].expand(bsize, n_tok))
            att_masks += [0] * n_tok
        lang_emb = m.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * (lang_emb.shape[-1] ** 0.5)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        state_emb = m.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        pad_masks.append(torch.ones(bsize, state_emb.shape[1], dtype=torch.bool, device=state_emb.device))
        att_masks += [1] * state_emb.shape[1]
        pe = torch.cat(embs, dim=1)
        ppm = torch.cat(pad_masks, dim=1)
        pam = torch.tensor(att_masks, dtype=torch.bool, device=ppm.device)[None, :].expand(bsize, -1)

        a2d = make_att_2d_masks(ppm, pam)
        pos = torch.cumsum(ppm, dim=1) - 1
        _, pkv = m.vlm_with_expert.forward(
            attention_mask=a2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[pe, None], use_cache=True,
        )
        K = torch.stack([pkv.layers[i].keys for i in range(len(pkv.layers))], dim=0)
        V = torch.stack([pkv.layers[i].values for i in range(len(pkv.layers))], dim=0)
        return K, V, ppm.to(torch.float32)


def truncate_vlm_layers(model, lo, hi, identity_norm):
    """Restrict vlm_with_expert to layers [lo:hi) for split-prefill export (prefill discards
    the final hidden, so norm can be Identity for the lower half)."""
    vwm = model.vlm_with_expert
    txt = vwm.get_vlm_model().text_model
    txt.layers = nn.ModuleList(list(txt.layers)[lo:hi])
    vwm.lm_expert.layers = nn.ModuleList(list(vwm.lm_expert.layers)[lo:hi])
    vwm.num_vlm_layers = hi - lo
    vwm.num_expert_layers = hi - lo
    if identity_norm:
        txt.norm = nn.Identity()


class PrefillEngineA(nn.Module):
    """Embed assembly + VLM layers [0:H) -> K,V for those layers + ppm + pam + raw hidden."""
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, img_embs, img_masks, lang_tokens, lang_masks, state):
        m = self.m
        wdt = m.state_proj.weight.dtype
        state = state.to(wdt)
        embs, pad_masks, att_masks = [], [], []
        ncam = img_embs.shape[1]
        for i in range(ncam):
            e = img_embs[:, i].to(wdt)
            e = e * torch.tensor(e.shape[-1] ** 0.5, dtype=e.dtype, device=e.device)
            bsize, n_tok = e.shape[:2]
            embs.append(e)
            pad_masks.append(img_masks[:, i][:, None].expand(bsize, n_tok))
            att_masks += [0] * n_tok
        lang_emb = m.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * (lang_emb.shape[-1] ** 0.5)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        state_emb = m.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        pad_masks.append(torch.ones(bsize, state_emb.shape[1], dtype=torch.bool, device=state_emb.device))
        att_masks += [1] * state_emb.shape[1]
        pe = torch.cat(embs, dim=1)
        ppm = torch.cat(pad_masks, dim=1)
        pam = torch.tensor(att_masks, dtype=torch.bool, device=ppm.device)[None, :].expand(bsize, -1)

        a2d = make_att_2d_masks(ppm, pam)
        pos = torch.cumsum(ppm, dim=1) - 1
        outs, pkv = self.m.vlm_with_expert.forward(
            attention_mask=a2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[pe, None], use_cache=True,
        )
        K = torch.stack([pkv.layers[i].keys for i in range(len(pkv.layers))], dim=0)
        V = torch.stack([pkv.layers[i].values for i in range(len(pkv.layers))], dim=0)
        return K, V, ppm.to(torch.float32), pam.to(torch.float32), outs[0]


class PrefillEngineB(nn.Module):
    """VLM layers [H:L) from raw hidden -> K,V for the upper layers."""
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, hidden, prefix_pad_masks, prefix_att_masks):
        ppm = prefix_pad_masks.to(torch.bool)
        pam = prefix_att_masks.to(torch.bool)
        a2d = make_att_2d_masks(ppm, pam)
        pos = torch.cumsum(ppm, dim=1) - 1
        _, pkv = self.m.vlm_with_expert.forward(
            attention_mask=a2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[hidden, None], use_cache=True,
        )
        K = torch.stack([pkv.layers[i].keys for i in range(len(pkv.layers))], dim=0)
        V = torch.stack([pkv.layers[i].values for i in range(len(pkv.layers))], dim=0)
        return K, V


class StateDecodeEngine(nn.Module):
    """Exact incremental prefill: recompute ONLY the state token's per-layer K/V given cached
    prefix K/V of the other tokens (they never attend to state — mathematically exact).
    Manual layer loop: the stock forward routes cached calls into cross-attn/expert mode."""
    def __init__(self, model, first_half: bool):
        super().__init__()
        self.m = model
        self.first = first_half

    def forward(self, state_or_hidden, K, V, prefix_pad_masks):
        m = self.m
        vwm = m.vlm_with_expert
        ppm = prefix_pad_masks.to(torch.bool)
        wdt = m.state_proj.weight.dtype
        if self.first:
            h = m.state_proj(state_or_hidden.to(wdt))
            h = h[:, None, :] if h.ndim == 2 else h
        else:
            h = state_or_hidden.to(wdt)
        plen = ppm.shape[1]
        bsz = ppm.shape[0]
        cache = DynamicCache()
        L = K.shape[0]
        for i in range(L):
            cache.update(K[i], V[i], i)
        att = torch.ones(bsz, 1, plen, dtype=torch.bool, device=h.device)
        pos = torch.full((bsz, 1), plen - 1, dtype=torch.long, device=h.device)
        models = [vwm.get_vlm_model().text_model, vwm.lm_expert]
        model_layers = vwm.get_model_layers(models)
        head_dim = vwm.vlm.config.text_config.head_dim
        for idx in range(L):
            att_outputs, cache = vwm.forward_attn_layer(
                model_layers, [h, None], idx, pos, att, bsz, head_dim,
                use_cache=True, past_key_values=cache)
            layer = model_layers[0][idx]
            ao = att_outputs[0]
            if ao.dtype != layer.self_attn.o_proj.weight.dtype:
                ao = ao.to(layer.self_attn.o_proj.weight.dtype)
            out = layer.self_attn.o_proj(ao[:, :1])
            out = out + h
            res = out.clone()
            out = layer.post_attention_layernorm(out)
            out = layer.mlp(out)
            out = out + res
            h = out
        Ks = torch.stack([cache.layers[i].keys[:, :, -1:, :] for i in range(L)], dim=0)
        Vs = torch.stack([cache.layers[i].values[:, :, -1:, :] for i in range(L)], dim=0)
        return Ks, Vs, h
