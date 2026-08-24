#!/usr/bin/env python3
"""SnapFlow-style 1-step distillation of SmolVLA-LIBERO's flow-matching action expert.

Recipe (arXiv 2604.05656, reimplemented — no public code exists):
  - freeze VLM tower; train action expert + a zero-init 2-layer MLP that embeds the
    target step size d and adds into the existing time embedding
  - loss = alpha * standard flow-matching loss (real obs-action pairs)
         + (1-alpha) * shortcut self-consistency: one d-sized jump must match two d/2 jumps
    (the frozen-ish current model is its own teacher; no external teacher net)
  - after training, inference uses a single step (d=1).
Usage: python distill_snapflow.py [steps] [batch] [lr]
"""
import os, sys, time, math
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
import engines  # fp32 sincos patch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
ALPHA = 0.5
ADIM_REAL = None  # resolved after policy load
DEV = "cuda"
OUT = "artifacts/snapflow_ckpt"

policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero").to(DEV)
policy.train()
m = policy.model
cfg = policy.config
ADIM_REAL = cfg.action_feature.shape[0]
print("real action dims:", ADIM_REAL)

# ---- freeze VLM tower; train action expert + projections ----
vlm = m.vlm_with_expert.get_vlm_model()
for p in vlm.parameters():
    p.requires_grad = False
trainable_mods = [m.vlm_with_expert.lm_expert, m.action_in_proj, m.action_out_proj,
                  m.action_time_mlp_in, m.action_time_mlp_out]
for mod in trainable_mods:
    for p in mod.parameters():
        p.requires_grad = True

# ---- zero-init step-size MLP added into the time embedding ----
tdim = m.action_time_mlp_in.in_features - m.action_in_proj.out_features  # time-emb width
class StepSizeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)  # zero-init output
    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))
step_mlp = StepSizeMLP(m.vlm_with_expert.expert_hidden_size).to(DEV)

# ---- frozen teacher: pristine copies of everything the student will mutate ----
import copy
T_expert = copy.deepcopy(m.vlm_with_expert.lm_expert).eval()
T_ain = copy.deepcopy(m.action_in_proj).eval(); T_aout = copy.deepcopy(m.action_out_proj).eval()
T_min = copy.deepcopy(m.action_time_mlp_in).eval(); T_mout = copy.deepcopy(m.action_time_mlp_out).eval()
for mod in (T_expert, T_ain, T_aout, T_min, T_mout):
    for p in mod.parameters(): p.requires_grad = False

import contextlib
@contextlib.contextmanager
def teacher_weights():
    """Swap frozen teacher modules in, restore student after."""
    orig = (m.vlm_with_expert.lm_expert, m.action_in_proj, m.action_out_proj,
            m.action_time_mlp_in, m.action_time_mlp_out)
    m.vlm_with_expert.lm_expert = T_expert; m.action_in_proj = T_ain
    m.action_out_proj = T_aout; m.action_time_mlp_in = T_min; m.action_time_mlp_out = T_mout
    try: yield
    finally:
        (m.vlm_with_expert.lm_expert, m.action_in_proj, m.action_out_proj,
         m.action_time_mlp_in, m.action_time_mlp_out) = orig

# hook: extend embed_suffix to accept a step-size d via module-level state
import lerobot.policies.smolvla.modeling_smolvla as M
_CUR_D = {"d": None}
_orig_sincos = M.create_sinusoidal_pos_embedding
def _sincos_with_d(t, dim, mn, mx, device="cpu"):
    emb = _orig_sincos(t, dim, mn, mx, device=device)
    if _CUR_D["d"] is not None:
        demb = _orig_sincos(_CUR_D["d"], dim, mn, mx, device=device)
        emb = emb + step_mlp(demb.to(emb.dtype))
    return emb
M.create_sinusoidal_pos_embedding = _sincos_with_d

def velocity(x_t, t, d, ppm, pam, pkv_kwargs):
    """v(x_t, t; d) — one denoise query with step-size conditioning."""
    _CUR_D["d"] = d  # None = no step-size conditioning (teacher/pretrained mode)
    try:
        suffix_embs, spm, sam = m.embed_suffix(x_t, t)
    finally:
        _CUR_D["d"] = None
    suffix_len = spm.shape[1]; bsz = ppm.shape[0]; plen = ppm.shape[1]
    p2d = ppm[:, None, :].expand(bsz, suffix_len, plen)
    s2d = make_att_2d_masks(spm, sam)
    full = torch.cat([p2d, s2d], dim=2)
    off = torch.sum(ppm, dim=-1)[:, None]
    pos = off + torch.cumsum(spm, dim=1) - 1
    outs, _ = m.vlm_with_expert.forward(
        attention_mask=full, position_ids=pos, past_key_values=pkv_kwargs["cache"](),
        inputs_embeds=[None, suffix_embs], use_cache=True)
    so = outs[1][:, -cfg.chunk_size:].to(dtype=m.action_out_proj.weight.dtype)
    return m.action_out_proj(so).to(torch.float32)

def prefix_cache(batch_imgs, img_masks, lang_tokens, lang_masks, state):
    """Frozen-VLM prefill once per batch; returns a factory producing fresh caches."""
    from transformers.cache_utils import DynamicCache
    with torch.no_grad():
        pe, ppm, pam = m.embed_prefix(batch_imgs, img_masks, lang_tokens, lang_masks, state=state)
        a2d = make_att_2d_masks(ppm, pam)
        pos = torch.cumsum(ppm, dim=1) - 1
        _, pkv = m.vlm_with_expert.forward(attention_mask=a2d, position_ids=pos,
            past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
        Ks = [pkv.layers[i].keys.detach() for i in range(len(pkv.layers))]
        Vs = [pkv.layers[i].values.detach() for i in range(len(pkv.layers))]
    def make():
        c = DynamicCache()
        for i, (k, v) in enumerate(zip(Ks, Vs)):
            c.update(k, v, i)
        return c
    return ppm, pam, make

# ---- data ----
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("HuggingFaceVLA/libero",
                    delta_timestamps={"action": [i / 10.0 for i in range(cfg.chunk_size)]})
dl = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=True, num_workers=4,
                                 drop_last=True, pin_memory=True)
print(f"dataset: {len(ds)} frames  chunk={cfg.chunk_size}")

params = [p for mod in trainable_mods for p in mod.parameters()] + list(step_mlp.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-5)
warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.02, total_iters=500)
cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS - 500)
sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[500])

from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.policies import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
_pcfg = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")
_pcfg.pretrained_path = "HuggingFaceVLA/smolvla_libero"
PRE, _ = make_pre_post_processors(policy_cfg=_pcfg, pretrained_path="HuggingFaceVLA/smolvla_libero",
    preprocessor_overrides={"device_processor": {"device": DEV}})

def prep_batch(b):
    batch = {"observation.images.image": b["observation.images.image"],
             "observation.images.image2": b["observation.images.image2"],
             "observation.state": b["observation.state"],
             "action": b["action"],
             "task": b.get("task", ["" for _ in range(BS)])}
    batch = PRE(batch)   # rename/tokenize/device/NORMALIZE with checkpoint stats
    return batch

print("starting distillation:", STEPS, "steps")
step = 0; t0 = time.time()
ema = {"flow": None, "cons": None}
def _ema(k, v):
    ema[k] = v if ema[k] is None else 0.98 * ema[k] + 0.02 * v
os.makedirs(OUT, exist_ok=True)
while step < STEPS:
    for b in dl:
        if step >= STEPS: break
        batch = prep_batch(b)
        # policy preprocessing: images+state prepared exactly as forward() does
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        actions = policy.prepare_action(batch)
        lt = batch[OBS_LANGUAGE_TOKENS]; lm = batch[OBS_LANGUAGE_ATTENTION_MASK]
        ppm, pam, cache_factory = prefix_cache(images, img_masks, lt, lm, state)
        pk = {"cache": cache_factory}
        bs = actions.shape[0]
        noise = torch.randn_like(actions)
        if step % 2 == 0:  # ---- flow-matching branch (alpha) ----
            t = torch.rand(bs, device=DEV)
            x_t = (1 - t)[:, None, None] * actions + t[:, None, None] * noise
            v_target = noise - actions
            d = torch.full((bs,), 1.0 / 16, device=DEV)  # smallest rung anchor
            v = velocity(x_t, t, d, ppm, pam, pk)
            loss = F.mse_loss(v[:, :, :ADIM_REAL], v_target[:, :, :ADIM_REAL])
            _ema("flow", loss.item())
        else:               # ---- shortcut self-consistency branch ----
            t = torch.rand(bs, device=DEV)
            rung = torch.randint(0, 4, (bs,), device=DEV)      # d in {1/8,1/4,1/2,1}
            d = (1.0 / 8) * (2.0 ** rung.float())
            d = torch.minimum(d, t.clamp(min=1.0 / 8))          # jump cannot exceed t
            x_t = (1 - t)[:, None, None] * actions + t[:, None, None] * noise
            with torch.no_grad(), teacher_weights():
                _CUR_D["d"] = None  # teacher has no step-size conditioning
                v1 = velocity(x_t, t, None, ppm, pam, pk)
                x_mid = x_t - (d / 2)[:, None, None] * v1
                v2 = velocity(x_mid, t - d / 2, None, ppm, pam, pk)
                v_teacher = (v1 + v2) / 2
            v = velocity(x_t, t, d, ppm, pam, pk)
            loss = F.mse_loss(v[:, :, :ADIM_REAL], v_teacher[:, :, :ADIM_REAL])
            _ema("cons", loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); step += 1
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}/{STEPS} flowEMA={ema['flow'] or 0:.4f} consEMA={ema['cons'] or 0:.5f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{el/step:.2f}s/step eta={(STEPS-step)*el/step/3600:.1f}h", flush=True)
        if step % 5000 == 0:
            torch.save({"expert": m.vlm_with_expert.lm_expert.state_dict(),
                        "action_in": m.action_in_proj.state_dict(),
                        "action_out": m.action_out_proj.state_dict(),
                        "mlp_in": m.action_time_mlp_in.state_dict(),
                        "mlp_out": m.action_time_mlp_out.state_dict(),
                        "step_mlp": step_mlp.state_dict(), "step": step},
                       f"{OUT}/ckpt_{step}.pt")
            print("checkpoint saved", step, flush=True)
torch.save({"expert": m.vlm_with_expert.lm_expert.state_dict(),
            "action_in": m.action_in_proj.state_dict(),
            "action_out": m.action_out_proj.state_dict(),
            "mlp_in": m.action_time_mlp_in.state_dict(),
            "mlp_out": m.action_time_mlp_out.state_dict(),
            "step_mlp": step_mlp.state_dict(), "step": step}, f"{OUT}/ckpt_final.pt")
print("DISTILLATION DONE")
