#!/usr/bin/env python3
"""v6: LoRA-constrained 1-step consistency distillation of SmolVLA-LIBERO.

Student = base policy + rank-R LoRA on the action expert's linear layers ONLY.
Objective: x0_student(x_t, t) matches frozen-teacher 4-substep Euler integration to t=0.
No step-size conditioning module. Deployment = 1 step at t=1.
Usage: python distill_lora.py [steps] [batch] [lr] [rank]
"""
import os, sys, time, copy
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
import engines
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from transformers.cache_utils import DynamicCache

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
RANK = int(sys.argv[4]) if len(sys.argv) > 4 else 32
T_SUB = 4       # teacher integration substeps
DEV = "cuda"
OUT = "artifacts/lora_ckpt"

policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero").to(DEV)
policy.eval()   # keep everything in eval mode; LoRA params still receive grads
m = policy.model
cfg = policy.config
AD = cfg.action_feature.shape[0]

for p in policy.parameters():
    p.requires_grad = False

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(rank, base.in_features, dtype=base.weight.dtype,
                                          device=base.weight.device) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype,
                                          device=base.weight.device))
        self.scale = alpha / rank
    @property
    def weight(self):  # attention code reads q_proj.weight.dtype etc.
        return self.base.weight
    @property
    def bias(self):
        return self.base.bias
    @property
    def in_features(self):
        return self.base.in_features
    @property
    def out_features(self):
        return self.base.out_features
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale

lora_params = []
n_wrapped = 0
for layer in m.vlm_with_expert.lm_expert.layers:
    for parent, attr in [(layer.self_attn, "q_proj"), (layer.self_attn, "k_proj"),
                         (layer.self_attn, "v_proj"), (layer.self_attn, "o_proj"),
                         (layer.mlp, "gate_proj"), (layer.mlp, "up_proj"), (layer.mlp, "down_proj")]:
        base = getattr(parent, attr, None)
        if isinstance(base, nn.Linear):
            w = LoRALinear(base, RANK)
            setattr(parent, attr, w)
            lora_params += [w.A, w.B]
            n_wrapped += 1
print(f"LoRA rank={RANK} wrapped {n_wrapped} linears; trainable={sum(p.numel() for p in lora_params)/1e6:.1f}M")

def velocity(x_t, t, ppm, pam, cache_factory):
    suffix_embs, spm, sam = m.embed_suffix(x_t, t)
    sl = spm.shape[1]; bsz = ppm.shape[0]; pl = ppm.shape[1]
    p2d = ppm[:, None, :].expand(bsz, sl, pl)
    s2d = make_att_2d_masks(spm, sam)
    full = torch.cat([p2d, s2d], dim=2)
    off = torch.sum(ppm, dim=-1)[:, None]
    pos = off + torch.cumsum(spm, dim=1) - 1
    outs, _ = m.vlm_with_expert.forward(attention_mask=full, position_ids=pos,
        past_key_values=cache_factory(), inputs_embeds=[None, suffix_embs], use_cache=True)
    so = outs[1][:, -cfg.chunk_size:].to(m.action_out_proj.weight.dtype)
    return m.action_out_proj(so).to(torch.float32)

class _NoLoRA:
    """Teacher mode: temporarily zero the LoRA contribution (B*A scale) via flag."""
    def __enter__(self):
        self.saved = [p.data.clone() for p in lora_params[1::2]]  # B matrices
        for p in lora_params[1::2]: p.data.zero_()
        return self
    def __exit__(self, *a):
        for p, s in zip(lora_params[1::2], self.saved): p.data.copy_(s)

def prefix_cache(images, img_masks, lt, lm_, state):
    with torch.no_grad():
        pe, ppm, pam = m.embed_prefix(images, img_masks, lt, lm_, state=state)
        a2d = make_att_2d_masks(ppm, pam)
        pos = torch.cumsum(ppm, dim=1) - 1
        _, pkv = m.vlm_with_expert.forward(attention_mask=a2d, position_ids=pos,
            past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
        Ks = [pkv.layers[i].keys.detach() for i in range(len(pkv.layers))]
        Vs = [pkv.layers[i].values.detach() for i in range(len(pkv.layers))]
    def make():
        c = DynamicCache()
        for i, (k, v) in enumerate(zip(Ks, Vs)): c.update(k, v, i)
        return c
    return ppm, pam, make

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
_pc = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")
_pc.pretrained_path = "HuggingFaceVLA/smolvla_libero"
PRE, _ = make_pre_post_processors(policy_cfg=_pc, pretrained_path="HuggingFaceVLA/smolvla_libero",
    preprocessor_overrides={"device_processor": {"device": DEV}})
ds = LeRobotDataset("HuggingFaceVLA/libero",
                    delta_timestamps={"action": [i / 10.0 for i in range(cfg.chunk_size)]})
dl = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=True, num_workers=4, drop_last=True)
print(f"dataset: {len(ds)} frames")

opt = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.0)
warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, total_iters=300)
cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS - 300)
sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[300])

os.makedirs(OUT, exist_ok=True)
step = 0; t0 = time.time(); ema = None
while step < STEPS:
    for b in dl:
        if step >= STEPS: break
        batch = PRE({"observation.images.image": b["observation.images.image"],
                     "observation.images.image2": b["observation.images.image2"],
                     "observation.state": b["observation.state"],
                     "action": b["action"], "task": b.get("task", [""] * BS)})
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        actions = policy.prepare_action(batch)
        lt = batch["observation.language.tokens"]; lm_ = batch["observation.language.attention_mask"]
        ppm, pam, cachef = prefix_cache(images, img_masks, lt, lm_, state)
        bs = actions.shape[0]
        noise = torch.randn_like(actions)
        # sample start time; weight toward t=1 (the deployment jump)
        t = torch.ones(bs, device=DEV) if step % 2 == 0 else (0.3 + 0.7 * torch.rand(bs, device=DEV))
        x_t = t[:, None, None] * noise + (1 - t)[:, None, None] * actions
        # frozen-teacher integration to t=0 in T_SUB substeps
        with torch.no_grad(), _NoLoRA():
            x = x_t.clone(); tt = t.clone()
            dt_sub = tt / T_SUB
            for s_ in range(T_SUB):
                v = velocity(x, tt, ppm, pam, cachef)
                x = x - dt_sub[:, None, None] * v
                tt = tt - dt_sub
            x0_T = x
        # student one-jump: x0_s = x_t - t * v_s(x_t, t)
        v_s = velocity(x_t, t, ppm, pam, cachef)
        x0_s = x_t - t[:, None, None] * v_s
        loss = F.mse_loss(x0_s[:, :, :AD], x0_T[:, :, :AD])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step(); sched.step(); step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}/{STEPS} lossEMA={ema:.5f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{el/step:.2f}s/step eta={(STEPS-step)*el/step/3600:.1f}h", flush=True)
        if step % 5000 == 0:
            torch.save({"lora": [p.detach().cpu() for p in lora_params], "rank": RANK, "step": step},
                       f"{OUT}/lora_{step}.pt")
torch.save({"lora": [p.detach().cpu() for p in lora_params], "rank": RANK, "step": step},
           f"{OUT}/lora_final.pt")
print("LORA DISTILLATION DONE")
