#!/usr/bin/env python3
"""LIBERO eval of the 1-step distilled SmolVLA student.
Loads distilled weights over the base checkpoint, applies the step-size (d) conditioning
used in training, sets num_steps=1, and runs the standard eval.
Usage: python eval_onestep.py <ckpt.pt> <n_episodes> [num_steps] [batch]"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
import engines  # fp32 sincos patch first
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import lerobot.policies.smolvla.modeling_smolvla as M
import lerobot.scripts.lerobot_eval as LE

CKPT = sys.argv[1]
N_EP = sys.argv[2] if len(sys.argv) > 2 else "10"
NSTEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
BS = sys.argv[4] if len(sys.argv) > 4 else "8"

sd = torch.load(CKPT, map_location="cpu")

class StepSizeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim); self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))

_step_mlp = {"m": None}
_CUR_D = {"d": None}
_base_sincos = M.create_sinusoidal_pos_embedding
def _sincos_with_d(t, dim, mn, mx, device="cpu"):
    emb = _base_sincos(t, dim, mn, mx, device=device)
    if _CUR_D["d"] is not None and _step_mlp["m"] is not None:
        d = torch.full_like(t, _CUR_D["d"])
        demb = _base_sincos(d, dim, mn, mx, device=device)
        emb = emb + _step_mlp["m"](demb.to(emb.dtype))
    return emb
M.create_sinusoidal_pos_embedding = _sincos_with_d

_orig_from_pretrained = SmolVLAPolicy.from_pretrained.__func__
def _patched_fp(cls, *a, **kw):
    policy = _orig_from_pretrained(cls, *a, **kw)
    m = policy.model
    m.vlm_with_expert.lm_expert.load_state_dict(sd["expert"])
    m.action_in_proj.load_state_dict(sd["action_in"])
    m.action_out_proj.load_state_dict(sd["action_out"])
    m.action_time_mlp_in.load_state_dict(sd["mlp_in"])
    m.action_time_mlp_out.load_state_dict(sd["mlp_out"])
    smlp = StepSizeMLP(m.vlm_with_expert.expert_hidden_size)
    smlp.load_state_dict(sd["step_mlp"])
    dev = next(m.parameters()).device
    _step_mlp["m"] = smlp.to(dev).to(next(m.action_in_proj.parameters()).dtype)
    policy.config.num_steps = NSTEPS
    _CUR_D["d"] = 1.0 / NSTEPS
    print(f"[onestep] distilled weights loaded (train step {sd.get('step')}), num_steps={NSTEPS}, d={1.0/NSTEPS}")
    return policy
SmolVLAPolicy.from_pretrained = classmethod(_patched_fp)

sys.argv = ["lerobot_eval",
            "--policy.path=HuggingFaceVLA/smolvla_libero",
            "--env.type=libero", "--env.task=libero_spatial",
            f"--eval.n_episodes={N_EP}", f"--eval.batch_size={BS}",
            f"--output_dir=artifacts/eval_onestep_s{NSTEPS}"]
LE.main()
print("ONESTEP EVAL DONE")
