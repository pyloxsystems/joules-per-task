#!/usr/bin/env python3
"""Eval base policy + LoRA adapters at N denoise steps. Usage: eval_lora.py <lora.pt> <n_eps> <steps> [task_ids]"""
import os, sys
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("PYTORCH_NVFUSER_DISABLE","1")
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn, torch.nn.functional as F
import engines
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
import lerobot.scripts.lerobot_eval as LE
TAU=0.05
STATS={'encodes':0,'skips':0}
_cam_counter={'i':0}
_cache={}
_orig_embed = SmolVLMWithExpertModel.embed_image
def _gated_embed(self, img):
    c = _cam_counter["i"]; _cam_counter["i"] += 1
    ent = _cache.get(c)
    if ent is not None:
        last, emb = ent
        if last.shape == img.shape:
            diff = (img - last).abs().mean().item()
            if diff < TAU:
                STATS["skips"] += 1
                return emb
    emb = _orig_embed(self, img)
    _cache[c] = (img.detach().clone(), emb.detach().clone())
    STATS["encodes"] += 1
    return emb
SmolVLMWithExpertModel.embed_image = _gated_embed

_orig_sel = SmolVLAPolicy.select_action
def _sel(self, batch, *a, **kw):
    _cam_counter["i"] = 0  # per-step camera index reset
    return _orig_sel(self, batch, *a, **kw)
SmolVLAPolicy.select_action = _sel

_orig_reset = SmolVLAPolicy.reset
def _reset(self):
    _cache.clear()  # never reuse across episodes
    return _orig_reset(self)
SmolVLAPolicy.reset = _reset



CKPT=sys.argv[1]; N_EP=sys.argv[2]; NSTEPS=int(sys.argv[3]); TIDS=sys.argv[4] if len(sys.argv)>4 else None
sd=torch.load(CKPT,map_location="cpu"); RANK=sd["rank"]

class LoRALinear(nn.Module):
    def __init__(self, base, A, B, alpha=16.0):
        super().__init__(); self.base=base
        self.A=nn.Parameter(A); self.B=nn.Parameter(B); self.scale=alpha/A.shape[0]
    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias
    def forward(self,x): return self.base(x)+F.linear(F.linear(x,self.A),self.B)*self.scale

_orig=SmolVLAPolicy.from_pretrained.__func__
def _fp(cls,*a,**kw):
    policy=_orig(cls,*a,**kw); m=policy.model
    params=[t for t in sd["lora"]]; i=0
    dev=next(m.parameters()).device
    for layer in m.vlm_with_expert.lm_expert.layers:
        for parent,attr in [(layer.self_attn,"q_proj"),(layer.self_attn,"k_proj"),
                            (layer.self_attn,"v_proj"),(layer.self_attn,"o_proj"),
                            (layer.mlp,"gate_proj"),(layer.mlp,"up_proj"),(layer.mlp,"down_proj")]:
            base=getattr(parent,attr,None)
            if isinstance(base,nn.Linear):
                A=params[i].to(dev); B=params[i+1].to(dev); i+=2
                setattr(parent,attr,LoRALinear(base,A,B))
    policy.config.num_steps=NSTEPS
    print(f"[lora] {i//2} adapters loaded (train step {sd.get('step')}), num_steps={NSTEPS}")
    return policy
SmolVLAPolicy.from_pretrained=classmethod(_fp)

argv=["lerobot_eval","--policy.path=HuggingFaceVLA/smolvla_libero",
 "--env.type=libero","--env.task=libero_spatial",
 f"--eval.n_episodes={N_EP}","--eval.batch_size=1",
 f"--output_dir=artifacts/eval_stack"]
if TIDS: argv.insert(4, f"--env.task_ids={TIDS}")
sys.argv=argv
LE.main()
tot=STATS["encodes"]+STATS["skips"]
print(f"STACK REUSE tau=0.05: skips={STATS['skips']} rate={100*STATS['skips']/max(tot,1):.1f}%")
print("STACK EVAL DONE")
