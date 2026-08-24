#!/usr/bin/env python3
"""Confirm the two-engine chain reproduces eager sample_actions before exporting."""
import os, sys
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE","1")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from engines import PrefixEngine, DenoiseEngine, euler_chain

dev="cuda"; torch.manual_seed(0)
p=SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(dev).eval()
c=p.config; m=p.model
B=1
images=torch.rand(B,3,3,512,512,device=dev)
img_masks=torch.ones(B,3,device=dev,dtype=torch.bool)
lang_tokens=torch.randint(0,32000,(B,48),device=dev)
lang_masks=torch.ones(B,48,device=dev,dtype=torch.bool)
state=torch.randn(B,32,device=dev)
noise=torch.randn(B,c.chunk_size,c.max_action_dim,device=dev)

pe_eng=PrefixEngine(m).eval(); de_eng=DenoiseEngine(m).eval()
with torch.no_grad():
    # reference: eager sample_actions with the SAME noise
    imgs=[images[:,i] for i in range(3)]; masks=[img_masks[:,i] for i in range(3)]
    ref=m.sample_actions(imgs,masks,lang_tokens,lang_masks,state,noise=noise)
    # two-engine chain
    got=euler_chain(pe_eng,de_eng,images,img_masks,lang_tokens,lang_masks,state,noise,c.num_steps)
maxerr=(ref-got).abs().max().item()
meanerr=(ref-got).abs().mean().item()
print(f"ref shape {tuple(ref.shape)}  chain shape {tuple(got.shape)}")
print(f"max abs err: {maxerr:.3e}   mean abs err: {meanerr:.3e}")
print("MATCH" if maxerr < 1e-3 else "MISMATCH")
