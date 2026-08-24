#!/usr/bin/env python3
"""Dump reference inputs + eager-fp16 outputs for on-Orin TRT validation."""
import os, sys
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE","1")
sys.path.insert(0, os.path.dirname(__file__))
import engines
import numpy as np, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

dev="cuda"; torch.manual_seed(0)
import os as _os
MODEL_ID=_os.environ.get("MODEL_ID","lerobot/smolvla_base")
p=SmolVLAPolicy.from_pretrained(MODEL_ID).to(dev).eval().float()
c=p.config; m=p.model
B=1
NCAM=sum(1 for k in c.input_features if "image" in k)
images=torch.rand(B,NCAM,3,512,512,device=dev)
img_masks=torch.ones(B,NCAM,device=dev,dtype=torch.bool)
lang_tokens=torch.randint(0,32000,(B,48),device=dev)
lang_masks=torch.ones(B,48,device=dev,dtype=torch.bool)
state=torch.randn(B,32,device=dev)
noise=torch.randn(B,c.chunk_size,c.max_action_dim,device=dev)
with torch.no_grad():
    imgs=[images[:,i] for i in range(NCAM)]; masks=[img_masks[:,i] for i in range(NCAM)]
    ref=m.sample_actions(imgs,masks,lang_tokens,lang_masks,state,noise=noise).float().cpu().numpy()
np.savez("artifacts/reference_io.npz",
    images=images.cpu().numpy().astype(np.float32),
    img_masks=img_masks.cpu().numpy(),
    lang_tokens=lang_tokens.cpu().numpy().astype(np.int64),
    lang_masks=lang_masks.cpu().numpy(),
    state=state.cpu().numpy().astype(np.float32),
    noise=noise.cpu().numpy().astype(np.float32),
    ref_actions=ref)
print("reference dumped:", ref.shape, "action[0,:3,:6]:"); print(ref[0,:3,:6])
