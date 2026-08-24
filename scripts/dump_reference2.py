#!/usr/bin/env python3
"""Dump eager-fp32 intermediates for per-stage TRT divergence hunting."""
import os, sys
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE","1")
sys.path.insert(0, os.path.dirname(__file__))
import engines
from engines import VisionEngine, PrefillEngine, DenoiseEngine
import numpy as np, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

dev="cuda"; torch.manual_seed(0)
import os as _os
MODEL_ID=_os.environ.get("MODEL_ID","lerobot/smolvla_base")
p=SmolVLAPolicy.from_pretrained(MODEL_ID).to(dev).eval().float()
c=p.config; m=p.model
r=np.load("artifacts/reference_io.npz")
images=torch.tensor(r["images"],device=dev); img_masks=torch.tensor(r["img_masks"],device=dev)
lang_tokens=torch.tensor(r["lang_tokens"],device=dev); lang_masks=torch.tensor(r["lang_masks"],device=dev)
state=torch.tensor(r["state"],device=dev); noise=torch.tensor(r["noise"],device=dev)
ve=VisionEngine(m).eval(); pf=PrefillEngine(m).eval(); de=DenoiseEngine(m).eval()
with torch.no_grad():
    ie=ve(images)
    K,V,ppm=pf(ie,img_masks,lang_tokens,lang_masks,state)
    tt=torch.tensor(1.0,dtype=torch.float32,device=dev).expand(1)
    v0=de(noise,tt,ppm,K,V)
np.savez("artifacts/reference_mid.npz",
    img_embs=ie.float().cpu().numpy(), K=K.float().cpu().numpy(), V=V.float().cpu().numpy(),
    ppm=ppm.float().cpu().numpy(), v0=v0.float().cpu().numpy())
print("mid dumped:", ie.shape, K.shape, v0.shape)
