#!/usr/bin/env python3
"""Run prefix.onnx + denoise.onnx (Euler loop) via ORT-CPU, compare to eager sample_actions."""
import os, sys
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE","1")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, onnxruntime as ort
import engines  # applies fp32 sincos patch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

dev="cuda"; torch.manual_seed(0)
p=SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(dev).eval()
p=p.float()
c=p.config; m=p.model
B=1
images=torch.rand(B,3,3,512,512,device=dev)
img_masks=torch.ones(B,3,device=dev,dtype=torch.bool)
lang_tokens=torch.randint(0,32000,(B,48),device=dev)
lang_masks=torch.ones(B,48,device=dev,dtype=torch.bool)
state=torch.randn(B,32,device=dev)
noise=torch.randn(B,c.chunk_size,c.max_action_dim,device=dev)

with torch.no_grad():
    imgs=[images[:,i] for i in range(3)]; masks=[img_masks[:,i] for i in range(3)]
    ref=m.sample_actions(imgs,masks,lang_tokens,lang_masks,state,noise=noise).cpu().numpy()

so=ort.SessionOptions(); prov=["CPUExecutionProvider"]
sa=ort.InferenceSession("artifacts/prefix_edge.onnx",so,providers=prov)
sb=ort.InferenceSession("artifacts/denoise_edge.onnx",so,providers=prov)
n=lambda t: t.detach().cpu().numpy()
Ka,Va,ppm = sa.run(None, {"images":n(images).astype(np.float32),"img_masks":n(img_masks),
    "lang_tokens":n(lang_tokens).astype(np.int64),"lang_masks":n(lang_masks),"state":n(state).astype(np.float32)})
num=c.num_steps; dt=-1.0/num; x=n(noise).astype(np.float32)
for step in range(num):
    t=np.float32(1.0+step*dt); tt=np.array([t],dtype=np.float32)
    v=sb.run(None,{"x_t":x,"timestep":tt,"prefix_pad_masks":ppm.astype(np.float32),"K":Ka,"V":Va})[0]
    x=x+dt*v
err=np.abs(ref-x); print(f"ONNX chain vs eager  max={err.max():.3e}  mean={err.mean():.3e}")
print("ONNX MATCH" if err.max()<1e-2 else "ONNX MISMATCH")
