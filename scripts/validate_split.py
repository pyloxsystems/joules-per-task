#!/usr/bin/env python3
"""Validate vision.onnx -> prefill.onnx -> 10x denoise.onnx chain vs eager (eager runs fp16 weights on GB10)."""
import os, sys
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE","1")
sys.path.insert(0, os.path.dirname(__file__))
import engines  # fp32-sincos patch
import numpy as np, torch, onnxruntime as ort
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

dev="cuda"; torch.manual_seed(0)
p=SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(dev).eval()
p=p.half()
c=p.config; m=p.model
B=1
images=torch.rand(B,3,3,512,512,device=dev)
img_masks=torch.ones(B,3,device=dev,dtype=torch.bool)
lang_tokens=torch.randint(0,32000,(B,48),device=dev)
lang_masks=torch.ones(B,48,device=dev,dtype=torch.bool)
state=torch.randn(B,32,device=dev)
noise=torch.randn(B,c.chunk_size,c.max_action_dim,device=dev)

with torch.no_grad():
    imgs=[images[:,i].half() for i in range(3)]; masks=[img_masks[:,i] for i in range(3)]
    ref=m.sample_actions(imgs,masks,lang_tokens,lang_masks,state.half(),noise=noise.half()).float().cpu().numpy()

so=ort.SessionOptions(); prov=["CPUExecutionProvider"]
sv=ort.InferenceSession("artifacts/vision_edge.onnx",so,providers=prov)
sp=ort.InferenceSession("artifacts/prefill_edge.onnx",so,providers=prov)
sb=ort.InferenceSession("artifacts/denoise_edge.onnx",so,providers=prov)
n=lambda t: t.detach().cpu().numpy()
ie = sv.run(None, {"images":n(images).astype(np.float32)})[0]
Ka,Va,ppm = sp.run(None, {"img_embs":ie,"img_masks":n(img_masks),
    "lang_tokens":n(lang_tokens).astype(np.int64),"lang_masks":n(lang_masks),"state":n(state).astype(np.float32)})
num=c.num_steps; dt=-1.0/num; x=n(noise).astype(np.float32)
for step in range(num):
    t=np.float32(1.0+step*dt); tt=np.array([t],dtype=np.float32)
    v=sb.run(None,{"x_t":x,"timestep":tt,"prefix_pad_masks":ppm.astype(np.float32),"K":Ka,"V":Va})[0]
    x=x+dt*v
err=np.abs(ref-x); print(f"split ONNX chain vs eager-fp16  max={err.max():.3e}  mean={err.mean():.3e}")
print("SPLIT MATCH" if err.max()<0.05 else "SPLIT MISMATCH")  # fp16 tolerance
