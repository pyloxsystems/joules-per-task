import os
#!/usr/bin/env python3
"""GB10 (workstation tier): energy per action-prediction for OpenFly-7B (OpenVLA drone VLA).
Loads the 7B model, feeds image(224)+prompt, times generate(), samples nvidia-smi power.draw
around a run of N inferences -> joules per inference. The heavy end of the JPT spectrum, and a
second hardware class (the 7B model CANNOT run on the 8GB Orin -> that's the edge-boundary finding)."""
import sys, time, subprocess, threading, re
import numpy as np, torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL = os.environ.get("OPENFLY_DIR", "./openfly")
N=int(sys.argv[1]) if len(sys.argv)>1 else 40
DEV="cuda"

class GPUPow(threading.Thread):
    def __init__(s,ms=50): super().__init__(daemon=True); s.i=ms/1000; s.samples=[]; s.stop=False
    def run(s):
        while not s.stop:
            try:
                w=float(subprocess.check_output(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],text=True).strip().split("\n")[0])
                s.samples.append((time.time(),w))
            except Exception: pass
            time.sleep(s.i)
def mp(sm,a,b):
    v=[w for t,w in sm if a<=t<=b]; return float(np.mean(v)) if v else float("nan")

print("loading OpenFly-7B ...")
proc=AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model=AutoModelForVision2Seq.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    low_cpu_mem_usage=True).to(DEV).eval()
nparams=sum(p.numel() for p in model.parameters())/1e9
print(f"loaded {nparams:.1f}B params")
img=Image.fromarray((np.random.rand(224,224,3)*255).astype(np.uint8))
prompt="In: What action should the drone take to reach the goal?\nOut:"
inputs=proc(prompt, img).to(DEV, dtype=torch.bfloat16)

def one():
    with torch.no_grad():
        _=model.generate(**inputs, max_new_tokens=7, do_sample=False)
    torch.cuda.synchronize()

for _ in range(3): one()  # warmup
ps=GPUPow(); ps.start(); time.sleep(2)
i0=time.time(); time.sleep(2); i1=time.time(); idle=mp(ps.samples,i0,i1)
lat=[]; t0=time.time()
for _ in range(N):
    s=time.time(); one(); lat.append((time.time()-s)*1000)
t1=time.time(); time.sleep(0.3); ps.stop=True
act=mp(ps.samples,t0,t1); per=(t1-t0)/N
jg=act*per; jn=(act-idle)*per
lat=np.array(lat)
print(f"\n===== OpenFly-7B | GB10 workstation | bf16 =====")
print(f"idle {idle:.0f} W | active {act:.0f} W | latency/inference {per*1000:.0f} ms (p50 {np.median(lat):.0f}) = {1/per:.1f} Hz")
print(f"JOULES/INFERENCE gross={jg:.2f} J  net={jn:.2f} J   ({nparams:.1f}B params)")
print(f"NOTE: 7B bf16 = ~{nparams*2:.0f}GB, exceeds 8GB Orin -> cannot run at edge tier (finding).")
