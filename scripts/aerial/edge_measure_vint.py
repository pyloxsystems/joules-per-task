#!/usr/bin/env python3
"""On-Orin: ViNT 30M visual-nav energy. obs (1,18,64,85) + goal (1,3,64,85) -> dist + 5x4 actions.
Modes: baseline | reuse <tau> (gate the whole forward on obs-stack change; ViNT has no separable
vision/policy split — encoder IS most of the net, so the gate skips the full inference and
holds the last action plan). Usage: edge_measure_vint.py [baseline|reuse] [tau] [n]"""
import sys, os, time, subprocess, threading, re
import numpy as np, tensorrt as trt
from cuda import cudart
MODE=sys.argv[1] if len(sys.argv)>1 else "baseline"
TAU=float(sys.argv[2]) if len(sys.argv)>2 else 0.01
N=int(sys.argv[3]) if len(sys.argv)>3 else 500
D = _o.environ.get("JPT_ENGINE_DIR", "./engines/")  # dir holding the built .plan files
TRT_DT={trt.float32:np.float32,trt.float16:np.float16,trt.int64:np.int64,trt.int32:np.int32,trt.bool:np.bool_}
def ck(e):
    e=e[0] if isinstance(e,tuple) else e; assert int(e)==0
def ck2(r):
    e,p=r; assert int(e)==0; return p
class E:
    def __init__(s,path):
        rt=trt.Runtime(trt.Logger(trt.Logger.ERROR)); s.e=rt.deserialize_cuda_engine(open(path,'rb').read())
        s.c=s.e.create_execution_context(); s.n=[s.e.get_tensor_name(i) for i in range(s.e.num_io_tensors)]
        s.dt={n:TRT_DT[s.e.get_tensor_dtype(n)] for n in s.n}; s.sh={n:tuple(s.e.get_tensor_shape(n)) for n in s.n}
        s.dev={n:ck2(cudart.cudaMalloc(int(np.prod(s.sh[n]))*np.dtype(s.dt[n]).itemsize)) for n in s.n}
    def infer(s,feeds,st):
        for n,a in feeds.items():
            a=np.ascontiguousarray(a.astype(s.dt[n])); ck(cudart.cudaMemcpy(s.dev[n],a.ctypes.data,a.nbytes,cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        for n in s.n: s.c.set_tensor_address(n,int(s.dev[n]))
        s.c.execute_async_v3(st); cudart.cudaStreamSynchronize(st)
class P(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.samples=[]; s.stop=False
    def run(s):
        p=subprocess.Popen(["tegrastats","--interval","10"],stdout=subprocess.PIPE,text=True)
        for ln in p.stdout:
            m=re.search(r"VDD_IN (\d+)mW",ln)
            if m: s.samples.append((time.time(),int(m.group(1))))
            if s.stop: break
        p.terminate()
def mp(sm,a,b):
    v=[w for t,w in sm if a<=t<=b]; return float(np.mean(v)) if v else float('nan')
def main():
    M=E(D+"vint_fp16.plan"); _,st=cudart.cudaStreamCreate()
    rng=np.random.default_rng(0)
    base=rng.random((3,64,85),dtype=np.float32)
    goal=rng.random((1,3,64,85),dtype=np.float32)
    last=[None]; sk=[0]; en=[0]
    def frame(i):
        return np.clip(base+0.002*np.sin(i/7.0)*rng.standard_normal((3,64,85)).astype(np.float32),0,1)
    def step(i):
        f=frame(i)
        if MODE=="reuse" and last[0] is not None and float(np.mean(np.abs(f-last[0])))<TAU:
            sk[0]+=1; return
        obs=np.tile(f,(6,1,1))[None]
        M.infer({"obs_img":obs,"goal_img":goal},st)
        last[0]=f; en[0]+=1
    for i in range(5): step(i)
    last[0]=None; sk[0]=0; en[0]=0
    ps=P(); ps.start(); time.sleep(2.5); a0=time.time(); time.sleep(2.0); a1=time.time(); idle=mp(ps.samples,a0,a1)
    t0=time.time()
    for i in range(N): step(i)
    t1=time.time(); time.sleep(0.3); ps.stop=True; time.sleep(0.2); act=mp(ps.samples,t0,t1)
    per=(t1-t0)/N
    print(f"===== ViNT 30M | Orin Nano | mode={MODE} tau={TAU} =====")
    print(f"idle {idle:.0f} mW | active {act:.0f} mW | latency/step {per*1000:.2f} ms ({1000/(per*1000):.0f} Hz)")
    print(f"encodes={en[0]} skips={sk[0]} skip-rate={100*sk[0]/max(en[0]+sk[0],1):.1f}%")
    print(f"JOULES/STEP gross={act/1000*per*1000:.2f} mJ  net={(act-idle)/1000*per*1000:.2f} mJ")
if __name__=="__main__": main()
