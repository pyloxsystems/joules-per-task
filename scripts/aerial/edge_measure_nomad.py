#!/usr/bin/env python3
"""On-Orin: NoMaD (19M diffusion visual-nav) energy. Chain: vision(obs 4-frame stack 96x96 + goal
+ mask) -> obs_cond(256); 10x DDPM denoise (ConditionalUnet1D) -> 8x2 trajectory.
Modes: baseline | reuse <tau> (skip vision when frame stack ~unchanged; drone flight frames are
temporally redundant). NSTEPS env overrides denoise iters (step-count energy lever).
Usage: python3 edge_measure_nomad.py [baseline|reuse] [tau] [n_steps_replay]"""
import sys, os, time, subprocess, threading, re
import numpy as np, tensorrt as trt
from cuda import cudart
MODE=sys.argv[1] if len(sys.argv)>1 else "baseline"
TAU=float(sys.argv[2]) if len(sys.argv)>2 else 0.01
N=int(sys.argv[3]) if len(sys.argv)>3 else 300
ITERS=int(os.environ.get("NSTEPS","10"))
REPLAY=os.environ.get("REPLAY")
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
        return {n:(lambda o=np.empty(s.sh[n],s.dt[n]): (ck(cudart.cudaMemcpy(o.ctypes.data,s.dev[n],o.nbytes,cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)),o)[1])() for n in s.n if n not in feeds}
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

sc=np.load(D+"ddpm_consts.npz")
TS=sc["timesteps"].astype(np.int64); AC=sc["alphas_cumprod"]; BE=sc["betas"]; AL=sc["alphas"]
def ddpm_step(eps,t,x,rng):
    a_t=AL[t]; ac_t=AC[t]; ac_prev=AC[t-1] if t>0 else 1.0; b_t=BE[t]
    x0=(x-np.sqrt(1-ac_t)*eps)/np.sqrt(ac_t)
    x0=np.clip(x0,-1,1)
    coef_x0=np.sqrt(ac_prev)*b_t/(1-ac_t); coef_xt=np.sqrt(a_t)*(1-ac_prev)/(1-ac_t)
    mean=coef_x0*x0+coef_xt*x
    if t>0:
        var=b_t*(1-ac_prev)/(1-ac_t)
        mean=mean+np.sqrt(max(var,1e-20))*rng.standard_normal(x.shape).astype(np.float32)
    return mean.astype(np.float32)

def main():
    V=E(D+"nomad_vision_fp16.plan"); U=E(D+"nomad_noise_fp16.plan")
    _,st=cudart.cudaStreamCreate()
    rng=np.random.default_rng(0)
    if REPLAY:
        FR=np.load(REPLAY)["frames"]  # (T,3,96,96) real footage
        print(f"REAL replay stream: {FR.shape[0]} frames from {REPLAY}")
        def frame(i): return FR[i%len(FR)]
    else:
        base=rng.random((3,96,96),dtype=np.float32)
        def frame(i):
            drift=0.002*np.sin(i/7.0)
            return np.clip(base+drift*rng.standard_normal((3,96,96)).astype(np.float32),0,1)
    goal=rng.random((1,3,96,96),dtype=np.float32); mask=np.ones((1,),np.int64)
    cached=[None]; last=[None]; sk=[0]; en=[0]
    def step(i):
        f=frame(i)
        if REPLAY:
            idx=[max(0,i-3),max(0,i-2),max(0,i-1),i]
            obs=np.concatenate([frame(j) for j in idx],axis=0)[None]  # real sliding stack
        else:
            obs=np.tile(f,(4,1,1))[None]  # (1,12,96,96)
        if MODE=="reuse" and cached[0] is not None and float(np.mean(np.abs(f-last[0])))<TAU:
            cond=cached[0]; sk[0]+=1
        else:
            cond=list(V.infer({"obs_img":obs,"goal_img":goal,"goal_mask":mask},st).values())[0]
            cached[0]=cond; last[0]=f; en[0]+=1
        x=rng.standard_normal((1,8,2)).astype(np.float32)
        ts_used=TS[-ITERS:] if ITERS<len(TS) else TS
        for t in ts_used:
            eps=U.infer({"naction":x,"timestep":np.array([t],np.int64),"obs_cond":cond},st)["noise"]
            x=ddpm_step(eps,int(t),x,rng)
        return x
    for i in range(5): step(i)
    cached[0]=None; last[0]=None; sk[0]=0; en[0]=0
    ps=P(); ps.start(); time.sleep(2.5); a0=time.time(); time.sleep(2.0); a1=time.time(); idle=mp(ps.samples,a0,a1)
    t0=time.time()
    for i in range(N): step(i)
    t1=time.time(); time.sleep(0.3); ps.stop=True; time.sleep(0.2); act=mp(ps.samples,t0,t1)
    per=(t1-t0)/N
    print(f"===== NoMaD 19M | Orin Nano | mode={MODE} tau={TAU} iters={ITERS} =====")
    print(f"idle {idle:.0f} mW | active {act:.0f} mW")
    print(f"latency/step {per*1000:.2f} ms ({1000/(per*1000):.0f} Hz)")
    print(f"vision: encodes={en[0]} skips={sk[0]} skip-rate={100*sk[0]/max(en[0]+sk[0],1):.1f}%")
    print(f"JOULES/STEP gross={act/1000*per*1000:.2f} mJ  net={(act-idle)/1000*per*1000:.2f} mJ")
if __name__=="__main__": main()
