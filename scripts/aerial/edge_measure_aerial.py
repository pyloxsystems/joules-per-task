#!/usr/bin/env python3
"""On-Orin: JPT-Aerial per-control-step energy for the DCE drone nav stack.
Chain per step: depth(1,1,270,480) -> VAE encoder -> 64 latent ; [latent|17 state] -> 81 obs
-> policy (GRU, carries hidden) -> action(3). Integrates tegrastats VDD_IN -> joules/step.
Fixed-size conv+MLP+GRU => per-step compute energy is input-distribution-independent, so a
representative-magnitude synthetic depth stream gives a faithful BASELINE number; the reuse
savings + joules-per-mission require recorded sim observations (separate track).
Usage: python3 edge_measure_aerial.py [n_steps]
"""
import sys, time, subprocess, threading, re
import numpy as np
import tensorrt as trt
from cuda import cudart

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
D = _o.environ.get("JPT_ENGINE_DIR", "./engines/")  # dir holding the built .plan files
STATE_DIM = 17  # 81 obs - 64 latent
TRT_DT = {trt.float32: np.float32, trt.float16: np.float16, trt.int64: np.int64,
          trt.int32: np.int32, trt.bool: np.bool_}

def ck(e):
    if isinstance(e, tuple): e = e[0]
    if int(e) != 0: raise RuntimeError(f"CUDA {e}")
def ck2(r):
    e, p = r
    if int(e) != 0: raise RuntimeError(f"malloc {e}")
    return p

class Engine:
    def __init__(self, path):
        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.e = rt.deserialize_cuda_engine(open(path, "rb").read())
        self.c = self.e.create_execution_context()
        self.names = [self.e.get_tensor_name(i) for i in range(self.e.num_io_tensors)]
        self.dt = {n: TRT_DT[self.e.get_tensor_dtype(n)] for n in self.names}
        self.sh = {n: tuple(self.e.get_tensor_shape(n)) for n in self.names}
        self.dev = {n: ck2(cudart.cudaMalloc(int(np.prod(self.sh[n])) * np.dtype(self.dt[n]).itemsize))
                    for n in self.names}
    def infer(self, feeds, stream):
        for n, a in feeds.items():
            a = np.ascontiguousarray(a.astype(self.dt[n]))
            ck(cudart.cudaMemcpy(self.dev[n], a.ctypes.data, a.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        for n in self.names:
            self.c.set_tensor_address(n, int(self.dev[n]))
        self.c.execute_async_v3(stream); cudart.cudaStreamSynchronize(stream)
        outs = {}
        for n in self.names:
            if n not in feeds:
                o = np.empty(self.sh[n], self.dt[n])
                ck(cudart.cudaMemcpy(o.ctypes.data, self.dev[n], o.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
                outs[n] = o
        return outs

class Power(threading.Thread):
    def __init__(s, ms=10):
        super().__init__(daemon=True); s.i = ms; s.samples = []; s.stop = False
    def run(s):
        p = subprocess.Popen(["tegrastats", "--interval", str(s.i)], stdout=subprocess.PIPE, text=True)
        for ln in p.stdout:
            m = re.search(r"VDD_IN (\d+)mW", ln)
            if m: s.samples.append((time.time(), int(m.group(1))))
            if s.stop: break
        try: p.terminate()
        except Exception: pass

def meanp(sm, a, b):
    v = [w for t, w in sm if a <= t <= b]
    return (float(np.mean(v)) if v else float("nan")), len(v)

def main():
    print("nvpmodel:", subprocess.getoutput("nvpmodel -q 2>/dev/null | tail -1").strip())
    V = Engine(D + "depth_encoder_fp16.plan")
    P = Engine(D + "policy_fp16.plan")
    _, stream = cudart.cudaStreamCreate()
    dname = V.names[0]; pin = [n for n in P.names]
    rng = np.random.default_rng(0)
    # representative depth: normalized [0,1] range as the VAE expects
    depth = rng.random((1, 1, 270, 480), dtype=np.float32)
    state = rng.standard_normal((1, STATE_DIM)).astype(np.float32)
    hid = np.zeros((1, 64), np.float32)

    def step():
        nonlocal hid
        z = V.infer({dname: depth}, stream)          # -> latent
        lat = list(z.values())[0].reshape(1, -1)     # (1,64)
        obs = np.concatenate([lat, state], axis=1)   # (1,81)
        out = P.infer({"obs": obs, "rnn": hid}, stream)
        hid = out["new_rnn"]
        return out["action"]

    for _ in range(5): step()  # warmup
    ps = Power(); ps.start(); time.sleep(2.5)
    ti0 = time.time(); time.sleep(2.0); ti1 = time.time()
    idle, ni = meanp(ps.samples, ti0, ti1)
    lat = []
    t0 = time.time()
    for _ in range(N):
        s = time.time(); step(); lat.append((time.time() - s) * 1000)
    t1 = time.time()
    time.sleep(0.3); ps.stop = True; time.sleep(0.3)
    act, na = meanp(ps.samples, t0, t1)
    lat = np.array(lat); per = (t1 - t0) / N
    jg = act / 1000 * per; jn = (act - idle) / 1000 * per
    print(f"\n===== JPT-Aerial | DCE nav stack | Jetson Orin Nano =====")
    print(f"idle VDD_IN {idle:.0f} mW ({ni}) | active {act:.0f} mW ({na})")
    print(f"latency/step {per*1000:.2f} ms  ({1000/(per*1000):.0f} Hz)  p50 {np.median(lat):.2f} ms")
    print(f"JOULES/STEP gross={jg*1000:.2f} mJ  net={jn*1000:.2f} mJ")

if __name__ == "__main__":
    main()
