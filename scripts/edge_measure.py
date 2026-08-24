#!/usr/bin/env python3
"""On-Orin: SmolVLA 3-engine TRT pipeline (vision -> prefill -> 10x denoise).

1) Validates the TRT chain against the GB10 fp32 eager reference (reference_io.npz).
2) Runs N chunks back-to-back sampling tegrastats VDD_IN -> joules per action-chunk.
Usage: python3 edge_measure.py [tag]   (expects vision_<tag>.plan etc. in cwd)
"""
import sys, time, subprocess, threading, re
import numpy as np
import tensorrt as trt
from cuda import cudart

TAG = sys.argv[1] if len(sys.argv) > 1 else "fp16"
NUM_STEPS = 10
N_CHUNKS = 100
CHUNK, ADIM = 50, 32

TRT_DT = {trt.float32: np.float32, trt.float16: np.float16, trt.int64: np.int64,
          trt.int32: np.int32, trt.bool: np.bool_}

def ck(err):
    if isinstance(err, tuple): err = err[0]
    if int(err) != 0: raise RuntimeError(f"CUDA error {err}")

def ck2(ret):
    err, ptr = ret
    if int(err) != 0: raise RuntimeError(f"cudaMalloc {err}")
    return ptr

class Engine:
    def __init__(self, path, refit_onnx=None):
        logger = trt.Logger(trt.Logger.ERROR)
        rt = trt.Runtime(logger)
        self.eng = rt.deserialize_cuda_engine(open(path, "rb").read())
        if refit_onnx:
            refitter = trt.Refitter(self.eng, logger)
            parser_refitter = trt.OnnxParserRefitter(refitter, logger)
            assert parser_refitter.refit_from_file(refit_onnx), f"refit from {refit_onnx} failed"
            assert refitter.refit_cuda_engine(), "refit_cuda_engine failed"
            print(f"refitted {path} from {refit_onnx}")
        self.ctx = self.eng.create_execution_context()
        self.names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.is_in = {n: self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in self.names}
        self.dt = {n: TRT_DT[self.eng.get_tensor_dtype(n)] for n in self.names}
        self.shape = {n: tuple(self.eng.get_tensor_shape(n)) for n in self.names}
        self.dev = {n: ck2(cudart.cudaMalloc(int(np.prod(self.shape[n])) * np.dtype(self.dt[n]).itemsize))
                    for n in self.names}
    def set_input(self, n, host_arr):
        a = np.ascontiguousarray(host_arr.astype(self.dt[n]))
        ck(cudart.cudaMemcpy(self.dev[n], a.ctypes.data, a.nbytes,
                             cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
    def run(self, stream):
        for n in self.names:
            self.ctx.set_tensor_address(n, int(self.dev[n]))
        self.ctx.execute_async_v3(stream)
    def get_output(self, n):
        out = np.empty(self.shape[n], self.dt[n])
        ck(cudart.cudaMemcpy(out.ctypes.data, self.dev[n], out.nbytes,
                             cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return out

class PowerSampler(threading.Thread):
    def __init__(self, interval_ms=10):
        super().__init__(daemon=True)
        self.interval = interval_ms; self.samples = []; self.stop_flag = False
    def run(self):
        p = subprocess.Popen(["tegrastats", "--interval", str(self.interval)],
                             stdout=subprocess.PIPE, text=True)
        self.proc = p
        for line in p.stdout:
            m = re.search(r"VDD_IN (\d+)mW", line)
            if m: self.samples.append((time.time(), int(m.group(1))))
            if self.stop_flag: break
        try: p.terminate()
        except Exception: pass
    def stop(self): self.stop_flag = True

def mean_between(samples, t0, t1):
    v = [mw for (t, mw) in samples if t0 <= t <= t1]
    return (float(np.mean(v)) if v else float("nan")), len(v)

def main():
    print("nvpmodel:", subprocess.getoutput("nvpmodel -q 2>/dev/null | tail -1").strip())
    Pa = Pb = None
    if len(sys.argv) > 5:  # split prefill: vision, prefillA, prefillB, denoise
        Vv = Engine(sys.argv[2]); Pa = Engine(sys.argv[3]); Pb = Engine(sys.argv[4]); Bb = Engine(sys.argv[5])
    elif len(sys.argv) > 4:
        Vv = Engine(sys.argv[2])
        Pp = Engine(sys.argv[3], refit_onnx=("prefill_edge.onnx" if "stripped" in sys.argv[3] else None))
        Bb = Engine(sys.argv[4])
    else:
        Vv = Engine(f"vision_{TAG}.plan"); Pp = Engine(f"prefill_{TAG}.plan"); Bb = Engine(f"denoise_{TAG}.plan")
    _, stream = cudart.cudaStreamCreate()
    ref = np.load("reference_io.npz")
    images = ref["images"]; img_masks = ref["img_masks"]
    lang_tokens = ref["lang_tokens"]; lang_masks = ref["lang_masks"]
    state = ref["state"]; noise = ref["noise"]; ref_actions = ref["ref_actions"]
    dt = -1.0 / NUM_STEPS

    def one_chunk(noise_arr):
        Vv.set_input("images", images)
        Vv.run(stream); cudart.cudaStreamSynchronize(stream)
        ie = Vv.get_output("img_embs")
        if Pa is not None:
            Pa.set_input("img_embs", ie); Pa.set_input("img_masks", img_masks)
            Pa.set_input("lang_tokens", lang_tokens); Pa.set_input("lang_masks", lang_masks)
            Pa.set_input("state", state)
            Pa.run(stream); cudart.cudaStreamSynchronize(stream)
            K1 = Pa.get_output("K"); V1 = Pa.get_output("V")
            ppm = Pa.get_output("prefix_pad_masks"); pam = Pa.get_output("prefix_att_masks")
            hidden = Pa.get_output("hidden")
            Pb.set_input("hidden", hidden); Pb.set_input("prefix_pad_masks", ppm)
            Pb.set_input("prefix_att_masks", pam)
            Pb.run(stream); cudart.cudaStreamSynchronize(stream)
            K2 = Pb.get_output("K"); V2 = Pb.get_output("V")
            K = np.concatenate([K1, K2], axis=0); Vt = np.concatenate([V1, V2], axis=0)
        else:
            Pp.set_input("img_embs", ie); Pp.set_input("img_masks", img_masks)
            Pp.set_input("lang_tokens", lang_tokens); Pp.set_input("lang_masks", lang_masks)
            Pp.set_input("state", state)
            Pp.run(stream); cudart.cudaStreamSynchronize(stream)
            K = Pp.get_output("K"); Vt = Pp.get_output("V"); ppm = Pp.get_output("prefix_pad_masks")
        Bb.set_input("prefix_pad_masks", ppm); Bb.set_input("K", K); Bb.set_input("V", Vt)
        x = noise_arr.astype(np.float32).copy()
        for step in range(NUM_STEPS):
            t = np.float32(1.0 + step * dt)
            Bb.set_input("x_t", x); Bb.set_input("timestep", np.array([t], np.float32))
            Bb.run(stream); cudart.cudaStreamSynchronize(stream)
            x = x + dt * Bb.get_output("v_t")
        return x

    # ---- validation vs GB10 fp32 eager ----
    acts = one_chunk(noise)
    err = np.abs(acts - ref_actions)
    print(f"TRT[{TAG}] vs eager-fp32: max={err.max():.4f}  mean={err.mean():.5f}")
    print("VALIDATION:", "PASS" if err.max() < 0.05 else "SUSPECT")

    for _ in range(3): one_chunk(noise)  # warmup

    ps = PowerSampler(); ps.start(); time.sleep(2.5)
    t0i = time.time(); time.sleep(2.0); t1i = time.time()
    idle_mw, nidle = mean_between(ps.samples, t0i, t1i)

    lat = []
    t0 = time.time()
    for _ in range(N_CHUNKS):
        s = time.time(); one_chunk(noise); lat.append((time.time() - s) * 1000)
    t1 = time.time()
    time.sleep(0.3); ps.stop(); time.sleep(0.3)
    act_mw, nact = mean_between(ps.samples, t0, t1)

    lat = np.array(lat); lat_chunk = (t1 - t0) / N_CHUNKS
    jg = act_mw / 1000.0 * lat_chunk
    jn = (act_mw - idle_mw) / 1000.0 * lat_chunk
    print(f"\n===== SmolVLA 3-engine TRT [{TAG}] | Jetson Orin Nano =====")
    print(f"idle VDD_IN: {idle_mw:.0f} mW ({nidle} samples) | active: {act_mw:.0f} mW ({nact} samples)")
    print(f"latency/chunk: {lat_chunk*1000:.1f} ms (p50 {np.median(lat):.1f})  actions/chunk: {CHUNK}  denoise steps: {NUM_STEPS}")
    print(f"JOULES/CHUNK gross={jg:.3f} J  net={jn:.3f} J   |  {jg/CHUNK*1000:.1f} mJ/action gross")

if __name__ == "__main__":
    main()
