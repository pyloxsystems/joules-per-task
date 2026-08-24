#!/usr/bin/env python3
"""On-Orin: replay real LIBERO episode observations through the TRT pipeline.

Modes:
  baseline — full pipeline every step (vision both cams + prefill + 10x denoise)
  reuse    — per-camera diff-gated vision reuse (skip SigLIP when frame ~unchanged)

Reports joules/step + joules/stream and the per-camera skip rates so the honest
tradeoff (energy vs staleness threshold) is explicit.
Usage: python3 edge_replay.py <baseline|reuse> [diff_threshold]
"""
import sys, time, subprocess, threading, re
import numpy as np
import tensorrt as trt
from cuda import cudart

MODE = sys.argv[1] if len(sys.argv) > 1 else "baseline"
TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01  # mean-abs pixel diff gate
import os as _o
NUM_STEPS = int(_o.environ.get("NSTEPS","10"))
DPLAN = _o.environ.get("DPLAN","denoise_fp16.plan")
REPEATS = 3  # loop the stream a few times for stable power stats

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
    def __init__(self, path):
        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.eng = rt.deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.eng.create_execution_context()
        self.names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
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
    d = "/home/pyloxsystems/jpt_edge/libero/"
    V1 = Engine(d + "vision1_fp16.plan")
    INCR = _o.environ.get("INCR") == "1"
    Sa = Engine(d + "stateA_fp32.plan") if INCR else None
    Sb = Engine(d + "stateB_fp32.plan") if INCR else None
    Pa = Engine(d + "prefillA_fp32.plan"); Pb = Engine(d + "prefillB_fp32.plan")
    Bb = Engine(d + DPLAN)
    _, stream = cudart.cudaStreamCreate()

    S = np.load(d + "replay_stream.npz")
    cam1 = S["observation__images__image"].astype(np.float32)      # (T,1,3,256,256)
    cam2 = S["observation__images__image2"].astype(np.float32)
    states = S["observation__state"].astype(np.float32)            # (T,1,8)
    lang_tok = S["observation__language__tokens"] if "observation__language__tokens" in S.files else None
    lang_msk = S["observation__language__attention_mask"] if "observation__language__attention_mask" in S.files else None
    T = cam1.shape[0]
    print(f"replay stream: {T} steps  mode={MODE} tau={TAU}")

    # policy preprocessing: images arrive 256x256; TRT vision engine wants 512x512 (policy resizes internally on GB10;
    # here we replicate with simple resize)
    import cv2
    def prep(im):  # (1,3,256,256) -> (1,3,512,512)
        x = im[0].transpose(1, 2, 0)
        x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)
        return x.transpose(2, 0, 1)[None]

    if lang_tok is None:
        lang_tokens = np.ones((1, 48), np.int64); lang_masks = np.ones((1, 48), bool)
    else:
        lt = lang_tok[0].astype(np.int64).reshape(1, -1)
        lm = lang_msk[0].astype(bool).reshape(1, -1)
        L = 48  # engine static lang length: pad with masked tokens (mirrors policy padding)
        lang_tokens = np.zeros((1, L), np.int64); lang_tokens[:, :lt.shape[1]] = lt
        lang_masks = np.zeros((1, L), bool); lang_masks[:, :lm.shape[1]] = lm
    img_masks = np.ones((1, 2), bool)

    # pad state 8 -> 32
    def pad_state(s):
        z = np.zeros((1, 32), np.float32); z[:, :s.shape[-1]] = s.reshape(1, -1); return z

    cached_emb = [None, None]
    last_frame = [None, None]
    skips = [0, 0]; encodes = [0, 0]
    cached_kv = {}; incr_hits = [0]
    rng = np.random.default_rng(0)

    def encode_cam(frame512):
        V1.set_input("images", frame512[None])   # (1,1,3,512,512)
        V1.run(stream); cudart.cudaStreamSynchronize(stream)
        return V1.get_output("img_embs").copy()  # (1,1,64,960)

    def one_step(i):
        frames = [prep(cam1[i]), prep(cam2[i])]
        embs = []
        for c in (0, 1):
            f = frames[c]
            if MODE == "reuse" and cached_emb[c] is not None:
                diff = float(np.mean(np.abs(f - last_frame[c])))
                if diff < TAU:
                    skips[c] += 1
                    embs.append(cached_emb[c]); continue
            e = encode_cam(f[0])
            cached_emb[c] = e; last_frame[c] = f; encodes[c] += 1
            embs.append(e)
        all_skipped = (len(cached_kv) > 0) and all(
            (MODE == "reuse" and cached_emb[c] is not None and embs[c] is cached_emb[c]) for c in (0, 1))
        if INCR and all_skipped:
            K176, V176, ppm, pam = cached_kv["K176"], cached_kv["V176"], cached_kv["ppm"], cached_kv["pam"]
            Sa.set_input("state", pad_state(states[i]))
            Sa.set_input("K", K176[:14]); Sa.set_input("V", V176[:14]); Sa.set_input("prefix_pad_masks", ppm)
            Sa.run(stream); cudart.cudaStreamSynchronize(stream)
            Ksa = Sa.get_output("Ks"); Vsa = Sa.get_output("Vs"); hid = Sa.get_output("hidden")
            Sb.set_input("hidden", hid); Sb.set_input("K", K176[14:]); Sb.set_input("V", V176[14:])
            Sb.set_input("prefix_pad_masks", ppm)
            Sb.run(stream); cudart.cudaStreamSynchronize(stream)
            Ksb = Sb.get_output("Ks"); Vsb = Sb.get_output("Vs")
            srow_K = np.concatenate([Ksa, Ksb], 0); srow_V = np.concatenate([Vsa, Vsb], 0)
            K = np.concatenate([K176, srow_K], axis=3)
            Vt = np.concatenate([V176, srow_V], axis=3)
            incr_hits[0] += 1
        else:
            ie = np.concatenate(embs, axis=1)  # (1,2,64,960)
            Pa.set_input("img_embs", ie); Pa.set_input("img_masks", img_masks)
            Pa.set_input("lang_tokens", lang_tokens); Pa.set_input("lang_masks", lang_masks)
            Pa.set_input("state", pad_state(states[i]))
            Pa.run(stream); cudart.cudaStreamSynchronize(stream)
            K1 = Pa.get_output("K"); Vv1 = Pa.get_output("V")
            ppm = Pa.get_output("prefix_pad_masks"); pam = Pa.get_output("prefix_att_masks")
            hidden = Pa.get_output("hidden")
            Pb.set_input("hidden", hidden); Pb.set_input("prefix_pad_masks", ppm)
            Pb.set_input("prefix_att_masks", pam)
            Pb.run(stream); cudart.cudaStreamSynchronize(stream)
            K = np.concatenate([K1, Pb.get_output("K")], 0)
            Vt = np.concatenate([Vv1, Pb.get_output("V")], 0)
            if INCR:
                cached_kv["K176"] = K[:, :, :, :176, :].copy()
                cached_kv["V176"] = Vt[:, :, :, :176, :].copy()
                cached_kv["ppm"] = ppm; cached_kv["pam"] = pam
        Bb.set_input("prefix_pad_masks", ppm); Bb.set_input("K", K); Bb.set_input("V", Vt)
        x = rng.standard_normal((1, 50, 32)).astype(np.float32)
        dt = -1.0 / NUM_STEPS
        for s in range(NUM_STEPS):
            t = np.float32(1.0 + s * dt)
            Bb.set_input("x_t", x); Bb.set_input("timestep", np.array([t], np.float32))
            Bb.run(stream); cudart.cudaStreamSynchronize(stream)
            x = x + dt * Bb.get_output("v_t")
        return x

    # warmup
    for i in range(3): one_step(i)
    cached_emb[:] = [None, None]; last_frame[:] = [None, None]
    skips[:] = [0, 0]; encodes[:] = [0, 0]

    ps = PowerSampler(); ps.start(); time.sleep(2.5)
    t0i = time.time(); time.sleep(2.0); t1i = time.time()
    idle_mw, _ = mean_between(ps.samples, t0i, t1i)

    t0 = time.time(); n = 0
    for r in range(REPEATS):
        cached_emb[:] = [None, None]; last_frame[:] = [None, None]; cached_kv.clear(); incr_hits[0] = 0
        for i in range(T):
            one_step(i); n += 1
    t1 = time.time()
    time.sleep(0.3); ps.stop(); time.sleep(0.3)
    act_mw, nact = mean_between(ps.samples, t0, t1)

    lat = (t1 - t0) / n
    jg = act_mw / 1000.0 * lat; jn = (act_mw - idle_mw) / 1000.0 * lat
    tot_e = sum(encodes); tot_s = sum(skips)
    print(f"\n===== REPLAY [{MODE} tau={TAU}] {n} steps ({REPEATS}x{T}) =====")
    print(f"idle {idle_mw:.0f} mW | active {act_mw:.0f} mW ({nact} samples)")
    print(f"latency/step {lat*1000:.1f} ms")
    print(f"vision: encodes={tot_e} skips={tot_s} skip-rate={100*tot_s/max(tot_e+tot_s,1):.1f}%  (cam1 {skips[0]}/{skips[0]+encodes[0]}, cam2 {skips[1]}/{skips[1]+encodes[1]})")
    print(f"incremental-prefill hits: {incr_hits[0]}/{T} last-repeat" if INCR else "incremental: off")
    print(f"JOULES/STEP gross={jg:.3f} net={jn:.3f}")

if __name__ == "__main__":
    main()
