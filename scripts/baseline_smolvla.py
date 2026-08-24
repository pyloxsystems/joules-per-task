#!/usr/bin/env python3
"""Load SmolVLA, inspect its I/O spec, run a timed forward pass = first baseline (GB10 sanity)."""
import os
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1+PTX")
import time, torch, numpy as np
for name in ["_jit_set_nvfuser_enabled", "_jit_set_texpr_fuser_enabled",
             "_jit_set_profiling_executor", "_jit_set_profiling_mode"]:
    fn = getattr(torch._C, name, None)
    if fn:
        try: fn(False)
        except Exception: pass

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("loading lerobot/smolvla_base ...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.to(dev).eval()

nparams = sum(p.numel() for p in policy.parameters())
print(f"SmolVLA params: {nparams/1e6:.1f}M  device={dev}")
cfg = policy.config
print("input_features:")
for k, v in cfg.input_features.items():
    print(f"   {k}: shape={getattr(v,'shape',None)} type={getattr(v,'type',None)}")
print("output_features:")
for k, v in cfg.output_features.items():
    print(f"   {k}: shape={getattr(v,'shape',None)}")

# build a dummy observation batch matching the input features
batch = {}
for k, v in cfg.input_features.items():
    shp = tuple(v.shape)
    batch[k] = torch.zeros((1, *shp), device=dev, dtype=torch.float32)

# tokenize the language instruction the way SmolVLA expects
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
tok = policy.model.vlm_with_expert.processor.tokenizer
maxlen = getattr(cfg, "tokenizer_max_length", 48)
enc = tok("pick up the cube", padding="max_length", max_length=maxlen,
          truncation=True, return_tensors="pt")
batch[OBS_LANGUAGE_TOKENS] = enc["input_ids"].to(dev)
batch[OBS_LANGUAGE_ATTENTION_MASK] = enc["attention_mask"].to(dev).bool()
print(f"language tokens: {tuple(enc['input_ids'].shape)}")
policy.reset()

# warmup
with torch.no_grad():
    for _ in range(3):
        try:
            a = policy.select_action(batch)
        except Exception as e:
            print("select_action needs adjusted batch:", repr(e)[:300]); raise
torch.cuda.synchronize() if dev == "cuda" else None

# time
lat = []
with torch.no_grad():
    for _ in range(30):
        t0 = time.time()
        a = policy.select_action(batch)
        torch.cuda.synchronize() if dev == "cuda" else None
        lat.append((time.time() - t0) * 1000)
lat = np.array(lat)
print(f"\n=== BASELINE (GB10, fp32) ===")
print(f"action shape: {tuple(a.shape)}")
print(f"latency/action: mean={lat.mean():.1f}ms  p50={np.median(lat):.1f}ms  -> {1000/lat.mean():.1f} actions/s")

# --- the number that actually costs energy: the full chunk forward ---
n_steps = getattr(cfg, "n_action_steps", None)
chunk = getattr(cfg, "chunk_size", None)
print(f"\nn_action_steps={n_steps}  chunk_size={chunk}")
clat = []
with torch.no_grad():
    for _ in range(20):
        policy.reset()  # empties queue -> forces a full chunk inference
        t0 = time.time()
        _ = policy.select_action(batch)
        torch.cuda.synchronize() if dev == "cuda" else None
        clat.append((time.time() - t0) * 1000)
clat = np.array(clat)
print(f"=== FULL CHUNK FORWARD (GB10, fp32) ===")
print(f"chunk inference: mean={clat.mean():.1f}ms  p50={np.median(clat):.1f}ms")
if n_steps:
    print(f"amortized/action over {n_steps} steps: {clat.mean()/n_steps:.2f}ms  -> {1000*n_steps/clat.mean():.0f} actions/s effective")

# --- realistic serving dtype: bf16 autocast ---
blat = []
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(20):
        policy.reset()
        t0 = time.time()
        _ = policy.select_action(batch)
        torch.cuda.synchronize() if dev == "cuda" else None
        blat.append((time.time() - t0) * 1000)
blat = np.array(blat)
print(f"\n=== FULL CHUNK FORWARD (GB10, bf16 autocast) ===")
print(f"chunk inference: mean={blat.mean():.1f}ms  p50={np.median(blat):.1f}ms  ({clat.mean()/blat.mean():.2f}x vs fp32)")
print(f"amortized/action over {n_steps} steps: {blat.mean()/n_steps:.2f}ms -> {1000*n_steps/blat.mean():.0f} actions/s effective")
