#!/usr/bin/env python3
"""LIBERO eval with per-camera diff-gated vision reuse injected into SmolVLA —
the exact logic measured for energy on the Orin, now validated for task success.
Usage: python eval_with_reuse.py <tau> <n_episodes>"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
import lerobot.scripts.lerobot_eval as LE

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
N_EP = sys.argv[2] if len(sys.argv) > 2 else "10"

STATS = {"encodes": 0, "skips": 0}
_cam_counter = {"i": 0}
_cache = {}

_orig_embed = SmolVLMWithExpertModel.embed_image
def _gated_embed(self, img):
    c = _cam_counter["i"]; _cam_counter["i"] += 1
    ent = _cache.get(c)
    if ent is not None:
        last, emb = ent
        if last.shape == img.shape:
            diff = (img - last).abs().mean().item()
            if diff < TAU:
                STATS["skips"] += 1
                return emb
    emb = _orig_embed(self, img)
    _cache[c] = (img.detach().clone(), emb.detach().clone())
    STATS["encodes"] += 1
    return emb
SmolVLMWithExpertModel.embed_image = _gated_embed

_orig_sel = SmolVLAPolicy.select_action
def _sel(self, batch, *a, **kw):
    _cam_counter["i"] = 0  # per-step camera index reset
    return _orig_sel(self, batch, *a, **kw)
SmolVLAPolicy.select_action = _sel

_orig_reset = SmolVLAPolicy.reset
def _reset(self):
    _cache.clear()  # never reuse across episodes
    return _orig_reset(self)
SmolVLAPolicy.reset = _reset

sys.argv = ["lerobot_eval",
            "--policy.path=HuggingFaceVLA/smolvla_libero",
            "--env.type=libero", "--env.task=libero_spatial",
            f"--eval.n_episodes={N_EP}", "--eval.batch_size=1",
            f"--output_dir=artifacts/eval_reuse_tau{TAU}"]
LE.main()
tot = STATS["encodes"] + STATS["skips"]
print(f"\nREUSE STATS tau={TAU}: encodes={STATS['encodes']} skips={STATS['skips']} "
      f"skip-rate={100*STATS['skips']/max(tot,1):.1f}%")
