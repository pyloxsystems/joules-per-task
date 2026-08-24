#!/usr/bin/env python3
"""Record the exact per-step policy-input batches during real LIBERO rollouts -> replay stream."""
import os, sys
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("PYTORCH_NVFUSER_DISABLE","1")
import numpy as np, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import lerobot.scripts.lerobot_eval as LE

STEPS=[]
_orig = SmolVLAPolicy.select_action
def _rec(self, batch, *a, **kw):
    entry={}
    for k,v in batch.items():
        if torch.is_tensor(v):
            arr=v.detach().float().cpu().numpy()
            entry[k]=arr.astype(np.float16) if arr.dtype==np.float32 and arr.size>1000 else arr
        elif isinstance(v,(list,str)):
            entry[k]=v
    STEPS.append(entry)
    return _orig(self, batch, *a, **kw)
SmolVLAPolicy.select_action=_rec

sys.argv=["lerobot_eval",
  "--policy.path=HuggingFaceVLA/smolvla_libero",
  "--env.type=libero","--env.task=libero_spatial","--env.task_ids=[0]",
  "--eval.n_episodes=3","--eval.batch_size=1",
  "--output_dir=artifacts/eval_record"]
LE.main()

print(f"recorded {len(STEPS)} steps")
keys=[k for k in STEPS[0] if isinstance(STEPS[0][k],np.ndarray)]
print("array keys:", keys)
out={}
for k in keys:
    out[k.replace(".","__")]=np.stack([s[k] for s in STEPS])
np.savez_compressed("artifacts/replay_stream.npz", **out)
print("saved artifacts/replay_stream.npz", {k:v.shape for k,v in out.items()})
