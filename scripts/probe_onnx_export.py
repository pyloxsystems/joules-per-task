#!/usr/bin/env python3
"""Derisk: can SmolVLA's sample_actions be exported to ONNX at all? Find the real snag."""
import os
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE","1")
import torch, torch.nn as nn
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

dev="cuda"
p = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(dev).eval()
cfg = p.config
m = p.model

class Wrap(nn.Module):
    """Tensor-in/tensor-out wrapper around sample_actions (no dict, no queue)."""
    def __init__(self, model): super().__init__(); self.model=model
    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        # sample_actions expects lists for images/img_masks
        imgs=[images[:,i] for i in range(images.shape[1])]
        masks=[img_masks[:,i] for i in range(img_masks.shape[1])]
        return self.model.sample_actions(imgs, masks, lang_tokens, lang_masks, state, noise=noise)

w=Wrap(m).eval()
B=1; ncam=3
images=torch.zeros(B,ncam,3,512,512,device=dev)      # embed_prefix resizes internally
img_masks=torch.ones(B,ncam,device=dev,dtype=torch.bool)
lang_tokens=torch.ones(B,48,device=dev,dtype=torch.long)
lang_masks=torch.ones(B,48,device=dev,dtype=torch.bool)
state=torch.zeros(B,32,device=dev)
noise=torch.zeros(B,cfg.chunk_size,cfg.max_action_dim,device=dev)

# sanity: does the wrapper run eager?
with torch.no_grad():
    try:
        out=w(images,img_masks,lang_tokens,lang_masks,state,noise)
        print("EAGER OK  action chunk:", tuple(out.shape))
    except Exception as e:
        print("EAGER FAILED:", repr(e)[:400]); raise

# attempt dynamo ONNX export
print("\n--- attempting torch.onnx.export(dynamo=True) ---")
try:
    with torch.no_grad():
        ep = torch.onnx.export(
            w, (images,img_masks,lang_tokens,lang_masks,state,noise),
            "artifacts/smolvla_sample.onnx", dynamo=True,
            input_names=["images","img_masks","lang_tokens","lang_masks","state","noise"],
            output_names=["actions"], opset_version=18,
        )
    print("EXPORT OK -> artifacts/smolvla_sample.onnx")
except Exception as e:
    import traceback
    print("EXPORT FAILED:")
    traceback.print_exc()
    print("\nERROR HEAD:", repr(e)[:500])
