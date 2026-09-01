#!/usr/bin/env python3
"""Export the DCE depth-VAE encoder (depth image -> 64-dim latent) to ONNX = the drone
'vision tower', the vision-reuse target for JPT-Aerial. Mirrors v1's VisionEngine export."""
import os, sys, torch, torch.nn as nn
VAEDIR = os.environ.get("JPT_VAE_DIR", "./vae")
sys.path.insert(0, VAEDIR)
from VAE import VAE
def clean_state_dict(sd):
    out={}
    for k,v in sd.items():
        k=k.replace("module.","").replace("dronet.","encoder.")
        out[k]=v
    return out

DEV="cuda"
WEIGHTS=os.path.join(VAEDIR,"weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth")
vae=VAE(input_dim=1, latent_dim=64).to(DEV).eval()
sd=clean_state_dict(torch.load(WEIGHTS, map_location=DEV))
missing,unexpected=vae.load_state_dict(sd, strict=False)
print(f"VAE loaded; missing={len(missing)} unexpected={len(unexpected)}")

class DepthEncoder(nn.Module):
    """depth (1,1,270,480) -> 64-dim mean latent (deterministic, inference)."""
    def __init__(self, vae): super().__init__(); self.enc=vae.encoder; self.ld=vae.latent_dim
    def forward(self, img):
        z=self.enc(img)               # (B, 128) = mean|logvar
        return z[:, :self.ld]         # deterministic mean latent

m=DepthEncoder(vae).to(DEV).eval()
x=torch.zeros(1,1,270,480,device=DEV)
with torch.no_grad(): z=m(x)
print("depth (1,1,270,480) -> latent", tuple(z.shape))
torch.onnx.export(m,(x,),"artifacts/depth_encoder.onnx",dynamo=True,opset_version=18,
    input_names=["depth"], output_names=["latent"])
print("exported artifacts/depth_encoder.onnx")
