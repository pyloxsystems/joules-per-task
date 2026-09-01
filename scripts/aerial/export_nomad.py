import os
#!/usr/bin/env python3
"""Export NoMaD (~30-45M diffusion visual-nav policy, quadrotor-deployed) to ONNX:
  1) vision encoder: obs stack (1, 3*(ctx+1)=12, 96, 96) + goal img + goal mask -> obs_cond (1,256)
  2) noise-pred single step: naction (1,8,2) + timestep + obs_cond -> noise  (DDPM loop on host, 10 iters)
Mirrors deployment/src/explore.py exactly."""
import sys, torch, yaml, torch.nn as nn
R = os.environ.get("VINT_REPO", "./visualnav-transformer")
sys.path.insert(0, R+"/train")
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
sys.path.insert(0, os.environ.get("JPT_VENDOR", "./vendor"))
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

cfg={**yaml.safe_load(open(R+"/train/config/defaults.yaml")), **yaml.safe_load(open(R+"/train/config/nomad.yaml"))}
ve=NoMaD_ViNT(obs_encoding_size=cfg["encoding_size"], context_size=cfg["context_size"],
    mha_num_attention_heads=cfg["mha_num_attention_heads"], mha_num_attention_layers=cfg["mha_num_attention_layers"],
    mha_ff_dim_factor=cfg["mha_ff_dim_factor"])
ve=replace_bn_with_gn(ve)
npn=ConditionalUnet1D(input_dim=2, global_cond_dim=cfg["encoding_size"], down_dims=cfg["down_dims"],
    cond_predict_scale=cfg["cond_predict_scale"])
model=NoMaD(vision_encoder=ve, noise_pred_net=npn, dist_pred_net=DenseNetwork(embedding_dim=cfg["encoding_size"]))
sd=torch.load(os.path.join(os.environ.get("JPT_WEIGHTS", "./weights"), "nomad.pth"), map_location="cpu")
missing,unexpected=model.load_state_dict(sd, strict=False)
print(f"NoMaD loaded; missing={len(missing)} unexpected={len(unexpected)}; params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
model.eval()

ctx=cfg["context_size"]; W,H=cfg["image_size"]; T=cfg["len_traj_pred"]
class VisionWrap(nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,obs_img,goal_img,goal_mask):
        return s.m("vision_encoder", obs_img=obs_img, goal_img=goal_img, input_goal_mask=goal_mask)
class NoiseWrap(nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,naction,timestep,obs_cond):
        return s.m("noise_pred_net", sample=naction, timestep=timestep, global_cond=obs_cond)

obs=torch.randn(1,3*(ctx+1),H,W); goal=torch.randn(1,3,H,W); mask=torch.ones(1,dtype=torch.long)
vw=VisionWrap(model).eval()
with torch.no_grad(): cond=vw(obs,goal,mask)
print("vision -> obs_cond", tuple(cond.shape))
na=torch.randn(1,T,2); ts=torch.tensor([5],dtype=torch.long)
nw=NoiseWrap(model).eval()
with torch.no_grad(): eps=nw(na,ts,cond)
print("noise_pred ->", tuple(eps.shape))

torch.onnx.export(vw,(obs,goal,mask),"artifacts/nomad_vision.onnx",dynamo=True,opset_version=18,
    input_names=["obs_img","goal_img","goal_mask"],output_names=["obs_cond"])
print("exported nomad_vision.onnx")
torch.onnx.export(nw,(na,ts,cond),"artifacts/nomad_noise.onnx",dynamo=True,opset_version=18,
    input_names=["naction","timestep","obs_cond"],output_names=["noise"])
print("exported nomad_noise.onnx")
