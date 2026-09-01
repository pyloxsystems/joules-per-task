import os
#!/usr/bin/env python3
"""Export ViNT (31M visual-nav transformer, quadrotor-deployed) to ONNX.
Inputs per deployment: obs_img (1, 3*(ctx+1)=18, H=64, W=85), goal_img (1,3,64,85)
Outputs: (dist_pred, action_pred)."""
import sys, torch, yaml
R = os.environ.get("VINT_REPO", "./visualnav-transformer")
sys.path.insert(0, R+"/train")
from vint_train.models.vint.vint import ViNT
# shim training-only dep referenced inside the pickled checkpoint
import types
ws=types.ModuleType("warmup_scheduler"); ws.__path__=[]
wss=types.ModuleType("warmup_scheduler.scheduler")
class GradualWarmupScheduler: pass
ws.GradualWarmupScheduler=GradualWarmupScheduler
wss.GradualWarmupScheduler=GradualWarmupScheduler
sys.modules["warmup_scheduler"]=ws
sys.modules["warmup_scheduler.scheduler"]=wss
cfg={**yaml.safe_load(open(R+"/train/config/defaults.yaml")), **yaml.safe_load(open(R+"/train/config/vint.yaml"))}
m=ViNT(context_size=cfg["context_size"], len_traj_pred=cfg["len_traj_pred"], learn_angle=cfg["learn_angle"],
    obs_encoder=cfg["obs_encoder"], obs_encoding_size=cfg["obs_encoding_size"], late_fusion=cfg.get("late_fusion",False),
    mha_num_attention_heads=cfg["mha_num_attention_heads"], mha_num_attention_layers=cfg["mha_num_attention_layers"],
    mha_ff_dim_factor=cfg["mha_ff_dim_factor"])
ck=torch.load(os.path.join(os.environ.get("JPT_WEIGHTS", "./weights"), "vint.pth"), map_location="cpu", weights_only=False)
lm=ck["model"]
try: sd=lm.module.state_dict()
except AttributeError: sd=lm.state_dict()
missing,unexpected=m.load_state_dict(sd, strict=False)
print(f"ViNT loaded; missing={len(missing)} unexpected={len(unexpected)}; params={sum(p.numel() for p in m.parameters())/1e6:.1f}M")
m.eval()
ctx=cfg["context_size"]; W,H=cfg["image_size"]
obs=torch.randn(1,3*(ctx+1),H,W); goal=torch.randn(1,3,H,W)
with torch.no_grad(): d,a=m(obs,goal)
print("dist",tuple(d.shape),"actions",tuple(a.shape))
torch.onnx.export(m,(obs,goal),"artifacts/vint.onnx",dynamo=True,opset_version=18,
    input_names=["obs_img","goal_img"],output_names=["dist","actions"])
print("exported artifacts/vint.onnx")
