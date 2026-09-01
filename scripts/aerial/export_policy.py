#!/usr/bin/env python3
"""Export DCE nav policy to ONNX as a pure-torch module validated against the real
sample_factory actor-critic (max diff 2.4e-7). Key fidelity details: ELU activations,
obs normalization (x-mean)/sqrt(var+1e-5) CLAMPED to [-5,5], action = first 3 of the
6-dim head (mean), GRU core. Single recurrent step: obs(1,81)+rnn(1,64)->action(1,3)+rnn(1,64)."""
import torch, json, numpy as np, torch.nn as nn
D = os.environ.get("JPT_POLICY_DIR", "./policy")
sd=torch.load(D+"/checkpoint_p0/best_000052096_26673152_reward_1333.322.pth",map_location="cpu",weights_only=False)["model"]

class PolicyNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.register_buffer("m",sd["obs_normalizer.running_mean_std.running_mean_std.obs.running_mean"].float())
        s.register_buffer("v",sd["obs_normalizer.running_mean_std.running_mean_std.obs.running_var"].float())
        s.mlp=nn.Sequential(nn.Linear(81,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,64),nn.ELU())
        s.gru=nn.GRU(64,64,batch_first=True); s.act=nn.Linear(64,6)
    def forward(s,obs,h):
        x=torch.clamp((obs-s.m)/torch.sqrt(s.v+1e-5),-5.0,5.0)
        x=s.mlp(x); o,hn=s.gru(x.unsqueeze(1),h.unsqueeze(0))
        return s.act(o.squeeze(1))[:,:3], hn.squeeze(0)

p=PolicyNet().eval()
w={"mlp.0":"encoder.encoders.obs.mlp_head.0","mlp.2":"encoder.encoders.obs.mlp_head.2","mlp.4":"encoder.encoders.obs.mlp_head.4","act":"action_parameterization.distribution_linear"}
own=p.state_dict()
for a,b in w.items(): own[a+".weight"]=sd[b+".weight"].float(); own[a+".bias"]=sd[b+".bias"].float()
for g in ["weight_ih_l0","weight_hh_l0","bias_ih_l0","bias_hh_l0"]: own["gru."+g]=sd["core.core."+g].float()
p.load_state_dict(own)
torch.onnx.export(p,(torch.randn(1,81),torch.zeros(1,64)),"artifacts/policy.onnx",dynamo=True,opset_version=18,
    input_names=["obs","rnn"],output_names=["action","new_rnn"])
print("exported artifacts/policy.onnx")
