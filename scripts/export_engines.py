#!/usr/bin/env python3
"""Export Engine A (prefix/VLM) and Engine B (denoise step) to ONNX."""
import os, sys, time
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE","1")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from engines import PrefixEngine, DenoiseEngine, VisionEngine, PrefillEngine, PrefillEngineA, PrefillEngineB, truncate_vlm_layers, StateDecodeEngine

dev="cuda"; torch.manual_seed(0)
import os as _os
MODEL_ID=_os.environ.get("MODEL_ID","lerobot/smolvla_base")
p=SmolVLAPolicy.from_pretrained(MODEL_ID).to(dev).eval()
LORA_CKPT=_os.environ.get("LORA_CKPT")
if LORA_CKPT:
    import torch as _t
    _sd=_t.load(LORA_CKPT,map_location="cpu"); _params=_sd["lora"]; _rank=_sd["rank"]; _i=0
    _scale=16.0/_rank
    for _layer in p.model.vlm_with_expert.lm_expert.layers:
        for _parent,_attr in [(_layer.self_attn,"q_proj"),(_layer.self_attn,"k_proj"),
                              (_layer.self_attn,"v_proj"),(_layer.self_attn,"o_proj"),
                              (_layer.mlp,"gate_proj"),(_layer.mlp,"up_proj"),(_layer.mlp,"down_proj")]:
            _base=getattr(_parent,_attr,None)
            if _base is not None and hasattr(_base,"weight"):
                _A=_params[_i].to(_base.weight.device,_base.weight.dtype)
                _B=_params[_i+1].to(_base.weight.device,_base.weight.dtype); _i+=2
                _base.weight.data += (_B @ _A) * _scale
    print(f"LoRA merged: {_i//2} adapters from {LORA_CKPT} (train step {_sd.get('step')})")
p=p.half()
c=p.config; m=p.model
os.makedirs("artifacts", exist_ok=True)

B=1
NCAM=sum(1 for k in c.input_features if "image" in k)
SDIM=c.max_state_dim
print(f"model={MODEL_ID} ncam={NCAM} state_pad={SDIM} chunk={c.chunk_size} adim={c.max_action_dim}")
images=torch.rand(B,NCAM,3,512,512,device=dev)
img_masks=torch.ones(B,NCAM,device=dev,dtype=torch.bool)
lang_tokens=torch.randint(0,32000,(B,48),device=dev)
lang_masks=torch.ones(B,48,device=dev,dtype=torch.bool)
state=torch.randn(B,SDIM,device=dev)

pe=PrefixEngine(m).eval()
which = sys.argv[1] if len(sys.argv)>1 else "both"

if which in ("A","both"):
    print(">>> exporting Engine A (prefix/VLM) ...")
    t0=time.time()
    with torch.no_grad():
        torch.onnx.export(pe,(images,img_masks,lang_tokens,lang_masks,state),
            "artifacts/prefix.onnx", dynamo=True, opset_version=18,
            input_names=["images","img_masks","lang_tokens","lang_masks","state"],
            output_names=["K","V","prefix_pad_masks"])
    print(f"    Engine A exported in {time.time()-t0:.0f}s")

if which in ("B","both"):
    print(">>> exporting Engine B (denoise step) ...")
    with torch.no_grad():
        K,V,ppm = pe(images,img_masks,lang_tokens,lang_masks,state)
    x_t=torch.randn(B,c.chunk_size,c.max_action_dim,device=dev)
    ts=torch.tensor(0.9,dtype=torch.float32,device=dev).expand(B)
    de=DenoiseEngine(m).eval()
    t0=time.time()
    with torch.no_grad():
        torch.onnx.export(de,(x_t,ts,ppm,K,V),
            "artifacts/denoise.onnx", dynamo=True, opset_version=18,
            input_names=["x_t","timestep","prefix_pad_masks","K","V"],
            output_names=["v_t"])
    print(f"    Engine B exported in {time.time()-t0:.0f}s")
if which in ("V","split"):
    print(">>> exporting VisionEngine (SigLIP tower) ...")
    ve=VisionEngine(m).eval(); t0=time.time()
    with torch.no_grad():
        torch.onnx.export(ve,(images,), "artifacts/vision.onnx", dynamo=True, opset_version=18,
            input_names=["images"], output_names=["img_embs"])
    print(f"    VisionEngine exported in {time.time()-t0:.0f}s")

if which in ("P","split"):
    print(">>> exporting PrefillEngine (VLM prefill from img embs) ...")
    ve=VisionEngine(m).eval()
    with torch.no_grad():
        img_embs=ve(images)
    print("    img_embs shape:", tuple(img_embs.shape))
    pf=PrefillEngine(m).eval(); t0=time.time()
    with torch.no_grad():
        torch.onnx.export(pf,(img_embs,img_masks,lang_tokens,lang_masks,state),
            "artifacts/prefill.onnx", dynamo=True, opset_version=18,
            input_names=["img_embs","img_masks","lang_tokens","lang_masks","state"],
            output_names=["K","V","prefix_pad_masks"])
    print(f"    PrefillEngine exported in {time.time()-t0:.0f}s")

if which == "V1":
    print(">>> exporting single-camera VisionEngine ...")
    ve=VisionEngine(m).eval(); t0=time.time()
    img1=torch.rand(B,1,3,512,512,device=dev)
    with torch.no_grad():
        torch.onnx.export(ve,(img1,), "artifacts/vision1.onnx", dynamo=True, opset_version=18,
            input_names=["images"], output_names=["img_embs"])
    print(f"    Vision1 exported in {time.time()-t0:.0f}s")

if which == "PA":
    print(">>> exporting PrefillEngineA (embeds + layers 0-7) ...")
    ve=VisionEngine(m).eval()
    with torch.no_grad(): img_embs=ve(images)
    TOTAL=len(m.vlm_with_expert.get_vlm_model().text_model.layers)
    HALF=int(_os.environ.get("SPLIT_AT", TOTAL//2))
    print(f"    splitting at layer {HALF} of {TOTAL}")
    truncate_vlm_layers(m, 0, HALF, identity_norm=True)
    pa=PrefillEngineA(m).eval(); t0=time.time()
    with torch.no_grad():
        torch.onnx.export(pa,(img_embs,img_masks,lang_tokens,lang_masks,state),
            "artifacts/prefillA.onnx", dynamo=True, opset_version=18,
            input_names=["img_embs","img_masks","lang_tokens","lang_masks","state"],
            output_names=["K","V","prefix_pad_masks","prefix_att_masks","hidden"])
    print(f"    PrefillEngineA exported in {time.time()-t0:.0f}s")

if which == "PB":
    print(">>> exporting PrefillEngineB (layers 8-15) ...")
    ve=VisionEngine(m).eval()
    with torch.no_grad():
        img_embs=ve(images)
        pf_full=PrefillEngine(m).eval()
        # get real hidden via a temp lower-half copy would need reload; use dummy hidden of right shape
    TOTAL=len(m.vlm_with_expert.get_vlm_model().text_model.layers)
    HALF=int(_os.environ.get("SPLIT_AT", TOTAL//2))
    print(f"    splitting at layer {HALF} of {TOTAL}")
    truncate_vlm_layers(m, HALF, TOTAL, identity_norm=False)
    with torch.no_grad():
        # dummy inputs shaped from config: prefix_len = ncam*64 + 48 + 1
        PLEN = images.shape[1]*64 + 48 + 1
        hidden=torch.randn(1,PLEN,960,device=dev,dtype=torch.float16)
        ppm=torch.ones(1,PLEN,device=dev); pam=torch.zeros(1,PLEN,device=dev); pam[:,-1]=1
    pb=PrefillEngineB(m).eval(); t0=time.time()
    with torch.no_grad():
        torch.onnx.export(pb,(hidden,ppm,pam),
            "artifacts/prefillB.onnx", dynamo=True, opset_version=18,
            input_names=["hidden","prefix_pad_masks","prefix_att_masks"],
            output_names=["K","V"])
    print(f"    PrefillEngineB exported in {time.time()-t0:.0f}s")

if which in ("SA","SB"):
    TOTAL=len(m.vlm_with_expert.get_vlm_model().text_model.layers)
    HALF=int(_os.environ.get("SPLIT_AT", TOTAL//2))
    NCAM2=sum(1 for k in c.input_features if "image" in k)
    PLEN=NCAM2*64+48+1
    if which=="SA":
        print(f">>> exporting StateDecodeEngine A (layers 0-{HALF-1}) ...")
        truncate_vlm_layers(m, 0, HALF, identity_norm=True)
        sa=StateDecodeEngine(m, first_half=True).eval()
        Kin=torch.randn(HALF,1,5,PLEN-1,64,device=dev,dtype=torch.float16)
        Vin=torch.randn(HALF,1,5,PLEN-1,64,device=dev,dtype=torch.float16)
        ppm2=torch.ones(1,PLEN,device=dev)
        st=torch.randn(1,32,device=dev)
        t0=time.time()
        with torch.no_grad():
            torch.onnx.export(sa,(st,Kin,Vin,ppm2),"artifacts/stateA.onnx",dynamo=True,opset_version=18,
                input_names=["state","K","V","prefix_pad_masks"],output_names=["Ks","Vs","hidden"])
        print(f"    StateA exported in {time.time()-t0:.0f}s")
    else:
        print(f">>> exporting StateDecodeEngine B (layers {HALF}-{TOTAL-1}) ...")
        truncate_vlm_layers(m, HALF, TOTAL, identity_norm=False)
        sb=StateDecodeEngine(m, first_half=False).eval()
        NB=TOTAL-HALF
        Kin=torch.randn(NB,1,5,PLEN-1,64,device=dev,dtype=torch.float16)
        Vin=torch.randn(NB,1,5,PLEN-1,64,device=dev,dtype=torch.float16)
        ppm2=torch.ones(1,PLEN,device=dev)
        hid=torch.randn(1,1,960,device=dev,dtype=torch.float16)
        t0=time.time()
        with torch.no_grad():
            torch.onnx.export(sb,(hid,Kin,Vin,ppm2),"artifacts/stateB.onnx",dynamo=True,opset_version=18,
                input_names=["hidden","K","V","prefix_pad_masks"],output_names=["Ks","Vs","hidden_out"])
        print(f"    StateB exported in {time.time()-t0:.0f}s")

print("DONE")
