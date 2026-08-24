#!/usr/bin/env python3
"""Produce a TRT-parseable ONNX with NO external-data re-serialization (avoids the tied-weight bloat).

Two graph-only edits, keeping every big tensor as an external reference to the ORIGINAL .data:
  1) strip the (no-op reduction='none') attribute off ScatterND  -> TRT 10.3 parser accepts it
  2) retype float index constants feeding Where->Gather-indices to int64, inlined so .data is untouched
Usage: make_trt_onnx.py in.onnx out.onnx   (out.onnx references in.onnx's .data by its stored basename)
"""
import sys, os, onnx, numpy as np
from onnx import numpy_helper, TensorProto, external_data_helper as edh

inp, out = sys.argv[1], sys.argv[2]
base_dir = os.path.dirname(os.path.abspath(inp))
m = onnx.load(inp, load_external_data=False)
g = m.graph
inits = {i.name: i for i in g.initializer}

def dtype_of(name):
    return inits[name].data_type if name in inits else None

# (1) strip ScatterND reduction
n_scatter = 0
for node in g.node:
    if node.op_type == "ScatterND":
        keep = [a for a in node.attribute if a.name != "reduction"]
        if len(keep) != len(node.attribute):
            n_scatter += 1
            del node.attribute[:]; node.attribute.extend(keep)

# (2) retype float index initializers feeding Where (int64/float mismatch) -> inline int64
n_type = 0
for node in g.node:
    if node.op_type != "Where":
        continue
    x, y = node.input[1], node.input[2]
    types = {nm: dtype_of(nm) for nm in (x, y) if nm in inits}
    FLOATS = (TensorProto.FLOAT, TensorProto.FLOAT16)
    if TensorProto.INT64 in types.values() and any(t in FLOATS for t in types.values()):
        for nm, dt in types.items():
            if dt in FLOATS:
                t = inits[nm]
                npdt = np.float32 if dt == TensorProto.FLOAT else np.float16
                if t.external_data:
                    # read just this small tensor's bytes from its external offset/length
                    meta = {e.key: e.value for e in t.external_data}
                    fpath = os.path.join(base_dir, meta["location"])
                    off = int(meta.get("offset", 0)); ln = int(meta.get("length", 0))
                    with open(fpath, "rb") as fh:
                        fh.seek(off); raw = fh.read(ln)
                else:
                    raw = t.raw_data
                a = np.frombuffer(raw, dtype=npdt).reshape(list(t.dims))
                assert np.allclose(a, np.round(a)), f"{nm} not integer-valued"
                newt = numpy_helper.from_array(a.astype(np.int64), nm)  # inline (raw_data)
                t.CopyFrom(newt)
                n_type += 1

print(f"stripped ScatterND reduction: {n_scatter}   inlined int64 index consts: {n_type}")
onnx.save(m, out)  # external tensors still point at the original .data basename
print(f"saved {out} (references '{os.path.basename(inp)}.data')")
