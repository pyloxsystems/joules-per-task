#!/usr/bin/env python3
"""Repair dynamo-exporter type-binding bugs: Where nodes that mix int64 index tensors with
float constants (they feed Gather-indices). Cast the float initializer operand to int64."""
import sys, onnx, numpy as np
from onnx import numpy_helper, TensorProto

inp = sys.argv[1]; out = sys.argv[2]
m = onnx.load(inp, load_external_data=True)
g = m.graph
inits = {i.name: i for i in g.initializer}
fixed = 0
for n in g.node:
    if n.op_type != "Where":
        continue
    x, y = n.input[1], n.input[2]
    types = {nm: inits[nm].data_type for nm in (x, y) if nm in inits}
    if TensorProto.INT64 in types.values() and TensorProto.FLOAT in types.values():
        for nm, dt in types.items():
            if dt == TensorProto.FLOAT:
                a = numpy_helper.to_array(inits[nm])
                assert np.allclose(a, np.round(a)), f"{nm} not integer-valued"
                inits[nm].CopyFrom(numpy_helper.from_array(a.astype(np.int64), nm))
                fixed += 1
                print(f"  retyped {nm}: float->int64 shape={a.shape}")
print(f"fixed {fixed} Where/int-index mismatches")
loc = out.split("/")[-1] + ".data"
onnx.save(m, out, save_as_external_data=True, location=loc, all_tensors_to_one_file=True)
print("saved", out)
