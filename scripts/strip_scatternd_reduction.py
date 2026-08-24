#!/usr/bin/env python3
"""TRT 10.3's parser rejects ScatterND that carries a `reduction` attribute even when it's 'none'.
Strip the (no-op) attribute. Graph-only edit -> keeps external-data references, no re-serialization."""
import sys, onnx
inp, out = sys.argv[1], sys.argv[2]
m = onnx.load(inp, load_external_data=False)  # keep external refs intact
n_strip = 0
for node in m.graph.node:
    if node.op_type == "ScatterND":
        keep = [a for a in node.attribute if a.name != "reduction"]
        removed = len(node.attribute) - len(keep)
        if removed:
            del node.attribute[:]
            node.attribute.extend(keep)
            n_strip += removed
print(f"stripped reduction from {n_strip} ScatterND nodes")
onnx.save(m, out)  # references the SAME external .data as `inp`
print("saved", out, "(references existing .onnx.data)")
