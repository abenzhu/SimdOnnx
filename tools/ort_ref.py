"""ONNX Runtime reference for the bench harness (debug aid).

usage: python ort_ref.py <model.onnx> <input.f32> <NxCxHxW> <out.f32> [tensor_name cut_model.onnx]

Runs the model (or, with tensor_name, the sub-graph that ends at that tensor, saved to
cut_model.onnx so the bench can run the same cut) on the raw float32 input and writes the
output as raw float32.
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort

model, inp, shape, out = sys.argv[1:5]
dims = [int(d) for d in shape.split("x")]
x = np.fromfile(inp, dtype=np.float32).reshape(dims)
path = model
if len(sys.argv) > 6:
    tensor, cut = sys.argv[5], sys.argv[6]
    m = onnx.load(model)
    # shape inference is not needed by ORT for the extra output: declare it untyped
    m.graph.output.clear()
    m.graph.output.append(onnx.helper.make_empty_tensor_value_info(tensor))
    onnx.save(m, cut)
    path = cut
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
y = s.run(None, {s.get_inputs()[0].name: x})[0]
print("ort output", y.shape, y.dtype)
np.ascontiguousarray(y, dtype=np.float32).tofile(out)
