"""Build a single-Conv ONNX model for kernel bisecting.
usage: python make_conv.py out.onnx ic oc h w [k] [group] [stride]"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

out = sys.argv[1]
ic, oc, h, w = (int(v) for v in sys.argv[2:6])
k = int(sys.argv[6]) if len(sys.argv) > 6 else 1
g = int(sys.argv[7]) if len(sys.argv) > 7 else 1
s = int(sys.argv[8]) if len(sys.argv) > 8 else 1
rng = np.random.default_rng(1)
W = rng.standard_normal((oc, ic // g, k, k)).astype(np.float32) * 0.3
B = rng.standard_normal(oc).astype(np.float32) * 0.1
node = helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[k, k], pads=[k // 2] * 4, group=g, strides=[s, s])
graph = helper.make_graph([node], "conv", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, ic, h, w])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
                          [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")])
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
onnx.save(m, out)
