"""Bisect helper: prints (index, op, first output name) of every node.
usage: python list_outputs.py <model.onnx> [start] [end]"""
import sys

import onnx

m = onnx.load(sys.argv[1])
s = int(sys.argv[2]) if len(sys.argv) > 2 else 0
e = int(sys.argv[3]) if len(sys.argv) > 3 else len(m.graph.node)
for i, n in enumerate(m.graph.node[s:e], s):
    print(i, n.op_type, n.output[0])
