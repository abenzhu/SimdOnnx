import onnx, sys
from onnx import helper, numpy_helper
path = sys.argv[1]
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 10**9)
m = onnx.load(path); g = m.graph
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
consts = {}
for n in g.node:
    if n.op_type == 'Constant':
        consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
def s(x):
    a = inits.get(x, consts.get(x))
    if a is not None:
        return 'C' + str(list(a.shape)) + (('=' + str(a.flatten()[:4].tolist())) if a.size <= 4 else '')
    return x[-16:]
for i, n in enumerate(g.node):
    if lo <= i <= hi and n.op_type not in ('Constant',):
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        print(i, n.op_type, [s(x) for x in n.input], '->', [o[-16:] for o in n.output], {k: (v if not isinstance(v, bytes) else v.decode()) for k, v in a.items()} if a else '')
