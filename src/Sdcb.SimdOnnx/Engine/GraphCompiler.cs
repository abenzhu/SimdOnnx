using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// Compiles an <see cref="OnnxModel"/> into an <see cref="ExecPlan"/> for a
/// concrete input shape. All ONNX dynamic-shape plumbing (Shape/Squeeze/Div/
/// Mod/Concat/Slice on int64 scalars) is evaluated at compile time; remaining
/// float ops are emitted with fully resolved shapes.
/// </summary>
internal static class GraphCompiler
{
    public static ExecPlan Compile(OnnxModel model, ReadOnlySpan<int> inputShape) =>
        Compile(model, inputShape, CompileOptions.Default);

    public static ExecPlan Compile(OnnxModel model, ReadOnlySpan<int> inputShape, CompileOptions opt)
    {
        var c = new Compiler(model);
        return c.Run(inputShape, opt);
    }

    /// <summary>Vulkan small-conv shaders (uphead skip recompute, convpool): dense stride-1
    /// conv, ic &lt; 8, oc a multiple of 8 (&lt;= 128), weights fit the 16 KB staging.</summary>
    internal static bool GpuSmallConv(ConvP cp, int[] wShape, int ic) =>
        cp.Group == 1 && ic < 8 && wShape.Length == 4 && wShape[0] % 8 == 0 && wShape[0] <= 128
        && (long)wShape[0] * ic * wShape[2] * wShape[3] <= 4096
        && cp.Strides.All(s => s == 1) && cp.Dilations.All(d => d == 1)
        && cp.Act is ActivationKind.None or ActivationKind.Relu;

    private sealed class Compiler
    {
        private readonly OnnxModel _m;
        private readonly List<CValue> _values = [];
        private readonly Dictionary<string, int> _byName = new(StringComparer.Ordinal);
        private readonly List<ExecNode> _nodes = [];
        private readonly List<int> _bufSize = [];
        private int _inputValue = -1;

        public Compiler(OnnxModel m) => _m = m;

        private int AddValue(CValue v)
        {
            int id = _values.Count;
            _values.Add(v);
            if (v.Name.Length != 0) _byName[v.Name] = id;
            return id;
        }

        private int NewBuf(int size)
        {
            int id = _bufSize.Count;
            _bufSize.Add(size);
            return id;
        }

        private CValue V(int id) => _values[id];

        private int Input(int i, OnnxNodeData n)
        {
            string name = n.Inputs[i];
            if (name.Length == 0) return -1;
            if (!_byName.TryGetValue(name, out int id))
                throw new InvalidDataException($"Node '{n.Name}' ({n.OpType}): input '{name}' not produced.");
            return id;
        }

        private int[] Inputs(OnnxNodeData n, int skip = 0)
        {
            int[] ids = new int[n.Inputs.Count - skip];
            for (int i = skip; i < n.Inputs.Count; i++) ids[i - skip] = Input(i, n);
            return ids;
        }

        /// <summary>Int list given as input <paramref name="i"/> (newer opsets) or as attribute <paramref name="attr"/> (older); null if neither.</summary>
        private long[]? IntsArg(OnnxNodeData n, int i, string attr)
        {
            if (n.Inputs.Count > i && n.Inputs[i].Length > 0) return V(Input(i, n)).Ints;
            long[] a = n.IntsAttr(attr);
            return a.Length > 0 || n.Attr(attr) is not null ? a : null;
        }

        private int _curNi;   // onnx node index currently being emitted (for SrcNode)

        private int Emit(ExecOp op, int[] ins, CValue output, object? p, OnnxNodeData n)
        {
            int outId = AddValue(output);
            _nodes.Add(new ExecNode { Op = op, In = ins, Out = [outId], P = p, SrcNode = _curNi, Name = n.Name });
            return outId;
        }

        public ExecPlan Run(ReadOnlySpan<int> inputShape, CompileOptions opt)
        {
            OnnxGraphData g = _m.Graph;
            foreach (OnnxTensorData t in g.Initializers)
            {
                int[] dims = t.DimsArray();
                if (t.DataType == OnnxTensorData.TypeFloat)
                    AddValue(new CValue { Name = t.Name, Shape = dims, DType = 1, Floats = t.AsFloats() });
                else
                    AddValue(new CValue { Name = t.Name, Shape = dims, DType = t.DataType, Ints = t.AsInt64s() });
            }
            if (g.Inputs.Count != 1) throw new NotSupportedException("Expected single graph input.");
            {
                string inName = g.Inputs[0].Name;
                int elems = 1;
                int[] shape = inputShape.ToArray();
                foreach (int d in shape) elems *= d;
                int buf = NewBuf(elems);
                _inputValue = AddValue(new CValue { Name = inName, Shape = shape, DType = 1, Buf = buf });
            }

            for (int ni = 0; ni < g.Nodes.Count; ni++)
            {
                OnnxNodeData n = g.Nodes[ni];
                _curNi = ni;
                EmitNode(n);
            }

            // graph outputs
            if (opt.OutputName is { } outName)
            {
                if (!_byName.TryGetValue(outName, out int ov))
                    throw new ArgumentException($"Tensor '{outName}' is not produced by the graph.", nameof(opt));
                return FuseAndPlan(_nodes, _values, _bufSize, _inputValue, ov, opt);
            }
            if (g.Outputs.Count != 1) throw new NotSupportedException("Expected single graph output.");
            int outputValue = _byName[g.Outputs[0].Name];

            return FuseAndPlan(_nodes, _values, _bufSize, _inputValue, outputValue, opt);
        }

        // ---------- fusion + buffer planning ----------
        private static ExecPlan FuseAndPlan(List<ExecNode> nodes, List<CValue> values,
            List<int> bufSize, int inputValue, int outputValue, CompileOptions opt)
        {
            // fusion passes operate on the logical graph and are layout-blind;
            // physical layout is decided afterwards by LayoutPlanner
            Prune(nodes, values, outputValue);
            bool fuse = Env.Flag("NOFUSE") != "1";
            if (fuse)
            {
                Fuse(nodes, values, outputValue, opt);
                RunPasses(PassStage.Logical, nodes, values, outputValue, opt);
                // NHWC-only kernels (CPU channels-last + GPU); the NCHW oracle plan keeps the plain ops
                if (opt.Nhwc && Env.Flag("NOFUSE_UP") != "1")
                    FuseResizeAdd(nodes, values, outputValue);
            }
            if (opt.Nhwc)
                LayoutPlanner.Run(nodes, values, bufSize, inputValue, ref outputValue, opt.PhysicalIo,
                    opt.PhysicalIo && !opt.LogicalOutput);
            if (opt.Nhwc && opt.MemFuse && CompileOptions.MemFusions && fuse)
            {
                RunPasses(PassStage.Physical, nodes, values, outputValue, opt);
                FuseMemory(nodes, values, outputValue, opt.Gpu);
            }

            PlanArena(nodes, values, bufSize, inputValue, outputValue,
                out int[] offsets, out int arenaFloats);
            if (Env.Flag("DUMP") == "2")
            {
                string Vs(int v) => v < 0 ? "-" : $"v{v}/b{values[v].Buf}[{string.Join('x', values[v].Shape)}]{(values[v].Layout is { } l && !Layout.IsIdentity(l) ? Layout.Str(l) : "")}";
                for (int i = 0; i < nodes.Count; i++)
                {
                    ExecNode m = nodes[i];
                    string extra = m.P switch
                    {
                        TransposeP t => $" perm=[{string.Join(',', t.Perm)}]",
                        ConvP c => $" k={c.Kernel[0]}x{c.Kernel[1]} s={c.Strides[0]} g={c.Group} add={Vs(c.AddValue)} act={c.Act}{(c.Pool is { } pl ? $" +maxpool{(pl.FuseRelu ? "+relu" : "")}" : "")}",
                        GemmP g => $" act={g.Act}",
                        ConcatP c => $" axis={c.Axis}",
                        EwP e => $" {e.Op} act={e.Act}",
                        _ => m.Ext is { } x ? $" {x.Name} {m.P}" : "",
                    };
                    Console.Error.WriteLine($"  #{i,-3} {m.Op,-14} {string.Join(' ', m.In.Select(Vs))} -> {string.Join(' ', m.Out.Select(Vs))}{extra}");
                }
            }
            if (Env.Flag("DUMP") == "1")
            {
                var hist = nodes.GroupBy(m => m.Ext?.Name ?? m.Op.ToString()).OrderByDescending(g => g.Count());
                Console.Error.WriteLine($"[plan] {nodes.Count} nodes, arena {arenaFloats * 4.0 / 1e6:F1} MB");
                foreach (var h in hist) Console.Error.WriteLine($"  {h.Key,-14} x{h.Count()}");
            }
            return new ExecPlan
            {
                Nodes = nodes.ToArray(),
                Values = values.ToArray(),
                BufSize = bufSize.ToArray(),
                BufOffset = offsets,
                ArenaFloats = arenaFloats,
                InputValue = inputValue,
                OutputValue = outputValue,
                Nhwc = opt.Nhwc,
                PhysicalIo = opt.Nhwc && opt.PhysicalIo,
            };
        }

        /// <summary>Drops nodes that do not contribute to the output (heads cut at an inner tensor).</summary>
        private static void Prune(List<ExecNode> nodes, List<CValue> values, int outputValue)
        {
            var live = new HashSet<int>();
            void Mark(int v) { while (v >= 0 && live.Add(v)) v = values[v].AliasOf; }
            Mark(outputValue);
            for (int i = nodes.Count - 1; i >= 0; i--)
            {
                ExecNode n = nodes[i];
                if (!n.Out.Any(live.Contains)) { nodes.RemoveAt(i); continue; }
                foreach (int v in n.In) Mark(v);
                foreach (int v in n.ParamRefs()) Mark(v);
            }
        }

        private static void RunPasses(PassStage stage, List<ExecNode> nodes, List<CValue> values, int outputValue, CompileOptions opt)
        {
            if (opt.Extensions is not { } ext) return;
            var ctx = new PassContext(nodes, values, outputValue, opt);
            foreach (GraphPass p in ext.Passes)
                if (p.Stage == stage) p.Run(ctx);
        }

        /// <summary>Structural fusions over the exec node list (single consumer chains).</summary>
        private static void Fuse(List<ExecNode> nodes, List<CValue> values, int outputValue, CompileOptions opt)
        {
            // pass 1: Pad -> Conv  (fold reflect/constant padding into the conv)
            // pass 2: Conv -> Relu (activation epilogue)
            // pass 3: Div(x,√2) Erf Add(x,1) Mul(x,0.5) Mul(x0) -> Gelu
            for (int i = 0; i < nodes.Count; i++)
            {
                ExecNode n = nodes[i];
                if (n.Op == ExecOp.Conv && n.P is ConvP cp && cp.Pads.All(p => p == 0))
                {
                    int src = n.In[0];
                    ExecNode? pad = nodes.FirstOrDefault(m => m.Out.Length == 1 && m.Out[0] == src && m.Op == ExecOp.Pad);
                    if (pad != null && SoleUse(nodes, src, i) && pad.P is PadP pp &&
                        PadsOnlySpatial(pp.Pads, values[src].Shape.Length))
                    {
                        cp.Pads = [pp.Pads[2], pp.Pads[3], pp.Pads[2 + values[src].Shape.Length], pp.Pads[3 + values[src].Shape.Length]];
                        cp.PadMode = pp.Mode;
                        cp.PadValue = pp.Value;
                        n.In[0] = pad.In[0];
                        nodes.RemoveAt(nodes.IndexOf(pad));
                        i--;
                        continue;
                    }
                }
            }
            // Relu/Sigmoid/elementwise-tail fusions
            FuseBiasAdd(nodes, values, outputValue);
            FuseLayerNorm(nodes, values);
            if (!opt.Gpu) FuseGatedAct(nodes, values, outputValue);
            for (int i = 0; i + 1 < nodes.Count; i++)
            {
                ExecNode a = nodes[i], b = nodes[i + 1];
                if (a.Op == ExecOp.Conv && b.Op == ExecOp.Relu && b.In[0] == a.Out[0] &&
                    SoleUse(nodes, a.Out[0], i + 1))
                {
                    ((ConvP)a.P!).Act = ActivationKind.Relu;
                    // rewrite consumers of b.Out to use a.Out
                    Redirect(nodes, values, b.Out[0], a.Out[0]);
                    nodes.RemoveAt(i + 1);
                    i--;
                }
                else if (a.Op == ExecOp.MaxPool && b.Op == ExecOp.Relu && b.In[0] == a.Out[0] &&
                    SoleUse(nodes, a.Out[0], i + 1))
                {
                    ((PoolP)a.P!).FuseRelu = true;
                    Redirect(nodes, values, b.Out[0], a.Out[0]);
                    nodes.RemoveAt(i + 1);
                    i--;
                }
                else if (a.Op == ExecOp.ResizeBilinear && b.Op == ExecOp.Relu && b.In[0] == a.Out[0] &&
                    SoleUse(nodes, a.Out[0], i + 1))
                {
                    ((ResizeP)a.P!).FuseRelu = true;
                    Redirect(nodes, values, b.Out[0], a.Out[0]);
                    nodes.RemoveAt(i + 1);
                    i--;
                }
                else if (a.Op == ExecOp.Elementwise && b.Op == ExecOp.Relu && b.In[0] == a.Out[0] &&
                    SoleUse(nodes, a.Out[0], i + 1))
                {
                    ((EwP)a.P!).Act = ActivationKind.Relu;
                    Redirect(nodes, values, b.Out[0], a.Out[0]);
                    nodes.RemoveAt(i + 1);
                    i--;
                }
            }
            FuseConvResidual(nodes, values, outputValue);
            FuseGelu(nodes, values);
            FuseConvTBias(nodes, values);
            CommuteActBeforeTranspose(nodes, values);
            FuseLateAct(nodes, values);
        }

        // act(T(x)) == T(act(x)) for elementwise unary act: move a sole-consumer
        // act node in front of its transpose producer so producer fusions (e.g.
        // Gemm->Gelu) can see it.
        private static void CommuteActBeforeTranspose(List<ExecNode> nodes, List<CValue> values)
        {
            for (int g = 0; g < nodes.Count; g++)
            {
                ExecNode act = nodes[g];
                if (act.Op is not (ExecOp.Relu or ExecOp.Gelu or ExecOp.Sigmoid or ExecOp.Erf)
                    || act.In.Length != 1) continue;
                int t = nodes.FindIndex(m => m.Out.Length == 1 && m.Out[0] == act.In[0]);
                if (t < 0 || t >= g || nodes[t].Op != ExecOp.Transpose) continue;
                ExecNode tr = nodes[t];
                if (tr.In.Length != 1 || !SoleUse(nodes, tr.Out[0], g)) continue;
                // rewire as x --act--> y --T--> z: swap the two Out ids so
                // downstream consumers of z read the transposed value as before.
                // y now holds act(x) in x's layout — fix its registered shape.
                int y = tr.Out[0], z = act.Out[0];
                values[y].Shape = values[tr.In[0]].Shape;
                act.In[0] = tr.In[0]; act.Out[0] = y;
                tr.In[0] = y; tr.Out[0] = z;
                nodes.RemoveAt(g);
                nodes.Insert(t, act);
            }
        }

        // Sole-consumer activation fusion, buffer-aware (consumers reached through
        // reshape/squeeze Buf-aliases count as uses of the producer's buffer) and
        // adjacency-independent. Retargets the act's output buffer onto the
        // producer's so downstream aliases keep their shape metadata.
        private static void FuseLateAct(List<ExecNode> nodes, List<CValue> values)
        {
            for (int i = 0; i < nodes.Count; i++)
            {
                ExecNode a = nodes[i];
                bool can = a.P switch
                {
                    ConvP c => c.Act == ActivationKind.None,
                    ConvTP c => c.Act == ActivationKind.None,
                    PoolP p => !p.FuseRelu,
                    ResizeP r => !r.FuseRelu,
                    EwP e => e.Act == ActivationKind.None,
                    GemmP g => g.Act == ActivationKind.None,
                    _ => false,
                };
                if (!can || a.Out.Length != 1) continue;
                int ob = values[a.Out[0]].Buf;
                if (ob < 0) continue;

                int u = -1, uses = 0;
                for (int j = 0; j < nodes.Count && uses <= 1; j++)
                    if (j != i && nodes[j].In.Any(iv => values[iv].Buf == ob)) { u = j; uses++; }
                if (uses != 1) continue;
                ExecNode b = nodes[u];
                if (b.In.Length != 1 || values[b.In[0]].Buf != ob) continue;
                ActivationKind got = b.Op switch
                {
                    ExecOp.Relu => ActivationKind.Relu,
                    ExecOp.Gelu => ActivationKind.Gelu,
                    ExecOp.Sigmoid => ActivationKind.Sigmoid,
                    ExecOp.HardSwish => ActivationKind.HardSwish,
                    ExecOp.Swish => ActivationKind.Swish,
                    _ => ActivationKind.None,
                };
                bool ok = a.P switch
                {
                    ConvP => got is ActivationKind.Relu or ActivationKind.Gelu or ActivationKind.HardSwish or ActivationKind.Swish,
                    ConvTP => got is ActivationKind.Relu or ActivationKind.Sigmoid,
                    PoolP or ResizeP or EwP => got == ActivationKind.Relu,
                    GemmP => got is ActivationKind.Relu or ActivationKind.Gelu or ActivationKind.HardSwish or ActivationKind.Swish,
                    _ => false,
                };
                if (!ok) continue;

                switch (a.P)
                {
                    case ConvP c: c.Act = got; break;
                    case ConvTP c: c.Act = got; break;
                    case PoolP p: p.FuseRelu = true; break;
                    case ResizeP r: r.FuseRelu = true; break;
                    case EwP e: e.Act = ActivationKind.Relu; break;
                    case GemmP g: g.Act = got; break;
                }
                int ab = values[b.Out[0]].Buf;
                if (ab >= 0 && ab != ob)
                {
                    foreach (CValue v in values)
                        if (v.Buf == ab) v.Buf = ob;
                    // b.Out is now an unproduced view of a.Out: record it so the
                    // layout pass can follow the alias chain
                    values[b.Out[0]].AliasOf = a.Out[0];
                }
                else
                {
                    Redirect(nodes, values, b.Out[0], a.Out[0]);
                }
                nodes.RemoveAt(u);
                if (u < i) i--;
            }
        }

        /// <summary>
        /// Conv -> MaxPool(2x2/2) memory fusion (after layout planning): the conv output's only
        /// reader is the pool, so the conv emits pooled rows directly (<see cref="ConvP.Pool"/>) and
        /// the full-resolution conv output is never stored. GPU plans only get it for small-ic stem
        /// convs (<see cref="GpuSmallConv"/>, staged weights).
        /// </summary>
        private static void FuseMemory(List<ExecNode> nodes, List<CValue> values, int outputValue, bool gpu)
        {
            var ctx = new PassContext(nodes, values, outputValue, new CompileOptions(true));
            for (int pi = 0; pi < nodes.Count; pi++)
            {
                ExecNode m = nodes[pi];
                if (m.Op != ExecOp.MaxPool || m.P is not PoolP pp || pp.Kernel is not [2, 2] || pp.Strides is not [2, 2]
                    || pp.Pads.Any(x => x != 0) || m.In.Length != 1) continue;
                int v = m.In[0], ci = ctx.Producer(v);
                if (ci < 0 || ci > pi || nodes[ci].P is not ConvP cp || cp.Group != 1 || cp.AddValue >= 0 || cp.Pool is not null
                    || ctx.Uses(v) != 1 || !ctx.IsNhwc4(v) || !ctx.IsNhwc4(m.Out[0])) continue;
                int[] cs = values[v].Shape, ps = values[m.Out[0]].Shape;
                if (cs[2] != ps[2] * 2 || cs[3] != ps[3] * 2) continue;
                int xin = nodes[ci].In[0];
                if (gpu && (!GpuSmallConv(cp, values[nodes[ci].In[1]].Shape, values[xin].Shape[1])
                    || values[xin].Shape[2] != cs[2] || values[xin].Shape[3] != cs[3])) continue;
                cp.Pool = pp;
                nodes[ci].Out = [m.Out[0]];
                nodes.RemoveAt(pi);
                pi--;
            }
        }

        /// <summary>
        /// Resize -> Add(., [Relu](skip)) becomes a Resize with an addend (In[1]) in NHWC plans, so
        /// the full-resolution resize output is never stored. The fused node takes the Add's position.
        /// </summary>
        private static void FuseResizeAdd(List<ExecNode> nodes, List<CValue> values, int outputValue)
        {
            var ctx = new PassContext(nodes, values, outputValue, new CompileOptions(true));
            for (int ai = 0; ai < nodes.Count; ai++)
            {
                ExecNode add = nodes[ai];
                if (add.Op != ExecOp.Elementwise || add.In.Length != 2 || add.P is not EwP { Op: EwOp.Add, Act: ActivationKind.None })
                    continue;
                ExecNode? rs = null, relu = null;
                int skip = -1;
                for (int k = 0; k < 2 && rs is null; k++)
                {
                    int u = add.In[k], o = add.In[1 - k];
                    int ri = ctx.Producer(u);
                    if (ri < 0 || nodes[ri].Op != ExecOp.ResizeBilinear || nodes[ri].In.Length != 1) continue;
                    if (ctx.Uses(u) != 1 || !ctx.SameShape(u, add.Out[0]) || !ctx.SameShape(o, add.Out[0]) || values[o].Buf < 0) continue;
                    rs = nodes[ri];
                    skip = o;
                    int pi = ctx.Producer(o);
                    if (pi >= 0 && nodes[pi].Op == ExecOp.Relu && nodes[pi].In.Length == 1 && ctx.Uses(o) == 1
                        && values[nodes[pi].In[0]].Buf >= 0 && ctx.SameShape(nodes[pi].In[0], o))
                    {
                        relu = nodes[pi];
                        skip = relu.In[0];
                    }
                }
                if (rs is null) continue;
                int rIdx = nodes.IndexOf(rs);
                rs.In = [rs.In[0], skip];
                ((ResizeP)rs.P!).AddRelu = relu is not null;
                rs.Out = [add.Out[0]];
                nodes[ai] = rs;      // producer precedes consumer: rIdx < ai
                nodes.RemoveAt(rIdx);
                if (relu is not null) nodes.Remove(relu);
                ai = -1;
            }
        }

        /// <summary>
        /// Conv -> Add(·, residual) (skip connections): the same-shape addend folds into the conv epilogue.
        /// Matches through zero-copy aliases (Identity / reshape views) and non-adjacent Adds, as long as
        /// the addend is produced before the conv and the conv output has no other reader.
        /// </summary>
        private static void FuseConvResidual(List<ExecNode> nodes, List<CValue> values, int outputValue)
        {
            int Root(int v)
            {
                while (values[v].AliasOf >= 0 && values[values[v].AliasOf].Buf == values[v].Buf) v = values[v].AliasOf;
                return v;
            }
            int Producer(int v) => nodes.FindIndex(m => m.Out.Length == 1 && m.Out[0] == v);
            int Readers(int buf, out int only)
            {
                int count = 0;
                only = -1;
                for (int j = 0; j < nodes.Count; j++)
                {
                    ExecNode m = nodes[j];
                    if (m.In.Any(iv => values[iv].Buf == buf) || m.ParamRefs().Any(iv => values[iv].Buf == buf))
                    {
                        count++;
                        only = j;
                    }
                }
                return count;
            }
            for (int bi = 0; bi < nodes.Count; bi++)
            {
                ExecNode b = nodes[bi];
                if (b.Op != ExecOp.Elementwise || b.In.Length != 2 || b.P is not EwP { Op: EwOp.Add } ep ||
                    ep.Act is not (ActivationKind.None or ActivationKind.Relu))
                    continue;
                for (int k = 0; k < 2; k++)
                {
                    int ci = Producer(Root(b.In[k]));
                    if (ci < 0 || ci > bi || nodes[ci].Op != ExecOp.Conv || nodes[ci].P is not ConvP cp) continue;
                    if (cp.Act != ActivationKind.None || cp.Group > 1 || cp.AddValue >= 0) continue;
                    int outV = nodes[ci].Out[0], other = b.In[1 - k];
                    int buf = values[outV].Buf;
                    if (buf < 0 || buf == values[outputValue].Buf || Readers(buf, out int only) != 1 || only != bi) continue;
                    if (!values[other].Shape.SequenceEqual(values[b.Out[0]].Shape) ||
                        !values[outV].Shape.SequenceEqual(values[b.Out[0]].Shape)) continue;
                    int op = Producer(Root(other));
                    if (op >= ci) continue;
                    cp.AddValue = other;
                    cp.Act = ep.Act;
                    // the conv now writes the Add's result; readers of the old Add output follow
                    int addOut = b.Out[0];
                    nodes[ci].Out = [addOut];
                    nodes.RemoveAt(bi);
                    bi--;
                    break;
                }
            }
        }

        internal static void Redirect(List<ExecNode> nodes, List<CValue> values, int oldValue, int newValue)
        {
            foreach (ExecNode m in nodes)
                for (int i = 0; i < m.In.Length; i++)
                    if (m.In[i] == oldValue) m.In[i] = newValue;
            if (values[oldValue].Name.Length != 0 && values[newValue].Name.Length == 0)
                values[newValue].Name = values[oldValue].Name;
        }

        /// <summary>True when value `v`'s only consumer is node index `useIdx`, once.</summary>
        internal static bool SoleUse(List<ExecNode> nodes, int v, int useIdx)
        {
            int count = 0;
            for (int i = 0; i < nodes.Count; i++)
                foreach (int inp in nodes[i].In)
                    if (inp == v)
                    {
                        if (i != useIdx) return false;
                        if (++count > 1) return false;
                    }
            return count == 1;
        }

        private static bool PadsOnlySpatial(int[] pads, int rank)
        {
            if (pads.Length != 2 * rank) return false;
            for (int i = 0; i < 2; i++) if (pads[i] != 0 || pads[i + rank] != 0) return false;
            return true;
        }

        private static void FuseGelu(List<ExecNode> nodes, List<CValue> values)
        {
            bool IsEw(ExecNode m, EwOp op) => m.Op == ExecOp.Elementwise && ((EwP)m.P!).Op == op && m.In.Length == 2;
            int Other(ExecNode m, int v) => m.In[0] == v ? m.In[1] : m.In[1] == v ? m.In[0] : -1;
            // paddle2onnx order: Div(X, sqrt2) -> Erf -> Add(+1) -> Mul(X, ·) -> Mul(·, 0.5)
            for (int i = 0; i + 4 < nodes.Count; i++)
            {
                ExecNode d = nodes[i], erf = nodes[i + 1], a1 = nodes[i + 2], mx = nodes[i + 3], mh = nodes[i + 4];
                if (!IsEw(d, EwOp.Div) || erf.Op != ExecOp.Erf || !IsEw(a1, EwOp.Add) || !IsEw(mx, EwOp.Mul) || !IsEw(mh, EwOp.Mul))
                    continue;
                if (erf.In[0] != d.Out[0] || !IsScalarConst(values, d.In[1], MathF.Sqrt(2f))) continue;
                int o1 = Other(a1, erf.Out[0]);
                if (o1 < 0 || !IsScalarConst(values, o1, 1f)) continue;
                if (Other(mx, a1.Out[0]) != d.In[0]) continue;
                int o2 = Other(mh, mx.Out[0]);
                if (o2 < 0 || !IsScalarConst(values, o2, 0.5f)) continue;
                if (!SoleUse(nodes, d.Out[0], i + 1) || !SoleUse(nodes, erf.Out[0], i + 2)
                    || !SoleUse(nodes, a1.Out[0], i + 3) || !SoleUse(nodes, mx.Out[0], i + 4)) continue;
                values[mh.Out[0]].Shape = values[d.In[0]].Shape;
                nodes[i] = new ExecNode { Op = ExecOp.Gelu, In = [d.In[0]], Out = mh.Out, P = null, SrcNode = d.SrcNode, Name = d.Name };
                nodes.RemoveRange(i + 1, 4);
            }
            // Div(X, sqrt2) -> Erf -> Add(+1) -> Mul(*0.5) -> Mul(*X)
            for (int i = 0; i + 4 < nodes.Count; i++)
            {
                ExecNode d = nodes[i], erf = nodes[i + 1], a1 = nodes[i + 2], mh = nodes[i + 3], mx = nodes[i + 4];
                if (d.Op != ExecOp.Elementwise || ((EwP)d.P!).Op != EwOp.Div ||
                    erf.Op != ExecOp.Erf ||
                    a1.Op != ExecOp.Elementwise || ((EwP)a1.P!).Op != EwOp.Add ||
                    mh.Op != ExecOp.Elementwise || ((EwP)mh.P!).Op != EwOp.Mul ||
                    mx.Op != ExecOp.Elementwise || ((EwP)mx.P!).Op != EwOp.Mul)
                    continue;
                if (erf.In[0] != d.Out[0] || a1.In[0] != erf.Out[0] && a1.In[1] != erf.Out[0] ||
                    (mh.In[0] != a1.Out[0] && mh.In[1] != a1.Out[0]) ||
                    (mx.In[0] != mh.Out[0] && mx.In[1] != mh.Out[0]))
                    continue;
                if (!IsScalarConst(values, d.In[1], MathF.Sqrt(2f))) continue;
                int a1Other = a1.In[0] == erf.Out[0] ? a1.In[1] : a1.In[0];
                int mhOther = mh.In[0] == a1.Out[0] ? mh.In[1] : mh.In[0];
                int mxOther = mx.In[0] == mh.Out[0] ? mx.In[1] : mx.In[0];
                if (!IsScalarConst(values, a1Other, 1f) || !IsScalarConst(values, mhOther, 0.5f)) continue;
                if (mxOther != d.In[0]) continue;
                // emit single Gelu
                values[mx.Out[0]].Shape = values[d.In[0]].Shape;
                nodes[i] = new ExecNode { Op = ExecOp.Gelu, In = [d.In[0]], Out = mx.Out, P = null, SrcNode = d.SrcNode, Name = d.Name };
                nodes.RemoveRange(i + 1, 4);
            }
        }

        private static bool IsScalarConst(List<CValue> values, int v, float expected)
        {
            CValue cv = values[v];
            return cv.Floats is { Length: 1 } f && MathF.Abs(f[0] - expected) < 1e-6f;
        }

        /// <summary>Conv/Gemm -> Add(per-channel const) folds into the producer's bias (the producer then
        /// writes the Add's output value directly).</summary>
        private static void FuseBiasAdd(List<ExecNode> nodes, List<CValue> values, int outputValue)
        {
            for (int i = 0; i < nodes.Count; i++)
            {
                ExecNode b = nodes[i];
                if (b.Op != ExecOp.Elementwise || b.P is not EwP { Op: EwOp.Add } ep || b.In.Length != 2) continue;
                for (int k = 0; k < 2; k++)
                {
                    int pv = b.In[k], cv = b.In[1 - k];
                    if (values[cv].Floats is not { } cf || values[pv].Buf < 0 || pv == outputValue) continue;
                    int pi = nodes.FindIndex(m => m.Out.Length == 1 && m.Out[0] == pv);
                    if (pi < 0 || pi > i || !SoleUse(nodes, pv, i) || values.Any(v => v.AliasOf == pv)) continue;
                    if (!values[b.Out[0]].Shape.SequenceEqual(values[pv].Shape)) continue;
                    ExecNode a = nodes[pi];
                    int[] os = values[pv].Shape, s = values[cv].Shape;
                    int ch, existing;
                    if (a.P is ConvP { Act: ActivationKind.None, AddValue: < 0 } cp && os.Length == 4)
                    {
                        ch = os[1];
                        // const right-aligned against [N,C,H,W]: only the channel dim may differ from 1
                        bool ok = s.Length >= 3 && s[^3] == ch && cf.Length == ch;
                        if (!ok) continue;
                        existing = cp.BiasValue;
                    }
                    else if (a.P is GemmP { Act: ActivationKind.None } gp)
                    {
                        ch = os[^1];
                        if (s.Length < 1 || s[^1] != ch || cf.Length != ch) continue;
                        existing = gp.BiasValue;
                    }
                    else continue;
                    float[] bias = (float[])cf.Clone();
                    if (existing >= 0)
                    {
                        float[] e = values[existing].Floats!;
                        for (int c = 0; c < ch; c++) bias[c] += e[c];
                    }
                    int bv = values.Count;
                    values.Add(new CValue { Name = values[cv].Name + "__bias", Shape = [ch], DType = 1, Floats = bias });
                    if (a.P is ConvP c2) { c2.BiasValue = bv; c2.Act = ep.Act; }
                    else if (a.P is GemmP g2) { g2.BiasValue = bv; g2.Act = ep.Act; }
                    a.Out = [b.Out[0]];
                    nodes.RemoveAt(i);
                    i--;
                    break;
                }
            }
        }

        /// <summary>CPU plans: Mul(x, HardSigmoid(x)) → HardSwish(x), Mul(x, Sigmoid(x)) → Swish(x); conv/gemm
        /// epilogues then absorb them (<see cref="FuseLateAct"/>).</summary>
        private static void FuseGatedAct(List<ExecNode> nodes, List<CValue> values, int outputValue)
        {
            for (int i = 0; i < nodes.Count; i++)
            {
                ExecNode mul = nodes[i];
                if (mul.Op != ExecOp.Elementwise || mul.P is not EwP { Op: EwOp.Mul, Act: ActivationKind.None } || mul.In.Length != 2) continue;
                for (int k = 0; k < 2; k++)
                {
                    int x = mul.In[k], g = mul.In[1 - k];
                    int gi = nodes.FindIndex(m => m.Out.Length == 1 && m.Out[0] == g);
                    if (gi < 0 || gi > i || nodes[gi].In[0] != x || g == outputValue || !SoleUse(nodes, g, i)
                        || values.Any(v => v.AliasOf == g)) continue;
                    ExecNode gate = nodes[gi];
                    ExecOp op;
                    if (gate.Op == ExecOp.Sigmoid) op = ExecOp.Swish;
                    else if (gate.Op == ExecOp.HardSigmoid && gate.P is HardSigmoidP hp
                        && MathF.Abs(hp.Alpha - 1f / 6f) < 1e-6f && hp.Beta == 0.5f) op = ExecOp.HardSwish;
                    else continue;
                    if (!values[mul.Out[0]].Shape.SequenceEqual(values[x].Shape)) continue;
                    nodes[i] = new ExecNode { Op = op, In = [x], Out = mul.Out, P = null, SrcNode = mul.SrcNode, Name = mul.Name };
                    nodes.RemoveAt(gi);
                    i--;
                    break;
                }
            }
        }

        /// <summary>paddle2onnx's decomposed layer norm over the last axis:
        /// m = mean(x); d = x - m; y = d / sqrt(mean(d·d) + eps) · gamma + beta.</summary>
        private static void FuseLayerNorm(List<ExecNode> nodes, List<CValue> values)
        {
            int Prod(int v) => nodes.FindIndex(m => m.Out.Length == 1 && m.Out[0] == v);
            int Uses(int v) => nodes.Sum(m => m.In.Count(x => x == v)) + values.Count(x => x.AliasOf == v);
            bool Ew(ExecNode m, EwOp op) => m.Op == ExecOp.Elementwise && m.P is EwP { Act: ActivationKind.None } e && e.Op == op && m.In.Length == 2;
            bool Const(int v) => values[v].Floats is not null;
            for (int i = 0; i < nodes.Count; i++)
            {
                ExecNode add = nodes[i];
                if (!Ew(add, EwOp.Add)) continue;
                int mulV = add.In[0], beta = add.In[1];
                if (!Const(beta)) (mulV, beta) = (beta, mulV);
                int mi = Prod(mulV);
                if (!Const(beta) || mi < 0 || !Ew(nodes[mi], EwOp.Mul) || Uses(mulV) != 1) continue;
                ExecNode mul = nodes[mi];
                int divV = mul.In[0], gamma = mul.In[1];
                if (!Const(gamma)) (divV, gamma) = (gamma, divV);
                int di = Prod(divV);
                if (!Const(gamma) || di < 0 || !Ew(nodes[di], EwOp.Div) || Uses(divV) != 1) continue;
                ExecNode div = nodes[di];
                int subV = div.In[0], sqrtV = div.In[1];
                int qi = Prod(sqrtV), si = Prod(subV);
                if (qi < 0 || si < 0 || nodes[qi].Op != ExecOp.Sqrt || Uses(sqrtV) != 1 || !Ew(nodes[si], EwOp.Sub)) continue;
                int epsV = nodes[qi].In[0], ei = Prod(epsV);
                if (ei < 0 || !Ew(nodes[ei], EwOp.Add) || Uses(epsV) != 1) continue;
                ExecNode epsAdd = nodes[ei];
                int varV = epsAdd.In[0], epsC = epsAdd.In[1];
                if (!Const(epsC)) (varV, epsC) = (epsC, varV);
                int vi = Prod(varV);
                if (values[epsC].Floats is not { Length: 1 } ef || vi < 0 || nodes[vi].Op != ExecOp.ReduceMean || Uses(varV) != 1) continue;
                int powV = nodes[vi].In[0], pi = Prod(powV);
                if (pi < 0 || !Ew(nodes[pi], EwOp.Mul) || nodes[pi].In[0] != subV || nodes[pi].In[1] != subV || Uses(powV) != 1) continue;
                ExecNode sub = nodes[si];
                int x = sub.In[0], meanV = sub.In[1], ri = Prod(meanV);
                if (ri < 0 || nodes[ri].Op != ExecOp.ReduceMean || nodes[ri].In[0] != x || Uses(meanV) != 1 || Uses(subV) != 3) continue;
                int c = values[x].Shape[^1];
                bool PerLast(int v) => values[v].Floats!.Length == c && values[v].Shape.Length >= 1 && values[v].Shape[^1] == c;
                if (!PerLast(gamma) || !PerLast(beta)) continue;
                var ln = new ExecNode
                {
                    Op = ExecOp.LayerNorm, In = [x, gamma, beta], Out = add.Out,
                    P = new LnP { Eps = ef[0] }, SrcNode = add.SrcNode, Name = add.Name,
                };
                ExecNode[] dead = [nodes[ri], sub, nodes[pi], nodes[vi], epsAdd, nodes[qi], div, mul];
                nodes[i] = ln;
                foreach (ExecNode d in dead) nodes.Remove(d);
                i = nodes.IndexOf(ln);
            }
        }

        /// <summary>ConvTranspose -> Add(const [1,oc,1,1]) folds into the transposed conv's bias.</summary>
        private static void FuseConvTBias(List<ExecNode> nodes, List<CValue> values)
        {
            for (int i = 0; i + 1 < nodes.Count; i++)
            {
                ExecNode a = nodes[i], b = nodes[i + 1];
                if (a.P is not ConvTP cp || cp.BiasValue >= 0
                    || b.Op != ExecOp.Elementwise || b.P is not EwP { Op: EwOp.Add } ep || b.In.Length != 2) continue;
                int o = b.In[0] == a.Out[0] ? b.In[1] : b.In[1] == a.Out[0] ? b.In[0] : -1;
                int oc = values[a.Out[0]].Shape[1];
                if (o < 0 || !SoleUse(nodes, a.Out[0], i + 1) || values[o].Floats is not { Length: var len } || len != oc
                    || values[o].Shape.Length == 4 && values[o].Shape[1] != oc) continue;
                cp.BiasValue = o;
                cp.Act = ep.Act;
                Redirect(nodes, values, b.Out[0], a.Out[0]);
                nodes.RemoveAt(i + 1);
                i--;
            }
        }

        /// <summary>Lifetime-based arena packing (def-order, free-list reuse).</summary>
        private static void PlanArena(List<ExecNode> nodes, List<CValue> values,
            List<int> bufSize, int inputValue, int outputValue,
            out int[] offsets, out int arenaFloats)
        {
            int bufCount = bufSize.Count;
            const int Align = 16; // floats (64B)
            int AlignUp(int v) => (v + Align - 1) & ~(Align - 1);

            int[] def = Enumerable.Repeat(int.MaxValue, bufCount).ToArray();
            int[] use = Enumerable.Repeat(-1, bufCount).ToArray();
            for (int ni = 0; ni < nodes.Count; ni++)
            {
                ExecNode n = nodes[ni];
                foreach (int iv in n.In)
                {
                    int b = values[iv].Buf;
                    if (b >= 0) { use[b] = Math.Max(use[b], ni); def[b] = Math.Min(def[b], ni); }
                }
                // value ids carried in params (conv bias/residual, gemm bias, extensions) count as uses
                foreach (int iv in n.ParamRefs())
                {
                    int b = values[iv].Buf;
                    if (b >= 0) { use[b] = Math.Max(use[b], ni); def[b] = Math.Min(def[b], ni); }
                }
                foreach (int ov in n.Out)
                {
                    int b = values[ov].Buf;
                    if (b >= 0) def[b] = Math.Min(def[b], ni);
                }
            }
            int inputBuf = values[inputValue].Buf;
            int outputBuf = values[outputValue].Buf;
            def[inputBuf] = -1;
            // keep input + output live for the whole run: callers may read
            // Input (e.g. two-stage pipelines reusing it) after Run().
            use[inputBuf] = nodes.Count;
            use[outputBuf] = nodes.Count;

            var order = Enumerable.Range(0, bufCount)
                .Where(b => def[b] != int.MaxValue)
                .OrderBy(b => def[b]).ThenByDescending(b => bufSize[b]).ToList();
            var active = new List<int>();   // placed buf ids still live
            var free = new List<(int off, int size)>();
            offsets = new int[bufCount];
            Array.Fill(offsets, -1);
            int bump = 0;
            foreach (int b in order)
            {
                // release dead bufs before allocating this one
                for (int i = active.Count - 1; i >= 0; i--)
                {
                    int ob = active[i];
                    if (use[ob] < def[b])
                    {
                        active.RemoveAt(i);
                        free.Add((offsets[ob], AlignUp(bufSize[ob])));
                    }
                }
                int need = AlignUp(bufSize[b]);
                int bestIdx = -1;
                for (int fi = 0; fi < free.Count; fi++)
                    if (free[fi].size >= need && (bestIdx < 0 || free[fi].size < free[bestIdx].size))
                        bestIdx = fi;
                if (bestIdx >= 0)
                {
                    var chunk = free[bestIdx];
                    offsets[b] = chunk.off;
                    free.RemoveAt(bestIdx);
                    if (chunk.size > need) free.Add((chunk.off + need, chunk.size - need));
                }
                else
                {
                    offsets[b] = bump;
                    bump += need;
                }
                active.Add(b);
            }
            arenaFloats = bump;

            // greedy by size (largest first, lowest-waste gap among lifetime-overlapping
            // placements): usually much tighter than def-order free lists; keep the better one
            {
                // the input (live for the whole run) goes first, at offset 0: pipelines can
                // then stack another plan's arena right after it
                var bySize = order.OrderByDescending(b => b == inputBuf).ThenByDescending(b => bufSize[b]).ThenBy(b => def[b]).ToList();
                var placed = new List<int>(bySize.Count);
                int[] off2 = new int[bufCount];
                Array.Fill(off2, -1);
                int top = 0;
                var conflicts = new List<(int off, int end)>();
                foreach (int b in bySize)
                {
                    int need = AlignUp(bufSize[b]);
                    conflicts.Clear();
                    foreach (int o in placed)
                        if (!(use[o] < def[b] || use[b] < def[o]))
                            conflicts.Add((off2[o], off2[o] + AlignUp(bufSize[o])));
                    conflicts.Sort();
                    int best = -1, bestWaste = int.MaxValue, cur = 0;
                    foreach (var (o, e) in conflicts)
                    {
                        if (o - cur >= need && o - cur - need < bestWaste) { best = cur; bestWaste = o - cur - need; }
                        cur = Math.Max(cur, e);
                    }
                    off2[b] = best >= 0 ? best : cur;
                    top = Math.Max(top, off2[b] + need);
                    placed.Add(b);
                }
                if (top <= bump)
                {
                    offsets = off2;
                    arenaFloats = top;
                }
            }
            if (Env.Flag("DUMP") == "3")
            {
                long peak = 0; int at = -1;
                for (int ni = -1; ni <= nodes.Count; ni++)
                {
                    long s = 0;
                    for (int b = 0; b < bufCount; b++) if (def[b] != int.MaxValue && def[b] <= ni && use[b] >= ni) s += AlignUp(bufSize[b]);
                    if (s > peak) { peak = s; at = ni; }
                }
                Console.Error.WriteLine($"[arena] {arenaFloats * 4.0 / 1e6:F1} MB (freelist {bump * 4.0 / 1e6:F1}), live-peak {peak * 4.0 / 1e6:F1} MB at #{at}, input {bufSize[inputBuf] * 4.0 / 1e6:F1} MB off {offsets[inputBuf]}");
                for (int b = 0; b < bufCount; b++)
                    if (def[b] != int.MaxValue && def[b] <= at && use[b] >= at)
                        Console.Error.WriteLine($"    b{b,-4} {bufSize[b] * 4.0 / 1e6,8:F1} MB  #{def[b]}..#{use[b]}  {(def[b] >= 0 && def[b] < nodes.Count ? nodes[def[b]].Op : "input")}");
            }
        }

        private static readonly HashSet<string> IntOps = new(StringComparer.Ordinal)
            { "Squeeze", "Unsqueeze", "Div", "Mod", "Add", "Sub", "Mul", "Reshape",
              "Concat", "Slice", "Gather", "Cast", "Floor", "Ceil", "Neg", "Abs",
              "Min", "Max", "Equal", "Less", "Where", "ConstantOfShape", "Expand",
              "Identity", "Tile" };

        private void EmitNode(OnnxNodeData n)
        {
            string op = n.OpType;
            // Shape → int tensor of input dims
            if (op == "Shape")
            {
                CValue x = V(Input(0, n));
                if (x.Shape is null) throw new InvalidDataException("Shape input has no known shape.");
                int start = (int)n.IntAttr("start", 0);
                int end = (int)n.IntAttr("end", x.Shape.Length);
                int rank = x.Shape.Length;
                if (start < 0) start += rank;
                if (end < 0) end += rank;
                start = Math.Clamp(start, 0, rank);
                end = Math.Clamp(end, 0, rank);
                long[] dims = x.Shape.AsSpan()[start..end].ToArray().Select(d => (long)d).ToArray();
                AddValue(new CValue { Name = n.Outputs[0], Shape = [dims.Length], DType = 7, Ints = dims });
                return;
            }
            if (op == "Size")
            {
                CValue x = V(Input(0, n));
                AddValue(new CValue { Name = n.Outputs[0], Shape = [], DType = 7, Ints = [x.ElementCount] });
                return;
            }
            if (op == "Constant")
            {
                OnnxAttributeData? a = n.Attr("value");
                if (a?.Tensor is OnnxTensorData t)
                {
                    if (t.DataType == OnnxTensorData.TypeFloat)
                        AddValue(new CValue { Name = n.Outputs[0], Shape = t.DimsArray(), DType = 1, Floats = t.AsFloats() });
                    else
                        AddValue(new CValue { Name = n.Outputs[0], Shape = t.DimsArray(), DType = t.DataType, Ints = t.AsInt64s() });
                    return;
                }
                throw new NotSupportedException("Constant without tensor value.");
            }

            // int-only evaluation: if every present input is a const int tensor
            // and the op is elementwise/shape math, evaluate now.
            bool allInt = n.Inputs.All(s => s.Length == 0 || (V(_byName[s]).IsInt));
            if (allInt && IntOps.Contains(op))
            {
                EvalInt(n);
                return;
            }
            // Cast on int → still int
            if (op == "Cast" && V(Input(0, n)).IsInt)
            {
                int to = (int)n.IntAttr("to");
                CValue x = V(Input(0, n));
                if (to == 1)
                    AddValue(new CValue { Name = n.Outputs[0], Shape = x.Shape, DType = 1, Floats = x.Ints!.Select(v => (float)v).ToArray() });
                else
                    AddValue(new CValue { Name = n.Outputs[0], Shape = x.Shape, DType = 7, Ints = x.Ints });
                return;
            }

            // ---- float ops ----
            switch (op)
            {
                case "Pad": EmitPad(n); break;
                case "Conv": EmitConv(n); break;
                case "MaxPool": EmitPool(n); break;
                case "Relu": EmitUnary(n, ExecOp.Relu); break;
                case "Sigmoid": EmitUnary(n, ExecOp.Sigmoid); break;
                case "Erf": EmitUnary(n, ExecOp.Erf); break;
                case "LayerNormalization": EmitLayerNorm(n); break;
                case "Gemm": EmitGemm(n); break;
                case "Resize": EmitResize(n); break;
                case "Transpose": EmitTranspose(n); break;
                case "Slice": EmitSlice(n); break;
                case "Concat": EmitConcat(n); break;
                case "Split": EmitSplit(n); break;
                case "Reshape": EmitReshape(n); break;
                case "Squeeze": EmitSqueeze(n); break;
                case "Unsqueeze": EmitUnsqueeze(n); break;
                case "Add" or "Sub" or "Mul" or "Div": EmitElementwise(n, op); break;
                case "Identity": Alias(n); break;
                case "HardSigmoid":
                    Emit(ExecOp.HardSigmoid, [Input(0, n)], ActivationValue(n.Outputs[0], V(Input(0, n)).Shape),
                        new HardSigmoidP { Alpha = n.FloatAttr("alpha", 0.2f), Beta = n.FloatAttr("beta", 0.5f) }, n);
                    break;
                case "GlobalAveragePool": EmitGlobalAvg(n); break;
                case "ReduceMean": EmitReduceMean(n); break;
                case "ConvTranspose": EmitConvTranspose(n); break;
                case "AveragePool": EmitPool(n, avg: true); break;
                case "BatchNormalization": EmitBatchNorm(n); break;
                case "MatMul": EmitMatMul(n); break;
                case "Softmax": EmitSoftmax(n); break;
                case "Sqrt": EmitUnary(n, ExecOp.Sqrt); break;
                case "Pow": EmitPow(n); break;
                default:
                    throw new NotSupportedException($"Op '{op}' not supported (node '{n.Name}').");
            }
        }

        // ---------- int evaluation ----------
        private void EvalInt(OnnxNodeData n)
        {
            string op = n.OpType;
            CValue In(int i) => V(Input(i, n));
            CValue Out(long[] data, int[] shape)
            {
                var v = new CValue { Name = n.Outputs[0], Shape = shape, DType = 7, Ints = data };
                AddValue(v);
                return v;
            }
            switch (op)
            {
                case "Squeeze":
                {
                    CValue x = In(0);
                    long[]? axes = IntsArg(n, 1, "axes");
                    int[] shape;
                    if (axes is null)
                        shape = x.Shape.Where(d => d != 1).ToArray();
                    else
                    {
                        var dims = x.Shape.ToList();
                        foreach (long a in axes.OrderByDescending(v => v))
                        {
                            int ax = (int)(a < 0 ? a + dims.Count : a);
                            if (dims[ax] != 1) throw new InvalidDataException("Squeeze axis not 1.");
                            dims.RemoveAt(ax);
                        }
                        shape = dims.ToArray();
                    }
                    Out(x.Ints!, shape);
                    return;
                }
                case "Unsqueeze":
                {
                    CValue x = In(0);
                    long[] axes = IntsArg(n, 1, "axes")!;
                    var dims = x.Shape.ToList();
                    foreach (long a in axes.OrderBy(v => v))
                    {
                        int ax = (int)(a < 0 ? a + dims.Count + 1 : a);
                        dims.Insert(ax, 1);
                    }
                    Out(x.Ints!, dims.ToArray());
                    return;
                }
                case "Reshape":
                {
                    CValue x = In(0);
                    long[] spec = In(1).Ints!;
                    bool allowZero = n.IntAttr("allowzero") == 1;
                    Out(x.Ints!, ResolveReshape(x.Shape, spec, allowZero, x.Ints!.Length));
                    return;
                }
                case "Concat":
                {
                    int axis = (int)n.IntAttr("axis");
                    var inputs = Inputs(n).Select(V).ToArray();
                    int rank = inputs[0].Shape.Length;
                    if (axis < 0) axis += rank;
                    int[] shape = (int[])inputs[0].Shape.Clone();
                    shape[axis] = inputs.Sum(v => v.Shape[axis]);
                    long[] data = IntConcat(inputs.Select(v => v.Ints!).ToArray(),
                        inputs.Select(v => v.Shape).ToArray(), axis, shape);
                    Out(data, shape);
                    return;
                }
                case "Slice":
                {
                    CValue x = In(0);
                    long[] starts = IntsArg(n, 1, "starts")!;
                    long[] ends = IntsArg(n, 2, "ends") ?? x.Shape.Select(d => (long)d).ToArray();
                    long[] axes = IntsArg(n, 3, "axes") ?? Enumerable.Range(0, starts.Length).Select(i => (long)i).ToArray();
                    long[] steps = IntsArg(n, 4, "steps") ?? starts.Select(_ => 1L).ToArray();
                    SliceBounds(x.Shape, starts, ends, axes, steps,
                        out int[] outShape, out int[] st, out int[] en, out int[] stepI);
                    long[] data = IntSlice(x.Ints!, x.Shape,
                        axes.Select(a => (int)(a < 0 ? a + x.Shape.Length : a)).ToArray(), st, stepI, outShape);
                    Out(data, outShape);
                    return;
                }
                case "Gather":
                {
                    CValue x = In(0); CValue idx = In(1);
                    int axis = (int)n.IntAttr("axis");
                    if (axis < 0) axis += x.Shape.Length;
                    if (x.Shape.Length != 1 || axis != 0) throw new NotSupportedException("Gather eval supports 1-D axis 0.");
                    Out(idx.Ints!.Select(i => x.Ints![checked((int)i)]).ToArray(), idx.Shape);
                    return;
                }
                case "Identity":
                {
                    CValue x = In(0);
                    Out(x.Ints!, x.Shape);
                    return;
                }
                case "Add" or "Sub" or "Mul" or "Div" or "Mod" or "Min" or "Max":
                {
                    long[] a = In(0).Ints!, b = In(1).Ints!;
                    long[] r = new long[Math.Max(a.Length, b.Length)];
                    for (int i = 0; i < r.Length; i++)
                    {
                        long av = a.Length == 1 ? a[0] : a[i], bv = b.Length == 1 ? b[0] : b[i];
                        r[i] = op switch
                        {
                            "Add" => av + bv, "Sub" => av - bv, "Mul" => av * bv,
                            "Div" => av / bv, "Mod" => ((av % bv) + bv) % bv,
                            "Min" => Math.Min(av, bv), _ => Math.Max(av, bv),
                        };
                    }
                    int[] shape = In(0).Shape.Length >= In(1).Shape.Length ? In(0).Shape : In(1).Shape;
                    Out(r, shape);
                    return;
                }
                case "Neg" or "Abs" or "Floor" or "Ceil":
                {
                    long[] a = In(0).Ints!;
                    // Floor/Ceil on ints are identity
                    long[] r = a.Select(v => op == "Neg" ? -v : op == "Abs" ? Math.Abs(v) : v).ToArray();
                    Out(r, In(0).Shape);
                    return;
                }
                default:
                    throw new NotSupportedException($"Int eval for '{op}' not implemented (node '{n.Name}').");
            }
        }

        // ---------- float ops ----------
        private CValue ActivationValue(string name, int[] shape)
        {
            int elems = 1;
            foreach (int d in shape) elems = checked(elems * d);
            return new CValue { Name = name, Shape = shape, DType = 1, Buf = NewBuf(elems) };
        }

        private void Alias(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            AddValue(new CValue { Name = n.Outputs[0], Shape = x.Shape, DType = 1, Buf = x.Buf, Floats = x.Floats, Ints = x.Ints, AliasOf = x.Buf >= 0 ? Input(0, n) : -1 });
        }

        private void EmitPad(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            long[] pads = V(Input(1, n)).Ints!;
            float val = n.Inputs.Count > 2 && n.Inputs[2].Length > 0 ? V(Input(2, n)).Floats![0] : 0f;
            PadMode mode = n.StringAttr("mode", "constant") switch
            {
                "reflect" => PadMode.Reflect,
                "constant" => PadMode.Constant,
                var s => throw new NotSupportedException($"Pad mode '{s}'"),
            };
            int[] shape = x.Shape.ToArray();
            for (int i = 0; i < shape.Length; i++)
                shape[i] += checked((int)(pads[i] + pads[i + shape.Length]));
            Emit(ExecOp.Pad, [Input(0, n)], ActivationValue(n.Outputs[0], shape),
                new PadP { Pads = pads.Select(p => (int)p).ToArray(), Mode = mode, Value = val }, n);
        }

        private void EmitConv(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            CValue w = V(Input(1, n));
            int[] wd = w.Shape;
            int[] pads = n.IntsAttr("pads").Select(v => (int)v).ToArray();
            if (pads.Length == 0) pads = new int[2 * (wd.Length - 2)];
            int[] strides = n.IntsAttr("strides").Select(v => (int)v).ToArray();
            if (strides.Length == 0) strides = Enumerable.Repeat(1, wd.Length - 2).ToArray();
            int[] dil = n.IntsAttr("dilations").Select(v => (int)v).ToArray();
            if (dil.Length == 0) dil = Enumerable.Repeat(1, wd.Length - 2).ToArray();
            if (n.StringAttr("auto_pad", "NOTSET") is "SAME_UPPER")
                pads = SameUpperPads(x.Shape.AsSpan()[2..].ToArray(), wd.AsSpan()[2..].ToArray(), strides, dil);
            int group = (int)n.IntAttr("group", 1);
            int[] outShape = new int[wd.Length];
            outShape[0] = x.Shape[0];
            outShape[1] = wd[0];
            for (int d = 0; d < wd.Length - 2; d++)
            {
                int eff = dil[d] * (wd[2 + d] - 1) + 1;
                outShape[2 + d] = (x.Shape[2 + d] + pads[d] + pads[d + wd.Length - 2] - eff) / strides[d] + 1;
            }
            int biasV = n.Inputs.Count > 2 && n.Inputs[2].Length > 0 ? Input(2, n) : -1;
            Emit(ExecOp.Conv, [Input(0, n), Input(1, n)], ActivationValue(n.Outputs[0], outShape),
                new ConvP
                {
                    Pads = pads, Strides = strides, Dilations = dil,
                    Kernel = wd.AsSpan()[2..].ToArray(), Group = group, BiasValue = biasV,
                }, n);
        }

        private void EmitPool(OnnxNodeData n, bool avg = false)
        {
            CValue x = V(Input(0, n));
            if (x.Shape.Length != 4) throw new NotSupportedException($"{n.OpType} '{n.Name}': rank-4 only.");
            if (n.IntAttr("ceil_mode", 0) != 0) throw new NotSupportedException($"{n.OpType} '{n.Name}': ceil_mode.");
            int[] kernel = n.IntsAttr("kernel_shape").Select(v => (int)v).ToArray();
            int[] strides = n.IntsAttr("strides").Select(v => (int)v).ToArray();
            if (strides.Length == 0) strides = (int[])kernel.Clone();
            int[] pads = n.IntsAttr("pads").Select(v => (int)v).ToArray();
            if (pads.Length == 0) pads = new int[2 * kernel.Length];
            if (n.StringAttr("auto_pad", "NOTSET") is "SAME_UPPER")
                pads = SameUpperPads(x.Shape.AsSpan()[2..].ToArray(), kernel, strides, Enumerable.Repeat(1, kernel.Length).ToArray());
            int[] outShape = (int[])x.Shape.Clone();
            for (int d = 0; d < kernel.Length; d++)
                outShape[2 + d] = (x.Shape[2 + d] + pads[d] + pads[d + kernel.Length] - kernel[d]) / strides[d] + 1;
            Emit(avg ? ExecOp.AvgPool : ExecOp.MaxPool, [Input(0, n)], ActivationValue(n.Outputs[0], outShape),
                new PoolP { Kernel = kernel, Strides = strides, Pads = pads, CountIncludePad = avg && n.IntAttr("count_include_pad", 0) != 0 }, n);
        }

        private static int[] SameUpperPads(int[] inSp, int[] k, int[] s, int[] d)
        {
            int r = k.Length;
            int[] pads = new int[2 * r];
            for (int i = 0; i < r; i++)
            {
                int outLen = (inSp[i] + s[i] - 1) / s[i];
                int total = Math.Max(0, (outLen - 1) * s[i] + (k[i] - 1) * d[i] + 1 - inSp[i]);
                pads[i] = total / 2;
                pads[i + r] = total - total / 2;
            }
            return pads;
        }

        private void EmitGlobalAvg(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            if (n.OpType == "ReduceMean")
            {
                long[] axes = n.IntsAttr("axes");
                if (axes.Length == 0 && n.Inputs.Count > 1 && n.Inputs[1].Length > 0) axes = V(Input(1, n)).Ints!;
                int rank = x.Shape.Length;
                long[] norm = axes.Select(a => a < 0 ? a + rank : a).OrderBy(a => a).ToArray();
                if (rank != 4 || !norm.SequenceEqual([2L, 3L]) || n.IntAttr("keepdims", 1) != 1)
                    throw new NotSupportedException($"ReduceMean '{n.Name}': only axes=[2,3] keepdims=1.");
            }
            Emit(ExecOp.GlobalAvgPool, [Input(0, n)], ActivationValue(n.Outputs[0], [x.Shape[0], x.Shape[1], 1, 1]), null, n);
        }

        private void EmitReduceMean(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            long[] axes = n.IntsAttr("axes");
            if (axes.Length == 0 && n.Inputs.Count > 1 && n.Inputs[1].Length > 0) axes = V(Input(1, n)).Ints!;
            int rank = x.Shape.Length;
            long[] norm = axes.Select(a => a < 0 ? a + rank : a).OrderBy(a => a).ToArray();
            bool keep = n.IntAttr("keepdims", 1) == 1;
            if (rank == 4 && norm.SequenceEqual([2L, 3L]) && keep) { EmitGlobalAvg(n); return; }
            if (norm.Length != 1 || norm[0] != rank - 1)
                throw new NotSupportedException($"ReduceMean '{n.Name}': axes [{string.Join(',', axes)}] (rank {rank}).");
            int[] shape = (int[])x.Shape.Clone();
            shape[^1] = 1;
            if (!keep) shape = shape.AsSpan()[..^1].ToArray();
            Emit(ExecOp.ReduceMean, [Input(0, n)], ActivationValue(n.Outputs[0], shape), null, n);
        }

        /// <summary>Inference BatchNormalization (not folded into a conv) as a per-channel affine: x·s + t.</summary>
        private void EmitBatchNorm(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            float[] gamma = V(Input(1, n)).Floats!, beta = V(Input(2, n)).Floats!;
            float[] mean = V(Input(3, n)).Floats!, variance = V(Input(4, n)).Floats!;
            float eps = n.FloatAttr("epsilon", 1e-5f);
            int c = gamma.Length, rank = x.Shape.Length;
            if (rank < 2 || x.Shape[1] != c) throw new NotSupportedException($"BatchNormalization '{n.Name}': channel mismatch.");
            float[] s = new float[c], t = new float[c];
            for (int i = 0; i < c; i++)
            {
                s[i] = gamma[i] / MathF.Sqrt(variance[i] + eps);
                t[i] = beta[i] - mean[i] * s[i];
            }
            // [C, 1, ..] broadcasts over the spatial dims of [N, C, ..]
            int[] cs = new int[rank - 1];
            Array.Fill(cs, 1);
            cs[0] = c;
            int sv = AddValue(new CValue { Name = n.Name + "__bn_scale", Shape = cs, DType = 1, Floats = s });
            int tv = AddValue(new CValue { Name = n.Name + "__bn_shift", Shape = cs, DType = 1, Floats = t });
            int scaled = Emit(ExecOp.Elementwise, [Input(0, n), sv], ActivationValue(n.Outputs[0] + "__bn_scaled", x.Shape), new EwP { Op = EwOp.Mul }, n);
            Emit(ExecOp.Elementwise, [scaled, tv], ActivationValue(n.Outputs[0], x.Shape), new EwP { Op = EwOp.Add }, n);
        }

        /// <summary>MatMul: a constant 2-D B lowers to <see cref="ExecOp.Gemm"/>, activation × activation to <see cref="ExecOp.MatMul"/>.</summary>
        private void EmitMatMul(OnnxNodeData n)
        {
            CValue a = V(Input(0, n)), b = V(Input(1, n));
            if (a.Shape.Length < 2 || b.Shape.Length < 2) throw new NotSupportedException($"MatMul '{n.Name}': rank < 2.");
            int k = a.Shape[^1];
            if (b.Shape[^2] != k) throw new InvalidDataException($"MatMul '{n.Name}': K mismatch.");
            if (b.Floats is not null && b.Shape.Length == 2)
            {
                int[] gs = (int[])a.Shape.Clone();
                gs[^1] = b.Shape[1];
                Emit(ExecOp.Gemm, [Input(0, n), Input(1, n)], ActivationValue(n.Outputs[0], gs), new GemmP(), n);
                return;
            }
            int[] ab = a.Shape.AsSpan()[..^2].ToArray(), bb = b.Shape.AsSpan()[..^2].ToArray();
            int[] batch = BroadcastShape(ab, bb);
            int[] shape = [.. batch, a.Shape[^2], b.Shape[^1]];
            Emit(ExecOp.MatMul, [Input(0, n), Input(1, n)], ActivationValue(n.Outputs[0], shape), null, n);
        }

        private void EmitSoftmax(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            int rank = x.Shape.Length;
            int axis = (int)n.IntAttr("axis", -1);
            if (axis < 0) axis += rank;
            if (axis != rank - 1) throw new NotSupportedException($"Softmax '{n.Name}': axis {axis} is not the last.");
            Emit(ExecOp.Softmax, [Input(0, n)], ActivationValue(n.Outputs[0], x.Shape), null, n);
        }

        private void EmitPow(OnnxNodeData n)
        {
            CValue e = V(Input(1, n));
            if (e.Floats is not { Length: 1 } f || f[0] != 2f)
                throw new NotSupportedException($"Pow '{n.Name}': only a constant exponent of 2.");
            CValue x = V(Input(0, n));
            Emit(ExecOp.Elementwise, [Input(0, n), Input(0, n)], ActivationValue(n.Outputs[0], x.Shape), new EwP { Op = EwOp.Mul }, n);
        }

        private void EmitConvTranspose(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            CValue w = V(Input(1, n));   // [ic, oc/group, kh, kw]
            int[] k = w.Shape.AsSpan()[2..].ToArray();
            int[] strides = n.IntsAttr("strides").Select(v => (int)v).ToArray();
            if (strides.Length == 0) strides = [1, 1];
            int[] pads = n.IntsAttr("pads").Select(v => (int)v).ToArray();
            int[] dil = n.IntsAttr("dilations").Select(v => (int)v).ToArray();
            if (n.IntAttr("group", 1) != 1 || !strides.SequenceEqual(k) || pads.Any(p => p != 0)
                || dil.Any(d => d != 1) || n.IntsAttr("output_padding").Any(p => p != 0) || w.Floats is null)
                throw new NotSupportedException($"ConvTranspose '{n.Name}': only group=1, stride==kernel, no pad.");
            int[] outShape = [x.Shape[0], w.Shape[1], x.Shape[2] * k[0], x.Shape[3] * k[1]];
            int biasV = n.Inputs.Count > 2 && n.Inputs[2].Length > 0 ? Input(2, n) : -1;
            Emit(ExecOp.ConvTranspose, [Input(0, n), Input(1, n)], ActivationValue(n.Outputs[0], outShape),
                new ConvTP { Kernel = k, BiasValue = biasV }, n);
        }

        private void EmitUnary(OnnxNodeData n, ExecOp op)
        {
            CValue x = V(Input(0, n));
            Emit(op, [Input(0, n)], ActivationValue(n.Outputs[0], x.Shape), null, n);
        }

        private void EmitLayerNorm(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            float eps = n.FloatAttr("epsilon", 1e-5f);
            int axis = (int)n.IntAttr("axis", -1);
            if (axis < 0) axis += x.Shape.Length;
            if (axis != x.Shape.Length - 1)
                throw new NotSupportedException("LayerNorm only over last dim.");
            Emit(ExecOp.LayerNorm, [Input(0, n), Input(1, n), Input(2, n)],
                ActivationValue(n.Outputs[0], x.Shape), new LnP { Eps = eps }, n);
        }

        private void EmitGemm(OnnxNodeData n)
        {
            CValue a = V(Input(0, n));
            CValue b = V(Input(1, n));
            bool transB = n.IntAttr("transB") == 1;
            float alpha = n.FloatAttr("alpha", 1f);
            if (alpha != 1f) throw new NotSupportedException("Gemm alpha != 1.");
            int k = transB ? b.Shape[1] : b.Shape[0];
            int ncol = transB ? b.Shape[0] : b.Shape[1];
            if (a.Shape[^1] != k) throw new InvalidDataException("Gemm K mismatch.");
            int[] outShape = (int[])a.Shape.Clone();
            outShape[^1] = ncol;
            int biasV = n.Inputs.Count > 2 && n.Inputs[2].Length > 0 ? Input(2, n) : -1;
            Emit(ExecOp.Gemm, [Input(0, n), Input(1, n)], ActivationValue(n.Outputs[0], outShape),
                new GemmP { TransB = transB, BiasValue = biasV }, n);
        }

        private void EmitResize(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            float[] scales;
            int[]? sizesI = null;
            if (n.Inputs.Count > 3 && n.Inputs[3].Length > 0)
            {
                // sizes input (int tensor): authoritative output shape
                long[] sizes = V(Input(3, n)).Ints!;
                sizesI = sizes.Select(s => (int)s).ToArray();
                scales = sizes.Select((s, i) => (float)s / x.Shape[i]).ToArray();
            }
            else
            {
                scales = V(Input(2, n)).Floats!;
            }
            if (n.StringAttr("mode") == "nearest")
            {
                if (n.StringAttr("coordinate_transformation_mode", "half_pixel") != "asymmetric"
                    || n.StringAttr("nearest_mode", "round_prefer_floor") != "floor")
                    throw new NotSupportedException("Resize nearest: only asymmetric/floor.");
                int[] os = sizesI ?? x.Shape.Select((d, i) => (int)MathF.Floor(d * scales[i])).ToArray();
                Emit(ExecOp.ResizeNearest, [Input(0, n)], ActivationValue(n.Outputs[0], os),
                    new ResizeNearestP { Scales = scales }, n);
                return;
            }
            if (n.StringAttr("mode") != "linear")
                throw new NotSupportedException("Resize mode != linear.");
            int[] outShape = sizesI ?? (int[])x.Shape.Clone();
            if (sizesI == null)
                for (int i = 0; i < outShape.Length; i++)
                    outShape[i] = (int)MathF.Floor(x.Shape[i] * scales[i]);
            Emit(ExecOp.ResizeBilinear, [Input(0, n)], ActivationValue(n.Outputs[0], outShape),
                new ResizeP { Scales = scales }, n);
        }

        private void EmitTranspose(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            int[] perm = n.IntsAttr("perm").Select(v => (int)v).ToArray();
            if (perm.Length == 0) perm = Enumerable.Range(0, x.Shape.Length).Reverse().ToArray();
            int[] outShape = perm.Select(p => x.Shape[p]).ToArray();
            Emit(ExecOp.Transpose, [Input(0, n)], ActivationValue(n.Outputs[0], outShape),
                new TransposeP { Perm = perm }, n);
        }

        private void EmitSlice(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            long[] starts = IntsArg(n, 1, "starts")!;
            long[] ends = IntsArg(n, 2, "ends")!;
            long[] axes = IntsArg(n, 3, "axes") ?? Enumerable.Range(0, starts.Length).Select(i => (long)i).ToArray();
            long[] steps = IntsArg(n, 4, "steps") ?? starts.Select(_ => 1L).ToArray();
            SliceBounds(x.Shape, starts, ends, axes, steps,
                out int[] outShape, out int[] st, out int[] en, out int[] stepI);
            Emit(ExecOp.Slice, [Input(0, n)], ActivationValue(n.Outputs[0], outShape),
                new SliceP { Starts = st, Ends = en,
                    Axes = axes.Select(a => (int)a).ToArray(), Steps = stepI }, n);
        }

        private void EmitConcat(OnnxNodeData n)
        {
            int axis = (int)n.IntAttr("axis");
            int[] ins = Inputs(n);
            int[] shape = (int[])V(ins[0]).Shape.Clone();
            int rank = shape.Length;
            if (axis < 0) axis += rank;
            shape[axis] = ins.Sum(id => V(id).Shape[axis]);
            Emit(ExecOp.Concat, ins, ActivationValue(n.Outputs[0], shape),
                new ConcatP { Axis = axis }, n);
        }

        private void EmitSplit(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            int axis = (int)n.IntAttr("axis");
            if (axis < 0) axis += x.Shape.Length;
            int[] sizes;
            if (n.Inputs.Count > 1 && n.Inputs[1].Length > 0)
                sizes = V(Input(1, n)).Ints!.Select(v => (int)v).ToArray();
            else
            {
                int outputs = (int)n.IntAttr("num_outputs", n.Outputs.Count);
                int each = x.Shape[axis] / outputs;
                sizes = Enumerable.Repeat(each, outputs).ToArray();
                sizes[^1] += x.Shape[axis] - each * outputs;
            }
            // Split is implemented as a per-output Slice at exec time.
            int[] starts = new int[x.Shape.Length];
            for (int i = 0; i < sizes.Length; i++)
            {
                int[] shape = (int[])x.Shape.Clone();
                shape[axis] = sizes[i];
                var slice = new SliceP
                {
                    Starts = (int[])starts.Clone(),
                    Ends = (int[])x.Shape.Clone(),
                    Axes = Enumerable.Range(0, x.Shape.Length).ToArray(),
                    Steps = Enumerable.Repeat(1, x.Shape.Length).ToArray(),
                };
                slice.Ends[axis] = starts[axis] + sizes[i];
                _nodes.Add(new ExecNode { Op = ExecOp.Slice, In = [Input(0, n)], Out = [AddValue(ActivationValue(n.Outputs[i], shape))], P = slice, SrcNode = _nodes.Count, Name = n.Name });
                starts[axis] += sizes[i];
            }
        }

        private void EmitReshape(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            long[] spec = V(Input(1, n)).Ints!;
            bool allowZero = n.IntAttr("allowzero") == 1;
            int[] shape = ResolveReshape(x.Shape, spec, allowZero, checked((int)x.ElementCount));
            // zero-copy alias (consts keep their payload so consumers see a materialized tensor)
            AddValue(new CValue { Name = n.Outputs[0], Shape = shape, DType = 1, Buf = x.Buf, Floats = x.Floats, AliasOf = x.Buf >= 0 ? Input(0, n) : -1 });
        }

        private void EmitSqueeze(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            long[]? axes = IntsArg(n, 1, "axes");
            int[] shape;
            if (axes is null) shape = x.Shape.Where(d => d != 1).ToArray();
            else
            {
                var dims = x.Shape.ToList();
                foreach (long a in axes.OrderByDescending(v => v)) dims.RemoveAt((int)(a < 0 ? a + dims.Count : a));
                shape = dims.ToArray();
            }
            AddValue(new CValue { Name = n.Outputs[0], Shape = shape, DType = 1, Buf = x.Buf, Floats = x.Floats, AliasOf = x.Buf >= 0 ? Input(0, n) : -1 });
        }

        private void EmitUnsqueeze(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            long[] axes = IntsArg(n, 1, "axes")!;
            var dims = x.Shape.ToList();
            foreach (long a in axes.OrderBy(v => v)) dims.Insert((int)(a < 0 ? a + dims.Count + 1 : a), 1);
            AddValue(new CValue { Name = n.Outputs[0], Shape = dims.ToArray(), DType = 1, Buf = x.Buf, Floats = x.Floats, AliasOf = x.Buf >= 0 ? Input(0, n) : -1 });
        }

        private void EmitElementwise(OnnxNodeData n, string op)
        {
            int a = Input(0, n), b = Input(1, n);
            CValue va = V(a), vb = V(b);
            int[] shape = BroadcastShape(va.Shape, vb.Shape);
            var eop = op switch { "Add" => EwOp.Add, "Sub" => EwOp.Sub, "Mul" => EwOp.Mul, "Div" => EwOp.Div, _ => throw new() };
            Emit(ExecOp.Elementwise, [a, b], ActivationValue(n.Outputs[0], shape), new EwP { Op = eop }, n);
        }

        // ---------- helpers ----------
        private static int[] BroadcastShape(int[] a, int[] b)
        {
            int rank = Math.Max(a.Length, b.Length);
            int[] r = new int[rank];
            for (int i = 0; i < rank; i++)
            {
                int ad = i - (rank - a.Length) >= 0 ? a[i - (rank - a.Length)] : 1;
                int bd = i - (rank - b.Length) >= 0 ? b[i - (rank - b.Length)] : 1;
                if (ad != bd && ad != 1 && bd != 1)
                    throw new InvalidDataException($"Broadcast mismatch {ad} vs {bd}.");
                r[i] = Math.Max(ad, bd);
            }
            return r;
        }

        internal static int[] ResolveReshape(int[] inShape, long[] spec, bool allowZero, int inElems)
        {
            int[] shape = new int[spec.Length];
            int neg = -1; long prod = 1;
            for (int i = 0; i < spec.Length; i++)
            {
                long s = spec[i];
                if (s == -1) { neg = i; shape[i] = 1; }
                else if (s == 0 && !allowZero) { shape[i] = inShape[i]; prod *= shape[i]; }
                else { shape[i] = checked((int)s); prod *= shape[i]; }
            }
            if (neg >= 0) shape[neg] = checked((int)(inElems / prod));
            return shape;
        }

        internal static void SliceBounds(int[] inShape, long[] starts, long[] ends,
            long[] axes, long[] steps, out int[] outShape, out int[] stOut, out int[] enOut, out int[] stepOut)
        {
            int rank = inShape.Length;
            stOut = new int[starts.Length];
            enOut = new int[starts.Length];
            stepOut = new int[starts.Length];
            outShape = (int[])inShape.Clone();
            for (int i = 0; i < starts.Length; i++)
            {
                int ax = (int)(axes[i] < 0 ? axes[i] + rank : axes[i]);
                int dim = inShape[ax];
                int step = (int)steps[i];
                long s = starts[i] < 0 ? starts[i] + dim : starts[i];
                long e = ends[i] < 0 ? ends[i] + dim : ends[i];
                if (step > 0)
                {
                    s = Math.Clamp(s, 0, dim);
                    e = Math.Clamp(e, 0, dim);
                }
                else
                {
                    s = Math.Clamp(s, -1, dim - 1);
                    e = Math.Clamp(e, -1, dim - 1);
                }
                int len = step > 0 ? Math.Max(0, (int)(e - s + step - 1) / step)
                                   : Math.Max(0, (int)(s - e - step - 1) / -step);
                stOut[i] = (int)s;
                enOut[i] = (int)e;
                stepOut[i] = step;
                outShape[ax] = len;
            }
        }

        private static long[] IntConcat(long[][] inputs, int[][] shapes, int axis, int[] outShape)
        {
            // small tensors only — generic n-d concat
            int outElems = outShape.Aggregate(1, (a, b) => a * b);
            long[] dst = new long[outElems];
            int[] strides = Strides(outShape);
            int[] coord = new int[outShape.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                int[] inStrides = Strides(shapes[i]);
                int inElems = shapes[i].Aggregate(1, (a, b) => a * b);
                for (int e = 0; e < inElems; e++)
                {
                    int rem = e, srcOff = 0;
                    for (int d = 0; d < shapes[i].Length; d++)
                    {
                        coord[d] = rem / inStrides[d];
                        srcOff += coord[d] * inStrides[d];
                        rem %= inStrides[d];
                    }
                    int dstOff = 0;
                    for (int d = 0; d < outShape.Length; d++)
                        dstOff += (coord[d] + (d == axis ? shapes.Take(i).Sum(s => s[axis]) : 0)) * strides[d];
                    dst[dstOff] = inputs[i][srcOff];
                }
            }
            return dst;
        }

        private static long[] IntSlice(long[] src, int[] inShape, int[] axes, int[] starts, int[] steps, int[] outShape)
        {
            int rank = inShape.Length;
            int[] fullStart = new int[rank], fullStep = Enumerable.Repeat(1, rank).ToArray();
            for (int i = 0; i < axes.Length; i++) { fullStart[axes[i]] = starts[i]; fullStep[axes[i]] = steps[i]; }
            int outElems = outShape.Aggregate(1, (a, b) => a * b);
            long[] dst = new long[outElems];
            int[] inStrides = Strides(inShape), outStrides = Strides(outShape);
            for (int e = 0; e < outElems; e++)
            {
                int rem = e, srcOff = 0;
                for (int d = 0; d < rank; d++)
                {
                    int cc = rem / outStrides[d];
                    rem %= outStrides[d];
                    srcOff += (fullStart[d] + cc * fullStep[d]) * inStrides[d];
                }
                dst[e] = src[srcOff];
            }
            return dst;
        }

        internal static int[] Strides(int[] shape)
        {
            int[] s = new int[shape.Length];
            int acc = 1;
            for (int i = shape.Length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
            return s;
        }
    }
}
