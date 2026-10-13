namespace Sdcb.SimdOnnx.Onnx;

/// <summary>
/// Shape-independent ONNX graph rewrites applied once at load: Constant nodes become
/// initializers, Conv → BatchNormalization folds into the conv weights, and parallel centered
/// convolution branches summed by an Add (RepLK-style 7x7 + 7x1 + 1x7) merge into one conv.
/// </summary>
internal static class OnnxRewrites
{
    public static void Apply(OnnxGraphData g, Dictionary<string, OnnxTensorData> init)
    {
        ConstantsToInitializers(g, init);
        HashSet<string> outputs = new(g.Outputs.Select(o => o.Name), StringComparer.Ordinal);
        FoldConvBatchNorm(g.Nodes, init, outputs);
        MergeParallelConvBranches(g.Nodes, init, outputs);
    }

    private static void ConstantsToInitializers(OnnxGraphData g, Dictionary<string, OnnxTensorData> init)
    {
        g.Nodes.RemoveAll(n =>
        {
            if (n.OpType != "Constant" || n.Outputs.Count != 1 || n.Outputs[0].Length == 0) return false;
            OnnxTensorData? t = n.Attr("value")?.Tensor;
            if (t is null)
            {
                if (n.Attr("value_float") is { } f) t = Floats(n.Outputs[0], [], [f.Float]);
                else if (n.Attr("value_floats") is { } fs) t = Floats(n.Outputs[0], [fs.Floats.Count], fs.Floats);
                else if (n.Attr("value_int") is { } i) { t = new() { DataType = OnnxTensorData.TypeInt64 }; t.Int64Data.Add(i.Int); }
                else if (n.Attr("value_ints") is { } iz) { t = new() { DataType = OnnxTensorData.TypeInt64 }; t.Dims.Add(iz.Ints.Count); t.Int64Data.AddRange(iz.Ints); }
                else return false;
            }
            t.Name = n.Outputs[0];
            init[t.Name] = t;
            g.Initializers.Add(t);
            return true;
        });
    }

    private static OnnxTensorData Floats(string name, IReadOnlyList<long> dims, IReadOnlyList<float> values)
    {
        OnnxTensorData t = new() { Name = name, DataType = OnnxTensorData.TypeFloat };
        t.Dims.AddRange(dims);
        t.FloatData.AddRange(values);
        return t;
    }

    private static string Unique(string baseName, Dictionary<string, OnnxTensorData> init)
    {
        string c = baseName;
        for (int i = 1; init.ContainsKey(c); i++) c = baseName + "_" + i;
        return c;
    }

    private static OnnxTensorData AddInit(Dictionary<string, OnnxTensorData> init, string baseName, IReadOnlyList<long> dims, float[] values)
    {
        OnnxTensorData t = Floats(Unique(baseName, init), dims, values);
        init[t.Name] = t;
        return t;
    }

    private static Dictionary<string, int> Consumers(List<OnnxNodeData> nodes)
    {
        var c = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (OnnxNodeData n in nodes)
            foreach (string i in n.Inputs)
                if (i.Length != 0) c[i] = c.GetValueOrDefault(i) + 1;
        return c;
    }

    private static void FoldConvBatchNorm(List<OnnxNodeData> nodes, Dictionary<string, OnnxTensorData> init, HashSet<string> graphOutputs)
    {
        var producer = new Dictionary<string, OnnxNodeData>(StringComparer.Ordinal);
        foreach (OnnxNodeData n in nodes)
            foreach (string o in n.Outputs) if (o.Length != 0) producer[o] = n;
        var consumers = Consumers(nodes);
        var folded = new HashSet<OnnxNodeData>();
        foreach (OnnxNodeData bn in nodes)
        {
            if (bn.OpType != "BatchNormalization" || bn.Inputs.Count != 5 || bn.Outputs.Count < 1) continue;
            if (!producer.TryGetValue(bn.Inputs[0], out OnnxNodeData? conv) || consumers.GetValueOrDefault(bn.Inputs[0]) != 1) continue;
            if (conv.OpType != "Conv" || conv.Inputs.Count is < 2 or > 3 || conv.Outputs.Count != 1 || graphOutputs.Contains(conv.Outputs[0])) continue;
            string[] names = [.. conv.Inputs.Skip(1).Concat(bn.Inputs.Skip(1))];
            if (names.Any(nm => !init.TryGetValue(nm, out OnnxTensorData? t) || t.DataType != OnnxTensorData.TypeFloat)) continue;
            OnnxTensorData wt = init[conv.Inputs[1]];
            if (wt.Dims.Count != 4) continue;
            float[] w = wt.AsFloats();
            float[] gamma = init[bn.Inputs[1]].AsFloats(), beta = init[bn.Inputs[2]].AsFloats();
            float[] mean = init[bn.Inputs[3]].AsFloats(), variance = init[bn.Inputs[4]].AsFloats();
            int oc = checked((int)wt.Dims[0]);
            if (gamma.Length != oc || beta.Length != oc || mean.Length != oc || variance.Length != oc || w.Length % oc != 0) continue;
            float eps = bn.Attr("epsilon")?.Float ?? 1e-5f;
            if (!float.IsFinite(eps) || eps < 0 || variance.Any(v => !float.IsFinite(v) || v < 0)) continue;
            float[] b0 = conv.Inputs.Count == 3 ? init[conv.Inputs[2]].AsFloats() : new float[oc];
            if (b0.Length != oc) continue;
            int per = w.Length / oc;
            float[] fw = new float[w.Length], fb = new float[oc];
            for (int c = 0; c < oc; c++)
            {
                float scale = gamma[c] / MathF.Sqrt(variance[c] + eps);
                fb[c] = (b0[c] - mean[c]) * scale + beta[c];
                for (int i = 0; i < per; i++) fw[c * per + i] = w[c * per + i] * scale;
            }
            conv.Inputs[1] = AddInit(init, conv.Inputs[1] + "__bn_w", wt.Dims, fw).Name;
            string bName = AddInit(init, (conv.Name.Length == 0 ? conv.Outputs[0] : conv.Name) + "__bn_b", [oc], fb).Name;
            if (conv.Inputs.Count == 2) conv.Inputs.Add(bName); else conv.Inputs[2] = bName;
            conv.Outputs[0] = bn.Outputs[0];
            folded.Add(bn);
        }
        nodes.RemoveAll(folded.Contains);
    }

    private readonly record struct Branch(string Input, int Oc, int Ic, int Kh, int Kw, float[] W, float[]? B);

    private static bool Mergeable(OnnxNodeData n, Dictionary<string, OnnxTensorData> init, HashSet<string> graphOutputs, out Branch br)
    {
        br = default;
        if (n.OpType != "Conv" || n.Inputs.Count is < 2 or > 3 || n.Outputs.Count != 1 || graphOutputs.Contains(n.Outputs[0])) return false;
        long[] st = n.IntsAttr("strides"), dl = n.IntsAttr("dilations");
        if (st.Any(v => v != 1) || dl.Any(v => v != 1) || n.IntAttr("group", 1) != 1) return false;
        string ap = n.StringAttr("auto_pad", "NOTSET");
        if (ap is not ("NOTSET" or "")) return false;
        if (!init.TryGetValue(n.Inputs[1], out OnnxTensorData? wt) || wt.DataType != OnnxTensorData.TypeFloat || wt.Dims.Count != 4) return false;
        int kh = checked((int)wt.Dims[2]), kw = checked((int)wt.Dims[3]);
        if ((kh & 1) == 0 || (kw & 1) == 0) return false;
        long[] pads = n.IntsAttr("pads");
        if (pads.Length == 0) pads = [0, 0, 0, 0];
        if (pads.Length != 4 || pads[0] != (kh - 1) / 2 || pads[2] != (kh - 1) / 2 || pads[1] != (kw - 1) / 2 || pads[3] != (kw - 1) / 2) return false;
        float[]? b = null;
        if (n.Inputs.Count == 3)
        {
            if (!init.TryGetValue(n.Inputs[2], out OnnxTensorData? bt) || bt.DataType != OnnxTensorData.TypeFloat) return false;
            b = bt.AsFloats();
            if (b.Length != wt.Dims[0]) return false;
        }
        br = new Branch(n.Inputs[0], checked((int)wt.Dims[0]), checked((int)wt.Dims[1]), kh, kw, wt.AsFloats(), b);
        return true;
    }

    private static void MergeParallelConvBranches(List<OnnxNodeData> nodes, Dictionary<string, OnnxTensorData> init, HashSet<string> graphOutputs)
    {
        for (bool merged = true; merged;)
        {
            merged = false;
            var producer = new Dictionary<string, OnnxNodeData>(StringComparer.Ordinal);
            foreach (OnnxNodeData n in nodes)
                foreach (string o in n.Outputs) if (o.Length != 0) producer[o] = n;
            var consumers = Consumers(nodes);
            foreach (OnnxNodeData add in nodes)
            {
                if (add.OpType != "Add" || add.Inputs.Count != 2 || add.Outputs.Count != 1) continue;
                if (!producer.TryGetValue(add.Inputs[0], out OnnxNodeData? c1) || !producer.TryGetValue(add.Inputs[1], out OnnxNodeData? c2) || c1 == c2) continue;
                if (!Mergeable(c1, init, graphOutputs, out Branch a) || !Mergeable(c2, init, graphOutputs, out Branch b)) continue;
                if (a.Input != b.Input || a.Oc != b.Oc || a.Ic != b.Ic) continue;
                if (consumers.GetValueOrDefault(c1.Outputs[0]) != 1 || consumers.GetValueOrDefault(c2.Outputs[0]) != 1) continue;
                int kh = Math.Max(a.Kh, b.Kh), kw = Math.Max(a.Kw, b.Kw);
                float[] w = new float[(long)a.Oc * a.Ic * kh * kw];
                Accumulate(w, a, kh, kw);
                Accumulate(w, b, kh, kw);
                c1.Inputs[1] = AddInit(init, c1.Inputs[1] + "__merged_w", [a.Oc, a.Ic, kh, kw], w).Name;
                if (a.B is not null || b.B is not null)
                {
                    float[] bias = new float[a.Oc];
                    for (int c = 0; c < bias.Length; c++) bias[c] = (a.B?[c] ?? 0f) + (b.B?[c] ?? 0f);
                    string bn = AddInit(init, c1.Inputs[1] + "_b", [a.Oc], bias).Name;
                    if (c1.Inputs.Count == 2) c1.Inputs.Add(bn); else c1.Inputs[2] = bn;
                }
                SetInts(c1, "kernel_shape", [kh, kw]);
                SetInts(c1, "pads", [(kh - 1) / 2, (kw - 1) / 2, (kh - 1) / 2, (kw - 1) / 2]);
                c1.Outputs[0] = add.Outputs[0];
                nodes.Remove(c2);
                nodes.Remove(add);
                merged = true;
                break;
            }
        }
    }

    private static void Accumulate(float[] target, Branch b, int kh, int kw)
    {
        int oy = (kh - b.Kh) / 2, ox = (kw - b.Kw) / 2, filters = b.Oc * b.Ic;
        for (int f = 0; f < filters; f++)
            for (int y = 0; y < b.Kh; y++)
                for (int x = 0; x < b.Kw; x++)
                    target[(f * kh + y + oy) * kw + x + ox] += b.W[(f * b.Kh + y) * b.Kw + x];
    }

    private static void SetInts(OnnxNodeData n, string name, long[] values)
    {
        OnnxAttributeData? a = n.Attr(name);
        if (a is null) { a = new OnnxAttributeData { Name = name }; n.Attributes.Add(a); }
        a.Ints.Clear();
        a.Ints.AddRange(values);
    }
}
