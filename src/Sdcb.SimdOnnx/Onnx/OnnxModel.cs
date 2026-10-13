namespace Sdcb.SimdOnnx.Onnx;

/// <summary>Validated ONNX model: graph nodes, initializers, IO metadata.</summary>
public sealed class OnnxModel
{
    private OnnxModel(int opset, OnnxGraphData graph, Dictionary<string, OnnxTensorData> initializers)
    {
        Opset = opset;
        Graph = graph;
        Initializers = initializers;
    }

    public int Opset { get; }
    internal OnnxGraphData Graph { get; }
    internal Dictionary<string, OnnxTensorData> Initializers { get; }

    public static OnnxModel Load(string path)
    {
        using FileStream fs = File.OpenRead(path);
        return Load(fs);
    }

    public static OnnxModel Load(Stream source) => Build(OnnxProtoReader.Parse(source));

    /// <summary>Parses the model straight from <paramref name="source"/> (pinned for the duration
    /// of the call, no intermediate copy); weights are decoded into model-owned arrays, so the
    /// memory may be released afterwards.</summary>
    public static unsafe OnnxModel Load(ReadOnlySpan<byte> source)
    {
        fixed (byte* p = source)
        {
            using var ums = new UnmanagedMemoryStream(p, source.Length);
            return Load(ums);
        }
    }

    private static OnnxModel Build(OnnxModelData parsed)
    {
        OnnxGraphData g = parsed.Graph!;
        int opset = checked((int)(parsed.Opsets.FirstOrDefault(
            static x => x.Domain.Length == 0 || x.Domain == "ai.onnx")?.Version ?? 0));
        Dictionary<string, OnnxTensorData> init = new(g.Initializers.Count);
        foreach (OnnxTensorData t in g.Initializers) init[t.Name] = t;
        OnnxRewrites.Apply(g, init);
        // rewrites add folded weights; drop the initializers no node reads any more
        HashSet<string> used = new(g.Nodes.SelectMany(n => n.Inputs), StringComparer.Ordinal);
        g.Initializers.Clear();
        g.Initializers.AddRange(init.Values.Where(t => used.Contains(t.Name)));
        init = g.Initializers.ToDictionary(t => t.Name, StringComparer.Ordinal);
        return new OnnxModel(opset, g, init);
    }
}
