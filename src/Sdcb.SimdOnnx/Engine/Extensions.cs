using System.Reflection;
using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;

namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// A consumer-defined executable operator (<see cref="ExecOp.Ext"/>). Graph passes create
/// nodes with <see cref="ExecNode.Ext"/> set; the core calls back for layout, CPU execution,
/// weight packing and GPU dispatch emission. Per-node parameters live in <see cref="ExecNode.P"/>.
/// A backend that does not implement the op must not be targeted by the fusing pass
/// (<see cref="PassContext.Target"/>): the core never falls back node by node.
/// </summary>
internal abstract class ExtOp
{
    public abstract string Name { get; }

    /// <summary>Value ids read through <see cref="ExecNode.P"/> (kept live by the arena planner).</summary>
    public virtual IEnumerable<int> ParamRefs(ExecNode n) => [];

    /// <summary>Layout pass: request input layouts and set the output's. Default: rank-4 NHWC in/out.</summary>
    public virtual void PlanLayout(LayoutPlanner lp, ExecNode n)
    {
        for (int k = 0; k < n.In.Length; k++)
            if (lp.Values[n.In[k]].IsActivation) lp.Need(n, k, Layout.Nhwc);
        foreach (int o in n.Out) lp.Values[o].Layout = Layout.Nhwc;
    }

    /// <summary>Packs CPU weights into <see cref="ExecNode.P"/> (idempotent; plans may be shared).</summary>
    public virtual void PackCpu(ExecPlan plan, ExecNode n, CpuKernels.Cache? cache) { }

    public abstract void RunCpu(InferenceSession s, ExecNode n);

    /// <summary>Consts the Vulkan dispatches bind, with the packing each needs.</summary>
    public virtual IEnumerable<(int value, ConstKind kind)> GpuConsts(ExecNode n) => [];

    public virtual IEnumerable<VkDispatch> EmitVulkan(VkGraphModel g, ExecNode n) =>
        throw new NotSupportedException($"Vulkan: extension op {Name} not implemented");

    public virtual IEnumerable<MtlDispatch> EmitMetal(MtlGraphModel g, ExecNode n) =>
        throw new NotSupportedException($"Metal: extension op {Name} not implemented");
}

/// <summary>When a <see cref="GraphPass"/> runs.</summary>
internal enum PassStage
{
    /// <summary>Logical graph, after the built-in fusions, before layout planning.</summary>
    Logical,
    /// <summary>Physical graph (layouts assigned), after the built-in memory fusions, before arena planning.</summary>
    Physical,
}

/// <summary>A consumer-defined rewrite over the exec node list.</summary>
internal abstract class GraphPass
{
    public abstract PassStage Stage { get; }
    public abstract void Run(PassContext ctx);
}

/// <summary>Registered extension operators and passes; pass via <see cref="CompileOptions.Extensions"/>.</summary>
internal sealed class EngineExtensions
{
    private readonly List<GraphPass> _passes = [];
    public IReadOnlyList<GraphPass> Passes => _passes;
    public EngineExtensions Add(GraphPass pass) { _passes.Add(pass); return this; }
}

/// <summary>Graph view handed to <see cref="GraphPass.Run"/>.</summary>
internal sealed class PassContext(List<ExecNode> nodes, List<CValue> values, int outputValue, CompileOptions options)
{
    public List<ExecNode> Nodes { get; } = nodes;
    public List<CValue> Values { get; } = values;
    public int OutputValue { get; } = outputValue;
    public CompileOptions Options { get; } = options;
    public PlanTarget Target => Options.Target;

    public int Producer(int v) => Nodes.FindIndex(m => m.Out.Contains(v));

    /// <summary>Readers of v's buffer (aliases and param operands included); the graph output counts as infinite.</summary>
    public int Uses(int v)
    {
        int b = Values[v].Buf;
        if (b < 0 || v == OutputValue || Values[OutputValue].Buf == b) return int.MaxValue;
        int n = 0;
        foreach (ExecNode m in Nodes)
        {
            foreach (int iv in m.In) if (Values[iv].Buf == b) n++;
            foreach (int iv in m.ParamRefs()) if (Values[iv].Buf == b) n++;
        }
        return n;
    }

    public bool SameShape(int a, int b) => Values[a].Shape.AsSpan().SequenceEqual(Values[b].Shape);

    public bool IsNhwc4(int v) => Values[v].Shape.Length == 4 && Values[v].Layout is [0, 2, 3, 1];

    /// <summary>Rewrites every reader of <paramref name="oldValue"/> to read <paramref name="newValue"/>.</summary>
    public void Redirect(int oldValue, int newValue)
    {
        foreach (ExecNode m in Nodes)
            for (int i = 0; i < m.In.Length; i++)
                if (m.In[i] == oldValue) m.In[i] = newValue;
        if (Values[oldValue].Name.Length != 0 && Values[newValue].Name.Length == 0)
            Values[newValue].Name = Values[oldValue].Name;
    }
}

/// <summary>
/// How a const tensor is laid out on a GPU (decided by its consumer). Built-ins cover the core
/// shaders; extensions create their own with a <see cref="Pack"/> function. Identity is by reference.
/// </summary>
internal sealed class ConstKind(string name, bool fp32, Func<CValue, float[]>? pack = null, Func<CValue, long>? elems = null)
{
    private static int s_next;
    /// <summary>Stable per-process id (const stores compare kind lists by it).</summary>
    public int Id { get; } = Interlocked.Increment(ref s_next);
    public string Name { get; } = name;
    /// <summary>Stored as fp32 even on fp16-storage graphs (bias, norm params).</summary>
    public bool Fp32 { get; } = fp32;
    public Func<CValue, float[]>? Pack { get; } = pack;
    /// <summary>Bound element count when it differs from the logical one (padded packings).</summary>
    public Func<CValue, long>? Elems { get; } = elems;
    public override string ToString() => Name;

    public static readonly ConstKind Raw = new("raw", false);
    public static readonly ConstKind F32 = new("f32", true);
    /// <summary>Conv [OC][IC][KH][KW] → [K][OC], k = (ky·KW + kx)·IC + ic (direct-conv shaders).</summary>
    public static readonly ConstKind ConvW = new("convw", false, GpuPack.Conv);
    /// <summary>Depthwise [C][1][KH][KW] → [tap][C].</summary>
    public static readonly ConstKind DwW = new("dww", false, GpuPack.Dw);
    /// <summary>Conv → [OC][Kpad] (k fastest, K padded to 8; matrix-core conv/GEMM shaders).</summary>
    public static readonly ConstKind ConvWT = new("convwt", false, GpuPack.ConvT, GpuPack.ConvTElems);
    /// <summary>Gemm B [K][N] → [N][K] (k fastest, matrix-core GEMM).</summary>
    public static readonly ConstKind GemmNK = new("gemmnk", false, GpuPack.GemmNK);
    /// <summary>ConvTranspose [IC][OC][KH][KW] → [tap][IC][OC].</summary>
    public static readonly ConstKind ConvTW = new("convtw", false, GpuPack.ConvTranspose);
}

/// <summary>Const packings shared by the Vulkan and Metal backends.</summary>
internal static class GpuPack
{
    public static float[] Conv(CValue w)
    {
        int oc = w.Shape[0], ic = w.Shape[1], taps = w.Shape[2] * w.Shape[3];
        float[] src = w.Floats!, dst = new float[src.Length];
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
                for (int t = 0; t < taps; t++)
                    dst[(t * ic + c) * oc + o] = src[(o * ic + c) * taps + t];
        return dst;
    }

    public static long ConvTElems(CValue w) => (long)w.Shape[0] * ((w.Shape[1] * w.Shape[2] * w.Shape[3] + 7) & ~7);

    public static float[] ConvT(CValue w)
    {
        int oc = w.Shape[0], ic = w.Shape[1], taps = w.Shape[2] * w.Shape[3], k = (taps * ic + 7) & ~7;
        float[] src = w.Floats!, dst = new float[oc * k];
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
                for (int t = 0; t < taps; t++)
                    dst[o * k + t * ic + c] = src[(o * ic + c) * taps + t];
        return dst;
    }

    public static float[] Dw(CValue w)
    {
        int c = w.Shape[0], taps = w.Shape[2] * w.Shape[3];
        float[] src = w.Floats!, dst = new float[src.Length];
        for (int ch = 0; ch < c; ch++)
            for (int t = 0; t < taps; t++) dst[t * c + ch] = src[ch * taps + t];
        return dst;
    }

    public static float[] GemmNK(CValue b)
    {
        int k = b.Shape[0], nn = b.Shape[1];
        float[] src = b.Floats!, dst = new float[src.Length];
        for (int i = 0; i < k; i++)
            for (int j = 0; j < nn; j++) dst[j * k + i] = src[i * nn + j];
        return dst;
    }

    public static float[] ConvTranspose(CValue w)
    {
        int ic = w.Shape[0], oc = w.Shape[1], taps = w.Shape[2] * w.Shape[3];
        float[] src = w.Floats!, dst = new float[src.Length];
        for (int c = 0; c < ic; c++)
            for (int o = 0; o < oc; o++)
                for (int t = 0; t < taps; t++) dst[(t * ic + c) * oc + o] = src[(c * oc + o) * taps + t];
        return dst;
    }

    /// <summary>Packed host bytes + per-const offsets (64 B aligned; fp16 unless the kind is fp32).</summary>
    public static (ulong[] offsets, byte[] host) Build(ExecPlan plan, IReadOnlyList<(int v, ConstKind k)> need, bool f16)
    {
        ulong total = 0;
        var offs = new ulong[need.Count];
        var items = new List<(ulong off, float[] data, bool half)>(need.Count);
        for (int i = 0; i < need.Count; i++)
        {
            var (v, k) = need[i];
            CValue cv = plan.Values[v];
            float[] data = k.Pack?.Invoke(cv) ?? cv.Floats!;
            bool half = f16 && !k.Fp32;
            total = (total + 63) & ~63UL;
            offs[i] = total;
            items.Add((total, data, half));
            total += (ulong)data.Length * (half ? 2UL : 4UL);
        }
        byte[] host = new byte[Math.Max((total + 63) & ~63UL, 64)];
        foreach (var (off, data, half) in items)
        {
            if (half)
            {
                Span<Half> dst = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Half>(host.AsSpan(checked((int)off), data.Length * 2));
                for (int i = 0; i < data.Length; i++) dst[i] = (Half)data[i];
            }
            else
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan()).CopyTo(host.AsSpan(checked((int)off)));
        }
        return (offs, host);
    }
}

/// <summary>Embedded GPU shader lookup across assemblies (the core and extension owners).</summary>
internal static class ShaderResources
{
    public static Stream? Open(Assembly asm, string fileName)
    {
        string suffix = ".Shaders." + fileName;
        string? res = asm.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(suffix, StringComparison.Ordinal));
        return res is null ? null : asm.GetManifestResourceStream(res);
    }
}
