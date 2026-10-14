using System.Collections.Concurrent;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Backends.Vulkan;

/// <summary>
/// Process-wide Vulkan device + per-(model,size) compiled graph cache.
/// Constructed lazily on first Vulkan <see cref="GcdrEnhancer"/> use.
/// </summary>
internal sealed unsafe class VkBackend : IDisposable
{
    internal readonly VkDevice Dev;
    // (model, shape, options) -> compiled graph (dispatch list; weights live in the per-model const store)
    private readonly ConcurrentDictionary<(OnnxModel m, string shape, CompileOptions opt), VkGraphModel> _graphs = new();

    private static VkBackend? s_default;
    private static System.Runtime.ExceptionServices.ExceptionDispatchInfo? s_createError;
    private static readonly object s_gate = new();

    /// <summary>
    /// The process-wide backend. After a device loss the next access creates a new device; objects
    /// built on the lost one (graphs, sessions, buffers) stay dead and must be re-created by callers.
    /// A failed first creation (no usable Vulkan) is cached and rethrown.
    /// </summary>
    internal static VkBackend Default
    {
        get
        {
            VkBackend? b = Volatile.Read(ref s_default);
            if (b is not null && !b.Dev.Lost) return b;
            lock (s_gate)
            {
                s_createError?.Throw();
                if (s_default is null || s_default.Dev.Lost)
                {
                    try { s_default = new VkBackend(); }
                    catch (Exception ex) when (s_default is null)
                    {
                        s_createError = System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex);
                        throw;
                    }
                }
                return s_default;
            }
        }
    }

    /// <summary>
    /// Plan options for GPU compiles: always channels-last (shaders address NHWC).
    /// Standalone sessions keep logical NCHW I/O (boundary relayouts in the plan);
    /// the fused pipeline keeps I/O physical and its glue shaders speak NHWC.
    /// </summary>
    internal static CompileOptions PlanOptions => new(Nhwc: true, PhysicalIo: false, Target: PlanTarget.Vulkan);
    internal static CompileOptions PipeOptions => new(Nhwc: true, PhysicalIo: true, Target: PlanTarget.Vulkan);

    /// <summary>fp16 activation/weight storage (needs storageBuffer16BitAccess; GCDRNET_VK_F16=0 forces fp32).</summary>
    internal bool F16 => Dev.StorageBuffer16 && Environment.GetEnvironmentVariable("GCDRNET_VK_F16") != "0";

    internal VkBackend()
    {
        Dev = VkDevice.Create();
    }

    /// <summary>Standalone session for one model at [n,c,h,w] (test path).</summary>
    public VkInferenceSession NewSession(OnnxModel m, int[] shape) => NewSession(m, shape, (EngineExtensions?)null);

    internal VkInferenceSession NewSession(OnnxModel m, int[] shape, EngineExtensions? ext)
    {
        VkGraphModel g = AcquireGraph(m, shape, physIo: false, ext);   // caller ref: later evictions can't free it under us
        try { return new VkInferenceSession(Dev, g.Plan, g); }
        finally { g.Release(); }
    }

    /// <summary>Compiled GPU graph for a model+shape (cached, bounded).</summary>
    internal VkGraphModel GraphFor(OnnxModel m, int[] shape, bool physIo = false, EngineExtensions? ext = null) =>
        GraphFor(m, shape, (physIo ? PipeOptions : PlanOptions) with { Extensions = ext });

    /// <summary>Compiled GPU graph for a model+shape under explicit plan options (Target forced to Vulkan).</summary>
    internal VkGraphModel GraphFor(OnnxModel m, int[] shape, CompileOptions opt)
    {
        opt = opt with { Target = PlanTarget.Vulkan, Nhwc = true };
        var key = (m, string.Join("x", shape), opt);
        if (_graphs.TryGetValue(key, out var g))
        {
            g.LastUse = Interlocked.Increment(ref _clock);
            return g;
        }
        // weights live in a per-model const store shared across sizes, so a graph is
        // just a plan + dispatch list (no per-size device memory)
        g = new VkGraphModel(Dev, GraphCompiler.Compile(m, shape, opt), F16, constKey: m);
        g.LastUse = Interlocked.Increment(ref _clock);
        if (!_graphs.TryAdd(key, g)) { g.Dispose(); return _graphs[key]; }
        // bound the cache: evict the least recently used other graph beyond the cap
        while (_graphs.Count > MaxGraphs)
        {
            var victim = _graphs.Where(kv => !kv.Key.Equals(key)).OrderBy(kv => kv.Value.LastUse).Select(kv => kv.Key).FirstOrDefault();
            if (victim.Equals(default)) break;
            if (_graphs.TryRemove(victim, out var old)) old.Release();
        }
        return g;
    }

    // graphs hold no device memory (consts are pooled per model); the bound is for the managed
    // plans (~0.4 MB each). OCR alone sees ~150 distinct det/rec shapes over a varied image set.
    private const int MaxGraphs = 256;
    private long _clock;

    /// <summary><see cref="GraphFor"/> plus a caller reference. Retries when a concurrent
    /// eviction freed the cached graph between lookup and acquire (never resurrects it).</summary>
    internal VkGraphModel AcquireGraph(OnnxModel m, int[] shape, bool physIo, EngineExtensions? ext = null) =>
        AcquireGraph(m, shape, (physIo ? PipeOptions : PlanOptions) with { Extensions = ext });

    internal VkGraphModel AcquireGraph(OnnxModel m, int[] shape, CompileOptions opt)
    {
        while (true)
        {
            VkGraphModel g = GraphFor(m, shape, opt);
            if (g.TryAcquire()) return g;
        }
    }

    /// <summary>Session over the graph for <paramref name="shape"/> under explicit plan options.</summary>
    internal VkInferenceSession NewSession(OnnxModel m, int[] shape, CompileOptions opt, VkSessionBuffers? shared = null)
    {
        VkGraphModel g = AcquireGraph(m, shape, opt);
        try { return new VkInferenceSession(Dev, g.Plan, g, shared); }
        finally { g.Release(); }
    }

    public void Dispose() => Dev.Dispose();
}
