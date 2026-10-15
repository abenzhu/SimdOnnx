using System.Collections.Concurrent;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Backends.Vulkan;

/// <summary>
/// Process-wide Vulkan device + per-(model,size) compiled graph cache.
/// Constructed lazily on first Vulkan <see cref="GcdrEnhancer"/> use.
/// </summary>
public sealed unsafe class VkBackend : IDisposable
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
    public static VkBackend Default
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
    public static CompileOptions PlanOptions => new(Nhwc: true, PhysicalIo: false, Target: PlanTarget.Vulkan);
    public static CompileOptions PipeOptions => new(Nhwc: true, PhysicalIo: true, Target: PlanTarget.Vulkan);

    /// <summary>fp16 activation/weight storage (needs storageBuffer16BitAccess; GCDRNET_VK_F16=0 forces fp32).</summary>
    internal bool F16 => Dev.StorageBuffer16 && Env.Flag("VK_F16") != "0";

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
        // one build per key: a background warm-up and a caller racing for the same shape share it
        var build = _building.GetOrAdd(key, k => new Lazy<VkGraphModel>(() => Build(k.m, shape, k.opt)));
        var entry = new KeyValuePair<(OnnxModel, string, CompileOptions), Lazy<VkGraphModel>>(key, build);
        try { g = build.Value; }
        catch { _building.TryRemove(entry); throw; }
        g.LastUse = Interlocked.Increment(ref _clock);
        // cache first, then retire the in-flight entry: no window for a second build
        bool added = _graphs.TryAdd(key, g);
        _building.TryRemove(entry);
        if (!added)
        {
            VkGraphModel cur = _graphs[key];
            if (!ReferenceEquals(cur, g)) g.Dispose();
            return cur;
        }
        // bound the cache: evict the least recently used other graph beyond the cap
        while (_graphs.Count > MaxGraphs)
        {
            var victim = _graphs.Where(kv => !kv.Key.Equals(key)).OrderBy(kv => kv.Value.LastUse).Select(kv => kv.Key).FirstOrDefault();
            if (victim.Equals(default)) break;
            if (_graphs.TryRemove(victim, out var old)) old.Release();
        }
        return g;
    }

    private readonly ConcurrentDictionary<(OnnxModel m, string shape, CompileOptions opt), Lazy<VkGraphModel>> _building = new();

    /// <summary>
    /// Starts building the graph of an upcoming shape in the background (a <see cref="GraphFor"/>
    /// of the same key then waits for that build instead of starting its own). Builds run on a few
    /// below-normal-priority threads: they use the cores while callers wait on the device instead of
    /// competing with the callers' own pre/post-processing.
    /// </summary>
    internal void Prefetch(OnnxModel m, int[] shape, CompileOptions opt)
    {
        opt = opt with { Target = PlanTarget.Vulkan, Nhwc = true };
        var key = (m, string.Join("x", shape), opt);
        if (_graphs.ContainsKey(key) || _building.ContainsKey(key)) return;
        s_prefetch.Add(() =>
        {
            try { GraphFor(m, shape, opt); }
            catch { /* the caller's own GraphFor reports it */ }
        });
    }

    private static readonly System.Collections.Concurrent.BlockingCollection<Action> s_prefetch = StartPrefetchers();

    private static System.Collections.Concurrent.BlockingCollection<Action> StartPrefetchers()
    {
        var q = new System.Collections.Concurrent.BlockingCollection<Action>();
        int n = Math.Max(1, Math.Min(4, Environment.ProcessorCount / 8));
        for (int i = 0; i < n; i++)
            new Thread(() => { foreach (Action a in q.GetConsumingEnumerable()) a(); })
            { IsBackground = true, Priority = ThreadPriority.BelowNormal, Name = "SimdOnnx graph prefetch" }.Start();
        return q;
    }

    private VkGraphModel Build(OnnxModel m, int[] shape, CompileOptions opt)
    {
        // weights live in a per-model const store shared across sizes, so a graph is
        // just a plan + dispatch list (no per-size device memory)
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        ExecPlan plan = GraphCompiler.Compile(m, shape, opt);
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        var g = new VkGraphModel(Dev, plan, F16, constKey: m);
        Stats.Record(t1 - t0, System.Diagnostics.Stopwatch.GetTimestamp() - t1);
        if (Stats.Verbose)
            Console.Error.WriteLine($"[simdonnx] graph {System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(m):x8} {string.Join("x", shape)} out={opt.OutputName ?? "-"} {(t1 - t0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F2} ms");
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
    /// <summary>A session over the graph of <paramref name="m"/> at <paramref name="shape"/> under
    /// <paramref name="options"/> (Target / NHWC are forced; extensions via <see cref="CompileOptions.Extensions"/>).</summary>
    public VkInferenceSession NewSession(OnnxModel m, int[] shape, CompileOptions options) => NewSession(m, shape, options, null);

    /// <summary>The device's name (e.g. "Intel(R) Arc(TM) B580 Graphics").</summary>
    public string DeviceName => Dev.DeviceName;

    internal VkInferenceSession NewSession(OnnxModel m, int[] shape, CompileOptions opt, VkSessionBuffers? shared)
    {
        VkGraphModel g = AcquireGraph(m, shape, opt);
        try { return new VkInferenceSession(Dev, g.Plan, g, shared); }
        finally { g.Release(); }
    }

    public void Dispose() => Dev.Dispose();

    /// <summary>Graph-cache misses and their cost; SIMDONNX_STATS=1 prints them at exit.</summary>
    internal static class Stats
    {
        private static long s_graphs, s_compileTicks, s_emitTicks;
        private static readonly bool s_print = Environment.GetEnvironmentVariable("SIMDONNX_STATS") is "1" or "2";
        internal static readonly bool Verbose = Environment.GetEnvironmentVariable("SIMDONNX_STATS") == "2";
        static Stats()
        {
            if (s_print)
                AppDomain.CurrentDomain.ProcessExit += (_, _) => Console.Error.WriteLine(
                    $"[simdonnx] vulkan graphs built: {s_graphs}, plan compile {Ms(s_compileTicks):F0} ms, graph emit {Ms(s_emitTicks):F0} ms");
        }
        private static double Ms(long ticks) => ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        internal static void Record(long compile, long emit)
        {
            Interlocked.Increment(ref s_graphs);
            Interlocked.Add(ref s_compileTicks, compile);
            Interlocked.Add(ref s_emitTicks, emit);
        }
    }
}
