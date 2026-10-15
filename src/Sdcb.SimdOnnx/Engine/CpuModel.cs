using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// CPU plans of one model for varying input shapes: each shape is compiled once (bounded LRU) and
/// every plan shares the packed weights (<see cref="CpuKernels.Cache"/>), so a new shape costs only
/// the graph lowering. Thread-safe.
/// </summary>
internal sealed class CpuModel
{
    private readonly Dictionary<string, (Lazy<ExecPlan> plan, long used)> _plans = new(StringComparer.Ordinal);
    private readonly CpuKernels.Cache _packs;
    private readonly int _capacity;
    private long _clock;

    /// <param name="packs">Packed-weight cache to share with other CPU models of the same graph
    /// (e.g. a head-truncated variant); null = private.</param>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal CpuModel(OnnxModel model, CompileOptions? options = null, int capacity = 64, CpuKernels.Cache? packs = null)
    {
        Model = model;
        Options = options ?? CompileOptions.Default;
        _capacity = Math.Max(1, capacity);
        _packs = packs ?? new CpuKernels.Cache();
    }

    internal OnnxModel Model { get; }
    internal CompileOptions Options { get; }
    internal CpuKernels.Cache Packs => _packs;

    /// <summary>The packed plan for <paramref name="shape"/> (compiled on first use).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal ExecPlan Plan(ReadOnlySpan<int> shape)
    {
        string key = string.Join(",", shape.ToArray());
        Lazy<ExecPlan> lazy;
        lock (_plans)
        {
            if (_plans.TryGetValue(key, out var e))
            {
                _plans[key] = (e.plan, ++_clock);
                lazy = e.plan;
            }
            else
            {
                // one compile per shape: concurrent callers of a new shape wait for the same plan
                int[] sh = shape.ToArray();
                lazy = new Lazy<ExecPlan>(() =>
                {
                    ExecPlan p = GraphCompiler.Compile(Model, sh, Options);
                    CpuKernels.Pack(p, _packs);
                    return p;
                }, LazyThreadSafetyMode.ExecutionAndPublication);
                _plans[key] = (lazy, ++_clock);
                if (_plans.Count > _capacity)
                {
                    string oldest = _plans.Where(kv => kv.Key != key).OrderBy(kv => kv.Value.used).First().Key;
                    _plans.Remove(oldest);
                }
            }
        }
        try { return lazy.Value; }
        catch
        {
            lock (_plans)
                if (_plans.TryGetValue(key, out var cur) && cur.plan == lazy) _plans.Remove(key);
            throw;
        }
    }
}

/// <summary>
/// A reshapable CPU session over a <see cref="CpuModel"/>: one grow-only native arena reused by the
/// plan of whatever shape is current. Not thread-safe (one in-flight run).
/// </summary>
internal sealed unsafe class CpuRunner : IDisposable
{
    private readonly CpuModel _model;
    private InferenceSession? _session;
    private nint _arena;
    private long _arenaBytes;

    internal CpuRunner(CpuModel model) => _model = model;

    internal int Threads { get; set; }
    internal InferenceSession Session => _session ?? throw new InvalidOperationException("Reshape first.");
    internal int[] InputShape => Session.InputShape;
    internal int[] OutputShape => Session.OutputShape;
    internal Span<float> Input => Session.Input;
    internal ReadOnlySpan<float> Output => Session.Output;
    internal long ArenaBytes => _arenaBytes;

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal void Reshape(ReadOnlySpan<int> shape)
    {
        if (_session is { } s && s.InputShape.AsSpan().SequenceEqual(shape)) return;
        ExecPlan plan = _model.Plan(shape);
        long need = Math.Max((long)plan.ArenaFloats * sizeof(float), 64);
        if (need > _arenaBytes)
        {
            Free();
            _arena = (nint)NativeMemory.AllocZeroed((nuint)need);
            _arenaBytes = need;
            GC.AddMemoryPressure(need);
        }
        _session = new InferenceSession(plan, _arena);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal void Run()
    {
        InferenceSession s = Session;
        s.Threads = Threads;
        s.Run();
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private void Free()
    {
        _session = null;
        if (_arena == 0) return;
        NativeMemory.Free((void*)_arena);
        GC.RemoveMemoryPressure(_arenaBytes);
        _arena = 0;
        _arenaBytes = 0;
    }

    public void Dispose() => Free();
}
