using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdOnnx.Kernels;

/// <summary>
/// Intra-op parallelism. Kernels read <see cref="Threads"/> / <see cref="Options"/> on the calling
/// thread, so a session can run with its own thread budget (<see cref="Use"/>) while other sessions
/// run concurrently with theirs. Default: SIMDONNX_THREADS (legacy GCDRNET_THREADS) or the
/// processor count.
/// </summary>
internal static class Par
{
    internal static readonly int DefaultThreads = InitThreads();
    private static readonly ParallelOptions?[] s_options = new ParallelOptions?[257];
    [ThreadStatic] private static int t_threads;

    private static int InitThreads() =>
        int.TryParse(Env.Flag("THREADS"), out int t) && t > 0 ? t : Environment.ProcessorCount;

    /// <summary>Thread budget of the current (calling) thread's run.</summary>
    internal static int Threads => t_threads > 0 ? t_threads : DefaultThreads;

    internal static ParallelOptions Options
    {
        get
        {
            int t = Math.Min(Threads, 256);
            return s_options[t] ??= new ParallelOptions { MaxDegreeOfParallelism = t };
        }
    }

    /// <summary>Sets the calling thread's budget until the returned scope is disposed (0 = default).</summary>
    internal static Scope Use(int threads)
    {
        int prev = t_threads;
        t_threads = threads;
        return new Scope(prev);
    }

    /// <summary>
    /// Parallel.For under the calling thread's budget; a budget of one (or a single index) runs inline —
    /// Parallel.For at MaxDegreeOfParallelism 1 still hops through the thread pool (tens of µs per call).
    /// </summary>
    internal static void For(int from, int to, Action<int> body)
    {
        if (to - from <= 1 || Threads <= 1)
        {
            for (int i = from; i < to; i++) body(i);
            return;
        }
        Parallel.For(from, to, Options, body);
    }

    internal readonly struct Scope(int prev) : IDisposable
    {
        public void Dispose() => t_threads = prev;
    }
}
