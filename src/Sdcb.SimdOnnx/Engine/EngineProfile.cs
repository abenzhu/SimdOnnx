using System.Collections.Concurrent;
using System.Diagnostics;

namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// Process-wide CPU operator profile for host benchmarks: per op name (ExecOp or extension name)
/// and per conv class, ticks and calls summed over every session while <see cref="Enabled"/>.
/// </summary>
internal static class EngineProfile
{
    internal static volatile bool Enabled = Environment.GetEnvironmentVariable("GCDRNET_PROF") == "1"
        || Env.Flag("PROF") == "1";

    private static readonly ConcurrentDictionary<string, long[]> s_ops = new(StringComparer.Ordinal);
    private static readonly ConcurrentDictionary<string, long[]> s_conv = new(StringComparer.Ordinal);

    internal static void Reset()
    {
        s_ops.Clear();
        s_conv.Clear();
    }

    internal static void Add(ExecNode n, long ticks)
    {
        Bump(s_ops, n.Ext?.Name ?? n.Op.ToString(), ticks);
        if (n.P is ConvP c) Bump(s_conv, ConvClass(c), ticks);
    }

    /// <summary>Time spent by a consumer-side operator (e.g. a head computed outside the plan).</summary>
    internal static void AddOp(string name, long ticks) => Bump(s_ops, name, ticks);

    private static void Bump(ConcurrentDictionary<string, long[]> d, string key, long ticks)
    {
        long[] slot = d.GetOrAdd(key, static _ => new long[2]);
        Interlocked.Add(ref slot[0], ticks);
        Interlocked.Increment(ref slot[1]);
    }

    internal static string ConvClass(ConvP c)
    {
        string kind = c.Group > 1 ? "Depthwise" : c.Kernel is [1, 1] ? "Conv1x1" : "Conv";
        string k = c.Kernel is [1, 1] && c.Group == 1 ? "" : $"{c.Kernel[0]}x{c.Kernel[1]}";
        return c.Strides[0] > 1 ? $"Stride{c.Strides[0]}{kind}{k}" : kind + k;
    }

    /// <summary>Milliseconds and call counts per operator name.</summary>
    internal static Dictionary<string, (double Ms, long Calls)> Operators() => Snap(s_ops);

    /// <summary>Milliseconds and call counts per conv class (e.g. Conv1x1, Depthwise3x3, Stride2Conv3x3).</summary>
    internal static Dictionary<string, (double Ms, long Calls)> ConvClasses() => Snap(s_conv);

    private static Dictionary<string, (double, long)> Snap(ConcurrentDictionary<string, long[]> d) =>
        d.ToDictionary(kv => kv.Key, kv => (Interlocked.Read(ref kv.Value[0]) * 1000.0 / Stopwatch.Frequency,
            Interlocked.Read(ref kv.Value[1])), StringComparer.Ordinal);
}
