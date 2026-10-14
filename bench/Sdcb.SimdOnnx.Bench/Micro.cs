using System.Diagnostics;
using System.Runtime.Intrinsics;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Kernels;

namespace Sdcb.SimdOnnx.Bench;

/// <summary>Kernel micro-benchmarks: <c>--micro ew|dw|conv ...</c> (single thread).</summary>
internal static class Micro
{
    public static void Run(string[] args)
    {
        switch (args[0])
        {
            case "ew":
            {
                int[] s = [1, 6, 80, 96];
                int n = s.Aggregate(1, (a, b) => a * b);
                float[] a = Rand(n, 1), b = Rand(n, 2), d = new float[n];
                Time("ew same-shape add 46080", () => Ops.Elementwise(a, s, b, s, d, s, EwOp.Add));
                int[] cs = [1, 1, 1, 96];
                float[] c = Rand(96, 3);
                Time("ew channel mul 46080", () => Ops.Elementwise(a, s, c, cs, d, s, EwOp.Mul));
                foreach (ActivationKind k in Enum.GetValues<ActivationKind>())
                    if (k != ActivationKind.None)
                        Time($"act {k} 46080", () => Ops.Activation(a, d, k));
                Time("relu 46080", () => Ops.Relu(a, d));
                                float[] big = Rand(40 * 6906, 5), outb = new float[big.Length];
                Time("exp128 276k", () => Exp128(big, outb), 200);
                Time("exp256 276k", () => Exp256(big, outb), 200);
                Time("softmax(noalloc) 40x6906", () => Ops.Softmax(big, outb, 6906), 200);
                break;
            }
            case "calib":
            {
                // dependent multiply chain: 1e9 × imul (3-cycle latency) → GHz estimate
                long x = 1;
                long t0 = Stopwatch.GetTimestamp();
                for (int i = 0; i < 1_000_000_000; i++) x = x * 3;
                double s = Stopwatch.GetElapsedTime(t0).TotalSeconds;
                Console.WriteLine($"calib: {s:F3}s → ~{3.0 / s:F2} GHz effective ({x & 1})");
                break;
            }
            case "dwk":
                unsafe
                {
                    int c = 64, taps = 25;
                    float* src = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(64 * 1024 * 4, 64);
                    float* w = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(taps * c * 4), 64);
                    float* b = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(c * 4), 64);
                    float* o = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(4 * c * 4), 64);
                    for (int i = 0; i < 64 * 1024; i++) src[i] = (i % 7) * 0.1f;
                    for (int i = 0; i < taps * c; i++) w[i] = (i % 5) * 0.1f;
                    for (int i = 0; i < c; i++) b[i] = 0.5f;
                    float** ptrs = stackalloc float*[taps * 4];
                    for (int p = 0; p < 4; p++)
                        for (int t = 0; t < taps; t++) ptrs[p * taps + t] = src + ((t / 5) * 80 + p + t % 5) * c;
                    const int N = 200000;
                    for (int i = 0; i < 1000; i++) Nhwc.DwBlockAvx(ptrs, taps, c, w, b, o, null, 4, ActivationKind.None);
                    long t0 = Stopwatch.GetTimestamp();
                    for (int i = 0; i < N; i++) Nhwc.DwBlockAvx(ptrs, taps, c, w, b, o, null, 4, ActivationKind.None);
                    double ns = Stopwatch.GetElapsedTime(t0).TotalMilliseconds * 1e6 / N;
                    Console.WriteLine($"DwBlockAvx 4px x 64ch x 25 taps: {ns:F1} ns/call  ({ns * 4.4 / (taps * 4 * 64 / 8.0):F2} cycles per vector FMA @4.4GHz)");
                }
                break;
            case "dw":
            {
                foreach ((int h, int w, int c, int k, int s) in new[] { (5, 80, 64, 5, 1), (40, 80, 8, 3, 1), (20, 80, 16, 3, 1), (6, 80, 48, 3, 1) })
                {
                    int pad = k / 2, oh = (h + 2 * pad - k) / s + 1, ow = (w + 2 * pad - k) / s + 1;
                    float[] src = Rand(h * w * c, 1), wt = Rand(c * k * k, 2), bias = Rand(c, 3), dst = new float[oh * ow * c];
                    var p = new ConvP { Pads = [pad, pad, pad, pad], Strides = [s, s], Dilations = [1, 1], Kernel = [k, k], Group = c, Act = Enum.Parse<ActivationKind>(Environment.GetEnvironmentVariable("MICRO_ACT") ?? "HardSwish") };
                    NhwcConvPlan kp = Nhwc.Pack(wt, [c, 1, k, k], bias, c, c);
                    double macs = (double)oh * ow * c * k * k;
                    Time($"dw{k}x{k}s{s} {h}x{w}x{c} ({macs / 1e6:F2}M MAC)", () => Nhwc.Conv(src, h, w, c, dst, oh, ow, c, p, kp, default));
                }
                break;
            }
        }
    }

    private static unsafe void Exp128(float[] s, float[] d)
    {
        fixed (float* ps = s, pd = d)
            for (int i = 0; i + 4 <= s.Length; i += 4)
                SimdMath.ExpCore(System.Runtime.Intrinsics.Vector128.Load(ps + i)).Store(pd + i);
    }

    private static unsafe void Exp256(float[] s, float[] d)
    {
        fixed (float* ps = s, pd = d)
            for (int i = 0; i + 8 <= s.Length; i += 8)
                SimdMath.ExpCore(System.Runtime.Intrinsics.Vector256.Load(ps + i)).Store(pd + i);
    }

    internal static float[] Rand(int n, int seed)
    {
        var r = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)r.NextDouble() * 2 - 1).ToArray();
    }

    internal static void Time(string name, Action a, int iters = 2000)
    {
        for (int i = 0; i < 50; i++) a();
        long t0 = Stopwatch.GetTimestamp();
        for (int i = 0; i < iters; i++) a();
        Console.WriteLine($"{name,-40} {Stopwatch.GetElapsedTime(t0).TotalMilliseconds * 1000 / iters,8:F2} us");
    }
}
