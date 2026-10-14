using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdOnnx.Kernels;

/// <summary>
/// Binary elementwise kernels (same shape / scalar / per-channel broadcast) with the op specialized
/// at JIT time (generic struct ops) and an optional fused ReLU. 256-bit vectors where hardware
/// accelerated, else 128-bit (NEON / SSE / WASM SIMD), then scalar. Parallel only for large tensors.
/// </summary>
internal static unsafe class Ew
{
    private const long ParallelMin = 1 << 16;

    internal interface IOp
    {
        static abstract Vector256<float> V(Vector256<float> a, Vector256<float> b);
        static abstract Vector128<float> V(Vector128<float> a, Vector128<float> b);
        static abstract float S(float a, float b);
    }

    private struct Add : IOp
    {
        public static Vector256<float> V(Vector256<float> a, Vector256<float> b) => a + b;
        public static Vector128<float> V(Vector128<float> a, Vector128<float> b) => a + b;
        public static float S(float a, float b) => a + b;
    }
    private struct Sub : IOp
    {
        public static Vector256<float> V(Vector256<float> a, Vector256<float> b) => a - b;
        public static Vector128<float> V(Vector128<float> a, Vector128<float> b) => a - b;
        public static float S(float a, float b) => a - b;
    }
    private struct Mul : IOp
    {
        public static Vector256<float> V(Vector256<float> a, Vector256<float> b) => a * b;
        public static Vector128<float> V(Vector128<float> a, Vector128<float> b) => a * b;
        public static float S(float a, float b) => a * b;
    }
    private struct Div : IOp
    {
        public static Vector256<float> V(Vector256<float> a, Vector256<float> b) => a / b;
        public static Vector128<float> V(Vector128<float> a, Vector128<float> b) => a / b;
        public static float S(float a, float b) => a / b;
    }
    // operands swapped: s op x for scalar / vector broadcasts on the left
    private struct RSub : IOp
    {
        public static Vector256<float> V(Vector256<float> a, Vector256<float> b) => b - a;
        public static Vector128<float> V(Vector128<float> a, Vector128<float> b) => b - a;
        public static float S(float a, float b) => b - a;
    }
    private struct RDiv : IOp
    {
        public static Vector256<float> V(Vector256<float> a, Vector256<float> b) => b / a;
        public static Vector128<float> V(Vector128<float> a, Vector128<float> b) => b / a;
        public static float S(float a, float b) => b / a;
    }

    private static bool V256 => Vector256.IsHardwareAccelerated;
    private static bool V128 => Vector128.IsHardwareAccelerated;

    /// <summary>d = a op b, elementwise.</summary>
    public static void Binary(float* a, float* b, float* d, long n, EwOp op, bool relu)
    {
        switch (op)
        {
            case EwOp.Add: Split<Add>(a, b, d, n, relu); break;
            case EwOp.Sub: Split<Sub>(a, b, d, n, relu); break;
            case EwOp.Mul: Split<Mul>(a, b, d, n, relu); break;
            default: Split<Div>(a, b, d, n, relu); break;
        }
    }

    /// <summary>d = a op s (or s op a when <paramref name="rev"/>).</summary>
    public static void Scalar(float* a, float s, float* d, long n, EwOp op, bool rev, bool relu)
    {
        switch (op)
        {
            case EwOp.Add: SplitS<Add>(a, s, d, n, relu); break;
            case EwOp.Mul: SplitS<Mul>(a, s, d, n, relu); break;
            case EwOp.Sub when rev: SplitS<RSub>(a, s, d, n, relu); break;
            case EwOp.Sub: SplitS<Sub>(a, s, d, n, relu); break;
            case EwOp.Div when rev: SplitS<RDiv>(a, s, d, n, relu); break;
            default: SplitS<Div>(a, s, d, n, relu); break;
        }
    }

    /// <summary>d[r, :] = full[r, :] op vec (vec op full when <paramref name="rev"/>), rows of c.</summary>
    public static void Channel(float* full, float* vec, float* d, long rows, int c, EwOp op, bool rev, bool relu)
    {
        switch (op)
        {
            case EwOp.Add: SplitC<Add>(full, vec, d, rows, c, relu); break;
            case EwOp.Mul: SplitC<Mul>(full, vec, d, rows, c, relu); break;
            case EwOp.Sub when rev: SplitC<RSub>(full, vec, d, rows, c, relu); break;
            case EwOp.Sub: SplitC<Sub>(full, vec, d, rows, c, relu); break;
            case EwOp.Div when rev: SplitC<RDiv>(full, vec, d, rows, c, relu); break;
            default: SplitC<Div>(full, vec, d, rows, c, relu); break;
        }
    }

    private static int Chunks(long work) =>
        work < ParallelMin ? 1 : (int)Math.Min(Par.Threads, Math.Max(1, work / (ParallelMin / 2)));

    private static void Split<T>(float* a, float* b, float* d, long n, bool relu) where T : struct, IOp
    {
        int k = Chunks(n);
        if (k == 1) { Run<T>(a, b, d, n, relu); return; }
        nint pa = (nint)a, pb = (nint)b, pd = (nint)d;
        long step = (n / k + 15) & ~15L;
        Par.For(0, k, t =>
        {
            long i0 = t * step, len = Math.Min(n - i0, step);
            if (len > 0) Run<T>((float*)pa + i0, (float*)pb + i0, (float*)pd + i0, len, relu);
        });
    }

    private static void SplitS<T>(float* a, float s, float* d, long n, bool relu) where T : struct, IOp
    {
        int k = Chunks(n);
        if (k == 1) { RunS<T>(a, s, d, n, relu); return; }
        nint pa = (nint)a, pd = (nint)d;
        long step = (n / k + 15) & ~15L;
        Par.For(0, k, t =>
        {
            long i0 = t * step, len = Math.Min(n - i0, step);
            if (len > 0) RunS<T>((float*)pa + i0, s, (float*)pd + i0, len, relu);
        });
    }

    private static void SplitC<T>(float* full, float* vec, float* d, long rows, int c, bool relu) where T : struct, IOp
    {
        int k = Chunks(rows * c);
        if (k == 1) { RunC<T>(full, vec, d, rows, c, relu); return; }
        nint pf = (nint)full, pv = (nint)vec, pd = (nint)d;
        long step = (rows + k - 1) / k;
        Par.For(0, k, t =>
        {
            long r0 = t * step, len = Math.Min(rows - r0, step);
            if (len > 0) RunC<T>((float*)pf + r0 * c, (float*)pv, (float*)pd + r0 * c, len, c, relu);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Run<T>(float* a, float* b, float* d, long n, bool relu) where T : struct, IOp
    {
        long i = 0;
        if (V256)
        {
            Vector256<float> z = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                Vector256<float> v = T.V(Vector256.Load(a + i), Vector256.Load(b + i));
                (relu ? Vector256.Max(v, z) : v).Store(d + i);
            }
        }
        else if (V128)
        {
            Vector128<float> z = Vector128<float>.Zero;
            for (; i + 4 <= n; i += 4)
            {
                Vector128<float> v = T.V(Vector128.Load(a + i), Vector128.Load(b + i));
                (relu ? Vector128.Max(v, z) : v).Store(d + i);
            }
        }
        for (; i < n; i++)
        {
            float v = T.S(a[i], b[i]);
            d[i] = relu ? MathF.Max(v, 0f) : v;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RunS<T>(float* a, float s, float* d, long n, bool relu) where T : struct, IOp
    {
        long i = 0;
        if (V256)
        {
            Vector256<float> z = Vector256<float>.Zero, vs = Vector256.Create(s);
            for (; i + 8 <= n; i += 8)
            {
                Vector256<float> v = T.V(Vector256.Load(a + i), vs);
                (relu ? Vector256.Max(v, z) : v).Store(d + i);
            }
        }
        else if (V128)
        {
            Vector128<float> z = Vector128<float>.Zero, vs = Vector128.Create(s);
            for (; i + 4 <= n; i += 4)
            {
                Vector128<float> v = T.V(Vector128.Load(a + i), vs);
                (relu ? Vector128.Max(v, z) : v).Store(d + i);
            }
        }
        for (; i < n; i++)
        {
            float v = T.S(a[i], s);
            d[i] = relu ? MathF.Max(v, 0f) : v;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RunC<T>(float* full, float* vec, float* d, long rows, int c, bool relu) where T : struct, IOp
    {
        for (long r = 0; r < rows; r++)
        {
            float* f = full + r * c, o = d + r * c;
            int i = 0;
            if (V256)
            {
                Vector256<float> z = Vector256<float>.Zero;
                for (; i + 8 <= c; i += 8)
                {
                    Vector256<float> v = T.V(Vector256.Load(f + i), Vector256.Load(vec + i));
                    (relu ? Vector256.Max(v, z) : v).Store(o + i);
                }
            }
            else if (V128)
            {
                Vector128<float> z = Vector128<float>.Zero;
                for (; i + 4 <= c; i += 4)
                {
                    Vector128<float> v = T.V(Vector128.Load(f + i), Vector128.Load(vec + i));
                    (relu ? Vector128.Max(v, z) : v).Store(o + i);
                }
            }
            for (; i < c; i++)
            {
                float v = T.S(f[i], vec[i]);
                o[i] = relu ? MathF.Max(v, 0f) : v;
            }
        }
    }
}
