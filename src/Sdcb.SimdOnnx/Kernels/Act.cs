using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Wasm;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdOnnx.Kernels;

/// <summary>
/// Elementwise activations d = f(s) (s and d may alias) with f specialized at JIT time. 256-bit
/// vectors where hardware accelerated, else 128-bit, then scalar; parallel only for large tensors.
/// </summary>
internal static unsafe class Act
{
    private const long ParallelMin = 1 << 15;

    internal interface IAct
    {
        Vector256<float> V(Vector256<float> x);
        Vector128<float> V(Vector128<float> x);
        Vector<float> V(Vector<float> x);
        float S(float x);
    }

    private struct Relu : IAct
    {
        public Vector<float> V(Vector<float> x) => Vector.Max(x, Vector<float>.Zero);
        public Vector256<float> V(Vector256<float> x) => Vector256.Max(x, Vector256<float>.Zero);
        public Vector128<float> V(Vector128<float> x) => Vector128.Max(x, Vector128<float>.Zero);
        public float S(float x) => MathF.Max(x, 0f);
    }
    private struct Sigmoid : IAct
    {
        public Vector<float> V(Vector<float> x) => SimdMath.SigmoidV(x);
        public Vector256<float> V(Vector256<float> x) => SimdMath.SigmoidCore(x);
        public Vector128<float> V(Vector128<float> x) => PackedSimd.IsSupported ? SimdMath.SigmoidPk(x) : SimdMath.SigmoidCore(x);
        public float S(float x) => 1f / (1f + MathF.Exp(-x));
    }
    private struct Gelu : IAct
    {
        public Vector<float> V(Vector<float> x) => SimdMath.GeluV(x);
        public Vector256<float> V(Vector256<float> x) => SimdMath.GeluCore(x);
        public Vector128<float> V(Vector128<float> x) => PackedSimd.IsSupported ? SimdMath.GeluPk(x) : SimdMath.GeluCore(x);
        public float S(float x) => 0.5f * x * (1f + SimdMath.ErfScalar(x * 0.70710678118654752f));
    }
    private struct Erf : IAct
    {
        public Vector<float> V(Vector<float> x) => SimdMath.ErfV(x);
        public Vector256<float> V(Vector256<float> x) => SimdMath.ErfCore(x);
        public Vector128<float> V(Vector128<float> x) => PackedSimd.IsSupported ? SimdMath.ErfPk(x) : SimdMath.ErfCore(x);
        public float S(float x) => SimdMath.ErfScalar(x);
    }
    private struct HardSwish : IAct
    {
        public Vector<float> V(Vector<float> x) => x * Vector.Min(Vector.Max(x * new Vector<float>(1f / 6f) + new Vector<float>(0.5f), Vector<float>.Zero), Vector<float>.One);
        public Vector256<float> V(Vector256<float> x) =>
            x * Vector256.Min(Vector256.Max(x * Vector256.Create(1f / 6f) + Vector256.Create(0.5f), Vector256<float>.Zero), Vector256<float>.One);
        public Vector128<float> V(Vector128<float> x) =>
            x * Vector128.Min(Vector128.Max(x * Vector128.Create(1f / 6f) + Vector128.Create(0.5f), Vector128<float>.Zero), Vector128<float>.One);
        public float S(float x) => x * Math.Clamp(x * (1f / 6f) + 0.5f, 0f, 1f);
    }
    private struct Swish : IAct
    {
        public Vector<float> V(Vector<float> x) => x * SimdMath.SigmoidV(x);
        public Vector256<float> V(Vector256<float> x) => x * SimdMath.SigmoidCore(x);
        public Vector128<float> V(Vector128<float> x) => x * (PackedSimd.IsSupported ? SimdMath.SigmoidPk(x) : SimdMath.SigmoidCore(x));
        public float S(float x) => x / (1f + MathF.Exp(-x));
    }

    public static void Apply(float* s, float* d, long n, ActivationKind act)
    {
        switch (act)
        {
            case ActivationKind.Relu: Split<Relu>(s, d, n); break;
            case ActivationKind.Sigmoid: Split<Sigmoid>(s, d, n); break;
            case ActivationKind.Gelu: Split<Gelu>(s, d, n); break;
            case ActivationKind.HardSwish: Split<HardSwish>(s, d, n); break;
            case ActivationKind.Swish: Split<Swish>(s, d, n); break;
            default: if (s != d) Buffer.MemoryCopy(s, d, n * 4, n * 4); break;
        }
    }

    public static void ApplyErf(float* s, float* d, long n) => Split<Erf>(s, d, n);

    private static void Split<T>(float* s, float* d, long n) where T : struct, IAct
    {
        int k = n < ParallelMin ? 1 : (int)Math.Min(Par.Threads, Math.Max(1, n / (ParallelMin / 2)));
        if (k == 1) { Run<T>(s, d, n); return; }
        nint ps = (nint)s, pd = (nint)d;
        long step = (n / k + 15) & ~15L;
        Par.For(0, k, t =>
        {
            long i0 = t * step, len = Math.Min(n - i0, step);
            if (len > 0) Run<T>((float*)ps + i0, (float*)pd + i0, len);
        });
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void Run<T>(float* s, float* d, long n) where T : struct, IAct
    {
        long i = 0;
        if (Vector256.IsHardwareAccelerated)
            for (; i + 8 <= n; i += 8) default(T).V(Vector256.Load(s + i)).Store(d + i);
        else if (Vector128.IsHardwareAccelerated)
            for (; i + 4 <= n; i += 4) default(T).V(Vector128.Load(s + i)).Store(d + i);
        else if (Vector.IsHardwareAccelerated)
            for (int vl = Vector<float>.Count; i + vl <= n; i += vl)
                Unsafe.WriteUnaligned(d + i, default(T).V(Unsafe.ReadUnaligned<Vector<float>>(s + i)));
        for (; i < n; i++) d[i] = default(T).S(s[i]);
    }
}
