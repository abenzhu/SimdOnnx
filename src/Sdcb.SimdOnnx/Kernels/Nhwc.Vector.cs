using System.Numerics;
using System.Runtime.CompilerServices;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdOnnx.Kernels;

/// <summary>
/// Vector&lt;float&gt; (8-wide) register-blocked tiers for runtimes without the hardware-intrinsic
/// APIs (netstandard2.0 on .NET Framework / Mono, or AVX2 hidden): the 6 × 16 dense tile and the
/// blocked depthwise kernels, mirroring their AVX2 counterparts with the same per-lane op order.
/// </summary>
internal static unsafe partial class Nhwc
{
    /// <summary>Hardware Vector&lt;float&gt; of 8 lanes and no intrinsic tier above it.</summary>
    internal static bool UseVec8 => !UseAvx && !UseNeon && !UsePacked && Vector.IsHardwareAccelerated && Vector<float>.Count == 8;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> Ld(float* p) => Unsafe.ReadUnaligned<Vector<float>>(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void St(float* p, Vector<float> v) => Unsafe.WriteUnaligned(p, v);

    /// <summary>Fused epilogue on one vector (transcendental activations via <see cref="SimdMath"/>).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> ActV(Vector<float> v, ActivationKind act) => act switch
    {
        ActivationKind.Relu => Vector.Max(v, Vector<float>.Zero),
        ActivationKind.HardSwish => v * Vector.Min(Vector.Max(v * new Vector<float>(1f / 6f) + new Vector<float>(0.5f), Vector<float>.Zero), Vector<float>.One),
        ActivationKind.Sigmoid => SimdMath.SigmoidV(v),
        ActivationKind.Swish => v * SimdMath.SigmoidV(v),
        ActivationKind.Gelu => SimdMath.GeluV(v),
        _ => v,
    };

    /// <summary>6 × 16 tile over Vector&lt;float&gt; (8 lanes): twelve register accumulators.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void TileVec8(float** ptrs, int rows, int taps, int ic, float* wb, float* bb,
        float* outp, float* addp, int oc, int lanes, ActivationKind act)
    {
        Vector<float> a0 = Ld(bb), b0 = Ld(bb + 8);
        Vector<float> a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0;
        Vector<float> b1 = b0, b2 = b0, b3 = b0, b4 = b0, b5 = b0;
        for (int tap = 0; tap < taps; tap++)
        {
            float** pp = ptrs + tap * Tile;
            float* p0 = pp[0], p1 = pp[1], p2 = pp[2], p3 = pp[3], p4 = pp[4], p5 = pp[5];
            float* w = wb + (long)tap * ic * Ob;
            for (int c = 0; c < ic; c++, w += Ob)
            {
                Vector<float> w0 = Ld(w), w1 = Ld(w + 8);
                var v0 = new Vector<float>(p0[c]); a0 += v0 * w0; b0 += v0 * w1;
                var v1 = new Vector<float>(p1[c]); a1 += v1 * w0; b1 += v1 * w1;
                var v2 = new Vector<float>(p2[c]); a2 += v2 * w0; b2 += v2 * w1;
                var v3 = new Vector<float>(p3[c]); a3 += v3 * w0; b3 += v3 * w1;
                var v4 = new Vector<float>(p4[c]); a4 += v4 * w0; b4 += v4 * w1;
                var v5 = new Vector<float>(p5[c]); a5 += v5 * w0; b5 += v5 * w1;
            }
        }
        float* part = stackalloc float[Tile * Ob];
        St(part, a0); St(part + 8, b0); St(part + 16, a1); St(part + 24, b1);
        St(part + 32, a2); St(part + 40, b2); St(part + 48, a3); St(part + 56, b3);
        St(part + 64, a4); St(part + 72, b4); St(part + 80, a5); St(part + 88, b5);
        for (int r = 0; r < rows; r++, part += Ob)
        {
            float* o = outp + (long)r * oc;
            float* ad = addp == null ? null : addp + (long)r * oc;
            int l = 0;
            for (; l + 8 <= lanes; l += 8)
            {
                Vector<float> v = Ld(part + l);
                if (ad != null) v += Ld(ad + l);
                St(o + l, ActV(v, act));
            }
            for (; l < lanes; l++)
                o[l] = ActS(ad != null ? part[l] + ad[l] : part[l], act);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DwStoreV(float* o, float* ad, int off, Vector<float> acc, ActivationKind act)
    {
        if (ad != null) acc += Ld(ad + off);
        St(o + off, ActV(acc, act));
    }

    /// <summary>Pointer-table depthwise for up to 4 pixels (see <see cref="DwBlockAvx"/>).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static int DwBlockVec(float** p, int taps, int c, float* w, float* b, float* o, float* ad, int np, ActivationKind act)
    {
        float** p1 = p + taps, p2 = p + 2 * taps, p3 = p + 3 * taps;
        int ch = 0;
        for (; ch + 8 <= c; ch += 8)
        {
            Vector<float> a0 = Ld(b + ch), a1 = a0, a2 = a0, a3 = a0;
            for (int t = 0; t < taps; t++)
            {
                Vector<float> wv = Ld(w + (long)t * c + ch);
                a0 += Ld(p[t] + ch) * wv;
                a1 += Ld(p1[t] + ch) * wv;
                a2 += Ld(p2[t] + ch) * wv;
                a3 += Ld(p3[t] + ch) * wv;
            }
            DwStoreV(o, ad, ch, a0, act);
            if (np > 1) DwStoreV(o, ad, c + ch, a1, act);
            if (np > 2) DwStoreV(o, ad, 2 * c + ch, a2, act);
            if (np > 3) DwStoreV(o, ad, 3 * c + ch, a3, act);
        }
        return ch;
    }

    /// <summary>4 interior pixels from kernel-row bases (see <see cref="DwRunAvx"/>); c % 8 == 0.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void DwRunVec(float** rows, int kh, int kw, int c, int ps, float* w, float* b, float* o, float* ad, ActivationKind act)
    {
        for (int ch = 0; ch < c; ch += 8)
        {
            Vector<float> a0 = Ld(b + ch), a1 = a0, a2 = a0, a3 = a0;
            for (int ky = 0; ky < kh; ky++)
            {
                float* s = rows[ky] + ch, wt = w + (long)ky * kw * c + ch;
                for (int kx = 0; kx < kw; kx++, s += c, wt += c)
                {
                    Vector<float> wv = Ld(wt);
                    a0 += Ld(s) * wv;
                    a1 += Ld(s + ps) * wv;
                    a2 += Ld(s + 2 * ps) * wv;
                    a3 += Ld(s + 3 * ps) * wv;
                }
            }
            DwStoreV(o, ad, ch, a0, act);
            DwStoreV(o, ad, c + ch, a1, act);
            DwStoreV(o, ad, 2 * c + ch, a2, act);
            DwStoreV(o, ad, 3 * c + ch, a3, act);
        }
    }
}
