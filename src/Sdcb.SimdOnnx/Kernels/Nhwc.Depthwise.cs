using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using Sdcb.SimdOnnx.Engine;

using F32x8 = System.Runtime.Intrinsics.Vector256<float>;
using V4 = System.Runtime.Intrinsics.Vector128<float>;

namespace Sdcb.SimdOnnx.Kernels;

/// <summary>
/// Register-blocked NHWC depthwise micro-kernels. A per-pixel tap loop is one FMA dependency
/// chain per channel vector (latency bound); these keep 8 independent chains in flight —
/// 2 pixels × 4 channel vectors, or 4 pixels × 1 vector for narrow tails. <c>ptrs</c> holds
/// <c>taps</c> source-row pointers for each of 4 pixels (missing pixels duplicate the last one;
/// only the first <c>np</c> pixels are stored). Same per-element op order as the scalar path.
/// </summary>
internal static unsafe partial class Nhwc
{
    /// <returns>Number of leading channels handled (multiple of the vector width).</returns>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static int DwBlockAvx(float** ptrs, int taps, int c, float* w, float* b, float* o, float* ad, int np, ActivationKind act)
    {
        int ch = 0;
        for (; ch + 32 <= c; ch += 32)
        {
            Dw2x32(ptrs, ptrs + taps, taps, c, ch, w, b, o, ad, np >= 2, act);
            if (np > 2)
                Dw2x32(ptrs + 2 * taps, ptrs + 3 * taps, taps, c, ch, w, b, o + 2 * c, ad == null ? null : ad + 2 * c, np >= 4, act);
        }
        for (; ch + 8 <= c; ch += 8) Dw4x8(ptrs, taps, c, ch, w, b, o, ad, np, act);
        return ch;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DwStore8(float* o, float* ad, int off, F32x8 acc, ActivationKind act)
    {
        if (ad != null) acc = Avx.Add(acc, Avx.LoadVector256(ad + off));
        Avx.Store(o + off, Act8(acc, act));
    }

    private static void Dw2x32(float** pa, float** pb, int taps, int c, int ch, float* w, float* b, float* o, float* ad, bool second, ActivationKind act)
    {
        F32x8 a0 = Avx.LoadVector256(b + ch), a1 = Avx.LoadVector256(b + ch + 8);
        F32x8 a2 = Avx.LoadVector256(b + ch + 16), a3 = Avx.LoadVector256(b + ch + 24);
        F32x8 b0 = a0, b1 = a1, b2 = a2, b3 = a3;
        for (int t = 0; t < taps; t++)
        {
            float* wt = w + (long)t * c + ch, sa = pa[t] + ch, sb = pb[t] + ch;
            F32x8 w0 = Avx.LoadVector256(wt), w1 = Avx.LoadVector256(wt + 8);
            F32x8 w2 = Avx.LoadVector256(wt + 16), w3 = Avx.LoadVector256(wt + 24);
            a0 = Fma.MultiplyAdd(Avx.LoadVector256(sa), w0, a0);
            a1 = Fma.MultiplyAdd(Avx.LoadVector256(sa + 8), w1, a1);
            a2 = Fma.MultiplyAdd(Avx.LoadVector256(sa + 16), w2, a2);
            a3 = Fma.MultiplyAdd(Avx.LoadVector256(sa + 24), w3, a3);
            b0 = Fma.MultiplyAdd(Avx.LoadVector256(sb), w0, b0);
            b1 = Fma.MultiplyAdd(Avx.LoadVector256(sb + 8), w1, b1);
            b2 = Fma.MultiplyAdd(Avx.LoadVector256(sb + 16), w2, b2);
            b3 = Fma.MultiplyAdd(Avx.LoadVector256(sb + 24), w3, b3);
        }
        DwStore8(o, ad, ch, a0, act); DwStore8(o, ad, ch + 8, a1, act);
        DwStore8(o, ad, ch + 16, a2, act); DwStore8(o, ad, ch + 24, a3, act);
        if (!second) return;
        o += c;
        if (ad != null) ad += c;
        DwStore8(o, ad, ch, b0, act); DwStore8(o, ad, ch + 8, b1, act);
        DwStore8(o, ad, ch + 16, b2, act); DwStore8(o, ad, ch + 24, b3, act);
    }

    private static void Dw4x8(float** p, int taps, int c, int ch, float* w, float* b, float* o, float* ad, int np, ActivationKind act)
    {
        F32x8 a0 = Avx.LoadVector256(b + ch), a1 = a0, a2 = a0, a3 = a0;
        float** p1 = p + taps, p2 = p + 2 * taps, p3 = p + 3 * taps;
        for (int t = 0; t < taps; t++)
        {
            F32x8 wv = Avx.LoadVector256(w + (long)t * c + ch);
            a0 = Fma.MultiplyAdd(Avx.LoadVector256(p[t] + ch), wv, a0);
            a1 = Fma.MultiplyAdd(Avx.LoadVector256(p1[t] + ch), wv, a1);
            a2 = Fma.MultiplyAdd(Avx.LoadVector256(p2[t] + ch), wv, a2);
            a3 = Fma.MultiplyAdd(Avx.LoadVector256(p3[t] + ch), wv, a3);
        }
        DwStore8(o, ad, ch, a0, act);
        if (np > 1) DwStore8(o, ad, c + ch, a1, act);
        if (np > 2) DwStore8(o, ad, 2 * c + ch, a2, act);
        if (np > 3) DwStore8(o, ad, 3 * c + ch, a3, act);
    }

    /// <summary>
    /// 4 interior pixels, all <paramref name="c"/> channels (a multiple of 8): taps addressed from
    /// <paramref name="rows"/> (kernel-row bases at pixel 0's first column), pixel stride
    /// <paramref name="ps"/> floats, tap stride <paramref name="c"/>.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void DwRunAvx(float** rows, int kh, int kw, int c, int ps, float* w, float* b, float* o, float* ad, ActivationKind act)
    {
        int ch = 0;
        for (; ch + 32 <= c; ch += 32)
        {
            Run2x32(rows, 0, kh, kw, c, ps, ch, w, b, o, ad, act);
            Run2x32(rows, 2L * ps, kh, kw, c, ps, ch, w, b, o + 2 * c, ad == null ? null : ad + 2 * c, act);
        }
        for (; ch < c; ch += 8) Run4x8(rows, kh, kw, c, ps, ch, w, b, o, ad, act);
    }

    private static void Run2x32(float** rows, long off, int kh, int kw, int c, int ps, int ch, float* w, float* b, float* o, float* ad, ActivationKind act)
    {
        F32x8 a0 = Avx.LoadVector256(b + ch), a1 = Avx.LoadVector256(b + ch + 8);
        F32x8 a2 = Avx.LoadVector256(b + ch + 16), a3 = Avx.LoadVector256(b + ch + 24);
        F32x8 b0 = a0, b1 = a1, b2 = a2, b3 = a3;
        for (int ky = 0; ky < kh; ky++)
        {
            float* sa = rows[ky] + off + ch, wt = w + (long)ky * kw * c + ch;
            for (int kx = 0; kx < kw; kx++, sa += c, wt += c)
            {
                float* sb = sa + ps;
                F32x8 w0 = Avx.LoadVector256(wt), w1 = Avx.LoadVector256(wt + 8);
                F32x8 w2 = Avx.LoadVector256(wt + 16), w3 = Avx.LoadVector256(wt + 24);
                a0 = Fma.MultiplyAdd(Avx.LoadVector256(sa), w0, a0);
                a1 = Fma.MultiplyAdd(Avx.LoadVector256(sa + 8), w1, a1);
                a2 = Fma.MultiplyAdd(Avx.LoadVector256(sa + 16), w2, a2);
                a3 = Fma.MultiplyAdd(Avx.LoadVector256(sa + 24), w3, a3);
                b0 = Fma.MultiplyAdd(Avx.LoadVector256(sb), w0, b0);
                b1 = Fma.MultiplyAdd(Avx.LoadVector256(sb + 8), w1, b1);
                b2 = Fma.MultiplyAdd(Avx.LoadVector256(sb + 16), w2, b2);
                b3 = Fma.MultiplyAdd(Avx.LoadVector256(sb + 24), w3, b3);
            }
        }
        DwStore8(o, ad, ch, a0, act); DwStore8(o, ad, ch + 8, a1, act);
        DwStore8(o, ad, ch + 16, a2, act); DwStore8(o, ad, ch + 24, a3, act);
        o += c;
        if (ad != null) ad += c;
        DwStore8(o, ad, ch, b0, act); DwStore8(o, ad, ch + 8, b1, act);
        DwStore8(o, ad, ch + 16, b2, act); DwStore8(o, ad, ch + 24, b3, act);
    }

    private static void Run4x8(float** rows, int kh, int kw, int c, int ps, int ch, float* w, float* b, float* o, float* ad, ActivationKind act)
    {
        F32x8 a0 = Avx.LoadVector256(b + ch), a1 = a0, a2 = a0, a3 = a0;
        for (int ky = 0; ky < kh; ky++)
        {
            float* s = rows[ky] + ch, wt = w + (long)ky * kw * c + ch;
            for (int kx = 0; kx < kw; kx++, s += c, wt += c)
            {
                F32x8 wv = Avx.LoadVector256(wt);
                a0 = Fma.MultiplyAdd(Avx.LoadVector256(s), wv, a0);
                a1 = Fma.MultiplyAdd(Avx.LoadVector256(s + ps), wv, a1);
                a2 = Fma.MultiplyAdd(Avx.LoadVector256(s + 2 * ps), wv, a2);
                a3 = Fma.MultiplyAdd(Avx.LoadVector256(s + 3 * ps), wv, a3);
            }
        }
        DwStore8(o, ad, ch, a0, act);
        DwStore8(o, ad, c + ch, a1, act);
        DwStore8(o, ad, 2 * c + ch, a2, act);
        DwStore8(o, ad, 3 * c + ch, a3, act);
    }

    /// <inheritdoc cref="DwRunAvx"/>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void DwRunNeon(float** rows, int kh, int kw, int c, int ps, float* w, float* b, float* o, float* ad, ActivationKind act)
    {
        int ch = 0;
        for (; ch + 16 <= c; ch += 16)
        {
            Run2x16(rows, 0, kh, kw, c, ps, ch, w, b, o, ad, act);
            Run2x16(rows, 2L * ps, kh, kw, c, ps, ch, w, b, o + 2 * c, ad == null ? null : ad + 2 * c, act);
        }
        for (; ch < c; ch += 4) Run4x4(rows, kh, kw, c, ps, ch, w, b, o, ad, act);
    }

    private static void Run2x16(float** rows, long off, int kh, int kw, int c, int ps, int ch, float* w, float* b, float* o, float* ad, ActivationKind act)
    {
        V4 a0 = AdvSimd.LoadVector128(b + ch), a1 = AdvSimd.LoadVector128(b + ch + 4);
        V4 a2 = AdvSimd.LoadVector128(b + ch + 8), a3 = AdvSimd.LoadVector128(b + ch + 12);
        V4 b0 = a0, b1 = a1, b2 = a2, b3 = a3;
        for (int ky = 0; ky < kh; ky++)
        {
            float* sa = rows[ky] + off + ch, wt = w + (long)ky * kw * c + ch;
            for (int kx = 0; kx < kw; kx++, sa += c, wt += c)
            {
                float* sb = sa + ps;
                V4 w0 = AdvSimd.LoadVector128(wt), w1 = AdvSimd.LoadVector128(wt + 4);
                V4 w2 = AdvSimd.LoadVector128(wt + 8), w3 = AdvSimd.LoadVector128(wt + 12);
                a0 = AdvSimd.FusedMultiplyAdd(a0, AdvSimd.LoadVector128(sa), w0);
                a1 = AdvSimd.FusedMultiplyAdd(a1, AdvSimd.LoadVector128(sa + 4), w1);
                a2 = AdvSimd.FusedMultiplyAdd(a2, AdvSimd.LoadVector128(sa + 8), w2);
                a3 = AdvSimd.FusedMultiplyAdd(a3, AdvSimd.LoadVector128(sa + 12), w3);
                b0 = AdvSimd.FusedMultiplyAdd(b0, AdvSimd.LoadVector128(sb), w0);
                b1 = AdvSimd.FusedMultiplyAdd(b1, AdvSimd.LoadVector128(sb + 4), w1);
                b2 = AdvSimd.FusedMultiplyAdd(b2, AdvSimd.LoadVector128(sb + 8), w2);
                b3 = AdvSimd.FusedMultiplyAdd(b3, AdvSimd.LoadVector128(sb + 12), w3);
            }
        }
        DwStore4(o, ad, ch, a0, act); DwStore4(o, ad, ch + 4, a1, act);
        DwStore4(o, ad, ch + 8, a2, act); DwStore4(o, ad, ch + 12, a3, act);
        o += c;
        if (ad != null) ad += c;
        DwStore4(o, ad, ch, b0, act); DwStore4(o, ad, ch + 4, b1, act);
        DwStore4(o, ad, ch + 8, b2, act); DwStore4(o, ad, ch + 12, b3, act);
    }

    private static void Run4x4(float** rows, int kh, int kw, int c, int ps, int ch, float* w, float* b, float* o, float* ad, ActivationKind act)
    {
        V4 a0 = AdvSimd.LoadVector128(b + ch), a1 = a0, a2 = a0, a3 = a0;
        for (int ky = 0; ky < kh; ky++)
        {
            float* s = rows[ky] + ch, wt = w + (long)ky * kw * c + ch;
            for (int kx = 0; kx < kw; kx++, s += c, wt += c)
            {
                V4 wv = AdvSimd.LoadVector128(wt);
                a0 = AdvSimd.FusedMultiplyAdd(a0, AdvSimd.LoadVector128(s), wv);
                a1 = AdvSimd.FusedMultiplyAdd(a1, AdvSimd.LoadVector128(s + ps), wv);
                a2 = AdvSimd.FusedMultiplyAdd(a2, AdvSimd.LoadVector128(s + 2 * ps), wv);
                a3 = AdvSimd.FusedMultiplyAdd(a3, AdvSimd.LoadVector128(s + 3 * ps), wv);
            }
        }
        DwStore4(o, ad, ch, a0, act);
        DwStore4(o, ad, c + ch, a1, act);
        DwStore4(o, ad, 2 * c + ch, a2, act);
        DwStore4(o, ad, 3 * c + ch, a3, act);
    }

    /// <inheritdoc cref="DwBlockAvx"/>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static int DwBlockNeon(float** ptrs, int taps, int c, float* w, float* b, float* o, float* ad, int np, ActivationKind act)
    {
        int ch = 0;
        for (; ch + 16 <= c; ch += 16)
        {
            Dw2x16(ptrs, ptrs + taps, taps, c, ch, w, b, o, ad, np >= 2, act);
            if (np > 2)
                Dw2x16(ptrs + 2 * taps, ptrs + 3 * taps, taps, c, ch, w, b, o + 2 * c, ad == null ? null : ad + 2 * c, np >= 4, act);
        }
        for (; ch + 4 <= c; ch += 4) Dw4x4(ptrs, taps, c, ch, w, b, o, ad, np, act);
        return ch;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DwStore4(float* o, float* ad, int off, V4 acc, ActivationKind act)
    {
        if (ad != null) acc = AdvSimd.Add(acc, AdvSimd.LoadVector128(ad + off));
        AdvSimd.Store(o + off, Act4(acc, act));
    }

    private static void Dw2x16(float** pa, float** pb, int taps, int c, int ch, float* w, float* b, float* o, float* ad, bool second, ActivationKind act)
    {
        V4 a0 = AdvSimd.LoadVector128(b + ch), a1 = AdvSimd.LoadVector128(b + ch + 4);
        V4 a2 = AdvSimd.LoadVector128(b + ch + 8), a3 = AdvSimd.LoadVector128(b + ch + 12);
        V4 b0 = a0, b1 = a1, b2 = a2, b3 = a3;
        for (int t = 0; t < taps; t++)
        {
            float* wt = w + (long)t * c + ch, sa = pa[t] + ch, sb = pb[t] + ch;
            V4 w0 = AdvSimd.LoadVector128(wt), w1 = AdvSimd.LoadVector128(wt + 4);
            V4 w2 = AdvSimd.LoadVector128(wt + 8), w3 = AdvSimd.LoadVector128(wt + 12);
            a0 = AdvSimd.FusedMultiplyAdd(a0, AdvSimd.LoadVector128(sa), w0);
            a1 = AdvSimd.FusedMultiplyAdd(a1, AdvSimd.LoadVector128(sa + 4), w1);
            a2 = AdvSimd.FusedMultiplyAdd(a2, AdvSimd.LoadVector128(sa + 8), w2);
            a3 = AdvSimd.FusedMultiplyAdd(a3, AdvSimd.LoadVector128(sa + 12), w3);
            b0 = AdvSimd.FusedMultiplyAdd(b0, AdvSimd.LoadVector128(sb), w0);
            b1 = AdvSimd.FusedMultiplyAdd(b1, AdvSimd.LoadVector128(sb + 4), w1);
            b2 = AdvSimd.FusedMultiplyAdd(b2, AdvSimd.LoadVector128(sb + 8), w2);
            b3 = AdvSimd.FusedMultiplyAdd(b3, AdvSimd.LoadVector128(sb + 12), w3);
        }
        DwStore4(o, ad, ch, a0, act); DwStore4(o, ad, ch + 4, a1, act);
        DwStore4(o, ad, ch + 8, a2, act); DwStore4(o, ad, ch + 12, a3, act);
        if (!second) return;
        o += c;
        if (ad != null) ad += c;
        DwStore4(o, ad, ch, b0, act); DwStore4(o, ad, ch + 4, b1, act);
        DwStore4(o, ad, ch + 8, b2, act); DwStore4(o, ad, ch + 12, b3, act);
    }

    private static void Dw4x4(float** p, int taps, int c, int ch, float* w, float* b, float* o, float* ad, int np, ActivationKind act)
    {
        V4 a0 = AdvSimd.LoadVector128(b + ch), a1 = a0, a2 = a0, a3 = a0;
        float** p1 = p + taps, p2 = p + 2 * taps, p3 = p + 3 * taps;
        for (int t = 0; t < taps; t++)
        {
            V4 wv = AdvSimd.LoadVector128(w + (long)t * c + ch);
            a0 = AdvSimd.FusedMultiplyAdd(a0, AdvSimd.LoadVector128(p[t] + ch), wv);
            a1 = AdvSimd.FusedMultiplyAdd(a1, AdvSimd.LoadVector128(p1[t] + ch), wv);
            a2 = AdvSimd.FusedMultiplyAdd(a2, AdvSimd.LoadVector128(p2[t] + ch), wv);
            a3 = AdvSimd.FusedMultiplyAdd(a3, AdvSimd.LoadVector128(p3[t] + ch), wv);
        }
        DwStore4(o, ad, ch, a0, act);
        if (np > 1) DwStore4(o, ad, c + ch, a1, act);
        if (np > 2) DwStore4(o, ad, 2 * c + ch, a2, act);
        if (np > 3) DwStore4(o, ad, 3 * c + ch, a3, act);
    }
}
