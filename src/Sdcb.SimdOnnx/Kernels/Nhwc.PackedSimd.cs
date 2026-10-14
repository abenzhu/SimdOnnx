using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Wasm;
using Sdcb.SimdOnnx.Engine;

using V4 = System.Runtime.Intrinsics.Vector128<float>;

namespace Sdcb.SimdOnnx.Kernels;

/// <summary>
/// WebAssembly SIMD128 (PackedSimd) tier, after SimdPaddleOCR's Nhwc.Conv.PackedSimd. WASM SIMD has
/// no FMA (relaxed SIMD is off) and V8 maps v128 values onto at most 15 allocatable XMM registers
/// on x64, so the conv / Gemm micro-kernel is <see cref="PkTile"/> pixels × 8 oc (8 accumulators +
/// 2 weights + 1 splat), run twice per 16-oc packed block; each input float is broadcast straight
/// from memory (v128.load32_splat). Every reduction keeps the Vector&lt;float&gt; tier's per-lane
/// order (bias-init, taps then channels ascending, separate multiply then add), so conv sums are
/// bit-identical to it; direct convs read NHWC kernel-row runs like the NEON tier (same k order).
/// </summary>
internal static unsafe partial class Nhwc
{
    /// <summary>PackedSimd tier switch: <c>PackedSimd.IsSupported</c> (browser-wasm with SIMD).
    /// Settable only for in-process tier comparisons (WasmBench <c>--tiers</c> / <c>--vector</c>).</summary>
    internal static bool UsePacked = PackedSimd.IsSupported;

    internal const int PkTile = 4;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static V4 LdP(float* p) => PackedSimd.LoadVector128(p);

    /// <summary>acc + a·b, rounded twice (no FMA): the Vector&lt;float&gt; tier's <c>acc += a * b</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static V4 MadP(V4 acc, V4 a, V4 b) => PackedSimd.Add(acc, PackedSimd.Multiply(a, b));

    // pmax(0, v): one maxps in V8 (f32x4.max needs a NaN / -0 fix-up); equals Max(v, 0) for finite v
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static V4 ActP(V4 v, ActivationKind act) => act switch
    {
        ActivationKind.Relu => PackedSimd.PseudoMax(V4.Zero, v),
        ActivationKind.Sigmoid => SimdMath.SigmoidPk(v),
        ActivationKind.Gelu => SimdMath.GeluPk(v),
        ActivationKind.HardSwish => PackedSimd.Multiply(v, PackedSimd.PseudoMin(Vector128.Create(1f),
            PackedSimd.PseudoMax(V4.Zero, MadP(Vector128.Create(0.5f), v, Vector128.Create(1f / 6f))))),
        ActivationKind.Swish => PackedSimd.Multiply(v, SimdMath.SigmoidPk(v)),
        _ => v,
    };

    /// <summary>
    /// <see cref="PkTile"/> rows × 16 oc: dst[r*ds + 0..15] = bias + Σ x·w, where ptrs[tap*ps + r] are
    /// row r's ic inputs of tap and wb + tap·wTap its [ic][16] weights. Two 8-oc passes; the k loop
    /// never calls out, so the accumulators stay in registers.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void TilePkK(float** ptrs, int ps, int taps, int ic, long wTap, float* wb, float* bb, float* dst, int ds)
    {
        for (int h = 0; h < Ob; h += 8)
        {
            V4 a0 = LdP(bb + h), b0 = LdP(bb + h + 4);
            V4 a1 = a0, a2 = a0, a3 = a0, b1 = b0, b2 = b0, b3 = b0;
            for (int tap = 0; tap < taps; tap++)
            {
                float** pp = ptrs + tap * ps;
                float* p0 = pp[0], p1 = pp[1], p2 = pp[2], p3 = pp[3];
                float* w = wb + tap * wTap + h;
                for (int c = 0; c < ic; c++, w += Ob)
                {
                    V4 w0 = LdP(w), w1 = LdP(w + 4);
                    V4 v = PackedSimd.LoadScalarAndSplatVector128(p0 + c);
                    a0 = MadP(a0, v, w0); b0 = MadP(b0, v, w1);
                    v = PackedSimd.LoadScalarAndSplatVector128(p1 + c);
                    a1 = MadP(a1, v, w0); b1 = MadP(b1, v, w1);
                    v = PackedSimd.LoadScalarAndSplatVector128(p2 + c);
                    a2 = MadP(a2, v, w0); b2 = MadP(b2, v, w1);
                    v = PackedSimd.LoadScalarAndSplatVector128(p3 + c);
                    a3 = MadP(a3, v, w0); b3 = MadP(b3, v, w1);
                }
            }
            float* d = dst + h;
            PackedSimd.Store(d, a0); PackedSimd.Store(d + 4, b0); d += ds;
            PackedSimd.Store(d, a1); PackedSimd.Store(d + 4, b1); d += ds;
            PackedSimd.Store(d, a2); PackedSimd.Store(d + 4, b2); d += ds;
            PackedSimd.Store(d, a3); PackedSimd.Store(d + 4, b3);
        }
    }

    /// <summary>Residual / activation epilogue of a finished [rows][16] scratch tile.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void StorePk(float* part, int rows, float* outp, float* addp, int oc, int lanes, ActivationKind act)
    {
        for (int r = 0; r < rows; r++, part += Ob)
        {
            float* o = outp + (long)r * oc;
            float* ad = addp == null ? null : addp + (long)r * oc;
            if (lanes == Ob)
            {
                for (int l = 0; l < Ob; l += 4)
                {
                    V4 v = LdP(part + l);
                    if (ad != null) v = PackedSimd.Add(v, LdP(ad + l));
                    PackedSimd.Store(o + l, ActP(v, act));
                }
                continue;
            }
            int l4 = 0;
            for (; l4 + 4 <= lanes; l4 += 4)
            {
                V4 v = LdP(part + l4);
                if (ad != null) v = PackedSimd.Add(v, LdP(ad + l4));
                PackedSimd.Store(o + l4, ActP(v, act));
            }
            for (int l = l4; l < lanes; l++)
                o[l] = ActS(ad != null ? part[l] + ad[l] : part[l], act);
        }
    }

    /// <summary><see cref="DenseRow"/> on PackedSimd: kh taps over NHWC kernel-row runs (kw·ic contiguous
    /// floats, see <see cref="DenseRowNeon"/>); border-crossing runs are gathered, constant-pad rows read PadVec.</summary>
    internal static void DenseRowPk(in DenseArgs a, int y, int blk, float* orow, float* arow)
    {
        int kh = a.Kh, kw = a.Kw, ic = a.Ic, ow = a.Ow, oc = a.Oc, run = kw * ic;
        float** ptrs = stackalloc float*[PkTile * kh];
        float* part = stackalloc float[PkTile * Ob];
        float* wb = a.W + (long)blk * kh * run * Ob;
        float* bb = a.Bias + blk * Ob;
        int lanes = Math.Min(Ob, oc - blk * Ob);
        float* gather = null;
        for (int x0 = 0; x0 < ow; x0 += PkTile)
        {
            int rows = Math.Min(PkTile, ow - x0);
            for (int ky = 0; ky < kh; ky++)
            {
                int r = a.Rm[y * kh + ky];
                for (int i = 0; i < PkTile; i++)
                {
                    int* cmx = a.Cm + (x0 + Math.Min(i, rows - 1)) * kw;
                    int cx = cmx[0];
                    float* p;
                    if (r < 0) p = a.PadVec;
                    else if (cx >= 0 && cmx[kw - 1] == cx + kw - 1) p = a.Src + ((long)r * a.Iw + cx) * ic;
                    else
                    {
                        if (gather == null) gather = Pinned(ref t_runs, PkTile * kh * run);
                        p = gather + (ky * PkTile + i) * run;
                        for (int kx = 0; kx < kw; kx++)
                        {
                            int c = cmx[kx];
                            Buffer.MemoryCopy(c < 0 ? a.PadVec : a.Src + ((long)r * a.Iw + c) * ic, p + kx * ic, ic * 4L, ic * 4L);
                        }
                    }
                    ptrs[ky * PkTile + i] = p;
                }
            }
            long o = (long)x0 * oc + blk * Ob;
            TilePkK(ptrs, PkTile, kh, run, (long)run * Ob, wb, bb, part, Ob);
            StorePk(part, rows, orow + o, arow != null ? arow + o : null, oc, lanes, a.Act);
        }
    }

    /// <summary><see cref="Gemm"/> on PackedSimd: <see cref="PkTile"/>-row tiles, 12 per task sharing one weight panel walk.</summary>
    internal static void GemmPk(ReadOnlySpan<float> a, int m, int k, Span<float> dst, int n, NhwcConvPlan kp, ActivationKind act)
    {
        const int G = 12;
        int nb = kp.Blocks, tiles = (m + PkTile - 1) / PkTile, groups = (tiles + G - 1) / G;
        fixed (float* ap0 = a, dp0 = dst, wp0 = kp.W, bp0 = kp.Bias)
        {
            float* ap = ap0, dp = dp0, wp = wp0, bp = bp0;
            For(groups * nb, (long)m * n * k / 64, t =>
            {
                int g = t / nb, blk = t % nb;
                float** ptrs = stackalloc float*[PkTile];
                float* part = stackalloc float[PkTile * Ob];
                float* wb = wp + (long)blk * k * Ob, bb = bp + blk * Ob;
                int lanes = Math.Min(Ob, n - blk * Ob);
                for (int tile = g * G, te = Math.Min(tiles, tile + G); tile < te; tile++)
                {
                    int m0 = tile * PkTile, rows = Math.Min(PkTile, m - m0);
                    for (int i = 0; i < PkTile; i++) ptrs[i] = ap + (long)(m0 + Math.Min(i, rows - 1)) * k;
                    TilePkK(ptrs, PkTile, 1, k, (long)k * Ob, wb, bb, part, Ob);
                    StorePk(part, rows, dp + (long)m0 * n + blk * Ob, null, n, lanes, act);
                }
            });
        }
    }

    /// <summary>
    /// One depthwise output pixel on PackedSimd: 16 then 4 channels per step, bias-initialized, taps
    /// ascending, + residual, activation. Returns the first channel left for the scalar tail.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static int DepthwisePixelPk(float** ptrs, int taps, int c, float* wp, float* bp, float* ad, float* o, ActivationKind act)
    {
        int ch = 0;
        for (; ch + 16 <= c; ch += 16)
        {
            V4 a0 = LdP(bp + ch), a1 = LdP(bp + ch + 4), a2 = LdP(bp + ch + 8), a3 = LdP(bp + ch + 12);
            float* w = wp + ch;
            for (int tap = 0; tap < taps; tap++, w += c)
            {
                float* s = ptrs[tap] + ch;
                a0 = MadP(a0, LdP(s), LdP(w));
                a1 = MadP(a1, LdP(s + 4), LdP(w + 4));
                a2 = MadP(a2, LdP(s + 8), LdP(w + 8));
                a3 = MadP(a3, LdP(s + 12), LdP(w + 12));
            }
            if (ad != null)
            {
                a0 = PackedSimd.Add(a0, LdP(ad + ch)); a1 = PackedSimd.Add(a1, LdP(ad + ch + 4));
                a2 = PackedSimd.Add(a2, LdP(ad + ch + 8)); a3 = PackedSimd.Add(a3, LdP(ad + ch + 12));
            }
            PackedSimd.Store(o + ch, ActP(a0, act)); PackedSimd.Store(o + ch + 4, ActP(a1, act));
            PackedSimd.Store(o + ch + 8, ActP(a2, act)); PackedSimd.Store(o + ch + 12, ActP(a3, act));
        }
        for (; ch + 4 <= c; ch += 4)
        {
            V4 acc = LdP(bp + ch);
            for (int tap = 0; tap < taps; tap++)
                acc = MadP(acc, LdP(ptrs[tap] + ch), LdP(wp + tap * c + ch));
            if (ad != null) acc = PackedSimd.Add(acc, LdP(ad + ch));
            PackedSimd.Store(o + ch, ActP(acc, act));
        }
        return ch;
    }


}
