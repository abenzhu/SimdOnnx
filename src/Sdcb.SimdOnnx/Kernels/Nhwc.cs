using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.Wasm;
using System.Runtime.Intrinsics.X86;
using Sdcb.SimdOnnx.Engine;

using F32x8 = System.Runtime.Intrinsics.Vector256<float>;
using V128 = System.Runtime.Intrinsics.Vector128<float>;
using V256 = System.Runtime.Intrinsics.Vector256;

namespace Sdcb.SimdOnnx.Kernels;

/// <summary>Backend-private packed conv weights for the NHWC CPU kernels (lives in <see cref="ConvP.KernelPlan"/>).</summary>
internal sealed class NhwcConvPlan
{
    public required float[] W;      // dense: [oc/16][tap][ic][16] (oc zero-padded); depthwise: [tap][c]
    public required float[] Bias;   // dense: padded to blocks*16; depthwise: [c]
    public bool Depthwise;
    public int Blocks;
    public float[]? WinoU;  // Winograd F(4x4,3x3) weights [36][oc/16][ic][16], null = direct conv
}

/// <summary>
/// Channels-last (NHWC, N==1) kernels. Shapes are passed in storage order
/// ([h,w,c]). Every pad mode goes through precomputed source row/col maps
/// (-1 = constant pad → a pad-value pixel vector), so the inner loops never
/// branch on borders. Dense/pointwise conv: register-tiled 6 pixels × 16 oc
/// AVX2 micro-kernel (after SimdPaddleOCR's Nhwc.Conv) with bias-init and a
/// fused residual + activation epilogue; AVX-512 widens it to 10 × 32 (Nhwc.Avx512.cs),
/// NEON keeps 6 × 16 (Nhwc.Neon.cs), WASM SIMD128 narrows it to 4 × 8 (Nhwc.PackedSimd.cs);
/// other tiers use Vector&lt;float&gt;.
/// </summary>
internal static unsafe partial class Nhwc
{
    internal const int Ob = 16, Tile = 6;
    internal static ParallelOptions PO => Par.Options;
    internal static bool UseAvx => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>
    /// body(0..n-1), split into min(threads, n) contiguous static shards when <paramref name="work"/>
    /// (≈ MAC/64 for dense convs) is worth a fork; rows of one kernel cost about the same.
    /// </summary>
    internal static void For(int n, long work, Action<int> body)
    {
        int threads = Par.Threads;
        if (work < 32768 || n == 1 || threads == 1) { for (int i = 0; i < n; i++) body(i); return; }
        int shards = Math.Min(threads, n);
        Par.For(0, shards, w =>
        {
            for (int i = (int)((long)n * w / shards), e = (int)((long)n * (w + 1) / shards); i < e; i++) body(i);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int Reflect(int i, int dim)
    {
        if (dim <= 1) return 0;
        while (i < 0 || i >= dim) i = i < 0 ? -i : 2 * dim - 2 - i;
        return i;
    }

    /// <summary>map[o*k+kk] = source index of output o, tap kk, or -1 for constant pad.</summary>
    internal static int[] Map(int outLen, int k, int stride, int pad, int inLen, PadMode mode)
    {
        int[] m = new int[outLen * k];
        FillMap(m, outLen, k, stride, pad, inLen, mode);
        return m;
    }

    private static void FillMap(int[] m, int outLen, int k, int stride, int pad, int inLen, PadMode mode)
    {
        for (int o = 0; o < outLen; o++)
            for (int kk = 0; kk < k; kk++)
            {
                int s = o * stride + kk - pad;
                m[o * k + kk] = (uint)s < (uint)inLen ? s : mode == PadMode.Reflect ? Reflect(s, inLen) : -1;
            }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float ActS(float v, ActivationKind act) =>
        act == ActivationKind.Relu ? MathF.Max(v, 0f) :
        act == ActivationKind.Sigmoid ? 1f / (1f + MathF.Exp(-v)) :
        act == ActivationKind.Gelu ? 0.5f * v * (1f + SimdMath.ErfScalar(v * 0.70710678f)) :
        act == ActivationKind.HardSwish ? v * MathF.Min(MathF.Max(v * (1f / 6f) + 0.5f, 0f), 1f) :
        act == ActivationKind.Swish ? v / (1f + MathF.Exp(-v)) : v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static F32x8 Act8(F32x8 v, ActivationKind act) => act switch
    {
        ActivationKind.Relu => V256.Max(v, F32x8.Zero),
        ActivationKind.Sigmoid => SimdMath.SigmoidCore(v),
        ActivationKind.Gelu => SimdMath.GeluCore(v),
        ActivationKind.HardSwish => v * V256.Min(V256.Max(Fma.MultiplyAdd(v, V256.Create(1f / 6f), V256.Create(0.5f)), F32x8.Zero), V256.Create(1f)),
        ActivationKind.Swish => v * SimdMath.SigmoidCore(v),
        _ => v,
    };

    // ======================== weight packing ========================
    public static NhwcConvPlan Pack(ReadOnlySpan<float> w, int[] wShape, ReadOnlySpan<float> bias, int group, int ic)
    {
        int oc = wShape[0], icg = wShape[1], taps = wShape[2] * wShape[3];
        if (group > 1)
        {
            if (group != ic || icg != 1 || oc != ic)
                throw new NotSupportedException($"NHWC grouped conv supports depthwise only (g={group}, ic={ic}, oc={oc}).");
            float[] pw = new float[taps * oc];
            for (int c = 0; c < oc; c++)
                for (int t = 0; t < taps; t++) pw[t * oc + c] = w[c * taps + t];
            float[] pb = new float[oc];
            if (!bias.IsEmpty) bias.CopyTo(pb);
            return new NhwcConvPlan { W = pw, Bias = pb, Depthwise = true };
        }
        int nb = (oc + Ob - 1) / Ob;
        float[] p = new float[checked(nb * taps * ic * Ob)];
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
                for (int t = 0; t < taps; t++)
                    p[(((o / Ob) * taps + t) * ic + c) * Ob + o % Ob] = w[(o * ic + c) * taps + t];
        float[] b = new float[nb * Ob];
        if (!bias.IsEmpty) bias.CopyTo(b);
        return new NhwcConvPlan { W = p, Bias = b, Blocks = nb };
    }

    /// <summary>Packs Gemm B ([K,N], or [N,K] when transB) as [N/16][K][16] — a 1x1 conv panel.</summary>
    public static NhwcConvPlan PackGemm(ReadOnlySpan<float> b, int k, int n, bool transB, ReadOnlySpan<float> bias)
    {
        int nb = (n + Ob - 1) / Ob;
        float[] p = new float[checked(nb * k * Ob)];
        for (int o = 0; o < n; o++)
            for (int c = 0; c < k; c++)
                p[((o / Ob) * k + c) * Ob + o % Ob] = transB ? b[o * k + c] : b[c * n + o];
        float[] pb = new float[nb * Ob];
        if (!bias.IsEmpty) bias.CopyTo(pb);
        return new NhwcConvPlan { W = p, Bias = pb, Blocks = nb };
    }

    /// <summary>dst[M,N] = act(a[M,K]·B + bias) through the 6x16 conv micro-kernel.</summary>
    public static void Gemm(ReadOnlySpan<float> a, int m, int k, Span<float> dst, int n,
        NhwcConvPlan kp, ActivationKind act)
    {
        if (UseAvx512) { Gemm512(a, m, k, dst, n, kp, act); return; }
        if (UsePacked) { GemmPk(a, m, k, dst, n, kp, act); return; }
        const int G = 8;   // row tiles per task sharing one weight panel walk
        int nb = kp.Blocks, tiles = (m + Tile - 1) / Tile, groups = (tiles + G - 1) / G;
        bool avx = UseAvx;
        fixed (float* ap0 = a, dp0 = dst, wp0 = kp.W, bp0 = kp.Bias)
        {
            float* ap = ap0, dp = dp0, wp = wp0, bp = bp0;
            For(groups * nb, (long)m * n * k / 64, t =>
            {
                int g = t / nb, blk = t % nb;
                float** ptrs = stackalloc float*[Tile];
                float* wb = wp + (long)blk * k * Ob, bb = bp + blk * Ob;
                int lanes = Math.Min(Ob, n - blk * Ob);
                for (int tile = g * G, te = Math.Min(tiles, tile + G); tile < te; tile++)
                {
                    int m0 = tile * Tile, rows = Math.Min(Tile, m - m0);
                    for (int i = 0; i < Tile; i++) ptrs[i] = ap + (long)(m0 + Math.Min(i, rows - 1)) * k;
                    float* outp = dp + (long)m0 * n + blk * Ob;
                    if (avx) TileAvx(ptrs, rows, 1, k, wb, bb, outp, null, n, lanes, act);
                    else TileVec(ptrs, rows, 1, k, wb, bb, outp, null, n, lanes, act);
                }
            });
        }
    }

    // ======================== conv ========================
    /// <summary>NHWC conv over <paramref name="batch"/> contiguous images ([batch, ih, iw, ic] → [batch, oh, ow, oc]).</summary>
    public static void Conv(ReadOnlySpan<float> src, int ih, int iw, int ic,
        Span<float> dst, int oh, int ow, int oc, ConvP p, NhwcConvPlan kp, ReadOnlySpan<float> add, int batch = 1)
    {
        if (!add.IsEmpty && add.Length != dst.Length)
            throw new NotSupportedException($"conv residual addend has {add.Length} elems, expected {dst.Length}.");
        if (batch > 1 && p.Kernel is [1, 1] && p.Strides is [1, 1] && p.Pads.All(x => x == 0))
        {
            // pointwise: the batch folds into the row dimension
            ih *= batch;
            oh *= batch;
            batch = 1;
        }
        if (batch > 1 && (kp.WinoU is not null || (UseAvx512 && !kp.Depthwise)))
        {
            int inImg = ih * iw * ic, outImg = oh * ow * oc;
            for (int b = 0; b < batch; b++)
                Conv(src.Slice(b * inImg, inImg), ih, iw, ic, dst.Slice(b * outImg, outImg), oh, ow, oc, p, kp,
                    add.IsEmpty ? add : add.Slice(b * outImg, outImg));
            return;
        }
        if (kp.WinoU is not null)
        {
            WinogradConv(src, ih, iw, ic, dst, oh, ow, oc, p, kp, add);
            return;
        }
        int kh = p.Kernel[0], kw = p.Kernel[1];
        // per-run tables from the shared pools: allocating them was most of a CPU run's garbage
        int[] rm = MapRented(oh, kh, p.Strides[0], p.Pads[0], ih, p.PadMode);
        int[] cm = MapRented(ow, kw, p.Strides[1], p.Pads[1], iw, p.PadMode);
        float[] padVec = PadRented(ic * kw + Ob, p.PadValue);   // one constant-pad kernel row (Dense512 row runs)
        try
        {
            if (kp.Depthwise)
                Depthwise(src, ih, iw, ic, dst, oh, ow, kh, kw, p.Strides[1], rm, cm, padVec, p.PadValue, kp, add, p.Act, batch);
            else if (UseAvx512)
                Dense512(src, iw, ic, dst, oh, ow, oc, kh, kw, rm, cm, padVec, kp, add, p.Act);
            else
                Dense(src, ih, iw, ic, dst, oh, ow, oc, kh, kw, rm, cm, padVec, kp, add, p.Act, batch);
        }
        finally
        {
            System.Buffers.ArrayPool<int>.Shared.Return(rm);
            System.Buffers.ArrayPool<int>.Shared.Return(cm);
            System.Buffers.ArrayPool<float>.Shared.Return(padVec);
        }
    }

    /// <summary><see cref="Map"/> into a pooled array (return it to ArrayPool&lt;int&gt;.Shared).</summary>
    internal static int[] MapRented(int outLen, int k, int stride, int pad, int inLen, PadMode mode)
    {
        int[] m = System.Buffers.ArrayPool<int>.Shared.Rent(outLen * k);
        FillMap(m, outLen, k, stride, pad, inLen, mode);
        return m;
    }

    /// <summary>A pooled array whose first <paramref name="n"/> entries are <paramref name="value"/>.</summary>
    internal static float[] PadRented(int n, float value)
    {
        float[] v = System.Buffers.ArrayPool<float>.Shared.Rent(n);
        v.AsSpan(0, n).Fill(value);
        return v;
    }

    internal static void Dense(ReadOnlySpan<float> src, int ih, int iw, int ic, Span<float> dst,
        int oh, int ow, int oc, int kh, int kw, int[] rm, int[] cm, float[] padVec,
        NhwcConvPlan kp, ReadOnlySpan<float> add, ActivationKind act, int batch = 1)
    {
        int taps = kh * kw, nb = kp.Blocks, perImg = oh * nb;
        long inImg = (long)ih * iw * ic, outImg = (long)oh * ow * oc;
        bool hasAdd = !add.IsEmpty, avx = UseAvx;
        fixed (float* sp0 = src, dp0 = dst, ap0 = add, wp0 = kp.W, bp0 = kp.Bias, pv0 = padVec)
        fixed (int* rm0 = rm, cm0 = cm)
        {
            var a = new DenseArgs(sp0, wp0, bp0, pv0, rm0, cm0, iw, ic, ow, oc, kh, kw, nb, act, avx);
            float* dp = dp0, ap = ap0;
            For(batch * perImg, batch * outImg * taps * ic / 64, t =>
            {
                int b = t / perImg, r = t - b * perImg, y = r / nb, blk = r % nb;
                long o = b * outImg + (long)y * ow * oc;
                DenseRow(b == 0 ? a : a.WithSrc(a.Src + b * inImg), y, blk, dp + o, hasAdd ? ap + o : null);
            });
        }
    }

    /// <summary>Per-call direct-conv constants (pointers into pinned/fixed storage).</summary>
    internal readonly unsafe struct DenseArgs(float* src, float* w, float* bias, float* padVec, int* rm, int* cm,
        int iw, int ic, int ow, int oc, int kh, int kw, int nb, ActivationKind act, bool avx)
    {
        public readonly float* Src = src, W = w, Bias = bias, PadVec = padVec;
        public readonly int* Rm = rm, Cm = cm;
        public readonly int Iw = iw, Ic = ic, Ow = ow, Oc = oc, Kh = kh, Kw = kw, Nb = nb;
        public readonly ActivationKind Act = act;
        public readonly bool Avx = avx;

        public DenseArgs WithSrc(float* s) => new(s, W, Bias, PadVec, Rm, Cm, Iw, Ic, Ow, Oc, Kh, Kw, Nb, Act, Avx);
    }

    /// <summary>Output row <paramref name="y"/>, oc block <paramref name="blk"/> of a direct conv
    /// into <paramref name="orow"/> (the row's first pixel; addend row likewise).</summary>
    internal static void DenseRow(in DenseArgs a, int y, int blk, float* orow, float* arow)
    {
        if (UseNeon)
        {
            DenseRowNeon(a, y, blk, orow, arow);
            return;
        }
        if (UsePacked)
        {
            DenseRowPk(a, y, blk, orow, arow);
            return;
        }
        // kh taps over kernel-row runs (kw·ic contiguous NHWC floats, weights [ky][kx·ic][16]) —
        // see DenseRowNeon; border-crossing runs are gathered, constant-pad rows read PadVec
        int kh = a.Kh, ic = a.Ic, ow = a.Ow, oc = a.Oc, kw = a.Kw, run = kw * ic;
        float** ptrs = stackalloc float*[Tile * kh];
        float* wb = a.W + (long)blk * kh * run * Ob;
        float* bb = a.Bias + blk * Ob;
        int lanes = Math.Min(Ob, oc - blk * Ob);
        float* gather = null;
        for (int x0 = 0; x0 < ow; x0 += Tile)
        {
            int rows = Math.Min(Tile, ow - x0);
            for (int ky = 0; ky < kh; ky++)
            {
                int r = a.Rm[y * kh + ky];
                for (int i = 0; i < Tile; i++)
                {
                    int* cmx = a.Cm + (x0 + Math.Min(i, rows - 1)) * kw;
                    int cx = cmx[0];
                    float* p;
                    if (r < 0) p = a.PadVec;
                    else if (cx >= 0 && cmx[kw - 1] == cx + kw - 1) p = a.Src + ((long)r * a.Iw + cx) * ic;
                    else
                    {
                        if (gather == null) gather = Pinned(ref t_runs, Tile * kh * run);
                        p = gather + (ky * Tile + i) * run;
                        for (int kx = 0; kx < kw; kx++)
                        {
                            int c = cmx[kx];
                            Buffer.MemoryCopy(c < 0 ? a.PadVec : a.Src + ((long)r * a.Iw + c) * ic, p + kx * ic, ic * 4L, ic * 4L);
                        }
                    }
                    ptrs[ky * Tile + i] = p;
                }
            }
            long o = (long)x0 * oc + blk * Ob;
            float* outp = orow + o;
            float* addp = arow != null ? arow + o : null;
            if (a.Avx) TileAvx(ptrs, rows, kh, run, wb, bb, outp, addp, oc, lanes, a.Act);
            else TileVec(ptrs, rows, kh, run, wb, bb, outp, addp, oc, lanes, a.Act);
        }
    }

    /// <summary>6 × 16 tile: <see cref="TileAvxK"/> into a scratch tile, then the
    /// residual / activation epilogue (<see cref="Store16"/>) per row.</summary>
    internal static void TileAvx(float** ptrs, int rows, int taps, int ic, float* wb, float* bb,
        float* outp, float* addp, int oc, int lanes, ActivationKind act)
    {
        float* part = stackalloc float[Tile * Ob];
        TileAvxK(ptrs, taps, ic, wb, bb, part, Ob);
        for (int r = 0; r < rows; r++, part += Ob)
            Store16(outp + (long)r * oc, addp == null ? null : addp + (long)r * oc,
                Avx.LoadVector256(part), Avx.LoadVector256(part + 8), lanes, act);
    }

    /// <summary>
    /// The 6 × 16 k loop: dst[r*ds + 0..15] = bias + Σ x·w for all <see cref="Tile"/> rows. It never
    /// calls out — passing the accumulators to a helper (by reference under the x64 ABIs) makes
    /// RyuJIT keep all twelve on the stack, a load + store around every FMA.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void TileAvxK(float** ptrs, int taps, int ic, float* wb, float* bb, float* dst, int ds)
    {
        F32x8 a0 = Avx.LoadVector256(bb), b0 = Avx.LoadVector256(bb + 8);
        F32x8 a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0;
        F32x8 b1 = b0, b2 = b0, b3 = b0, b4 = b0, b5 = b0;
        for (int tap = 0; tap < taps; tap++)
        {
            float** pp = ptrs + tap * Tile;
            float* p0 = pp[0], p1 = pp[1], p2 = pp[2], p3 = pp[3], p4 = pp[4], p5 = pp[5];
            float* w = wb + (long)tap * ic * Ob;
            for (int c = 0; c < ic; c++, w += Ob)
            {
                F32x8 w0 = Avx.LoadVector256(w), w1 = Avx.LoadVector256(w + 8);
                F32x8 v0 = Avx.BroadcastScalarToVector256(p0 + c), v1 = Avx.BroadcastScalarToVector256(p1 + c);
                a0 = Fma.MultiplyAdd(v0, w0, a0); b0 = Fma.MultiplyAdd(v0, w1, b0);
                a1 = Fma.MultiplyAdd(v1, w0, a1); b1 = Fma.MultiplyAdd(v1, w1, b1);
                F32x8 v2 = Avx.BroadcastScalarToVector256(p2 + c), v3 = Avx.BroadcastScalarToVector256(p3 + c);
                a2 = Fma.MultiplyAdd(v2, w0, a2); b2 = Fma.MultiplyAdd(v2, w1, b2);
                a3 = Fma.MultiplyAdd(v3, w0, a3); b3 = Fma.MultiplyAdd(v3, w1, b3);
                F32x8 v4 = Avx.BroadcastScalarToVector256(p4 + c), v5 = Avx.BroadcastScalarToVector256(p5 + c);
                a4 = Fma.MultiplyAdd(v4, w0, a4); b4 = Fma.MultiplyAdd(v4, w1, b4);
                a5 = Fma.MultiplyAdd(v5, w0, a5); b5 = Fma.MultiplyAdd(v5, w1, b5);
            }
        }
        Avx.Store(dst, a0); Avx.Store(dst + 8, b0); dst += ds;
        Avx.Store(dst, a1); Avx.Store(dst + 8, b1); dst += ds;
        Avx.Store(dst, a2); Avx.Store(dst + 8, b2); dst += ds;
        Avx.Store(dst, a3); Avx.Store(dst + 8, b3); dst += ds;
        Avx.Store(dst, a4); Avx.Store(dst + 8, b4); dst += ds;
        Avx.Store(dst, a5); Avx.Store(dst + 8, b5);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Store16(float* o, float* add, F32x8 a, F32x8 b, int lanes, ActivationKind act)
    {
        if (lanes == Ob)
        {
            if (add != null) { a = Avx.Add(a, Avx.LoadVector256(add)); b = Avx.Add(b, Avx.LoadVector256(add + 8)); }
            Avx.Store(o, Act8(a, act)); Avx.Store(o + 8, Act8(b, act));
            return;
        }
        // channel tail (e.g. oc=8 blocks, the final oc=3 conv): only the valid lanes touch memory
        if (lanes >= 8)
        {
            if (add != null) a = Avx.Add(a, Avx.LoadVector256(add));
            Avx.Store(o, Act8(a, act));
            if (lanes == 8) return;
            o += 8;
            if (add != null) add += 8;
            lanes -= 8;
            a = b;
        }
        Vector256<int> mask = Vector256.LessThan(Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7), Vector256.Create(lanes));
        if (add != null) a = Avx.Add(a, Avx.MaskLoad(add, mask.AsSingle()));
        Avx.MaskStore(o, mask.AsSingle(), Act8(a, act));
    }

    internal static void TileVec(float** ptrs, int rows, int taps, int ic, float* wb, float* bb,
        float* outp, float* addp, int oc, int lanes, ActivationKind act)
    {
        if (UseNeon)
        {
            TileNeon(ptrs, rows, taps, ic, wb, bb, outp, addp, oc, lanes, act);
            return;
        }
        int VL = Vector<float>.Count;
        // software Vector<T> (DOTNET_EnableHWIntrinsic=0) is far slower than plain scalar
        bool vec = Vector.IsHardwareAccelerated && Ob % VL == 0;
        if (vec && VL == 4)
        {
            TileVec4(ptrs, rows, taps, ic, wb, bb, outp, addp, oc, lanes, act);
            return;
        }
        if (vec && VL == 8)
        {
            TileVec8(ptrs, rows, taps, ic, wb, bb, outp, addp, oc, lanes, act);
            return;
        }
        float* acc = stackalloc float[Ob];
        for (int i = 0; i < rows; i++)
        {
            for (int l = 0; l < Ob; l++) acc[l] = bb[l];
            for (int tap = 0; tap < taps; tap++)
            {
                float* p = ptrs[tap * Tile + i];
                float* w = wb + (long)tap * ic * Ob;
                for (int c = 0; c < ic; c++, w += Ob)
                {
                    float v = p[c];
                    if (vec)
                    {
                        var vv = new Vector<float>(v);
                        for (int l = 0; l < Ob; l += VL)
                            Simd.VectorStore(acc + l, Simd.VectorLoad(acc + l) + Simd.VectorLoad(w + l) * vv);
                    }
                    else
                        for (int l = 0; l < Ob; l++) acc[l] += v * w[l];
                }
            }
            float* o = outp + (long)i * oc;
            float* ad = addp == null ? null : addp + (long)i * oc;
            for (int l = 0; l < lanes; l++)
                o[l] = ActS(ad != null ? acc[l] + ad[l] : acc[l], act);
        }
    }

    /// <summary>
    /// 128-bit Vector&lt;float&gt; tile (AVX2 disabled): two pixel rows × 16 oc in eight register
    /// accumulators (+4 weight vectors), instead of a stack accumulator round-tripped through
    /// memory on every multiply-add. Same per-lane op order (acc + w·x) as the generic loop.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void TileVec4(float** ptrs, int rows, int taps, int ic, float* wb, float* bb,
        float* outp, float* addp, int oc, int lanes, ActivationKind act)
    {
        float* tmp = stackalloc float[2 * Ob];
        for (int i = 0; i < rows; i += 2)
        {
            Vector<float> a0 = Simd.VectorLoad(bb), a1 = Simd.VectorLoad(bb + 4),
                a2 = Simd.VectorLoad(bb + 8), a3 = Simd.VectorLoad(bb + 12);
            Vector<float> c0 = a0, c1 = a1, c2 = a2, c3 = a3;
            for (int tap = 0; tap < taps; tap++)
            {
                // ptrs holds Tile entries per tap (short tiles repeat their last pixel)
                float* p = ptrs[tap * Tile + i], q = ptrs[tap * Tile + i + 1];
                float* w = wb + (long)tap * ic * Ob;
                for (int c = 0; c < ic; c++, w += Ob)
                {
                    Vector<float> w0 = Simd.VectorLoad(w), w1 = Simd.VectorLoad(w + 4),
                        w2 = Simd.VectorLoad(w + 8), w3 = Simd.VectorLoad(w + 12);
                    var vp = new Vector<float>(p[c]);
                    a0 += w0 * vp; a1 += w1 * vp; a2 += w2 * vp; a3 += w3 * vp;
                    var vq = new Vector<float>(q[c]);
                    c0 += w0 * vq; c1 += w1 * vq; c2 += w2 * vq; c3 += w3 * vq;
                }
            }
            Simd.VectorStore(tmp, a0); Simd.VectorStore(tmp + 4, a1);
            Simd.VectorStore(tmp + 8, a2); Simd.VectorStore(tmp + 12, a3);
            Simd.VectorStore(tmp + 16, c0); Simd.VectorStore(tmp + 20, c1);
            Simd.VectorStore(tmp + 24, c2); Simd.VectorStore(tmp + 28, c3);
            for (int r = 0; r < 2 && i + r < rows; r++)
            {
                float* o = outp + (long)(i + r) * oc;
                float* ad = addp == null ? null : addp + (long)(i + r) * oc;
                float* t = tmp + r * Ob;
                for (int l = 0; l < lanes; l++)
                    o[l] = ActS(ad != null ? t[l] + ad[l] : t[l], act);
            }
        }
    }

    [ThreadStatic] private static float[]? t_padRow;

    internal static void Depthwise(ReadOnlySpan<float> src, int ih, int iw, int c, Span<float> dst,
        int oh, int ow, int kh, int kw, int sw, int[] rm, int[] cm, float[] padVec, float padValue,
        NhwcConvPlan kp, ReadOnlySpan<float> add, ActivationKind act, int batch = 1)
    {
        bool hasAdd = !add.IsEmpty;
        // a whole constant-pad source row, so out-of-range kernel rows of interior pixels need no pointer table
        float* padRow = null;
        if (Array.IndexOf(rm, -1) >= 0)
        {
            padRow = Pinned(ref t_padRow, iw * c);
            new Span<float>(padRow, iw * c).Fill(padValue);
        }
        fixed (float* sp0 = src, dp0 = dst, ap0 = add, wp0 = kp.W, bp0 = kp.Bias, pv0 = padVec)
        fixed (int* rm0 = rm, cm0 = cm)
        {
            var a = new DwArgs(sp0, dp0, hasAdd ? ap0 : null, wp0, bp0, pv0, padRow, rm0, cm0, iw, c, oh, ow, kh, kw, sw,
                (long)ih * iw * c, (long)oh * ow * c, padValue == 0f, act);
            long macs = (long)batch * oh * ow * c * kh * kw;
            if (Par.Threads == 1 || macs < 1_000_000)
                for (int t = 0; t < batch * oh; t++) DwRow(a, t);
            else
                For(batch * oh, macs, t => DwRow(a, t));
        }
    }

    /// <summary>Per-call depthwise constants (pointers into pinned/fixed storage).</summary>
    internal readonly struct DwArgs(float* sp, float* dp, float* ap, float* wp, float* bp, float* pv, float* padRow,
        int* rm, int* cm, int iw, int c, int oh, int ow, int kh, int kw, int sw, long inImg, long outImg, bool clipRows, ActivationKind act)
    {
        public readonly float* Sp = sp, Dp = dp, Ap = ap, Wp = wp, Bp = bp, Pv = pv, PadRow = padRow;
        public readonly int* Rm = rm, Cm = cm;
        public readonly int Iw = iw, C = c, Oh = oh, Ow = ow, Kh = kh, Kw = kw, Sw = sw;
        public readonly long InImg = inImg, OutImg = outImg;
        /// <summary>Constant pad 0: out-of-range kernel rows contribute nothing and are skipped.</summary>
        public readonly bool ClipRows = clipRows;
        public readonly ActivationKind Act = act;
    }

    // a static row body: pointer stores in a closure would force every captured value to be reloaded
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void DwRow(in DwArgs a, int t)
    {
        int img = t / a.Oh, y = t - img * a.Oh;
        float* sp = a.Sp + img * a.InImg, dp = a.Dp + img * a.OutImg, ap = a.Ap == null ? null : a.Ap + img * a.OutImg;
        float* wp = a.Wp, bp = a.Bp, pv = a.Pv;
        int* cmp = a.Cm, rmy = a.Rm + y * a.Kh;
        int iw = a.Iw, c = a.C, ow = a.Ow, kh = a.Kh, kw = a.Kw;
        if (a.ClipRows)
        {
            // zero-padded rows: run only the in-range kernel rows (contiguous for constant padding)
            int kyB = 0, kyE = kh;
            while (kyB < kyE && rmy[kyB] < 0) kyB++;
            while (kyE > kyB && rmy[kyE - 1] < 0) kyE--;
            rmy += kyB;
            wp += (long)kyB * kw * c;
            kh = kyE - kyB;
        }
        int taps = kh * kw;
        ActivationKind act = a.Act;
        bool avx = UseAvx, neon = UseNeon;
        int VL = Vector<float>.Count, sw = a.Sw;
        float** ptrs4 = stackalloc float*[taps * 4];
        // interior groups of 4 pixels address taps from kh row bases: no per-pixel pointer table
        bool vec8 = !avx && !neon && UseVec8;
        bool runs = avx || vec8 ? c % 8 == 0 : neon && c % 4 == 0;
        float** rows = stackalloc float*[kh];
        for (int ky = 0; ky < kh; ky++)
        {
            int r = rmy[ky];
            rows[ky] = r >= 0 ? sp + (long)r * iw * c : a.PadRow;
        }
        for (int x0 = 0; x0 < ow; x0 += 4)
        {
            int np = Math.Min(4, ow - x0);
            if (runs && np == 4 && Interior(cmp + x0 * kw, kw, sw))
            {
                long ob = ((long)y * ow + x0) * c, col = (long)cmp[x0 * kw] * c;
                float** rp = ptrs4;
                for (int ky = 0; ky < kh; ky++) rp[ky] = rows[ky] + col;
                if (avx) DwRunAvx(rp, kh, kw, c, sw * c, wp, bp, dp + ob, ap == null ? null : ap + ob, act);
                else if (neon) DwRunNeon(rp, kh, kw, c, sw * c, wp, bp, dp + ob, ap == null ? null : ap + ob, act);
                else DwRunVec(rp, kh, kw, c, sw * c, wp, bp, dp + ob, ap == null ? null : ap + ob, act);
                continue;
            }
            float** pp = ptrs4;
            for (int px = 0; px < 4; px++)
            {
                int* cmx = cmp + (x0 + Math.Min(px, np - 1)) * kw;
                for (int ky = 0; ky < kh; ky++)
                {
                    int r = rmy[ky];
                    float* row = sp + (long)r * iw * c;
                    for (int kx = 0; kx < kw; kx++)
                    {
                        int cc = cmx[kx];
                        *pp++ = r < 0 || cc < 0 ? pv : row + (long)cc * c;
                    }
                }
            }
            long ob0 = ((long)y * ow + x0) * c;
            float* ad0 = ap == null ? null : ap + ob0;
            int done = avx ? DwBlockAvx(ptrs4, taps, c, wp, bp, dp + ob0, ad0, np, act)
                : neon ? DwBlockNeon(ptrs4, taps, c, wp, bp, dp + ob0, ad0, np, act)
                : vec8 ? DwBlockVec(ptrs4, taps, c, wp, bp, dp + ob0, ad0, np, act) : 0;
            if (done == c) continue;
            for (int px = 0; px < np; px++)
            {
                float** ptrs = ptrs4 + px * taps;
                float* o = dp + ob0 + (long)px * c;
                float* ad = ad0 == null ? null : ad0 + (long)px * c;
                int ch = done;
                if (avx || neon || vec8) { }
                else if (UsePacked)
                    ch = DepthwisePixelPk(ptrs, taps, c, wp, bp, ad, o, act);
                else if (Vector.IsHardwareAccelerated && act is ActivationKind.None or ActivationKind.Relu)
                {
                    for (; ch + VL <= c; ch += VL)
                    {
                        var acc = Simd.VectorLoad(bp + ch);
                        for (int tap = 0; tap < taps; tap++)
                            acc += Simd.VectorLoad(ptrs[tap] + ch) * Simd.VectorLoad(wp + tap * c + ch);
                        if (ad != null) acc += Simd.VectorLoad(ad + ch);
                        if (act == ActivationKind.Relu) acc = Vector.Max(acc, Vector<float>.Zero);
                        Simd.VectorStore(o + ch, acc);
                    }
                }
                for (; ch < c; ch++)
                {
                    float acc = bp[ch];
                    for (int tap = 0; tap < taps; tap++) acc += ptrs[tap][ch] * wp[tap * c + ch];
                    if (ad != null) acc += ad[ch];
                    o[ch] = ActS(acc, act);
                }
            }
        }
    }

    /// <summary>Whether 4 pixels starting at <paramref name="cmx"/> read contiguous, in-range, stride-<paramref name="sw"/> columns.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Interior(int* cmx, int kw, int sw)
    {
        int c0 = cmx[0];
        if (c0 < 0) return false;
        for (int p = 0; p < 4; p++)
            for (int kx = 0; kx < kw; kx++)
                if (cmx[p * kw + kx] != c0 + p * sw + kx) return false;
        return true;
    }

    // ======================== pixel-vector helpers ========================
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Copy(float* s, float* d, int n) =>
        Buffer.MemoryCopy(s, d, (long)n * 4, (long)n * 4);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Fill(float* d, int n, float v) => new Span<float>(d, n).Fill(v);

    // ======================== conv + 2x2 max pool ========================
    [ThreadStatic] internal static float[]? t_band;

    internal static float[] BandBuf(long floats) =>
        t_band is { } b && b.Length >= floats ? b : (t_band = new float[checked((int)floats)]);

    /// <summary>
    /// Dense / Winograd conv with a fused 2x2 stride-2 max pool (<see cref="ConvP.Pool"/>):
    /// each task produces a band of conv rows (2 direct, 4 = one Winograd tile row) in a
    /// thread-local buffer and pools it straight into <paramref name="dst"/> ([ph, pw, oc]),
    /// so the full-resolution conv output is never stored. Per-element math is the same as
    /// the separate Conv → MaxPool nodes.
    /// </summary>
    public static void ConvPool(ReadOnlySpan<float> src, int ih, int iw, int ic,
        Span<float> dst, int ph, int pw, int oc, ConvP p, NhwcConvPlan kp)
    {
        int oh = ph * 2, ow = pw * 2;
        if (kp.Depthwise || p.AddValue >= 0 || dst.Length != (long)ph * pw * oc)
            throw new NotSupportedException("ConvPool shape/kind mismatch.");
        bool relu = p.Pool!.FuseRelu, wino = kp.WinoU is not null;
        int band = wino ? 4 : 2, tasks = (oh + band - 1) / band;
        int kh = p.Kernel[0], kw = p.Kernel[1];
        float[] padVec = new float[ic * kw + Ob];   // one constant-pad kernel row (Dense512Row)
        Array.Fill(padVec, p.PadValue);
        int[] rm = Map(oh, kh, p.Strides[0], p.Pads[0], ih, p.PadMode);
        int[] cm = Map(ow, kw, p.Strides[1], p.Pads[1], iw, p.PadMode);
        fixed (float* sp = src, dp0 = dst, wp = kp.W, up = kp.WinoU, bp = kp.Bias, pv = padVec)
        fixed (int* rmp = rm, cmp = cm)
        {
            var da = new DenseArgs(sp, wp, bp, pv, rmp, cmp, iw, ic, ow, oc, kh, kw, kp.Blocks, p.Act, UseAvx);
            var wa = wino ? new WinoArgs(sp, up, bp, pv, ih, iw, ic, oh, ow, oc, kp.Blocks, p) : default;
            float* dp = dp0;
            For(tasks, (long)oh * ow * oc * kh * kw * ic / 64, t =>
            {
                int y0 = t * band, rows = Math.Min(band, oh - y0);
                float[] buf = BandBuf((long)band * ow * oc);
                fixed (float* sb = buf)
                {
                    if (wino)
                        for (int t0 = y0 / 4 * wa.Tw, te = t0 + wa.Tw; t0 < te; t0 += wa.T)
                            WinoTiles(wa, t0, Math.Min(wa.T, te - t0), sb, null, y0);
                    else
                        for (int r = 0; r < rows; r++)
                            ConvRow(da, y0 + r, sb + (long)r * ow * oc, null);
                    PoolBand(sb, rows / 2, ow, oc, dp + (long)(y0 / 2) * pw * oc, pw, relu);
                }
            });
        }
    }

    /// <summary>2x2/2 max over conv row pairs of a band; same op order as <see cref="MaxPool"/>.</summary>
    internal static void PoolBand(float* sb, int prows, int ow, int c, float* drow, int pw, bool relu)
    {
        int VL = Vector<float>.Count;
        for (int r = 0; r < prows; r++)
        {
            float* top = sb + (long)(2 * r) * ow * c, bot = top + (long)ow * c;
            float* orow = drow + (long)r * pw * c;
            for (int x = 0; x < pw; x++)
            {
                float* o = orow + (long)x * c;
                float* s0 = top + (long)(2 * x) * c, s1 = s0 + c, s2 = bot + (long)(2 * x) * c, s3 = s2 + c;
                int ch = 0;
                for (; ch + VL <= c; ch += VL)
                {
                    var v = Vector.Max(Simd.VectorLoad(s0 + ch), Simd.VectorLoad(s1 + ch));
                    v = Vector.Max(v, Simd.VectorLoad(s2 + ch));
                    Simd.VectorStore(o + ch, Vector.Max(v, Simd.VectorLoad(s3 + ch)));
                }
                for (; ch < c; ch++) o[ch] = MathF.Max(MathF.Max(MathF.Max(s0[ch], s1[ch]), s2[ch]), s3[ch]);
                if (relu)
                    for (ch = 0; ch < c; ch++) o[ch] = MathF.Max(o[ch], 0f);
            }
        }
    }

    // ======================== maxpool ========================
    public static void MaxPool(ReadOnlySpan<float> src, int ih, int iw, int c,
        Span<float> dst, int oh, int ow, PoolP p)
    {
        int kh = p.Kernel[0], kw = p.Kernel[1];
        int[] rm = Map(oh, kh, p.Strides[0], p.Pads[0], ih, PadMode.Constant);
        int[] cm = Map(ow, kw, p.Strides[1], p.Pads[1], iw, PadMode.Constant);
        bool relu = p.FuseRelu;
        int VL = Vector<float>.Count;
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            if (UseAvx512 && c % 16 == 0) { MaxPool512(sp, iw, c, dp, oh, ow, kh, kw, rm, cm, relu); return; }
            For(oh, (long)oh * ow * c * kh * kw, y =>
            {
                for (int x = 0; x < ow; x++)
                {
                    float* o = dp + ((long)y * ow + x) * c;
                    bool any = false;
                    for (int ky = 0; ky < kh; ky++)
                    {
                        int r = rm[y * kh + ky];
                        if (r < 0) continue;
                        for (int kx = 0; kx < kw; kx++)
                        {
                            int cc = cm[x * kw + kx];
                            if (cc < 0) continue;
                            float* s = sp + ((long)r * iw + cc) * c;
                            if (!any) { Copy(s, o, c); any = true; continue; }
                            int ch = 0;
                            for (; ch + VL <= c; ch += VL)
                                Simd.VectorStore(o + ch, Vector.Max(Simd.VectorLoad(o + ch), Simd.VectorLoad(s + ch)));
                            for (; ch < c; ch++) o[ch] = MathF.Max(o[ch], s[ch]);
                        }
                    }
                    if (!any) Fill(o, c, 0f);
                    else if (relu)
                        for (int ch = 0; ch < c; ch++) o[ch] = MathF.Max(o[ch], 0f);
                }
            });
        }
    }

    // ======================== bilinear resize (half-pixel) ========================
    internal static void ResizeAxis(int outLen, int inLen, float scale, int[] i0, int[] i1, float[] wt)
    {
        for (int o = 0; o < outLen; o++)
        {
            float f = (o + 0.5f) / scale - 0.5f;
            float fl = MathF.Floor(f);
            i0[o] = Math.Clamp((int)fl, 0, inLen - 1);
            i1[o] = Math.Clamp((int)fl + 1, 0, inLen - 1);
            wt[o] = f - fl;
        }
    }

    /// <summary>One output pixel of the bilinear resize into <paramref name="o"/> (c channels):
    /// v = [relu](lerp) + [relu](add) — the same per-element ops as separate Resize/Relu/Add nodes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ResizePixel(float* a, float* b, float* e, float* f, float wX, float wY,
        Vector<float> vX, Vector<float> vY, int c, bool relu, float* add, bool addRelu, float* o)
    {
        int VL = Vector<float>.Count, ch = 0;
        for (; ch + VL <= c; ch += VL)
        {
            var va = Simd.VectorLoad(a + ch); var ve = Simd.VectorLoad(e + ch);
            var top = va + (Simd.VectorLoad(b + ch) - va) * vX;
            var bot = ve + (Simd.VectorLoad(f + ch) - ve) * vX;
            var v = top + (bot - top) * vY;
            if (relu) v = Vector.Max(v, Vector<float>.Zero);
            if (add != null)
            {
                var s = Simd.VectorLoad(add + ch);
                v += addRelu ? Vector.Max(s, Vector<float>.Zero) : s;
            }
            Simd.VectorStore(o + ch, v);
        }
        for (; ch < c; ch++)
        {
            float top = a[ch] + (b[ch] - a[ch]) * wX;
            float bot = e[ch] + (f[ch] - e[ch]) * wX;
            float v = top + (bot - top) * wY;
            if (relu) v = MathF.Max(v, 0f);
            if (add != null) v += addRelu ? MathF.Max(add[ch], 0f) : add[ch];
            o[ch] = v;
        }
    }

    /// <summary>NHWC bilinear resize; optional same-shape addend (out = [relu](resize) + [relu](add)).</summary>
    public static void ResizeBilinear(ReadOnlySpan<float> src, int ih, int iw, int c,
        Span<float> dst, int oh, int ow, float sy, float sx, ActivationKind act,
        ReadOnlySpan<float> add = default, bool addRelu = false)
    {
        if (!add.IsEmpty && add.Length != dst.Length)
            throw new NotSupportedException($"resize addend has {add.Length} elems, expected {dst.Length}.");
        int[] y0 = new int[oh], y1 = new int[oh], x0 = new int[ow], x1 = new int[ow];
        float[] wy = new float[oh], wx = new float[ow];
        ResizeAxis(oh, ih, sy, y0, y1, wy);
        ResizeAxis(ow, iw, sx, x0, x1, wx);
        bool relu = act == ActivationKind.Relu, hasAdd = !add.IsEmpty, v512 = UseAvx512 && c % 16 == 0;
        fixed (float* sp0 = src, dp0 = dst, ap0 = add)
        {
            float* sp = sp0, dp = dp0, ap = ap0;
            For(oh, (long)oh * ow * c, y =>
            {
                float* r0 = sp + (long)y0[y] * iw * c, r1 = sp + (long)y1[y] * iw * c;
                var vY = new Vector<float>(wy[y]);
                float wY = wy[y];
                if (v512)
                {
                    var vY16 = Vector512.Create(wY);
                    for (int x = 0; x < ow; x++)
                    {
                        long ob = ((long)y * ow + x) * c;
                        ResizePixel512(r0 + (long)x0[x] * c, r0 + (long)x1[x] * c, r1 + (long)x0[x] * c, r1 + (long)x1[x] * c,
                            Vector512.Create(wx[x]), vY16, c, relu, hasAdd ? ap + ob : null, addRelu, dp + ob);
                    }
                    return;
                }
                for (int x = 0; x < ow; x++)
                {
                    long ob = ((long)y * ow + x) * c;
                    ResizePixel(r0 + (long)x0[x] * c, r0 + (long)x1[x] * c, r1 + (long)x0[x] * c, r1 + (long)x1[x] * c,
                        wx[x], wY, new Vector<float>(wx[x]), vY, c, relu, hasAdd ? ap + ob : null, addRelu, dp + ob);
                }
            });
        }
    }

    internal static void ActRow(float* p, int n, ActivationKind act, bool avx)
    {
        int i = 0;
        if (avx)
            for (; i + 8 <= n; i += 8) Avx.Store(p + i, Act8(Avx.LoadVector256(p + i), act));
        else if (UseNeon)
            for (; i + 4 <= n; i += 4) AdvSimd.Store(p + i, Act4(AdvSimd.LoadVector128(p + i), act));
        else if (UsePacked)
            for (; i + 4 <= n; i += 4) PackedSimd.Store(p + i, ActP(LdP(p + i), act));
        else if (act == ActivationKind.Sigmoid)
        {
            var row = new Span<float>(p + i, n - i);
            SimdMath.Sigmoid(row, row);
            return;
        }
        for (; i < n; i++) p[i] = ActS(p[i], act);
    }

    // ======================== pad (spatial only) ========================
    public static void Pad(ReadOnlySpan<float> src, int ih, int iw, int c,
        Span<float> dst, int oh, int ow, PadP p)
    {
        int rank = p.Pads.Length / 2;
        if (rank != 4) throw new NotSupportedException("NHWC Pad supports rank-4 only.");
        int[] rm = Map(oh, 1, 1, p.Pads[2], ih, p.Mode);
        int[] cm = Map(ow, 1, 1, p.Pads[3], iw, p.Mode);
        float val = p.Value;
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            For(oh, (long)oh * ow * c, y =>
            {
                int r = rm[y];
                for (int x = 0; x < ow; x++)
                {
                    float* o = dp + ((long)y * ow + x) * c;
                    int cc = cm[x];
                    if (r < 0 || cc < 0) Fill(o, c, val);
                    else Copy(sp + ((long)r * iw + cc) * c, o, c);
                }
            });
        }
    }


}
