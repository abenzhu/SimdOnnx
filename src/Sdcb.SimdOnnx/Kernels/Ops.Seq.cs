using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdOnnx.Kernels;

// Sequence-model ops (PP-OCR cls/rec heads, SVTR attention): batched MatMul, Softmax,
// last-axis mean, Sqrt, average pooling.
internal static unsafe partial class Ops
{
    /// <summary>dst = act(src) (layout-agnostic; src and dst may alias).</summary>
    public static void Activation(ReadOnlySpan<float> src, Span<float> dst, ActivationKind act)
    {
        fixed (float* s = src, d = dst) Act.Apply(s, d, src.Length, act);
    }

    public static void Sqrt(ReadOnlySpan<float> src, Span<float> dst)
    {
        int i = 0, VL = Vector<float>.Count;
        if (Vector.IsHardwareAccelerated)
            for (; i + VL <= src.Length; i += VL)
                Simd.VectorStore(dst, i, Vector.SquareRoot(Simd.VectorLoad(src, i)));
        for (; i < src.Length; i++) dst[i] = MathF.Sqrt(src[i]);
    }

    /// <summary>Row softmax over contiguous rows of length <paramref name="c"/>.</summary>
    public static void Softmax(ReadOnlySpan<float> src, Span<float> dst, int c)
    {
        int rows = src.Length / c;
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            Nhwc.For(rows, (long)rows * c * 8, r => SoftmaxRow(sp + (long)r * c, dp + (long)r * c, c));
        }
    }

    private static void SoftmaxRow(float* s, float* d, int c)
    {
        float max = float.NegativeInfinity;
        int i = 0;
        if (Vector128.IsHardwareAccelerated && c >= 4)
        {
            var vm = Vector128.Create(float.NegativeInfinity);
            for (; i + 4 <= c; i += 4) vm = Vector128.Max(vm, Vector128.Load(s + i));
            max = MathF.Max(MathF.Max(vm[0], vm[1]), MathF.Max(vm[2], vm[3]));
        }
        for (; i < c; i++) max = MathF.Max(max, s[i]);
        float sum = 0;
        i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            var vmax = Vector128.Create(max);
            var acc = Vector128<float>.Zero;
            for (; i + 4 <= c; i += 4)
            {
                var e = SimdMath.ExpCore(Vector128.Load(s + i) - vmax);
                e.Store(d + i);
                acc += e;
            }
            sum = Vector128.Sum(acc);
        }
        for (; i < c; i++) { float e = MathF.Exp(s[i] - max); d[i] = e; sum += e; }
        float inv = 1f / sum;
        i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            var vi = Vector128.Create(inv);
            for (; i + 4 <= c; i += 4) (Vector128.Load(d + i) * vi).Store(d + i);
        }
        for (; i < c; i++) d[i] *= inv;
    }

    /// <summary>dst[r] = mean of row r (rows of length <paramref name="c"/>).</summary>
    public static void ReduceMeanLast(ReadOnlySpan<float> src, Span<float> dst, int c)
    {
        int rows = src.Length / c;
        float inv = 1f / c;
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> row = src.Slice(r * c, c);
            int i = 0, VL = Vector<float>.Count;
            var acc = Vector<float>.Zero;
            for (; i + VL <= c; i += VL) acc += Simd.VectorLoad(row, i);
            float s = Vector.Sum(acc);
            for (; i < c; i++) s += row[i];
            dst[r] = s * inv;
        }
    }

    /// <summary>Batched row-major MatMul: a [.., M, K] × b [.., K, N] → [.., M, N] (numpy batch broadcast).</summary>
    public static void MatMul(ReadOnlySpan<float> a, int[] aShape, ReadOnlySpan<float> b, int[] bShape,
        Span<float> dst, int[] outShape)
    {
        int rank = outShape.Length;
        int m = outShape[^2], n = outShape[^1], k = aShape[^1];
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= outShape[i];
        int[] aStr = BatchStrides(aShape, rank, m * k), bStr = BatchStrides(bShape, rank, k * n);
        bool avx = Avx2.IsSupported && Fma.IsSupported;
        for (int bi = 0; bi < batch; bi++)
        {
            int rem = bi, ao = 0, bo = 0;
            for (int d = rank - 3; d >= 0; d--)
            {
                int idx = rem % outShape[d];
                rem /= outShape[d];
                ao += idx * aStr[d];
                bo += idx * bStr[d];
            }
            ReadOnlySpan<float> ab = a.Slice(ao, m * k), bb = b.Slice(bo, k * n);
            Span<float> db = dst.Slice(bi * m * n, m * n);
            if (avx && (long)m * n * k >= 32768) Avx2Kernels.GemmKN(ab, m, k, bb, n, default, db);
            else GemmKN(ab, m, k, bb, n, default, db);
        }
    }

    /// <summary>Per-batch-dim element strides (0 for broadcast dims) of a [.., X, Y] operand aligned to rank.</summary>
    private static int[] BatchStrides(int[] shape, int rank, int matSize)
    {
        int[] s = new int[Math.Max(rank - 2, 0)];
        int acc = matSize;
        for (int d = rank - 3; d >= 0; d--)
        {
            int sd = d - (rank - shape.Length);
            int dim = sd >= 0 ? shape[sd] : 1;
            s[d] = dim == 1 ? 0 : acc;
            acc *= dim;
        }
        return s;
    }

    /// <summary>NCHW global average pool (reference path).</summary>
    public static void GlobalAvgPool(ReadOnlySpan<float> src, int[] inShape, Span<float> dst)
    {
        int nc = inShape[0] * inShape[1], hw = inShape[2] * inShape[3];
        for (int c = 0; c < nc; c++)
        {
            float s = 0;
            foreach (float v in src.Slice(c * hw, hw)) s += v;
            dst[c] = s / hw;
        }
    }

    /// <summary>NCHW average pool (reference path for non-NHWC plans).</summary>
    public static void AvgPool(ReadOnlySpan<float> src, int[] inShape, Span<float> dst, int[] outShape, PoolP p)
    {
        int nc = inShape[0] * inShape[1], ih = inShape[2], iw = inShape[3];
        int oh = outShape[2], ow = outShape[3];
        int kh = p.Kernel[0], kw = p.Kernel[1], sh = p.Strides[0], sw = p.Strides[1], pt = p.Pads[0], pl = p.Pads[1];
        for (int c = 0; c < nc; c++)
        {
            ReadOnlySpan<float> s = src.Slice(c * ih * iw, ih * iw);
            Span<float> d = dst.Slice(c * oh * ow, oh * ow);
            for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
            {
                float sum = 0; int cnt = 0;
                for (int ky = 0; ky < kh; ky++)
                {
                    int sy = y * sh + ky - pt;
                    if ((uint)sy >= (uint)ih) continue;
                    for (int kx = 0; kx < kw; kx++)
                    {
                        int sx = x * sw + kx - pl;
                        if ((uint)sx >= (uint)iw) continue;
                        sum += s[sy * iw + sx];
                        cnt++;
                    }
                }
                d[y * ow + x] = sum / (p.CountIncludePad ? kh * kw : Math.Max(cnt, 1));
            }
        }
    }
}

internal static unsafe partial class Nhwc
{
    /// <summary>NHWC average pool of one image ([ih, iw, c] → [oh, ow, c]).</summary>
    public static void AvgPool(ReadOnlySpan<float> src, int ih, int iw, int c,
        Span<float> dst, int oh, int ow, PoolP p)
    {
        int kh = p.Kernel[0], kw = p.Kernel[1];
        int[] rm = Map(oh, kh, p.Strides[0], p.Pads[0], ih, PadMode.Constant);
        int[] cm = Map(ow, kw, p.Strides[1], p.Pads[1], iw, PadMode.Constant);
        bool v = V128, incl = p.CountIncludePad;
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            For(oh, (long)oh * ow * c * kh * kw, y =>
            {
                for (int x = 0; x < ow; x++)
                {
                    float* o = dp + ((long)y * ow + x) * c;
                    Fill(o, c, 0f);
                    int cnt = 0;
                    for (int ky = 0; ky < kh; ky++)
                    {
                        int r = rm[y * kh + ky];
                        if (r < 0) continue;
                        for (int kx = 0; kx < kw; kx++)
                        {
                            int cc = cm[x * kw + kx];
                            if (cc < 0) continue;
                            float* s = sp + ((long)r * iw + cc) * c;
                            int ch = 0;
                            if (v)
                                for (; ch + 4 <= c; ch += 4)
                                    (Vector128.Load(o + ch) + Vector128.Load(s + ch)).Store(o + ch);
                            for (; ch < c; ch++) o[ch] += s[ch];
                            cnt++;
                        }
                    }
                    float inv = 1f / (incl ? kh * kw : Math.Max(cnt, 1));
                    int k = 0;
                    if (v)
                    {
                        var vi = Vector128.Create(inv);
                        for (; k + 4 <= c; k += 4) (Vector128.Load(o + k) * vi).Store(o + k);
                    }
                    for (; k < c; k++) o[k] *= inv;
                }
            });
        }
    }
}
