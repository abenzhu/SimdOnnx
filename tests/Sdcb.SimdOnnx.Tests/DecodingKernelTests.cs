using Sdcb.SimdOnnx.Decoding;
using Sdcb.SimdOnnx.Decoding.Kernels;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>CTC projection + argmax kernels (moved from SimdPaddleOCR with the kernels) and the public decoding API.</summary>
public class DecodingKernelTests
{
    [Theory]
    [InlineData(5, 32, 40)]
    [InlineData(4, 64, 1040)]
    [InlineData(5, 64, 1040)]
    [InlineData(4, 32, 39)]
    [InlineData(6, 32, 37)]
    [InlineData(7, 96, 1037)]
    public void MatMul_VectorMatchesReference(int rows, int inner, int columns)
    {
        const int batch = 2;
        float[] input = Ramp(batch * rows * inner, 0.01f);
        float[] weights = Ramp(inner * columns, 0.02f);
        float[]? packed = columns >= 16 ? PackMatMul(weights, inner, columns) : null;

        float[] expected = new float[batch * rows * columns];
        MatMulRef(input, weights, expected, batch, rows, inner, columns);

        float[] actual = new float[expected.Length];
        Assert.True(MatMul.Try(input, weights, actual, batch, rows, inner, columns, packed));
        AssertClose(expected, actual, rtol: 5e-5f, atol: 5e-5f);

        float[] vectorActual = new float[expected.Length];
        Assert.True(MatMul.TryVector(input, weights, vectorActual, batch, rows, inner, columns, packed));
        AssertClose(expected, vectorActual, rtol: 5e-5f, atol: 5e-5f);
    }

    [Fact]
    public void MatMul_Rows8PackedHandlesPartialColumnTile()
    {
        const int batch = 1, rows = 8, inner = 64, columns = 1050;
        float[] input = Ramp(batch * rows * inner, 0.01f);
        float[] weights = Ramp(inner * columns, -0.02f);
        float[] packed = PackMatMul(weights, inner, columns);
        float[] expected = new float[batch * rows * columns];
        MatMulRef(input, weights, expected, batch, rows, inner, columns);
        float[] actual = new float[expected.Length];
        Assert.True(MatMul.Try(input, weights, actual, batch, rows, inner, columns, packed));
        AssertClose(expected, actual, rtol: 5e-5f, atol: 5e-5f);
    }

    [Theory]
    [InlineData(4, 64, 1040)]
    [InlineData(5, 96, 1037)]
    [InlineData(3, 64, 1030)]
    [InlineData(7, 96, 6624)]
    public void MatMul_ArgMaxMatchesReference(int rows, int inner, int columns)
    {
        // Random data: a ramp over the inner dim produces near-tied column scores,
        // and argmax then flips between rounding orders (FMA vs mul+add).
        const int batch = 2;
        float[] input = RandomFill(batch * rows * inner, 1f, seed: 1234);
        float[] weights = RandomFill(inner * columns, 0.25f, seed: 5678);
        float[] bias = RandomFill(columns, 0.5f, seed: 9012);
        float[] packed = PackMatMul(weights, inner, columns);

        float[] logits = new float[batch * rows * columns];
        MatMulRef(input, weights, logits, batch, rows, inner, columns);
        int[] expectedIndex = new int[batch * rows];
        float[] expectedScore = new float[batch * rows];
        for (int r = 0; r < batch * rows; r++)
        {
            int best = 0;
            float max = logits[r * columns] + bias[0];
            for (int c = 1; c < columns; c++)
            {
                float value = logits[r * columns + c] + bias[c];
                if (value > max) { max = value; best = c; }
            }
            expectedIndex[r] = best;
            expectedScore[r] = max;
        }

        int[] indices = new int[batch * rows];
        float[] scores = new float[batch * rows];
        Assert.True(MatMul.TryArgMax(input, weights, bias, indices, scores,
            batch, rows, inner, columns, packed, threads: 3));
        Assert.Equal(expectedIndex, indices);
        AssertClose(expectedScore, scores, rtol: 5e-5f, atol: 5e-5f);
    }

    internal static float[] PackMatMul(ReadOnlySpan<float> weights, int inner, int columns)
    {
        int fullTiles = columns / 16;
        float[] packed = new float[fullTiles * inner * 16];
        for (int tile = 0; tile < fullTiles; tile++)
            for (int k = 0; k < inner; k++)
                weights.Slice(k * columns + tile * 16, 16).CopyTo(packed.AsSpan((tile * inner + k) * 16, 16));
        return packed;
    }

    internal static void MatMulRef(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        Span<float> output, int batch, int rows, int inner, int columns)
    {
        for (int b = 0; b < batch; b++)
            for (int row = 0; row < rows; row++)
                for (int col = 0; col < columns; col++)
                {
                    float sum = 0;
                    for (int k = 0; k < inner; k++)
                        sum += input[(b * rows + row) * inner + k] * weights[k * columns + col];
                    output[(b * rows + row) * columns + col] = sum;
                }
    }

    internal static float[] Ramp(int length, float scale)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = (i % 17 - 8) * scale;
        return values;
    }

    internal static float[] RandomFill(int length, float amplitude, int seed)
    {
        var rng = new Random(seed);
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1) * amplitude;
        return values;
    }

    internal static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual,
        float rtol = 2e-5f, float atol = 2e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = atol + rtol * Math.Max(Math.Abs(e), Math.Abs(a));
            if (Math.Abs(e - a) > tol)
                Assert.Fail($"index {i}: expected {e}, actual {a}, tol {tol}");
        }
    }

    [Theory]
    [InlineData(4, 1)]
    [InlineData(256, 4)]
    [InlineData(512, 4)]
    [InlineData(516, 4)]
    [InlineData(1024, 4)]
    [InlineData(2048, 1)]
    public void ReusedWorkspaceResetsMaximaAndTieIndices(int rows, int threads)
    {
        const int batch = 2, inner = 64, columns = 1040;
        float[] input = new float[batch * rows * inner];
        float[] weights = new float[inner * columns];
        float[] packed = PackMatMul(weights, inner, columns);
        float[] bias = new float[columns];
        int[] indices = new int[batch * rows];
        float[] scores = new float[indices.Length];
        for (int iteration = 0; iteration < 8; iteration++)
        {
            bool tied = (iteration & 1) != 0;
            Array.Fill(bias, -20f);
            int expected = tied ? 0 : columns - 1 - iteration;
            if (!tied) bias[expected] = 5f;
            Assert.True(MatMul.TryArgMax(input, weights, bias, indices, scores,
                batch, rows, inner, columns, packed, threads));
            Assert.All(indices, value => Assert.Equal(expected, value));
            Assert.All(scores, value => Assert.Equal(tied ? -20f : 5f, value));
        }
    }

    [Fact]
    public void CtcProjection_MatchesKernel()
    {
        const int batch = 2, rows = 8, inner = 64, columns = 1040;
        float[] input = RandomFill(batch * rows * inner, 1f, 3), weights = RandomFill(inner * columns, 0.2f, 4), bias = RandomFill(columns, 0.5f, 5);
        var head = new CtcProjection(weights, bias, inner, columns);
        Assert.True(head.IsFused);
        int[] a = new int[batch * rows], b = new int[a.Length];
        float[] sa = new float[a.Length], sb = new float[a.Length];
        Assert.True(head.ArgMax(input, batch, rows, a, sa, 4));
        float[] logits = new float[batch * rows * columns];
        MatMulRef(input, weights, logits, batch, rows, inner, columns);
        for (int r = 0; r < a.Length; r++)
        {
            for (int c = 0; c < columns; c++) logits[r * columns + c] += bias[c];
            b[r] = CtcGreedyDecoder.ArgMax(logits.AsSpan(r * columns, columns), out sb[r]);
        }
        Assert.Equal(b, a);
        for (int r = 0; r < a.Length; r++) Assert.True(Math.Abs(sa[r] - sb[r]) < 1e-3f, $"row {r}: {sa[r]} vs {sb[r]}");
    }

    [Fact]
    public void Collapse_MergesRepeatsAndDropsBlanks()
    {
        int[] idx = [0, 3, 3, 0, 3, 5, 5, 5];
        float[] sc = [.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f];
        var tokens = new CtcToken[idx.Length];
        int n = CtcGreedyDecoder.Collapse(idx, sc, tokens);
        Assert.Equal(3, n);
        Assert.Equal(new CtcToken(3, .2f, 1, 3), tokens[0]);
        Assert.Equal(new CtcToken(3, .5f, 4, 5), tokens[1]);
        Assert.Equal(new CtcToken(5, .6f, 5, 8), tokens[2]);
    }
}
