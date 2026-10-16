using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdOnnx.Tests;

/// <summary><see cref="Layout"/>'s allocation-free helpers against their original LINQ forms.</summary>
public class LayoutTests
{
    [Fact]
    public void Equivalent_And_Reshaped_MatchReference()
    {
        var rng = new Random(11);
        int views = 0;
        for (int it = 0; it < 20000; it++)
        {
            int rank = rng.Next(1, 6);
            int[] src = new int[rank];
            for (int d = 0; d < rank; d++) src[d] = rng.Next(3) == 0 ? 1 : rng.Next(2, 7);
            int[] l = Enumerable.Range(0, rank).OrderBy(_ => rng.Next()).ToArray();
            if (rng.Next(3) == 0) l = Enumerable.Range(0, rank).ToArray();
            int[] b = Enumerable.Range(0, rank).OrderBy(_ => rng.Next()).ToArray();
            Assert.Equal(EquivalentRef(src, l, b), Layout.Equivalent(src, l, b));

            int[] dst = RandomReshape(rng, src);
            int[]? want = ReshapedRef(src, l, dst), got = Layout.Reshaped(src, l, dst);
            Assert.Equal(want is null, got is null);
            if (want is not null) { Assert.Equal(want, got); views++; }
        }
        Assert.True(views > 1000, $"only {views} view reshapes exercised");
    }

    // a random reshape of the same element count: split / merge adjacent dims, sprinkle 1s
    private static int[] RandomReshape(Random rng, int[] src)
    {
        var dims = new List<int>();
        foreach (int d in src)
        {
            if (d == 1) continue;
            if (d == 4 && rng.Next(2) == 0) { dims.Add(2); dims.Add(2); }
            else if (d == 6 && rng.Next(2) == 0) { dims.Add(2); dims.Add(3); }
            else dims.Add(d);
        }
        for (int k = 0; k + 1 < dims.Count; k++)
            if (rng.Next(3) == 0) { dims[k] *= dims[k + 1]; dims.RemoveAt(k + 1); }
        int ones = rng.Next(3);
        for (int k = 0; k < ones; k++) dims.Insert(rng.Next(dims.Count + 1), 1);
        if (dims.Count == 0) dims.Add(1);
        return [.. dims];
    }

    private static bool EquivalentRef(int[] shape, int[] a, int[] b) =>
        a.Where(d => shape[d] != 1).SequenceEqual(b.Where(d => shape[d] != 1));

    private static int[]? ReshapedRef(int[] src, int[] l, int[] dst)
    {
        int[] sd = Enumerable.Range(0, src.Length).Where(d => src[d] != 1).ToArray();
        int[] td = Enumerable.Range(0, dst.Length).Where(d => dst[d] != 1).ToArray();
        int[] rank = new int[src.Length];
        int r = 0;
        foreach (int d in l) if (src[d] != 1) rank[d] = r++;
        double[] key = new double[dst.Length];
        int i = 0, j = 0;
        while (i < sd.Length || j < td.Length)
        {
            if (i >= sd.Length || j >= td.Length) return null;
            var gs = new List<int> { sd[i] }; long ps = src[sd[i++]];
            var gt = new List<int> { td[j] }; long pt = dst[td[j++]];
            while (ps != pt)
            {
                if (ps < pt) { if (i >= sd.Length) return null; gs.Add(sd[i]); ps *= src[sd[i++]]; }
                else { if (j >= td.Length) return null; gt.Add(td[j]); pt *= dst[td[j++]]; }
            }
            for (int k = 1; k < gs.Count; k++)
                if (rank[gs[k]] != rank[gs[k - 1]] + 1) return null;
            for (int k = 0; k < gt.Count; k++)
                key[gt[k]] = rank[gs[0]] + k / (double)(gt.Count + 1);
        }
        double prev = -1;
        for (int e = 0; e < dst.Length; e++)
        {
            if (dst[e] != 1) prev = key[e];
            else key[e] = prev + 1e-3 * (e + 1) / (dst.Length + 1);
        }
        return Enumerable.Range(0, dst.Length).OrderBy(e => key[e]).ThenBy(e => e).ToArray();
    }
}
