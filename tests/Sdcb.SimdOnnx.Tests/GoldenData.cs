using System.Text.Json;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>
/// GCDRNet onnxruntime golden fixtures (Sdcb.GcdrNet tools/GenGolden), read from the sibling
/// repo checkout (../Sdcb.GcdrNet/tests/testdata) or SIMDONNX_GCDR_TESTDATA.
/// </summary>
internal static class Golden
{
    public static string Dir { get; } = FindDir();

    private static string FindDir()
    {
        string? env = Environment.GetEnvironmentVariable("SIMDONNX_GCDR_TESTDATA");
        if (env is not null && File.Exists(Path.Combine(env, "manifest.json"))) return env;
        var d = new DirectoryInfo(AppContext.BaseDirectory);
        while (d != null)
        {
            string cand = Path.Combine(d.FullName, "Sdcb.GcdrNet", "tests", "testdata", "manifest.json");
            if (File.Exists(cand)) return Path.GetDirectoryName(cand)!;
            d = d.Parent;
        }
        throw new DirectoryNotFoundException("GCDRNet testdata not found (set SIMDONNX_GCDR_TESTDATA).");
    }

    /// <summary>"shadow" / "output" are the committed onnxruntime outputs; "input" / "dr_in" are
    /// derived from {tag}_src.png exactly as Sdcb.GcdrNet's tests/GoldenDerive.cs does.</summary>
    public static float[] F32(string tag, string name)
    {
        var (_, _, ph, pw, oh, ow) = Dims(tag);
        switch (name)
        {
            case "input":
            {
                byte[] bgr = Bgr(tag);
                int h = oh + ph, w = ow + pw;
                var x = new float[3 * h * w];
                for (int c = 0; c < 3; c++)
                    for (int y = 0; y < h; y++)
                        for (int xx = 0; xx < w; xx++)
                            x[(c * h + y) * w + xx] = bgr[(Math.Max(y - ph, 0) * ow + Math.Max(xx - pw, 0)) * 3 + c] / 255f;
                return x;
            }
            case "dr_in":
            {
                float[] input = F32(tag, "input"), shadow = F32(tag, "shadow");
                int n = input.Length;
                var d = new float[2 * n];
                input.CopyTo(d, 0);
                for (int i = 0; i < n; i++)
                    d[n + i] = Math.Min(Math.Max(input[i] / Math.Max(shadow[i], 1e-6f), 0f), 1f);
                return d;
            }
            default:
                return FromFile<float>(Path.Combine(Dir, $"{tag}_{name}.f32"));
        }
    }

    /// <summary>Expected final BGR24 output (cropped), derived from the DR-Net golden output.</summary>
    public static byte[] U8(string tag)
    {
        var (_, _, ph, pw, oh, ow) = Dims(tag);
        float[] output = F32(tag, "output");
        int h = oh + ph, w = ow + pw;
        var u8 = new byte[oh * ow * 3];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < oh; y++)
                for (int xx = 0; xx < ow; xx++)
                {
                    double v = output[c * h * w + (y + ph) * w + xx + pw] * 255.0;
                    u8[(y * ow + xx) * 3 + c] = (byte)Math.Min(Math.Max((int)v, 0), 255);
                }
        return u8;
    }

    /// <summary>Tight BGR24 pixels of {tag}_src.png (8-bit RGB, non-interlaced).</summary>
    public static byte[] Bgr(string tag)
    {
        byte[] rgb = Png.DecodeRgb8(File.ReadAllBytes(Path.Combine(Dir, $"{tag}_src.png")), out _, out _);
        for (int i = 0; i < rgb.Length; i += 3) (rgb[i], rgb[i + 2]) = (rgb[i + 2], rgb[i]);
        return rgb;
    }

    public static int[] ShapeOf(string tag, string name)
    {
        var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "manifest.json")));
        var arr = doc.RootElement.GetProperty(tag).GetProperty("shapes").GetProperty(name);
        return arr.EnumerateArray().Select(e => e.GetInt32()).ToArray();
    }

    public static (int h, int w, int ph, int pw, int oh, int ow) Dims(string tag)
    {
        var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "manifest.json")));
        var e = doc.RootElement.GetProperty(tag);
        return (e.GetProperty("h").GetInt32(), e.GetProperty("w").GetInt32(),
            e.GetProperty("ph").GetInt32(), e.GetProperty("pw").GetInt32(),
            e.GetProperty("orig_h").GetInt32(), e.GetProperty("orig_w").GetInt32());
    }

    private static T[] FromFile<T>(string path) where T : unmanaged
    {
        byte[] raw = File.ReadAllBytes(path);
        T[] dst = new T[raw.Length / System.Runtime.InteropServices.Marshal.SizeOf<T>()];
        Buffer.BlockCopy(raw, 0, dst, 0, raw.Length);
        return dst;
    }

    public static (float max, double rmse, float mean) Diff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float max = 0; double sum = 0; double amean = 0;
        for (int i = 0; i < a.Length; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d > max) max = d;
            sum += (double)d * d;
            amean += a[i];
        }
        return (max, Math.Sqrt(sum / a.Length), (float)(amean / a.Length));
    }

    public static string ModelDir { get; } = Path.Combine(Dir, "models");
}
