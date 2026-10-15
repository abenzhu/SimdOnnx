using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>
/// Whole-graph GPU vs CPU checks of the ops and fused epilogues the OCR / enhancement models need
/// beyond plain conv (AveragePool, Softmax, batched MatMul, gated hardswish / swish after conv,
/// depthwise and Gemm), on every GPU backend present: Vulkan here, Metal on the macOS runners.
/// </summary>
public class GpuOpTests
{
    public static TheoryData<string> Backends() => new() { "vulkan", "metal" };

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void AvgPool_MatchesCpu(string backend)
    {
        // c % 4 == 0 (vec4 kernel) and odd c, batch 2, padding with / without count_include_pad
        foreach ((int c, int k, int s, int pad, int inc) in new[] { (8, 2, 2, 0, 0), (6, 3, 1, 1, 0), (12, 3, 2, 1, 1), (5, 2, 1, 0, 0) })
        {
            byte[] m = new OnnxBuilder().Input("x", 2, c, 9, 14)
                .Node("AveragePool", ["x"], ["y"], ("kernel_shape", new[] { k, k }), ("strides", new[] { s, s }),
                    ("pads", new[] { pad, pad, pad, pad }), ("count_include_pad", inc))
                .Output("y").Build();
            Check(backend, OnnxModel.Load(m), [2, c, 9, 14], $"avgpool c{c} k{k} s{s} p{pad} inc{inc}");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void SoftmaxMatMul_MatchesCpu(string backend)
    {
        // attention-shaped: softmax(x · W) · V with W broadcast over the batch, V batched
        var rng = new Random(5);
        byte[] m = new OnnxBuilder().Input("x", 2, 3, 10, 8)
            .Init("W", Rand(rng, 8 * 37, 0.5f), 8, 37)
            .Init("V", Rand(rng, 2 * 3 * 37 * 12, 0.5f), 2, 3, 37, 12)
            .Node("MatMul", ["x", "W"], ["s"])
            .Node("Softmax", ["s"], ["a"], ("axis", -1))
            .Node("MatMul", ["a", "V"], ["y"])
            .Output("y").Build();
        Check(backend, OnnxModel.Load(m), [2, 3, 10, 8], "softmax / matmul");
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void GatedActs_MatchCpu(string backend)
    {
        foreach (bool hard in new[] { true, false })
        {
            var rng = new Random(hard ? 11 : 12);
            int c = 16, oc = 24;
            var b = new OnnxBuilder().Input("x", 1, c, 12, 18)
                .Init("W1", Rand(rng, oc * c * 9, 0.3f), oc, c, 3, 3).Init("B1", Rand(rng, oc, 0.5f), oc)
                .Init("W2", Rand(rng, oc * 9, 0.3f), oc, 1, 3, 3).Init("B2", Rand(rng, oc, 0.5f), oc)
                .Init("W3", Rand(rng, oc * 32, 0.3f), oc, 32).Init("B3", Rand(rng, 32, 0.5f), 32)
                .Node("Conv", ["x", "W1", "B1"], ["c1"], ("kernel_shape", new[] { 3, 3 }), ("pads", new[] { 1, 1, 1, 1 }));
            Act(b, "c1", "a1", hard);
            b.Node("Conv", ["a1", "W2", "B2"], ["c2"], ("kernel_shape", new[] { 3, 3 }), ("pads", new[] { 1, 1, 1, 1 }), ("group", oc));
            Act(b, "c2", "a2", hard);
            b.Node("Transpose", ["a2"], ["t"], ("perm", new[] { 0, 2, 3, 1 }))
                .Node("MatMul", ["t", "W3"], ["g"])
                .Node("Add", ["g", "B3"], ["gb"]);
            Act(b, "gb", "y", hard);
            OnnxModel m = OnnxModel.Load(b.Output("y").Build());
            ActivationKind want = hard ? ActivationKind.HardSwish : ActivationKind.Swish;
            // CPU plans fold the gates into the conv / gemm epilogues
            ExecPlan plan = GraphCompiler.Compile(m, [1, c, 12, 18]);
            Assert.Equal(2, plan.Nodes.Count(x => x.P is ConvP cp && cp.Act == want));
            Assert.Contains(plan.Nodes, x => x.P is GemmP gp && gp.Act == want);
            Check(backend, m, [1, c, 12, 18], hard ? "hardswish" : "swish");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void OneByOneSpatial_MatchesCpu(string backend)
    {
        // a 1x1-spatial value may keep an NCHW-ordered physical shape: spatial ops must read
        // [n,h,w,c] from the logical shape (the depthwise conv saw ic = 1 on Metal)
        var rng = new Random(3);
        int c = 24;
        byte[] m = new OnnxBuilder().Input("x", 1, c, 6, 10)
            .Init("Wd", Rand(rng, c, 0.5f), c, 1, 1, 1).Init("Bd", Rand(rng, c, 0.5f), c)
            .Init("Wp", Rand(rng, 16 * c, 0.3f), 16, c, 1, 1)
            .Node("GlobalAveragePool", ["x"], ["g"])
            .Node("Conv", ["g", "Wd", "Bd"], ["d"], ("kernel_shape", new[] { 1, 1 }), ("group", c))
            .Node("Relu", ["d"], ["r"])
            .Node("Conv", ["r", "Wp"], ["y"], ("kernel_shape", new[] { 1, 1 }))
            .Output("y").Build();
        Check(backend, OnnxModel.Load(m), [1, c, 6, 10], "1x1 spatial");
    }

    // x * HardSigmoid(x; 1/6, 0.5) (hardswish) or x * Sigmoid(x) (swish)
    private static void Act(OnnxBuilder b, string x, string y, bool hard)
    {
        if (hard) b.Node("HardSigmoid", [x], [x + "_g"], ("alpha", 1f / 6f), ("beta", 0.5f));
        else b.Node("Sigmoid", [x], [x + "_g"]);
        b.Node("Mul", [x, x + "_g"], [y]);
    }

    private static void Check(string backend, OnnxModel m, int[] shape, string what)
    {
        Func<float[], float[]>? gpu = Gpu(backend, m, shape);
        Skip.If(gpu is null, $"no {backend} device");
        using var cpu = new InferenceSession(GraphCompiler.Compile(m, shape));
        float[] input = Rand(new Random(shape.Sum()), cpu.Input.Length, 2f);
        input.CopyTo(cpu.Input);
        cpu.Run();
        float[] want = cpu.Output.ToArray();
        float[] got = gpu!(input);
        Assert.Equal(want.Length, got.Length);
        float tol = 0.01f * want.Max(Math.Abs) + 0.02f;   // fp16 storage
        float maxd = 0;
        int at = 0;
        for (int i = 0; i < got.Length; i++)
        {
            float d = Math.Abs(got[i] - want[i]);
            if (!(d <= maxd)) { maxd = d; at = i; }
        }
        Assert.True(maxd <= tol, $"{backend} {what}: max |gpu-cpu| {maxd} (tol {tol}) at {at}: gpu={got[at]} cpu={want[at]}");
    }

    private static Func<float[], float[]>? Gpu(string backend, OnnxModel m, int[] shape)
    {
        if (backend == "vulkan")
        {
            VkBackend be;
            try { be = VkBackend.Default; _ = be.DeviceName; }
            catch { return null; }
            return input =>
            {
                using VkInferenceSession s = be.NewSession(m, shape);
                input.CopyTo(s.Input);
                s.Run();
                return s.Output.ToArray();
            };
        }
        if (!OperatingSystem.IsMacOS()) return null;
        MetalBackend mb;
        try { mb = MetalBackend.Default; _ = mb.DeviceName; }
        catch { return null; }
        return input =>
        {
            using MtlInferenceSession s = mb.NewSession(m, shape);
            input.CopyTo(s.Input);
            s.Run();
            return s.Output.ToArray();
        };
    }

    private static float[] Rand(Random rng, int n, float scale) =>
        Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();
}
