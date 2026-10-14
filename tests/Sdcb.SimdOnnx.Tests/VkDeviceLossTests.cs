using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Onnx;
using Xunit;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>
/// Marks the process-wide Vulkan device lost, which disturbs every other Vulkan test running in the
/// process: opt-in (SIMDONNX_TEST_DEVICELOSS=1), run it on its own (--filter DeviceLoss).
/// </summary>
public class VkDeviceLossTests
{
    [Fact]
    public void DefaultBackend_RecreatesDevice_AfterLoss()
    {
        if (Environment.GetEnvironmentVariable("SIMDONNX_TEST_DEVICELOSS") != "1") return;
        VkBackend before;
        try { before = VkBackend.Default; }
        catch { return; }   // no Vulkan device

        OnnxModel m = OnnxModel.Load(Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx"));
        float[] input = Golden.F32("small", "input");
        int[] shape = Golden.ShapeOf("small", "input");
        float[] want;
        using (var s = before.NewSession(m, shape))
        {
            input.CopyTo(s.Input);
            s.Run();
            want = s.Output.ToArray();

            before.Dev.Lost = true;
            input.CopyTo(s.Input);
            Assert.Throws<VkDeviceLostException>(s.Run);
        }

        VkBackend after = VkBackend.Default;
        Assert.NotSame(before, after);
        Assert.False(after.Dev.Lost);
        using var s2 = after.NewSession(m, shape);
        input.CopyTo(s2.Input);
        s2.Run();
        Assert.Equal(want, s2.Output.ToArray());
    }
}
