using System.Reflection;

namespace Sdcb.SimdOnnx.Backends.Vulkan;

/// <summary>Loads SPIR-V blobs embedded next to their .comp sources (core or an extension assembly).</summary>
internal static class VkShaders
{
    private static readonly Assembly s_core = typeof(VkShaders).Assembly;

    public static byte[] Load(string name, Assembly? owner = null)
    {
        Assembly asm = owner ?? s_core;
        using Stream? s = Engine.ShaderResources.Open(asm, name + ".spv")
            ?? throw new InvalidOperationException($"Embedded shader '{name}.spv' not found in {asm.GetName().Name}.");
        byte[] b = new byte[s.Length];
        s.ReadExactly(b);
        return b;
    }
}
