using System.Runtime.CompilerServices;

namespace Sdcb.SimdOnnx;

/// <summary><see cref="MethodImplOptions"/> values missing from netstandard2.0 (same numeric value; ignored by runtimes that predate them).</summary>
internal static class MethodImplCompat
{
    public const MethodImplOptions AggressiveOptimization = (MethodImplOptions)512;
}
