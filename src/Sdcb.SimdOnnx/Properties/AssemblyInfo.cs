using System.Runtime.CompilerServices;

// First-party consumers build against engine internals (extension ops, fused pipelines);
// they pin an exact Sdcb.SimdOnnx version, so internal changes cannot skew at runtime.
[assembly: InternalsVisibleTo("Sdcb.SimdPaddleOCR")]
[assembly: InternalsVisibleTo("Sdcb.SimdGcdrNet")]
[assembly: InternalsVisibleTo("Sdcb.SimdOnnx.Tests")]
[assembly: InternalsVisibleTo("Sdcb.SimdOnnx.Bench")]
