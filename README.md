# Sdcb.SimdOnnx [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdOnnx.svg)](https://www.nuget.org/packages/Sdcb.SimdOnnx) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

<img src="assets/icon.svg" width="96" align="right" alt="icon" />

Pure C# inference engine for a **subset** of ONNX. It is the shared engine behind
[SimdPaddleOCR](https://github.com/sdcb/SimdPaddleOCR) and
[SimdGcdrNet](https://github.com/sdcb/Sdcb.GcdrNet).

> Not affiliated with Microsoft ONNX Runtime. Only the operators those models need are
> implemented; unsupported graphs fail at compile time with the offending node named.

- **Graph compiler**: shape-specialized lowering, fusion (conv + activation / residual / pool,
  GELU, transposed-conv bias, …), physical layout planning (NHWC, zero-copy transposes and
  reshapes) and a static, lifetime-packed memory arena.
- **CPU**: hand-written kernels for AVX-512, AVX2+FMA, AArch64 NEON, WASM SIMD128,
  `Vector<T>` and scalar, selected at runtime.
- **GPU** (net10.0): Vulkan (fp16 storage, cooperative-matrix conv/GEMM where available) and
  Metal (macOS); no native binaries, everything goes through P/Invoke.
- **Extensible**: consumers register model-specific fused operators and GPU kernels without
  touching the core.

Targets `net10.0` and `netstandard2.0` (CPU only, `Vector<T>` and scalar kernels).

## Status

Pre-release. The public API is minimal (`OnnxModel`, `CompiledModel`, `InferenceSession`);
engine internals are shared with the first-party consumers above, which pin an exact version.

## Performance

Before/after numbers for both consumers (interleaved A/B on the same machine) and the open
performance work are in [docs/perf.md](docs/perf.md); trade-offs are logged in
[docs/decisions.md](docs/decisions.md).

## Device verification

Apple Silicon (Metal), arm64 Linux/Android (NEON) and mobile Vulkan were ported best effort
and are checked by CI for correctness only. `bench/Sdcb.SimdOnnx.Bench/run.sh` runs the
correctness + performance matrix on a device; see its header for the expected output.

## License

Apache-2.0
