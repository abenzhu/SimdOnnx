# Decisions

Trade-offs taken while the maintainer was away. Each entry: what, why, how to revisit.

## 2026-10-13 — Core seeded from the GcdrNet engine

The GcdrNet `ExecPlan` design (shape-specialized lowering → fusion → layout → arena, one IR for
CPU / Vulkan / Metal) is the trunk; SimdPaddleOCR's strengths are ported into it.
Spike (`Sdcb.GcdrNet` branch `spike/simdcore-det`) showed PP-OCR det on this engine at CPU parity
(multi-thread), lower peak memory, Vulkan correct.

## 2026-10-13 — First-party consumers use InternalsVisibleTo

SimdPaddleOCR and SimdGcdrNet register extension ops and drive fused GPU pipelines through engine
internals (`ExecPlan`, `VkGraphModel`, …). Designing a stable public extension API up front would
freeze types that are still moving. Instead the core grants `InternalsVisibleTo` to both, and they
reference an exact version (`[x.y.z]`) so a mismatched internal API cannot load.
Revisit when a third-party extension is needed.

## 2026-10-13 — GCDR glue stays in SimdGcdrNet

`bgr2nhwc` / `div6` / `nhwc2bgr` shaders, `Vk/MtlGcdrPipeline` and `PixelOps` are GCDR-specific and
were not copied into the core.

## 2026-10-13 — netstandard2.0: polyfills, not a separate kernel set

The ns2.0 build compiles the same sources. `Compatibility/Intrinsics.cs` stubs the x86/Arm/Wasm
intrinsic types with `IsSupported == false`, so the `Vector<T>` and scalar tiers run;
`Compatibility/Polyfills.cs` supplies the missing BCL members (C# 14 extension members).
Static-abstract interface kernels (`IOp`, `IAct`) became instance interfaces called on
`default(T)` (same codegen on net10). GPU backends are net10-only. Friend test assemblies that
consume both builds reference the engine with an alias (`simdonnx_ns20`) to avoid duplicate
polyfill types.

## 2026-10-13 — Intel Arc B580: avoid the 1x1 cooperative-matrix BN=32 variant

Intermittent `VK_ERROR_DEVICE_LOST` on the B580 (subgroup 16 coopmat path) also reproduces on
the old GcdrNet main. It correlates with the BN=32 1x1 coopmat conv; on subgroup-16 devices 1x1
convs now always use the BN=64 variant (`conv_cm_n64`). Stress runs went from flaky to clean
(det/rec 8/8, GcdrNet 5/5). `GCDRNET_VK_CM1X1=0` disables coopmat 1x1 entirely if it recurs.
A rare DeviceLost in GcdrNet's size-switching Vulkan test (~1 in 10 full-suite runs) remains.

## 2026-10-13 — Ship with Paddle Vulkan slower than the old hand-fused backend

CPU met the "not slower" bar everywhere (see `docs/perf.md`); Paddle Vulkan did not (tiny +40%,
small +20%, medium +5%), and Vulkan peak working set is higher. Rather than port the old
PP-OCR-specific fused shaders into the generic engine under time pressure, the PR ships the
generic path (correct, accuracy equal or better) and lists the gap as the first follow-up.
Revisit: fusion work in `docs/perf.md` "Known gaps"; do not merge the Paddle PR to main until the
maintainer accepts or closes the gap.

## 2026-10-13 — Engine-internal tests/benches move to SimdOnnx

Paddle's kernel tests for removed kernels (`NhwcKernelTests`, conv parts of
`KernelCorrectnessTests`, `GpuSessionShapeTests`, `InferenceSessionRunContractTests`), the
`GpuBench` project and the AndroidBench GPU probes (caps, gemm, layer probe, profiling) tested
engine internals that no longer live in Paddle. They were removed; equivalent coverage lives in
`Sdcb.SimdOnnx.Tests` and `bench/Sdcb.SimdOnnx.Bench` (`gcdr-vkdiff`, `--micro`). Paddle keeps
its own tests for what it still owns (CTC/MatMul argmax head, preprocessing, pipeline).

## 2026-10-13 — Pre-release packages on the ProGet test feed

Consumers pin `Sdcb.SimdOnnx` exactly and restore it from the ProGet test feed through a
`nuget.config` with package-source mapping (only `Sdcb.SimdOnnx` comes from ProGet). Before
merging to main: publish the engine to nuget.org and delete the ProGet source.
