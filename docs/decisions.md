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
*Update 2026-10-14:* most of that flakiness was test interference, see below.

## 2026-10-13 — Ship with Paddle Vulkan slower than the old hand-fused backend

*Update 2026-10-14:* the gap is now 4–9% in the last interleaved run (noisy: 0.97–1.43 across
runs), memory +15–23% instead of +100%; see `docs/perf.md`. The remaining gap is per-new-shape
setup, not kernels.

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

## 2026-10-14 — GPU consts pooled per model, not per identical const list

The const store used to be shared only between graphs whose const lists matched exactly; the
CTC/logits graphs (truncated outputs) of every rec shape therefore re-packed and re-uploaded the
whole model (~170 MB of allocations each, medium rec), which was the "Vulkan memory doubled".
Now each (device, model, precision) has one pool of packed entries keyed by source array +
packing kind (content-compared on a reference miss); a graph uploads only missing entries as a
new chunk, and const binds address the chunk buffer directly (`VkSlot.Direct`). Metal mirrors it.

## 2026-10-14 — Device loss: recover instead of failing forever

`VK_ERROR_DEVICE_LOST` raises `VkDeviceLostException` and marks the device lost;
`VkBackend.Default` then creates a new device on next use. Paddle finishes the failing call on the
CPU and goes back to the GPU (permanent CPU fallback only after 3 losses per session); GcdrNet
retries the image once on a fresh pipeline pool. Objects of the lost device are abandoned, not
destroyed. Tests for this are opt-in (`SIMDONNX_TEST_DEVICELOSS=1`) because they disturb every
other Vulkan test in the process.

## 2026-10-14 — B580 flakiness: what was found

- GcdrNet's Vulkan tests failed ~1 in 4–10 full runs ("size-switch differs from a fresh
  worker", earlier also DeviceLost). Two tests changed process-wide state while others ran in
  parallel: `NoFuse_MatchesFused` set `GCDRNET_NOFUSE` (other tests then compiled unfused graphs)
  and the ConvDot route test flipped `ConvDot` on the shared device. Fixed with
  `CompileOptions.Fuse` and a private device for the ConvDot test: 0 failures in 25 + 10 runs.
- Validation layers (core, thread safety) on a 4-worker Paddle Vulkan run: clean. GPU-assisted
  validation on the GcdrNet Vulkan tests: no out-of-bounds access; it reports shared-memory races in
  `conv_cm` between `coopMatStore` and the epilogue loads of the same subgroup, separated by
  `subgroupMemoryBarrierShared(); subgroupBarrier()` — taken as false positives of GPU-AV's race
  model for cooperative-matrix ops, not changed.
- Paddle Vulkan e2e stress (tiny, 4 workers): 0 crashes in 46 runs; an access violation inside the driver's
  `vkQueueSubmit` was seen 3 times in one cluster (while VS Code was installing an update) and
  once a run silently fell back to the CPU; neither reproduced afterwards. Fallbacks now log one
  line to stderr. `GCDRNET_VK_QUEUE=gfx` (graphics queue instead of async compute) was tried:
  no stability difference, 2x slower, so it stays a debug knob.
