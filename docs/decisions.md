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
