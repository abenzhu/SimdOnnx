# Performance: consumers before / after the SimdOnnx switch

Machine: AMD Ryzen 9 5950X (16C/32T, AVX2, no AVX-512), Intel Arc B580 (Vulkan), Windows, .NET 10.
Every row is an **interleaved** A/B: the old (main) and new (feature/simdonnx) binaries run back
to back per case, alternating which goes first. The machine's absolute speed drifted by up to 35%
between sessions, so only compare numbers that come from the same table.

Reproduce: `tools/ab-paddle.ps1` (raw JSON in `docs/ab/paddle/{old,new}`), `tools/ab-gcdr.ps1`.

Last run 2026-10-15: SimdOnnx `fb775f4` (consumers built with `UseLocalSimdOnnx=true`);
new = Paddle `feature/simdonnx` `ec9645d`, GcdrNet `feature/simdonnx` `a2a0bc9` + `main` 1.0.0
merged; old = Paddle `main` `33ff8a8`, GcdrNet `main` `2b8fdfe`.

## SimdPaddleOCR (e2e over the 100-image dataset, median ms per image)

`ratio` = new / old (< 1 is faster). Stage columns are old/new mean ms. `exact` = exactly
recognised lines (old/new).

| case | old | new | ratio | det | cls | rec | peak WS MB | exact |
|---|---:|---:|---:|---|---|---|---|---|
| tiny cpu 1w | 84.5 | 82.4 | **0.98** | 22.6/20.5 | 10.9/12.1 | 42.8/41.4 | 501/533 | 742/742 |
| tiny cpu 4w | 50.5 | 46.6 | **0.92** | 23.8/22.2 | 13.0/17.5 | 60.8/60.9 | 520/563 | 742/742 |
| tiny cpu 16w | 47.3 | 46.5 | **0.98** | 23.4/21.8 | 14.3/21.0 | 112.2/97.5 | 539/596 | 742/742 |
| small cpu 1w | 298.5 | 230.9 | **0.77** | 80.1/36.6 | 12.8/12.9 | 182.2/161.5 | 603/666 | 950/950 |
| small cpu 4w | 202.5 | 141.6 | **0.70** | 91.1/46.2 | 23.3/22.8 | 311.5/279.7 | 650/760 | 950/950 |
| small cpu 16w | 214.2 | 158.6 | **0.74** | 98.9/48.2 | 33.9/24.9 | 640.5/578.9 | 719/872 | 950/950 |
| medium cpu 1w | 1084.7 | 1094.6 | 1.01 | 218.3/201.5 | 11.0/14.7 | 827.4/845.2 | 1105/1190 | 1004/1004 |
| medium cpu 4w | 532.1 | 526.3 | **0.99** | 226.3/207.7 | 22.2/25.8 | 1051.5/1061.7 | 1172/1446 | 1004/1004 |
| medium cpu 16w | 652.3 | 654.9 | 1.00 | 231.5/213.2 | 19.4/55.9 | 2368.7/2483.8 | 1269/1776 | 1004/1004 |
| tiny vulkan 4w | 19.2 | 22.7 | 1.18 | 5.6/4.8 | 1.3/1.2 | 4.9/5.9 | 580/668 | 740/741 |
| small vulkan 4w | 26.2 | 175.5 | 6.69 † | 7.5/51.2 | 1.2/25.0 | 13.0/289.5 | 660/1121 | 950/944 |
| medium vulkan 4w | 53.8 | 58.4 | 1.09 | 22.2/16.8 | 1.3/1.2 | 24.9/31.4 | 955/1177 | 1003/1003 |

† The new binary lost the Vulkan device in this run (see below) and ran on the CPU from then on.
Clean Vulkan reruns, interleaved, 3 rounds (`docs/ab/paddle-vk/summary.txt`, median ms):

| model | old | new | ratio |
|---|---:|---:|---:|
| tiny | 16.9–18.3 | 21.7–22.8 | 1.19–1.33 |
| small | 26.4–28.4 | 37.7–41.4 | 1.36–1.50 |
| medium | 53.8–54.5 | 58.5–60.8 | 1.07–1.13 |

- **CPU**: at parity or faster in every configuration (small det is ~2x faster). Accuracy identical.
- **Vulkan**: correct (accuracy equal or better) but **slower than the old hand-fused Paddle GPU
  path**: tiny +19–33%, small +36–50%, medium +7–13% in the 2026-10-15 runs (the 2026-10-14
  run had shown 4–9%; it does not reproduce). Stage means (det/cls/rec) are close, medium det is
  faster; the gap is per-new-shape setup (plan compile + graph + session, ~3–8 ms each, ~150 new
  shapes per 100 images) and the CPU stages it competes with.
- **Vulkan stability**: 2 device losses (`VK_ERROR_DEVICE_LOST` in `vkQueueSubmit`) in 13 new
  4-worker runs (tiny once, small once), 0 in 12 old runs. Recovery did not bring the GPU back:
  after the loss, re-creating the device failed repeatedly (5 log lines) with `vkCreateInstance:
  ErrorIncompatibleDriver`, so the rest of the run fell back to the CPU (results still correct).
- **Memory**: Vulkan peak working set +15–23%; CPU +5–40%, mostly at 16 workers on medium
  (per-shape plan cache + per-session arenas).

## SimdGcdrNet (`Sdcb.SimdGcdrNet.Benchmark`, e2e ms)

| case | old | new | ratio | peak WS MB |
|---|---:|---:|---:|---|
| cpu 1600x1152 t=1 | 1091.0 | 893.3 | **0.82** | 307/304 |
| cpu 1600x1152 t=4 | 323.7 | 292.2 | **0.90** | 308/312 |
| cpu 1600x1152 t=16 | 208.1 | 163.3 | **0.78** | 341/344 |
| vulkan 1600x1152 | 16.6 | 16.5 | 0.99 | 175/176 |
| cpu 3200x2304 t=16 | 676.7 | 614.7 | **0.91** | 905/904 |
| vulkan 3200x2304 | 65.1 | 65.0 | 1.00 | 205/208 |

CPU 10–22% faster, Vulkan at parity, memory unchanged; matches the previous run.

## Known gaps / next steps

1. **Vulkan device loss on the B580** (Paddle, 2 of 13 runs on 2026-10-15) and device re-creation
   failing afterwards (`ErrorIncompatibleDriver`), which turns a transient loss into a CPU run.
2. **Per-shape setup cost** on GPU (the old backend derived per-shape schedules from one prebuilt
   graph): faster fusion passes (several are O(nodes²) with LINQ; `FuseConvResidual` done), fewer
   allocations per compile (~1 MB), or a shape-polymorphic plan.
3. **CPU memory at high worker counts**: per-shape plan LRU (64) × 3 plan kinds, arenas per worker.
4. **Metal**: ported best effort (incl. the const pool), compiled but not run on Apple hardware.

## What closed most of the Vulkan gap (2026-10-14)

| change | effect |
|---|---|
| per-model const pool (entries shared across shapes and output truncations) | medium peak WS 1946 → 1175 MB; no more ~170 MB weight re-pack + upload per CTC/logits graph |
| two-pass GAP (`gap_part`/`gap_sum`, chunked over pixels) | det GPU time 3.9 → 2.7 ms (GAP was a third of it) |
| zero-copy staging (`Input`/`Output` are the mapped staging buffers) | det 736x960 5.6 → 3.8 ms wall |
| LRU graph cache 96 → 256 | rec graphs 154 compiles → 88 (no thrash) |
| Paddle: compile device graphs at `Reshape`/`ReshapeMany` on the thread pool | det/rec setup overlaps preprocessing |
| Paddle: CTC head (MatMul+ArgMax) on the thread pool in growing batches | rec CPU head 1879 → 1412 ms per 100 images (medium) |
| GPU-timestamp profiler (`GCDRNET_PROF=1`, VK_QUERY_TYPE_TIMESTAMP was 1 instead of 2) | per-dispatch device times instead of submit round trips |
