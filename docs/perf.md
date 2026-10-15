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

Vulkan rows of the first run lost the device (since fixed, see decisions). Current Vulkan
(SimdOnnx `56aa5d5`, Paddle `0c0c77f`), interleaved, 3 rounds, 4 workers (`.rerun/alt.sh`):

| model | old | new | ratio | det graph | rec graph | det pre / post, crop, rec pre (sum) | peak WS MB | exact |
|---|---:|---:|---:|---|---|---|---|---|
| tiny | 17.1–23.4 | 23.5–26.4 | 1.19 | 5.57/4.19 | 6.20/6.22 | 10.5/13.4 | 580/602 | 740/741 |
| small | 31.7–32.6 | 34.7–35.6 | 1.09 | 8.11/6.17 | 16.0/15.5 | 9.8/12.6 | 661/725 | 950/955 |
| medium | 52.3–55.9 | 54.1–54.9 | **1.00** | 21.4/19.1 | 26.6/26.3 | 7.9/8.9 | 953/1169 | 1003/1003 |

(the old engine's tiny median wandered 17–23 ms within this session; earlier sessions measured
small at 1.04 and medium at 0.99.)

- **CPU**: at parity or faster in every configuration (small det is ~2x faster). Accuracy identical.
  Memory: per-key packing and plans (one build shared by concurrent workers) cut medium 16w peak
  1776 → 1388 MB, still above old (1262); medium 4w 1280 vs 1172, small 16w 812 vs 717. The live
  heap after a full GC is ~90 MB above old: same reachable weights, but twice the allocation churn
  per run (956 vs 510 MB) and per-shape statics.
- **Vulkan**: device losses fixed (0 in 50 stress runs, was ~1 in 7). Every GPU stage is as fast
  or faster than the old hand-fused path (det graph 10–25% faster). The remaining gap is the CPU
  stages around them, 25–60% slower in Vulkan mode only (open, see decisions); medium is at
  parity, tiny / small still behind. Accuracy equal or better. Peak working set +4–23%.
- **Metal**: the macOS runners now run the Paddle smoke on Metal (device engaged, detected-line
  counts equal to the CPU) and `GpuOpTests` on M1; not timed against the old engine.

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

Mixed-size corpus (every image a different size, so per-shape compile counts), interleaved:
Vulkan 35.6 vs 36.3 ms per image, peak 1192 vs 1249 MB; CPU 349 vs 415 ms, peak 1145 vs 1150 MB.
PSNR goldens identical.

## Known gaps / next steps

1. **Paddle Vulkan tiny / small**: CPU stages slower next to the new GPU path (see decisions);
   candidates: Paddle's fused operators (attention, SE, concat-resize, ...) as Paddle extensions,
   the old engine's per-vendor kernels (UHD 770 8-lane gemm, Adreno gemm_nc_d / ds, RDNA conv_dw4a).
2. **Memory** above old on Paddle (CPU ~+9–13% peak, Vulkan +4–23%): allocation churn per run,
   per-shape caches.
3. **Device re-creation after a loss** still fails with `ErrorIncompatibleDriver` on the B580
   (no longer triggered).
4. **Metal performance** unmeasured against the old engines (correctness checked in CI).

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
