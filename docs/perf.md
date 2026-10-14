# Performance: consumers before / after the SimdOnnx switch

Machine: AMD Ryzen 9 5950X (16C/32T, AVX2, no AVX-512), Intel Arc B580 (Vulkan), Windows, .NET 10.
Every row is an **interleaved** A/B: the old (main) and new (feature/simdonnx) binaries run back
to back per case, alternating which goes first. The machine's absolute speed drifted by up to 35%
between sessions, so only compare numbers that come from the same table.

Reproduce: `tools/ab-paddle.ps1` (raw JSON in `docs/ab/paddle/{old,new}`), `tools/ab-gcdr.ps1`.

## SimdPaddleOCR (e2e over the 100-image dataset, median ms per image)

`ratio` = new / old (< 1 is faster). Stage columns are old/new mean ms. `exact` = exactly
recognised lines (old/new).

| case | old | new | ratio | det | cls | rec | peak WS MB | exact |
|---|---:|---:|---:|---|---|---|---|---|
| tiny cpu 1w | 121.5 | 116.8 | **0.96** | 30.9/28.7 | 12.6/14.4 | 65.8/62.5 | 500/528 | 742/742 |
| tiny cpu 4w | 64.8 | 62.0 | **0.96** | 30.7/29.0 | 19.7/24.5 | 82.4/76.9 | 518/559 | 742/742 |
| tiny cpu 16w | 62.0 | 60.6 | **0.98** | 30.6/29.1 | 22.1/31.0 | 142.6/132.8 | 541/593 | 742/742 |
| small cpu 1w | 423.3 | 341.4 | **0.81** | 99.1/55.4 | 13.4/12.9 | 280.4/246.8 | 599/666 | 950/950 |
| small cpu 4w | 212.0 | 148.4 | **0.70** | 99.8/52.0 | 23.2/24.8 | 326.1/294.2 | 650/750 | 950/950 |
| small cpu 16w | 225.0 | 177.7 | **0.79** | 104.0/56.8 | 32.4/24.8 | 636.0/629.1 | 717/885 | 950/950 |
| medium cpu 1w | 1137.6 | 1148.2 | 1.01 | 234.0/214.2 | 11.8/15.8 | 863.7/875.6 | 1105/1189 | 1004/1004 |
| medium cpu 4w | 556.3 | 522.2 | **0.94** | 232.0/203.7 | 22.3/26.2 | 1124.0/1046.3 | 1173/1427 | 1004/1004 |
| medium cpu 16w | 686.9 | 671.8 | **0.98** | 249.0/220.8 | 19.7/54.8 | 2496.3/2548.1 | 1269/1786 | 1004/1004 |
| tiny vulkan 4w | 21.9 | 30.7 | 1.40 | 6.2/8.5 | 1.3/1.4 | 6.4/10.3 | 577/748 | 740/741 |
| small vulkan 4w | 32.7 | 39.4 | 1.20 | 7.7/9.8 | 1.3/1.3 | 16.3/22.9 | 656/933 | 950/955 |
| medium vulkan 4w | 56.6 | 59.2 | 1.05 | 22.4/16.9 | 1.4/1.3 | 26.5/35.6 | 951/1946 | 1003/1003 |

- **CPU**: at parity or faster in every configuration (small det is ~1.9x faster). Accuracy identical.
- **Vulkan**: correct (accuracy equal or better) but **slower than the old hand-fused Paddle GPU
  path** on tiny/small, and peak working set is higher (up to 2x on medium). Medium det is
  faster. Open work, see below.
- **Memory (CPU)**: peak working set is 5–40% higher, mostly at 16 workers on medium (per-shape
  plan cache + per-session arenas). Open work.

## SimdGcdrNet (`Sdcb.SimdGcdrNet.Benchmark`, e2e ms)

| case | old | new | ratio | peak WS MB |
|---|---:|---:|---:|---|
| cpu 1600x1152 t=1 | 1229.9 | 1002.2 | **0.81** | 303/303 |
| cpu 1600x1152 t=4 | 517.6 | 458.4 | **0.89** | 307/311 |
| cpu 1600x1152 t=16 | 265.6 | 248.3 | **0.93** | 339/343 |
| vulkan 1600x1152 | 16.6 | 16.6 | 1.00 | 173/173 |
| cpu 3200x2304 t=16 | 835.3 | 730.6 | **0.87** | 903/909 |
| vulkan 3200x2304 | 65.2 | 64.9 | 1.00 | 205/206 |

## Known gaps / next steps

1. **Paddle Vulkan latency** (tiny +40%, small +20%). The old backend ran PP-OCR through
   hand-written fused shaders (SE block, conv+bn+hardswish chains, fused rec head); the generic
   graph emits more dispatches. Candidates: HardSigmoid epilogue / SE fusion, two-pass GAP
   (one workgroup per 64 channels per image underuses the GPU), batched submission of unit graphs.
2. **Paddle Vulkan memory** (medium peak 951 → 1946 MB): audit per-shape `VkGraphModel` host
   allocations and the shared session buffers' high-water marks.
3. **CPU memory at high worker counts**: per-shape plan LRU (64) × 3 plan kinds, arenas per worker.
4. **Metal**: ported best effort, compiled but not run on Apple hardware in this change.
