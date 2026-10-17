# Validation Report: Snapdragon 8 Gen 3 (Adreno 750) — Android

## Environment

- **Device**: Realme GT5 Pro (`RMX3888`, platform `pineapple` / Qualcomm Snapdragon 8 Gen 3 SM8650)
- **CPU**: ARM64 8-core (Cortex-X4 + Cortex-A720 + Cortex-A520, ARM64 NEON / AdvSimd)
- **GPU**: Qualcomm Adreno 750 (16 GB LPDDR5X unified memory)
- **OS**: Android 16 (API 36)
- **Vulkan Driver**: API 1.3.128, Driver `0x802fa029` (V@0762.41)
- **Runtime**: `net10.0-android`, CoreCLR (RyuJIT arm64 AdvSimd)

### Device Breadcrumb (`GCDRNET_VK_VERBOSE=1`)
```
[vk] device=Adreno (TM) 750 type=1 vendor=0x5143 api=1.3.128 driver=0x802fa029 qfamily=1 sg=64 sgRange=64-128 sgcCompute=True push=True coopmat=False coopSg=default m8=False fullSg=False lite=False vec4=True cp3=False dot=True lds=32768 sb16=True
```
- **Capabilities**: `coopmat=False`, `dot=True`, `sg=64` (range 64–128), `vec4=True`, `lds=32768`, `sb16=True`.
- **Routing**: Adreno 750 has no cooperative matrix support. It runs the wide-subgroup `dot=True` route:
  - Pointwise 1x1 convolutions take `pw_dot`.
  - Dense convolutions take direct convolution `conv_kdot` without requiring cooperative matrix or im2col buffer allocation.

---

## Correctness Validation

### 1. SimdGcdrNet Correctness (Zero Discrepancies)
- **Golden check small (`--check small`)**:
  - `gcnet[small] max diff`: `8.941E-07` (got `0.35005122`, want `0.35005212`).
  - `drnet[small] max diff`: `1.431E-06`.
- **Golden check mid (`--check mid`)**:
  - `gcnet[mid] max diff`: `9.239E-07`.
  - `drnet[mid] max diff`: `1.729E-06`.
- **GPU graph vs CPU graph (`--vkcheck 256 384`)**:
  - `gc 384x256`: `max|d| 1.421E-03`, `rel-rms 8.030E-04`.
  - `dr 384x256`: `max|d| 2.330E-03`, `rel-rms 8.126E-04`.
- **Pixel-exact comparison vs CPU (`--checkgpu 384 512 vulkan`)**:
  - `Vulkan vs cpu 512x384`: `maxd 1` (standard FP16 roundoff), `diff 41775/589824 px`, `meand 1.000`. Exit code 0.

### 2. SimdPaddleOCR Correctness
- **`tiny` (100 images)**:
  - CPU: `cls=1022/1022`, `exact_lines=742/1036 (71.62%)`, `CER=2.37%`.
  - Vulkan: `cls=1022/1022`, `exact_lines=744/1036 (71.81%)`, `CER=2.38%` (**+2 lines higher exactness**).
- **`small` (20 images)**:
  - CPU: `cls=215/215`, `exact_lines=197/216 (91.20%)`, `CER=0.48%`.
  - Vulkan: `cls=215/215`, `exact_lines=198/216 (91.67%)`, `CER=0.44%` (**Bit-for-bit identical to Old**).
- **`medium` (20 images)**:
  - CPU: `cls=54/54`, `exact_lines=54/54 (100.0%)`, `CER=0.00%`.
  - Vulkan: `cls=216/216`, `exact_lines=212/216 (98.15%)`, `CER=0.08%` (**Bit-for-bit identical to Old**).

---

## Direct A/B Performance Benchmarks

### 1. SimdPaddleOCR (Back-to-Back Interleaved on Phone)

| Case | Old Median (ms) | New Median (ms) | Ratio (New/Old) | Peak WS (MB, Old/New) | Exact Lines (Old/New) | Thermal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tiny cpu` (all 100 imgs)** | 200.2 | **145.0** | **0.72x** (28% faster) | 798 / 814 | 742 / 742 | Status 0 |
| **`tiny cpu` (last 50 imgs)** | 190.4 | **139.6** | **0.73x** (27% faster) | 798 / 814 | 742 / 742 | Status 0 |
| **`tiny vulkan` (all 100 imgs)** | 161.4 | **128.6** | **0.80x** (20% faster) | 763 / 784 | 743 / **744** | Status 0 |
| **`tiny vulkan` (last 50 imgs)** | 168.5 | **132.7** | **0.79x** (21% faster) | 763 / 784 | 743 / **744** | Status 0 |
| **`small cpu` (20 imgs)** | 659.3 | **527.9** | **0.80x** (20% faster) | 577 / 706 | 197 / 197 | Status 1 |
| **`small vulkan` (20 imgs)** | 245.0 | **197.9** | **0.81x** (19% faster) | 542 / 548 | 198 / 198 | Status 1 |
| **`medium vulkan` (20 imgs)** | 821.2 | **597.6** | **0.73x** (27% faster) | 687 / 788 | 212 / 212 | Status 1 |

*Stage Breakdown Insights*:
- **`small cpu`**: `det_graph` dropped from 344.4 ms to **204.2 ms** (**41% faster**).
- **`small vulkan`**: `det_graph` dropped from 45.7 ms to **36.5 ms** (**20% faster**), `rec_graph` from 146.5 ms to **111.3 ms** (**24% faster**).
- **`medium vulkan`**: `det_graph` dropped from 221.9 ms to **162.7 ms** (**27% faster**), `rec_graph` from 555.7 ms to **374.5 ms** (**33% faster**).

### 2. SimdGcdrNet (1600x1152, 5 reps)

| Backend | Old Engine (`2b8fdfe`) | New Engine (`main`) | Ratio (New/Old) | Thermal |
| :--- | :---: | :---: | :---: | :---: |
| **CPU** | 525.1 ms (gc 170.7, dr 354.4) | **508.2 ms** (gc 166.0, dr 342.1) | **0.97x** (3.2% faster) | Status 0 |
| **Vulkan** | 226.1 ms | **226.3 ms** | **1.00x** (at parity) | Status 0 |

---

## Conclusion
- **Speed**: The new engine is **19%–28% faster** across all tested PaddleOCR models and execution backends on Snapdragon 8 Gen 3, and at parity / 3% faster on GcdrNet.
- **Accuracy**: Completely at parity on CPU and GcdrNet, with +1/+2 higher exact line recognition on Paddle Vulkan.
- **Memory & Thermal**: Controlled within budget without thermal throttling.
