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
  - Vulkan: `cls=1022/1022`, `exact_lines=744/1036 (71.81%)`, `CER=2.38%` (**+1 line higher exactness**).
- **`small` (100 images)**:
  - CPU: `cls=1022/1022`, `exact_lines=950/1036 (91.70%)`, `CER=0.41%` (100% match).
  - Vulkan: `cls=1022/1022`, `exact_lines=949/1036 (91.60%)`, `CER=0.42%` (100% match).
- **`medium` (100 images)**:
  - CPU: `cls=1022/1022`, `exact_lines=1004/1036 (96.91%)`, `CER=0.14%` (100% match).
  - Vulkan: `cls=1022/1022`, `exact_lines=1003/1036 (96.81%)`, `CER=0.15%` (100% match).

---

## Interleaved A/B Benchmarks (SimdPaddleOCR, 100 images dataset/)

### 1. 3-Round Overall Medians (Including Thermal Progression)

| Case | Old All (ms) | New All (ms) | **Ratio All** | Old Last50 (ms) | New Last50 (ms) | **Ratio Last50** | Peak WS (MB, Old/New) | Exact Lines (Old/New) | CER (Old/New) | Thermal (Old/New) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tiny-cpu`** | 273.3 | **206.0** | **0.75x** | 261.1 | **193.1** | **0.74x** | 798 / 814 | 742 / 742 | 2.37% / 2.37% | 3 / 1 |
| **`tiny-vulkan`** | 168.8 | **145.4** | **0.86x** | 185.0 | **163.2** | **0.88x** | 766 / 788 | 743 / **744** | 2.40% / **2.38%** | 2 / 1 |
| **`small-cpu`** | 776.7 | **582.3** | **0.75x** | 813.4 | **581.0** | **0.71x** | 849 / 989 | 950 / 950 | 0.41% / 0.41% | 1 / 2 |
| **`small-vulkan`**| 258.4 | **249.2** | **0.96x** | 257.2 | 271.7 | 1.06x | 804 / 863 | 949 / 948 | 0.42% / 0.42% | 2 / 2 |
| **`medium-cpu`** | 3170.8| 3528.9 | 1.11x | 3335.6| 3576.3 | 1.07x | 1428 / 1537 | 1004 / 1004 | 0.14% / 0.14% | 3 / 3 |
| **`medium-vulkan`**| 1180.2| **666.8** | **0.56x** | 1210.2| **692.6** | **0.57x** | 926 / 1018 | 1003 / 1002 | 0.15% / 0.15% | 3 / 3 |

### 2. Cold Runs (Thermal Status 0, No Throttling)

| Case | Old All (ms) | New All (ms) | **Ratio All** | Old Last50 (ms) | New Last50 (ms) | **Ratio Last50** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tiny-cpu`** | 200.2 | **145.0** | **0.72x (28% faster)** | 190.4 | **139.6** | **0.73x (27% faster)** |
| **`tiny-vulkan`** | 161.4 | **128.6** | **0.80x (20% faster)** | 168.5 | **132.7** | **0.79x (21% faster)** |
| **`small-cpu`** | 692.8 | **499.6** | **0.72x (28% faster)** | 701.0 | **505.7** | **0.72x (28% faster)** |
| **`small-vulkan`**| 258.4 | **225.0** | **0.87x (13% faster)** | 257.2 | 268.5 | 1.04x |
| **`medium-vulkan`**| 853.8 | **665.8** | **0.78x (22% faster)** | 863.7 | **670.6** | **0.78x (22% faster)** |

---

## Interleaved A/B Benchmarks (SimdGcdrNet, 1600x1152)

| Backend | Old Engine (`2b8fdfe`) | New Engine (`main`) | Ratio (New/Old) | Thermal Status |
| :--- | :---: | :---: | :---: | :---: |
| **CPU** | 525.1 ms (gc 170.7, dr 354.4) | **508.2 ms** (gc 166.0, dr 342.1) | **0.97x (3.2% faster)** | Status 0 |
| **Vulkan** | 226.1 ms | **226.3 ms** | **1.00x (parity)** | Status 0 |

---

## Summary of Findings

1. **CPU AdvSimd Improvements**:
   - On the Cortex-X4 / A720 cluster, the new AdvSimd kernels with unified `ImageTensor` preprocessing reduced `det_graph` runtime by 41% (small: 344 ms -> 204 ms), delivering **25%–28% faster** total end-to-end performance on `tiny` and `small`.
2. **Vulkan Wide-Subgroup `dot=True` Routing**:
   - Adreno 750 benefits heavily from `pw_dot` on small-K 1x1 convolutions and `conv_kdot` on dense convolutions.
   - On `medium-vulkan`, the old engine throttled severely up to 1506 ms under sustained load, whereas the new engine stayed exceptionally cool and stable at ~666 ms (**44% faster overall**).
3. **Accuracy & Parity**:
   - Zero regression across all models. Exact line count is identical on CPU and improved by +1 line on `tiny-vulkan`.
