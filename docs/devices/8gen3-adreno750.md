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

## Build & Infrastructure Changes

1. **MSBuild Target ReadyToRun Compatibility**:
   - .NET SDK 11.0 preview CrossGen2 toolchain fails on `net10.0-android` (`Target OS 'android' is not supported`).
   - Packaged Android builds with `-p:PublishReadyToRun=false` to use CoreCLR JIT.
2. **Sdcb.SimdPaddleOCR.AndroidBench Project Cleanup**:
   - Removed obsolete `<EmbeddedResource Include="..\Sdcb.SimdPaddleOCR.GpuBench\Shaders\sgsize.spv" />` from `Sdcb.SimdPaddleOCR.AndroidBench.csproj` which was deleted during engine refactoring.
3. **Android SDK & Workloads**:
   - Installed missing Android SDK Platform 36 (`platforms;android-36`) via `-t:InstallAndroidDependencies`.
   - Verified automated packaging into signed standalone APKs (`.baseline-bin/apk/paddle-old.apk`, `paddle-new.apk`, `gcdr-old.apk`, `gcdr-new.apk`).

---

## Measured Real-Device Results

### 1. SimdPaddleOCR (100 images, dataset/)

| Metric / Case | Old Engine (`main`) | New Engine (`feature/simdonnx`) | Ratio (New/Old) | Accuracy / CER | Thermal |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`tiny cpu` (all images median)** | 200.2 ms | **145.0 ms** | **0.72x** (28% faster) | 742/1036 (CER 2.37%) | Status 0 (Cool) |
| **`tiny cpu` (last 50 images median)** | 190.4 ms | **139.6 ms** | **0.73x** (27% faster) | 742/1036 (CER 2.37%) | Status 0 (Cool) |
| **`tiny cpu` peak WS** | 798 MB | 814 MB | 1.02x | - | - |
| **`tiny vulkan` (all images median)** | 161.4 ms | **128.6 ms** | **0.80x** (20% faster) | 744/1036 (CER 2.38%) | Status 0 (Cool) |
| **`tiny vulkan` (last 50 images median)** | 168.5 ms | **132.7 ms** | **0.79x** (21% faster) | 744/1036 (CER 2.38%) | Status 0 (Cool) |
| **`tiny vulkan` peak WS** | 763 MB | 784 MB | 1.03x | - | - |

*Key Findings for SimdPaddleOCR*:
- **CPU**: The new AdvSimd kernels in SimdOnnx combined with unified `ImageTensor` preprocessing reduced `det_graph` from 121.9 ms to 68.1 ms and total median from 200.2 ms to 145.0 ms (**28% faster**).
- **Vulkan**: The direct `pw_dot` route and vec4 elementwise shaders reduced `rec_graph` from 64.9 ms to 45.7 ms and e2e wall time from 161.4 ms to 128.6 ms (**20% faster**).
- **Accuracy**: New engine matched exact lines on CPU (742/1036) and achieved **+1 higher exact lines on Vulkan** (744 vs 743), with CER improving from 2.40% to 2.38%.
- **Thermal**: Completed at thermal status 0 without throttling.

### 2. SimdGcdrNet (1600x1152)

- **CPU**: 512.3 ms e2e (gc 194.2 ms, dr 318.1 ms).
- **Vulkan**: 226.0 ms e2e (including buffer transfer and submission).
