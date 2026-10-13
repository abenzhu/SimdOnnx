#!/usr/bin/env bash
# Device run (macOS arm64 / Linux arm64). Needs ../SimdPaddleOCR and ../Sdcb.GcdrNet checked out next to this repo.
# usage: bench/Sdcb.SimdOnnx.Bench/run.sh [models dir] [gpu engines, e.g. "gcdr-mtl paddle-mtl"]
set -euo pipefail
cd "$(dirname "$0")/../.."
MODELS=${1:-../SimdPaddleOCR/models}
GPU=${2:-"gcdr-mtl paddle-mtl"}
mkdir -p artifacts/spike
ls artifacts/spike/det_*.f32 >/dev/null 2>&1 || python3 bench/Sdcb.SimdOnnx.Bench/gen_input.py ../Sdcb.GcdrNet/tests/testdata/realdae/02_color.webp artifacts/spike
dotnet build bench/Sdcb.SimdOnnx.Bench -c Release -v q -nologo
EXE="dotnet bench/Sdcb.SimdOnnx.Bench/bin/Release/net10.0/Sdcb.SimdOnnx.Bench.dll"
NCPU=$(getconf _NPROCESSORS_ONLN)
for m in det small_det medium_det; do
  for in in artifacts/spike/det_*.f32; do
    echo "=== $m $(basename "$in")"
    # CPU: each engine in its own process (clean peak memory); correctness in a shared one
    for t in 1 4 "$NCPU"; do
      for e in gcdr paddle; do $EXE "$MODELS/$m.onnx" "$in" $e "$t" 15 | sed -n 2p | sed "s/^/t=$t /"; done
    done
    $EXE "$MODELS/$m.onnx" "$in" paddle,gcdr "$NCPU" 3 | tail -n 1
    for g in $GPU; do $EXE "$MODELS/$m.onnx" "$in" "paddle,$g" "$NCPU" 20 | tail -n 2; done
  done
done
