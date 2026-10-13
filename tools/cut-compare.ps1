# Bisect helper: run a model cut at a tensor through ORT and the SimdOnnx engines.
# usage: tools/cut-compare.ps1 -Model m.onnx -In dir\in_NxCxHxW.f32 -Tensor name [-Engines gcdr-ref,gcdr]
param(
    [Parameter(Mandatory)] [string]$Model,
    [Parameter(Mandatory)] [string]$In,
    [Parameter(Mandatory)] [string]$Tensor,
    [string]$Engines = "gcdr-ref,gcdr"
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$dir = Split-Path $In -Parent
$shape = [IO.Path]::GetFileNameWithoutExtension($In).Split('_')[-1]
$cut = Join-Path $dir 'cut.onnx'
$ref = Join-Path $dir 'cut_ref.f32'
python (Join-Path $root 'tools/ort_ref.py') $Model $In $shape $ref $Tensor $cut | Out-Null
$bench = Join-Path $root 'bench/Sdcb.SimdOnnx.Bench/bin/Release/net10.0/Sdcb.SimdOnnx.Bench.dll'
dotnet $bench $cut $In "file:$ref,$Engines" 16 1 2>&1 | Select-String 'vs file|Exception'
