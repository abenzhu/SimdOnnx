# CPU A/B matrix: SimdPaddleOCR engine vs SimdOnnx per model / input / thread count
# (each engine in its own process). Prints "model input threads paddle_ms simdonnx_ms ratio".
# usage: tools/cpu-matrix.ps1 [-Models det,small_det] [-Inputs rand:1x3x48x320,...] [-Threads 1,4,16] [-Iters 15]
param(
    [string[]]$Models = @('det', 'small_det', 'medium_det'),
    [string[]]$Inputs = @("$PSScriptRoot\..\artifacts\spike\det_960x864.f32"),
    [int[]]$Threads = @(1, 4, 16),
    [int]$Iters = 15,
    [string]$ModelDir = "$PSScriptRoot\..\..\SimdPaddleOCR\models",
    [string]$Engine = 'gcdr'
)
$ErrorActionPreference = 'Stop'
$bench = "$PSScriptRoot\..\bench\Sdcb.SimdOnnx.Bench\bin\Release\net10.0\Sdcb.SimdOnnx.Bench.dll"
function Median([string]$line) { if ($line -match 'median=\s*([\d.]+)ms') { [double]$Matches[1] } else { [double]::NaN } }
foreach ($m in $Models) {
    foreach ($in in $Inputs) {
        foreach ($t in $Threads) {
            $p = Median ((dotnet $bench "$ModelDir\$m.onnx" $in paddle $t $Iters) | Select-Object -Index 1)
            $g = Median ((dotnet $bench "$ModelDir\$m.onnx" $in $Engine $t $Iters) | Select-Object -Index 1)
            $label = if ($in -like 'rand:*') { $in.Substring(5) } else { [IO.Path]::GetFileNameWithoutExtension($in) }
            '{0,-12} {1,-14} t={2,-3} paddle={3,8:F2} simdonnx={4,8:F2} ratio={5,5:F2}' -f $m, $label, $t, $p, $g, ($g / $p)
        }
    }
}
