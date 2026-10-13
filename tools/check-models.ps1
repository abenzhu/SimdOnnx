# Correctness sweep: every PP-OCR model through SimdOnnx vs the SimdPaddleOCR engine (CPU).
# Fails (exit 1) on max|d| > 1e-3 or any argmax / det-mask disagreement beyond the thresholds.
param(
    [string]$ModelDir = "$PSScriptRoot\..\..\SimdPaddleOCR\models",
    [string]$Engine = 'gcdr',
    [int]$Threads = 4
)
$ErrorActionPreference = 'Stop'
$bench = "$PSScriptRoot\..\bench\Sdcb.SimdOnnx.Bench\bin\Release\net10.0\Sdcb.SimdOnnx.Bench.dll"
$det = "$PSScriptRoot\..\artifacts\spike\det_960x864.f32"
$cases = @(
    @('cls', 'rand:1x3x80x160'), @('cls', 'rand:5x3x80x160'),
    @('rec', 'rand:1x3x48x320'), @('rec', 'rand:3x3x48x200'),
    @('small_rec', 'rand:1x3x48x320'), @('small_rec', 'rand:3x3x48x200'),
    @('medium_rec', 'rand:1x3x48x320'), @('medium_rec', 'rand:2x3x48x96'),
    @('det', $det), @('small_det', $det), @('medium_det', $det)
)
$bad = 0
foreach ($c in $cases) {
    $line = (dotnet $bench "$ModelDir\$($c[0]).onnx" $c[1] "paddle,$Engine" $Threads 1 2>&1 | Select-Object -Last 1) -join ''
    $ok = $line -match 'max\|d\|=([\d.E+-]+).*flips=(\d+) / (\d+)'
    if ($ok) {
        $max = [double]$Matches[1]; $flips = [int]$Matches[2]; $total = [int]$Matches[3]
        $am = if ($line -match 'argmax-diff=(\d+)/') { [int]$Matches[1] } else { 0 }
        $isDet = $c[0] -like '*det'
        $ok = if ($isDet) { $flips -le [Math]::Max(1, $total / 10000) } else { $max -le 1e-3 -and $am -eq 0 }
    }
    if (-not $ok) { $bad++ }
    '{0} {1,-11} {2,-18} {3}' -f ($(if ($ok) { 'ok  ' } else { 'FAIL' })), $c[0], ($c[1] -replace '.*[\\/]', ''), $line.Trim()
}
if ($bad) { Write-Host "$bad case(s) failed"; exit 1 }
