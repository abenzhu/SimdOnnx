# Interleaved A/B of the SimdPaddleOCR e2e benchmark: old (main) vs new (SimdOnnx) test exe, run
# back to back per case so both sides see the same machine state. Writes
# docs/ab/paddle/{old,new}/<case>.json and prints a comparison table.
param(
    [string]$Old = "$PSScriptRoot\..\.baseline-bin\paddle\Sdcb.SimdPaddleOCR.Tests.exe",
    [string]$New = "$PSScriptRoot\..\.baseline-bin\paddle-new\Sdcb.SimdPaddleOCR.Tests.exe",
    [string]$Out = "$PSScriptRoot\..\docs\ab\paddle",
    [string]$Dataset = "F:\_\code\SimdPaddleOCR\dataset",
    [string[]]$Models = @('tiny', 'small', 'medium'),
    [string[]]$Configs = @('sharp:1', 'sharp:4', 'sharp:16', 'vulkan:4'),
    [int]$Count = 100
)
$ErrorActionPreference = 'Stop'
foreach ($side in 'old', 'new') { New-Item -ItemType Directory -Force (Join-Path $Out $side) | Out-Null }
$i = 0
foreach ($m in $Models) {
    foreach ($c in $Configs) {
        $engine, $w = $c.Split(':')
        $id = "$m-$engine-${w}w"
        # alternate which side goes first to cancel warm-up/thermal drift
        $order = if ($i++ % 2) { @('new', 'old') } else { @('old', 'new') }
        foreach ($side in $order) {
            $json = Join-Path $Out "$side\$id.json"
            if (Test-Path $json) { continue }
            $exe = if ($side -eq 'old') { $Old } else { $New }
            Write-Host "== $id ($side)"
            & $exe --benchmark --count $Count --warmup 2 --engine $engine --workers $w --model $m --input $Dataset --case-id "$id-$side" --out $json | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "$id $side failed: $LASTEXITCODE" }
        }
    }
}
$rows = foreach ($f in Get-ChildItem (Join-Path $Out 'new') -Filter *.json | Sort-Object Name) {
    $o = Get-Content (Join-Path $Out "old\$($f.Name)") -Raw | ConvertFrom-Json
    $n = Get-Content $f.FullName -Raw | ConvertFrom-Json
    [pscustomobject]@{
        case = $f.BaseName
        old_ms = [math]::Round($o.summary.total_ms.median, 1)
        new_ms = [math]::Round($n.summary.total_ms.median, 1)
        ratio = [math]::Round($n.summary.total_ms.median / $o.summary.total_ms.median, 2)
        det = '{0:F1}/{1:F1}' -f $o.summary.stage_ms_mean.det_graph, $n.summary.stage_ms_mean.det_graph
        cls = '{0:F1}/{1:F1}' -f $o.summary.stage_ms_mean.cls_graph, $n.summary.stage_ms_mean.cls_graph
        rec = '{0:F1}/{1:F1}' -f $o.summary.stage_ms_mean.rec_graph, $n.summary.stage_ms_mean.rec_graph
        peak_mb = '{0:F0}/{1:F0}' -f $o.meta.working_set_mb_peak, $n.meta.working_set_mb_peak
        exact = '{0}/{1}' -f $o.meta.accuracy.exact_lines, $n.meta.accuracy.exact_lines
    }
}
$rows | Format-Table -AutoSize | Out-String -Width 200
