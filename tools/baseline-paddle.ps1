# Phase-0 baselines: SimdPaddleOCR main (sharp CPU x 1/4/16 workers, vulkan) over tiny/small/medium.
param([string]$Out = "$PSScriptRoot\..\docs\baseline\paddle", [string]$Exe = "$PSScriptRoot\..\.baseline-bin\paddle\Sdcb.SimdPaddleOCR.Tests.exe",
      [string]$Dataset = "F:\_\code\SimdPaddleOCR\dataset", [string[]]$Models = @('tiny','small','medium'), [int]$Count = 100)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $Out | Out-Null
foreach ($m in $Models) {
    foreach ($cfg in @(@{e='sharp';w=1}, @{e='sharp';w=4}, @{e='sharp';w=16}, @{e='vulkan';w=4})) {
        $id = "$m-$($cfg.e)-$($cfg.w)w"
        $json = Join-Path $Out "$id.json"
        if (Test-Path $json) { continue }
        Write-Host "== $id"
        & $Exe --benchmark --count $Count --warmup 2 --engine $cfg.e --workers $cfg.w --model $m --input $Dataset --case-id $id --out $json
        if ($LASTEXITCODE -ne 0) { throw "$id failed: $LASTEXITCODE" }
    }
}
