# Phase-0 baselines: SimdGcdrNet main benchmark (CPU e2e at 1/16 threads + per-net, Vulkan e2e) and memory.
param([string]$Out = "$PSScriptRoot\..\docs\baseline\gcdr.txt",
      [string]$Proj = "$PSScriptRoot\..\.baseline-bin\gcdr-main\bench\Sdcb.SimdGcdrNet.Benchmark")
$ErrorActionPreference = 'Stop'
$exe = Get-ChildItem "$Proj\bin\Release\net10.0" -Filter *.exe | Select-Object -First 1
$env:GCDRNET_TESTDATA = (Resolve-Path "$PSScriptRoot\..\..\Sdcb.GcdrNet\tests\testdata").Path
$lines = @()
function Run([string]$label, [string[]]$a) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $exe.FullName -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\gcdr-bl.txt"
    $peak = 0
    while (-not $p.HasExited) { try { $p.Refresh(); $peak = [math]::Max($peak, $p.PeakWorkingSet64) } catch {}; Start-Sleep -Milliseconds 50 }
    $o = (Get-Content "$env:TEMP\gcdr-bl.txt") -join ' | '
    $script:lines += "{0,-28} {1}  peakWS={2:N0}MB" -f $label, $o, ($peak / 1MB)
    Write-Host $script:lines[-1]
}
foreach ($size in @(@(1152,1600), @(2304,3200))) {
    $h, $w = $size
    foreach ($t in 1, 4, 16) {
        $env:GCDRNET_THREADS = "$t"
        Run "cpu-e2e ${w}x${h} t=$t" @('--benchcpu', "$h", "$w", '5')
        Run "cpu-nets ${w}x${h} t=$t" @('--bench', "$h", "$w", '5')
    }
    Remove-Item env:GCDRNET_THREADS
    Run "vulkan-e2e ${w}x${h}" @('--benchvk', "$h", "$w", '10')
}
$lines | Set-Content $Out
