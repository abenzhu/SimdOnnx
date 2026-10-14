# Interleaved A/B of the SimdGcdrNet benchmark: main vs SimdOnnx branch, back to back per case.
param(
    [string]$Old = "$PSScriptRoot\..\.baseline-bin\gcdr-main\bench\Sdcb.SimdGcdrNet.Benchmark\bin\Release\net10.0",
    [string]$New = "$PSScriptRoot\..\..\Sdcb.GcdrNet\bench\Sdcb.SimdGcdrNet.Benchmark\bin\Release\net10.0",
    [string]$Out = "$PSScriptRoot\..\docs\ab\gcdr.txt"
)
$ErrorActionPreference = 'Stop'
$env:GCDRNET_TESTDATA = (Resolve-Path "$PSScriptRoot\..\..\Sdcb.GcdrNet\tests\testdata").Path
function Run([string]$dir, [string[]]$a) {
    $exe = (Get-ChildItem $dir -Filter *.exe | Select-Object -First 1).FullName
    $p = Start-Process -FilePath $exe -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\gcdr-ab.txt"
    $peak = 0
    while (-not $p.HasExited) { try { $p.Refresh(); $peak = [math]::Max($peak, $p.PeakWorkingSet64) } catch {}; Start-Sleep -Milliseconds 50 }
    $o = (Get-Content "$env:TEMP\gcdr-ab.txt") -join ' '
    $ms = if ($o -match ':\s*([\d.]+) ms') { [double]$Matches[1] } else { [double]::NaN }
    [pscustomobject]@{ ms = $ms; peak = [math]::Round($peak / 1MB) }
}
$cases = @(
    @{ n = 'cpu 1600x1152 t=1'; t = 1; a = @('--benchcpu', '1152', '1600', '5') },
    @{ n = 'cpu 1600x1152 t=4'; t = 4; a = @('--benchcpu', '1152', '1600', '5') },
    @{ n = 'cpu 1600x1152 t=16'; t = 16; a = @('--benchcpu', '1152', '1600', '5') },
    @{ n = 'vulkan 1600x1152'; t = 0; a = @('--benchvk', '1152', '1600', '20') },
    @{ n = 'cpu 3200x2304 t=16'; t = 16; a = @('--benchcpu', '2304', '3200', '5') },
    @{ n = 'vulkan 3200x2304'; t = 0; a = @('--benchvk', '2304', '3200', '20') }
)
$i = 0
$rows = foreach ($c in $cases) {
    if ($c.t) { $env:GCDRNET_THREADS = "$($c.t)" } else { Remove-Item env:GCDRNET_THREADS -ErrorAction SilentlyContinue }
    $r = @{}
    foreach ($side in $(if ($i++ % 2) { 'new', 'old' } else { 'old', 'new' })) {
        $r[$side] = Run $(if ($side -eq 'old') { $Old } else { $New }) $c.a
    }
    $row = [pscustomobject]@{ case = $c.n; old_ms = $r.old.ms; new_ms = $r.new.ms
        ratio = [math]::Round($r.new.ms / $r.old.ms, 2); peak_mb = "$($r.old.peak)/$($r.new.peak)" }
    Write-Host ($row | Format-Table -HideTableHeaders | Out-String).Trim()
    $row
}
Remove-Item env:GCDRNET_THREADS -ErrorAction SilentlyContinue
$t = $rows | Format-Table -AutoSize | Out-String -Width 200
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$t | Set-Content $Out
$t
