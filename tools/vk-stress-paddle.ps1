# Repeats the SimdPaddleOCR Vulkan e2e benchmark and counts crashed / failed runs.
#   tools/vk-stress-paddle.ps1 -Runs 15 -Model tiny -Env "GCDRNET_VK_QUEUE=gfx"
param([int]$Runs = 15, [string]$Model = 'tiny', [int]$Workers = 4, [string]$Env = "",
      [string]$Exe = "$PSScriptRoot\..\.baseline-bin\paddle-new\Sdcb.SimdPaddleOCR.Tests.exe",
      [string]$Dataset = "F:\_\code\SimdPaddleOCR\dataset")
$saved = @{}
foreach ($kv in ($Env -split ';' | Where-Object { $_ })) {
    $k, $v = $kv.Split('=', 2)
    $saved[$k] = [Environment]::GetEnvironmentVariable($k)
    [Environment]::SetEnvironmentVariable($k, $v)
}
$bad = 0
$ms = @()
for ($i = 1; $i -le $Runs; $i++) {
    $o = & $Exe --benchmark --count 100 --warmup 2 --engine vulkan --workers $Workers --model $Model --input $Dataset --case-id stress --out "$env:TEMP\vkstress.json" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $bad++
        Write-Host "run $i exit=$LASTEXITCODE $(($o | Select-String 'Fatal|Exception|at Sdcb' | Select-Object -First 2 | ForEach-Object Line) -join ' | ')"
    }
    else { $ms += (Get-Content "$env:TEMP\vkstress.json" -Raw | ConvertFrom-Json).summary.total_ms.median }
}
foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
"env='$Env' model=$Model failed=$bad/$Runs median_ms=$(if ($ms) { '{0:F1}' -f ($ms | Measure-Object -Average).Average })"
