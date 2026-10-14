# Loops the GcdrNet Vulkan tests (optionally several test processes at once to raise GPU
# contention) and counts runs with a failure (on the B580 these are device losses).
#   tools/vk-stress-gcdr.ps1 -Runs 40 -Parallel 3 -Env "GCDRNET_VK_QUEUE=gfx"
param([int]$Runs = 40, [int]$Parallel = 1, [string]$Env = "")
$ErrorActionPreference = 'Continue'
$saved = @{}
foreach ($kv in ($Env -split ';' | Where-Object { $_ })) {
    $k, $v = $kv.Split('=', 2)
    $saved[$k] = [Environment]::GetEnvironmentVariable($k)
    [Environment]::SetEnvironmentVariable($k, $v)
}
$repo = (Resolve-Path "$PSScriptRoot\..\..\Sdcb.GcdrNet").Path
$fail = 0
for ($i = 1; $i -le $Runs; $i++) {
    $procs = for ($j = 0; $j -lt $Parallel; $j++) {
        Start-Process dotnet -ArgumentList @('test', "$repo\tests\Sdcb.SimdGcdrNet.Tests", '-c', 'Release', '--no-build',
            '--filter', 'FullyQualifiedName~Vulkan', '--logger', 'console;verbosity=normal') `
            -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\vkstress-$j.txt" -RedirectStandardError "$env:TEMP\vkstress-$j.err"
    }
    $procs | Wait-Process
    for ($j = 0; $j -lt $Parallel; $j++) {
        $t = Select-String -Path "$env:TEMP\vkstress-$j.txt" -Pattern '\[FAIL\]' | Select-Object -First 1
        if ($t) {
            $fail++
            $why = Select-String -Path "$env:TEMP\vkstress-$j.txt" -Pattern 'Exception :' | Select-Object -First 1
            Write-Host "run $i.$j : $($t.Line.Trim()) | $($why.Line.Trim())"
        }
    }
}
foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
"env='$Env' parallel=$Parallel fails=$fail/$($Runs * $Parallel)"
