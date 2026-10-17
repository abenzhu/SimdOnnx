# Android interleaved A/B benchmark for SimdPaddleOCR (old main vs new SimdOnnx)
param(
    [int]$Rounds = 3,
    [int]$Cool = 60,
    [string[]]$Models = @('tiny', 'small', 'medium'),
    [string[]]$Backends = @('cpu', 'vulkan'),
    [string]$OldApk = "$PSScriptRoot\..\.baseline-bin\apk\paddle-old.apk",
    [string]$NewApk = "$PSScriptRoot\..\.baseline-bin\apk\paddle-new.apk",
    [string]$Adb = "C:\Users\sdfly\AppData\Local\Android\Sdk\platform-tools\adb.exe",
    [string]$AndroidSdk = "C:\Users\sdfly\AppData\Local\Android\Sdk",
    [string]$Jdk = "C:\Program Files\Android\openjdk\jdk-21.0.8",
    [string]$OutDir = "$PSScriptRoot\..\docs\ab\android-paddle"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$runScript = Resolve-Path "$PSScriptRoot\..\..\SimdPaddleOCR\test\Sdcb.SimdPaddleOCR.AndroidBench\run.ps1"

$currentInstalled = ""

function Ensure-Apk([string]$side) {
    if ($script:currentInstalled -eq $side) { return }
    $apk = if ($side -eq 'old') { $OldApk } else { $NewApk }
    Write-Host "== Installing $side APK: $apk"
    & $Adb install -r -g $apk | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "adb install failed for $side" }
    $script:currentInstalled = $side
}

$results = @()

for ($r = 1; $r -le $Rounds; $r++) {
    Write-Host "`n==================== ROUND $r OF $Rounds ===================="
    $sides = if ($r % 2 -eq 1) { @('old', 'new') } else { @('new', 'old') }

    foreach ($m in $Models) {
        foreach ($b in $Backends) {
            foreach ($side in $sides) {
                $id = "paddle-$m-$b-$side-r$r"
                $logLocal = "$OutDir\$id.log"
                $jsonLocal = "$OutDir\$id.json"

                if (Test-Path $logLocal) {
                    Write-Host "== Skipping existing $id"
                } else {
                    Ensure-Apk $side
                    Write-Host "== Cooling down $Cool s before $id..."
                    Start-Sleep -Seconds $Cool

                    $preThermal = (& $Adb shell "dumpsys thermalservice | grep -m 1 'mName=skin'") -join ""
                    Write-Host "[$id] Starting (pre-run thermal: $preThermal)"

                    & $runScript -Bench "--e2e --model $m --backend $b --out $id.json" -Run $id -TimeoutSec 1200 -Adb $Adb -AndroidSdk $AndroidSdk -Jdk $Jdk | Out-Host

                    # Pull result
                    $phoneRemote = "/sdcard/Android/data/com.sdcb.simdocr.bench/files/out"
                    & $Adb pull "$phoneRemote/$id.log" $logLocal | Out-Null
                    & $Adb pull "$phoneRemote/$id.json" $jsonLocal | Out-Null
                }

                if (Test-Path $logLocal) {
                    $logText = Get-Content $logLocal -Raw
                    $allMed = 0.0; $last50Med = 0.0; $wsPeak = 0; $thermal = -1
                    $exact = 0; $cer = 0.0

                    if ($logText -match 'steady: all-med ([\d.]+) last50-med ([\d.]+) ws-peak (\d+)MB thermal=(-?\d+)') {
                        $allMed = [double]$Matches[1]
                        $last50Med = [double]$Matches[2]
                        $wsPeak = [int]$Matches[3]
                        $thermal = [int]$Matches[4]
                    }
                    if ($logText -match 'exact_lines=(\d+)/\d+.*CER=([\d.]+)%') {
                        $exact = [int]$Matches[1]
                        $cer = [double]$Matches[2]
                    }

                    $row = [PSCustomObject]@{
                        Round = $r
                        Model = $m
                        Backend = $b
                        Side = $side
                        AllMed = $allMed
                        Last50Med = $last50Med
                        WsPeak = $wsPeak
                        Thermal = $thermal
                        Exact = $exact
                        CER = $cer
                    }
                    $results += $row
                    Write-Host ("--> {0,-6} {1,-6} {2,-4} r{3}: all={4,6} last50={5,6} ws={6,4}MB th={7} exact={8} cer={9}%" -f `
                        $m, $b, $side, $r, $allMed, $last50Med, $wsPeak, $thermal, $exact, $cer)
                }
            }
        }
    }
}

$results | Export-Csv -Path "$OutDir\summary.csv" -NoTypeInformation
Write-Host "`nDone! Summary saved to $OutDir\summary.csv"
