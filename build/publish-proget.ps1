# Packs Sdcb.SimdOnnx and pushes it to the ProGet test feed (PROGET_API_KEY, machine env var).
param([string]$Version, [switch]$NoPush,
      [string]$Source = 'https://proget.starworks.cc:88/nuget/test/v3/index.json')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot
$out = Join-Path $PSScriptRoot 'nupkgs'
New-Item -ItemType Directory -Force $out | Out-Null
$args_ = @('pack', "$root/src/Sdcb.SimdOnnx/Sdcb.SimdOnnx.csproj", '-c', 'Release', '-o', $out, '--nologo')
if ($Version) { $args_ += "-p:SdcbSimdOnnxVersion=$Version" }
dotnet @args_
if ($LASTEXITCODE -ne 0) { throw "pack failed: $LASTEXITCODE" }
if ($NoPush) { return }
$key = [Environment]::GetEnvironmentVariable('PROGET_API_KEY', 'Machine')
if (-not $key) { $key = $env:PROGET_API_KEY }
if (-not $key) { throw 'PROGET_API_KEY is not set' }
$pkg = Get-ChildItem $out -Filter "Sdcb.SimdOnnx.$(if ($Version) { $Version } else { '*' }).nupkg" |
    Where-Object { $_.Name -notlike '*.symbols.nupkg' } | Sort-Object LastWriteTime | Select-Object -Last 1
dotnet nuget push $pkg.FullName -s $Source -k $key --skip-duplicate
if ($LASTEXITCODE -ne 0) { throw "push failed: $LASTEXITCODE" }
