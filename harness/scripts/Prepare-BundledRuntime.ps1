[CmdletBinding()]
param([string]$InstallRoot, [string]$StateRoot)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$harnessRoot = Split-Path -Parent $PSScriptRoot
if (-not $InstallRoot) { $InstallRoot = Split-Path -Parent $harnessRoot }
if (-not $StateRoot) { $StateRoot = Join-Path $InstallRoot 'data' }
$node = Join-Path $harnessRoot 'bootstrap\node.exe'
if (-not (Test-Path -LiteralPath $node -PathType Leaf)) {
    $node = Get-ChildItem -LiteralPath (Join-Path $harnessRoot 'runtime') -Directory -Filter 'node-v*-win-x64' |
        ForEach-Object { Join-Path $_.FullName 'node.exe' } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
}
if (-not $node) { throw 'The bundled MoE4All Node runtime is missing.' }
$workerCount = [Math]::Max(4, [Math]::Min(16, [Environment]::ProcessorCount))
$env:UV_THREADPOOL_SIZE = [string]$workerCount
& $node (Join-Path $PSScriptRoot 'Prepare-BundledRuntime.cjs') ([IO.Path]::GetFullPath($InstallRoot)) ([IO.Path]::GetFullPath($StateRoot))
exit $LASTEXITCODE
