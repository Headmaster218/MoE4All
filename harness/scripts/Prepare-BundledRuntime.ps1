[CmdletBinding()]
param([string]$InstallRoot, [string]$StateRoot)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$harnessRoot = Split-Path -Parent $PSScriptRoot
if (-not $InstallRoot) { $InstallRoot = Split-Path -Parent $harnessRoot }
if (-not $StateRoot) { $StateRoot = Join-Path $harnessRoot 'agent-state' }
$node = Get-ChildItem -LiteralPath (Join-Path $harnessRoot 'runtime') -Directory -Filter 'node-v*-win-x64' |
    ForEach-Object { Join-Path $_.FullName 'node.exe' } |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $node) { throw 'The bundled MoE4All Node runtime is missing.' }
& $node (Join-Path $PSScriptRoot 'Prepare-BundledRuntime.cjs') ([IO.Path]::GetFullPath($InstallRoot)) ([IO.Path]::GetFullPath($StateRoot))
exit $LASTEXITCODE
