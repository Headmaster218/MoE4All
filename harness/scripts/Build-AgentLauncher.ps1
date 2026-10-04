[CmdletBinding()]
param([string]$OutputPath)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $OutputPath) { $OutputPath = Join-Path $root 'MoE4All.exe' }
$source = Join-Path $root 'harness\launcher\MoE4All.cs'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { throw 'The .NET Framework C# compiler is unavailable.' }
$nativeArgs = @('/nologo', '/target:exe', '/optimize+', ('/out:' + $OutputPath), $source)
& $compiler @nativeArgs
if ($LASTEXITCODE -ne 0) { throw "Launcher compilation failed with exit code $LASTEXITCODE" }
Write-Host "MoE4All launcher: $OutputPath"
