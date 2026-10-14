[CmdletBinding()]
param(
    [string]$OutputPath,
    [switch]$Console
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $OutputPath) { $OutputPath = Join-Path $root 'MoE4All.exe' }
$source = Join-Path $root 'harness\launcher\MoE4All.cs'
$vendor = Join-Path $root 'harness\launcher\vendor\webview2'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { throw 'The .NET Framework C# compiler is unavailable.' }
$core = Join-Path $vendor 'Microsoft.Web.WebView2.Core.dll'
$forms = Join-Path $vendor 'Microsoft.Web.WebView2.WinForms.dll'
foreach ($dependency in @($core, $forms, (Join-Path $vendor 'WebView2Loader.dll'))) {
    if (-not (Test-Path -LiteralPath $dependency -PathType Leaf)) { throw "WebView2 launcher dependency is missing: $dependency" }
}
$target = if ($Console) { 'exe' } else { 'winexe' }
$nativeArgs = @(
    '/nologo', ("/target:$target"), '/platform:x64', '/optimize+',
    '/reference:System.dll', '/reference:System.Core.dll', '/reference:System.Drawing.dll', '/reference:System.Windows.Forms.dll',
    ("/reference:$core"), ("/reference:$forms"),
    ('/out:' + $OutputPath), $source
)
& $compiler @nativeArgs
if ($LASTEXITCODE -ne 0) { throw "Launcher compilation failed with exit code $LASTEXITCODE" }
$destination = Split-Path -Parent ([IO.Path]::GetFullPath($OutputPath))
foreach ($name in @(
    'Microsoft.Web.WebView2.Core.dll',
    'Microsoft.Web.WebView2.WinForms.dll',
    'WebView2Loader.dll',
    'Microsoft.Web.WebView2.LICENSE.txt'
)) {
    Copy-Item -LiteralPath (Join-Path $vendor $name) -Destination (Join-Path $destination $name) -Force
}
Write-Host "MoE4All launcher: $OutputPath"
