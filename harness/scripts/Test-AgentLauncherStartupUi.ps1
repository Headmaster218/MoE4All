[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$tempRoot = [IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetTempPath()) ('moe4all-startup-ui-' + [guid]::NewGuid().ToString('N'))))
$tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
if (-not $tempRoot.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unexpected startup UI test directory: $tempRoot"
}

try {
    [IO.Directory]::CreateDirectory($tempRoot) | Out-Null
    $launcher = Join-Path $tempRoot 'MoE4All.exe'
    & (Join-Path $PSScriptRoot 'Build-AgentLauncher.ps1') -OutputPath $launcher -Console
    & $launcher '--startup-ui-test'
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Startup UI layout test failed with exit code $exitCode"
    }
    Write-Host 'Startup UI test passed: responsive layout at compact, standard and large window sizes.'
} finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
