[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('MoE4All entry policy ' + [guid]::NewGuid().ToString('N'))
$previous = $env:PSExecutionPolicyPreference
try {
    New-Item -ItemType Directory -Path (Join-Path $fixture 'scripts') -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $root 'Start-INFR-Wizard.cmd') -Destination $fixture
    $script = Join-Path $fixture 'scripts\infr-wizard.ps1'
    [IO.File]::WriteAllText($script, @'
if ((Get-ExecutionPolicy) -ne 'Bypass') { throw 'Entry policy was not applied.' }
Write-Output 'ENTRY_POLICY_OK'
exit 42
'@, [Text.UTF8Encoding]::new($false))
    Set-Content -LiteralPath $script -Stream Zone.Identifier -Value "[ZoneTransfer]`r`nZoneId=3" -Encoding ASCII
    $env:PSExecutionPolicyPreference = 'Restricted'
    Push-Location $fixture
    try {
        $output = & $env:ComSpec /d /c Start-INFR-Wizard.cmd 2>&1
        $exitCode = $LASTEXITCODE
    } finally { Pop-Location }
    if ($exitCode -ne 0 -or ($output | Out-String) -notmatch 'ENTRY_POLICY_OK') {
        throw ('Entry policy test failed: ' + ($output | Out-String))
    }
    Write-Output 'Test-WizardEntryPolicy: downloaded script runs under process-local Bypass; no elevation.'
} finally {
    $env:PSExecutionPolicyPreference = $previous
    $resolved = [IO.Path]::GetFullPath($fixture)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing cleanup outside the temporary directory.'
    }
    if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture -Recurse -Force }
}
