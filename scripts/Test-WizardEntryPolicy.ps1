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

    [IO.File]::WriteAllText($script, @'
Write-Output 'START_FAILURE_FIXTURE'
exit 7
'@, [Text.UTF8Encoding]::new($false))
    Push-Location $fixture
    try {
        $output = ' ' | & $env:ComSpec /d /c Start-INFR-Wizard.cmd 2>&1
        $exitCode = $LASTEXITCODE
    } finally { Pop-Location }
    $failureText = $output | Out-String
    if ($exitCode -ne 7 -or $failureText -notmatch 'START_FAILURE_FIXTURE' -or $failureText -notmatch 'Review the messages above') {
        throw ('Entry failure retention test failed: ' + $failureText)
    }
    [IO.File]::WriteAllText($script,"Write-Output 'DASHBOARD_CLOSED'`nexit 44`n",[Text.UTF8Encoding]::new($false))
    Push-Location $fixture
    try { $closedOutput=& $env:ComSpec /d /c Start-INFR-Wizard.cmd 2>&1; $closedExit=$LASTEXITCODE } finally { Pop-Location }
    $closedText=$closedOutput|Out-String
    if($closedExit -ne 0 -or $closedText -notmatch 'DASHBOARD_CLOSED' -or $closedText -match 'Press any key|launch failed'){throw 'Closing dashboard must exit without another pause'}
    foreach($nativeCode in @(45,46)){
        [IO.File]::WriteAllText($script,("Write-Output 'NATIVE_FINISHED'`nexit "+$nativeCode+"`n"),[Text.UTF8Encoding]::new($false))
        Push-Location $fixture
        try{$nativeOutput=& $env:ComSpec /d /c Start-INFR-Wizard.cmd 2>&1;$nativeExit=$LASTEXITCODE}finally{Pop-Location}
        $expected=if($nativeCode -eq 45){0}else{1}
        if($nativeExit -ne $expected -or ($nativeOutput|Out-String) -match 'Press any key|launch failed'){throw 'Native completion must preserve success/failure without another console pause'}
    }
    Write-Output 'Test-WizardEntryPolicy: downloaded script runs under process-local Bypass; dashboard close exits directly; no elevation.'
} finally {
    $env:PSExecutionPolicyPreference = $previous
    $resolved = [IO.Path]::GetFullPath($fixture)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing cleanup outside the temporary directory.'
    }
    if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture -Recurse -Force }
}
