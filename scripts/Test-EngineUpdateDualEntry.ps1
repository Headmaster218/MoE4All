[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9]{8}-[0-9]{6}-[A-Za-z0-9-]+$')][string]$TaskId,
    [string]$WindowLauncherPath = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($PSVersionTable.PSEdition -ne 'Desktop') { throw 'Run this inbox .NET fixture suite with powershell.exe (Windows PowerShell 5.1).' }
$repoRoot = Split-Path -Parent $PSScriptRoot
$workRoot = if ($env:MOE4ALL_WORK_ROOT) { [IO.Path]::GetFullPath($env:MOE4ALL_WORK_ROOT) } else { Join-Path $repoRoot 'local-work' }
$runId = [guid]::NewGuid().ToString('N')
$staging = Join-Path $workRoot "staging\moe4all\$TaskId\ota\$runId"
$evidence = Join-Path $workRoot "evidence\moe4all\$TaskId\ota\$runId"
$temp = Join-Path $workRoot "tmp\moe4all\$TaskId\ota\$runId"
foreach ($directory in @($staging, $evidence, $temp)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$env:TEMP = $temp
$env:TMP = $temp
Write-Output "Staging: $staging"
Write-Output "Evidence: $evidence"
Write-Output "TEMP/TMP: $temp"
$results = [System.Collections.Generic.List[object]]::new()

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Write-Text {
    param([string]$Path, [string]$Text)
    [IO.File]::WriteAllText($Path, $Text, [Text.UTF8Encoding]::new($false))
}

function New-ManagedManifest {
    param([string]$Root, [string]$Version)
    $files = @(Get-ChildItem -LiteralPath $Root -File -Recurse | Where-Object { $_.Name -ne 'install-manifest.json' } | ForEach-Object {
        [ordered]@{
            path = $_.FullName.Substring($Root.Length + 1).Replace('\', '/')
            size = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
    $manifest = [ordered]@{ schema_version = 1; updater_protocol = 1; product = 'moe4all-engine'; version = $Version; tag = "release-$Version"; files = $files }
    Write-Text (Join-Path $Root 'install-manifest.json') ($manifest | ConvertTo-Json -Depth 6)
}

function New-FixturePackage {
    param([string]$Root, [string]$Version, [bool]$HasWindow, [string]$FailureMode = 'ok',
        [string]$WindowName = 'MoE4All Engine.exe')
    New-Item -ItemType Directory -Path (Join-Path $Root 'scripts') -Force | Out-Null
    Copy-Item -LiteralPath $enginePath -Destination (Join-Path $Root 'infr.exe')
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'moe4all-update-common.ps1') -Destination (Join-Path $Root 'scripts\moe4all-update-common.ps1')
    Write-Text (Join-Path $Root 'engine-version.txt') $Version
    Write-Text (Join-Path $Root 'probe-mode.txt') $FailureMode
    Write-Text (Join-Path $Root 'scripts\infr-wizard.ps1') $wizardFixture
    Write-Text (Join-Path $Root 'Start-INFR-Wizard.cmd') "@echo off`r`npowershell.exe -NoLogo -NoProfile -STA -File `"%~dp0scripts\infr-wizard.ps1`" -ClassicWizard %*`r`nexit /b %ERRORLEVEL%`r`n"
    if ($HasWindow) { Copy-Item -LiteralPath $launcherPath -Destination (Join-Path $Root $WindowName) }
    New-ManagedManifest $Root $Version
}

function Get-ProductHashes {
    param([string]$Root)
    $manifest = Get-Content -LiteralPath (Join-Path $Root 'install-manifest.json') -Raw | ConvertFrom-Json
    $hashes = @{}
    foreach ($relative in (@($manifest.files | ForEach-Object { [string]$_.path }) + @('install-manifest.json'))) {
        $hashes[$relative] = (Get-FileHash -LiteralPath (Join-Path $Root $relative) -Algorithm SHA256).Hash
    }
    return $hashes
}

try {
    # Load only the matching function, without running the updater transaction.
    $tokens = $null; $parseErrors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'apply-engine-update.ps1'), [ref]$tokens, [ref]$parseErrors)
    Assert-True ($parseErrors.Count -eq 0) 'Updater parse failed'
    $matcher = $ast.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-WindowLauncherProcess' }, $true)
    . ([scriptblock]::Create($matcher.Extent.Text))
    foreach ($name in @('MoE4All Engine.exe', 'Start-INFR-Wizard.exe')) {
        $root = Join-Path $staging 'process matching'
        Assert-True (Test-WindowLauncherProcess ([pscustomobject]@{ Name = $name; ExecutablePath = (Join-Path $root $name) }) $root) 'Installed launcher did not match'
        Assert-True (-not (Test-WindowLauncherProcess ([pscustomobject]@{ Name = $name; ExecutablePath = (Join-Path $staging $name) }) $root)) 'Unrelated same-name launcher matched'
        Assert-True (-not (Test-WindowLauncherProcess ([pscustomobject]@{ Name = $name; ExecutablePath = $null }) $root)) 'Unknown process path matched'
    }
    Write-Output 'PASS exact launcher process paths (new and legacy)'
    # Only two tiny inbox .NET fixtures are compiled. No Rust, GPU or network.
    $provider = [Microsoft.CSharp.CSharpCodeProvider]::new()
    $compiler = [System.CodeDom.Compiler.CompilerParameters]::new()
    $compiler.GenerateExecutable = $true
    [void]$compiler.ReferencedAssemblies.Add('System.dll')
    $compiler.TempFiles = [System.CodeDom.Compiler.TempFileCollection]::new($temp, $false)
    $enginePath = Join-Path $staging 'fixture-engine.exe'
    $compiler.OutputAssembly = $enginePath
    $engineSource = @'
using System;
using System.IO;
class EngineFixture {
    static int Main(string[] args) {
        Console.WriteLine("infr " + File.ReadAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "engine-version.txt")).Trim());
        return 0;
    }
}
'@
    $compiled = $provider.CompileAssemblyFromSource($compiler, $engineSource)
    if ($compiled.Errors.HasErrors) { throw ($compiled.Errors | Out-String) }
    if ($WindowLauncherPath) {
        $launcherPath = (Resolve-Path -LiteralPath $WindowLauncherPath).Path
    } else {
        $launcherPath = Join-Path $staging 'MoE4All Engine.exe'
        $compiler.OutputAssembly = $launcherPath
        $compiler.CompilerOptions = '/target:winexe'
        $launcherSource = @'
using System;
using System.Diagnostics;
using System.IO;
class WindowEntryFixture {
    static int Main(string[] args) {
        var start = new ProcessStartInfo("powershell.exe");
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        start.Arguments = "-NoLogo -NoProfile -STA -File \"" + Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scripts", "infr-wizard.ps1") + "\" -WindowWizard";
        foreach (var arg in args) start.Arguments += " \"" + arg.Replace("\"", "\\\"") + "\"";
        using (var child = Process.Start(start)) { child.WaitForExit(); return child.ExitCode; }
    }
}
'@
        $compiled = $provider.CompileAssemblyFromSource($compiler, $launcherSource)
        if ($compiled.Errors.HasErrors) { throw ($compiled.Errors | Out-String) }
    }
    $provider.Dispose()
    $wizardFixture = @'
param([switch]$ClassicWizard, [switch]$WindowWizard, [switch]$SkipUpdateCheck,
    [string]$PostUpdateProbePath = '', [string]$ExpectedPostUpdateVersion = '', [switch]$ExitAfterPostUpdateProbe)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$version = [IO.File]::ReadAllText((Join-Path $root 'engine-version.txt')).Trim()
$mode = [IO.File]::ReadAllText((Join-Path $root 'probe-mode.txt')).Trim()
$entry = if ($WindowWizard) { 'window' } elseif ($ClassicWizard) { 'console' } else { 'unknown' }
$update = Join-Path $root '.update'
New-Item -ItemType Directory -Path $update -Force | Out-Null
$parentId = (Get-CimInstance Win32_Process -Filter "ProcessId = $PID").ParentProcessId
$parentPath = (Get-CimInstance Win32_Process -Filter "ProcessId = $parentId").ExecutablePath
$record = @{ version = $version; entry = $entry; launcher = [IO.Path]::GetFileName($parentPath); skip = [bool]$SkipUpdateCheck; probe_only = [bool]$ExitAfterPostUpdateProbe; sta = [Threading.Thread]::CurrentThread.GetApartmentState().ToString() }
[IO.File]::AppendAllText((Join-Path $update 'launches.jsonl'), (($record | ConvertTo-Json -Compress) + "`n"))
if ($mode -eq 'exit-before-probe') { exit 46 }
if ($PostUpdateProbePath) {
    if ($mode -eq 'wrong-probe') { $version = '99.0.0' }
    if ($mode -eq 'wrong-entry') { $entry = if ($entry -eq 'window') { 'console' } else { 'window' } }
    $probe = @{ version = $version }
    if ($mode -ne 'legacy-probe') { $probe.entry = $entry }
    [IO.File]::WriteAllText($PostUpdateProbePath, ($probe | ConvertTo-Json))
}
exit 0
'@

    $cases = @(
        @{ Name = 'rename-legacy-window'; Entry = 'window'; OldWindow = $true; NewWindow = $true; OldName = 'Start-INFR-Wizard.exe'; Failure = 'ok' }
        @{ Name = 'rename-legacy-rollback'; Entry = 'window'; OldWindow = $true; NewWindow = $true; OldName = 'Start-INFR-Wizard.exe'; Failure = 'wrong-probe' }
        @{ Name = 'legacy-target-window'; Entry = 'window'; OldWindow = $true; NewWindow = $true; NewName = 'Start-INFR-Wizard.exe'; Failure = 'ok' }
        @{ Name = 'prefer-new-preserve-unmanaged-legacy'; Entry = 'window'; OldWindow = $true; NewWindow = $true; UnmanagedLegacy = $true; Failure = 'ok' }
        @{ Name = 'default-console'; Entry = ''; OldWindow = $true; NewWindow = $true; Failure = 'ok' }
        @{ Name = 'explicit-console'; Entry = 'console'; OldWindow = $true; NewWindow = $true; Failure = 'ok' }
        @{ Name = 'window'; Entry = 'window'; OldWindow = $true; NewWindow = $true; Failure = 'ok' }
        @{ Name = 'legacy-upgrade-console'; Entry = 'console'; OldWindow = $false; NewWindow = $true; Failure = 'ok' }
        @{ Name = 'legacy-target-fallback'; Entry = 'window'; OldWindow = $true; NewWindow = $false; Failure = 'ok' }
        @{ Name = 'console-probe-rollback'; Entry = 'console'; OldWindow = $true; NewWindow = $true; Failure = 'wrong-probe' }
        @{ Name = 'window-probe-rollback'; Entry = 'window'; OldWindow = $true; NewWindow = $true; Failure = 'wrong-probe' }
        @{ Name = 'window-start-rollback'; Entry = 'window'; OldWindow = $true; NewWindow = $true; Failure = 'exit-before-probe' }
        @{ Name = 'legacy-rollback-fallback'; Entry = 'window'; OldWindow = $false; NewWindow = $true; Failure = 'wrong-probe' }
        @{ Name = 'console-entry-rollback'; Entry = 'console'; OldWindow = $true; NewWindow = $true; Failure = 'wrong-entry' }
        @{ Name = 'window-entry-rollback'; Entry = 'window'; OldWindow = $true; NewWindow = $true; Failure = 'wrong-entry' }
        @{ Name = 'legacy-console-probe'; Entry = 'window'; OldWindow = $true; NewWindow = $false; Failure = 'legacy-probe' }
        @{ Name = 'window-missing-entry-rollback'; Entry = 'window'; OldWindow = $true; NewWindow = $true; Failure = 'legacy-probe'; MustRollback = $true }
    )
    foreach ($case in $cases) {
        # Spaces and non-ASCII exercise the real updater's launch quoting.
        $caseRoot = Join-Path $staging ($case.Name + ' space ' + [char]0x4e2d)
        $install = Join-Path $caseRoot 'installed app'
        $package = Join-Path $caseRoot 'staged package'
        $oldName = if ($case.ContainsKey('OldName')) { $case.OldName } else { 'MoE4All Engine.exe' }
        $newName = if ($case.ContainsKey('NewName')) { $case.NewName } else { 'MoE4All Engine.exe' }
        New-FixturePackage $install '1.0.0' $case.OldWindow 'ok' $oldName
        New-FixturePackage $package '1.0.1' $case.NewWindow $case.Failure $newName
        $originalHashes = Get-ProductHashes $install
        $userFiles = @('infr.toml', 'gui-data\wizard-state.json', 'kv-sessions\keep.bin', 'cache\kv-sessions\keep.bin', 'data\settings.json', 'custom-user-file.txt')
        if ($case.ContainsKey('UnmanagedLegacy')) { $userFiles += 'Start-INFR-Wizard.exe' }
        $userHashes = @{}
        foreach ($relative in $userFiles) {
            $path = Join-Path $install $relative
            New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($path)) -Force | Out-Null
            Write-Text $path ('user-owned:' + $relative)
            $userHashes[$relative] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        }
        $nativeArgs = @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', (Join-Path $PSScriptRoot 'apply-engine-update.ps1'),
            '-InstallRoot', $install, '-StagedPackageRoot', $package, '-ExpectedVersion', '1.0.1', '-RelaunchProbeOnly')
        if ($case.Entry) { $nativeArgs += @('-RelaunchEntry', $case.Entry) }
        # Windows PowerShell reports a child stderr stream as NativeCommandError;
        # capture the expected failing transaction without terminating this suite.
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $output = & powershell.exe @nativeArgs 2>&1 | Out-String
            $exitCode = $LASTEXITCODE
        } finally { $ErrorActionPreference = $savedPreference }
        Write-Text (Join-Path $evidence ($case.Name + '.log')) $output
        $rollback = $case.Failure -notin @('ok', 'legacy-probe') -or ($case.ContainsKey('MustRollback') -and $case.MustRollback)
        Assert-True (($exitCode -eq 0) -eq (-not $rollback)) "$($case.Name): unexpected updater exit $exitCode. $output"
        $result = Get-Content -LiteralPath (Join-Path $install '.update\last-update-result.json') -Raw | ConvertFrom-Json
        $expectedStatus = if ($rollback) { 'error' } else { 'success' }
        Assert-True ($result.status -eq $expectedStatus) "$($case.Name): incorrect update status"
        if ($rollback) {
            foreach ($relative in $originalHashes.Keys) {
                Assert-True ((Get-FileHash -LiteralPath (Join-Path $install $relative) -Algorithm SHA256).Hash -eq $originalHashes[$relative]) "$($case.Name): rollback changed $relative"
            }
            foreach ($name in @('MoE4All Engine.exe', 'Start-INFR-Wizard.exe')) {
                if (-not $originalHashes.ContainsKey($name)) {
                    Assert-True (-not (Test-Path -LiteralPath (Join-Path $install $name))) "Rollback left an added EXE: $name"
                }
            }
        } else {
            . (Join-Path $PSScriptRoot 'moe4all-update-common.ps1')
            [void](Read-Moe4AllInstallManifest -ManifestPath (Join-Path $install 'install-manifest.json') -PackageRoot $install -VerifyFiles)
            if ($case.OldWindow -and (-not $case.NewWindow -or $oldName -ne $newName)) {
                Assert-True (-not (Test-Path -LiteralPath (Join-Path $install $oldName))) 'Obsolete managed launcher was not removed'
            }
        }
        foreach ($relative in $userHashes.Keys) {
            Assert-True ((Get-FileHash -LiteralPath (Join-Path $install $relative) -Algorithm SHA256).Hash -eq $userHashes[$relative]) "$($case.Name): modified user data $relative"
        }
        $launchLog = Join-Path $install '.update\launches.jsonl'
        $expectedLaunchCount = if ($rollback) { 2 } else { 1 }
        $deadline = [DateTime]::UtcNow.AddSeconds(10)
        do {
            $launches = @(if (Test-Path -LiteralPath $launchLog) { Get-Content -LiteralPath $launchLog | ForEach-Object { $_ | ConvertFrom-Json } })
            if ($launches.Count -ge $expectedLaunchCount) { break }
            Start-Sleep -Milliseconds 100
        } while ([DateTime]::UtcNow -lt $deadline)
        Assert-True ($launches.Count -eq $expectedLaunchCount) "$($case.Name): incorrect launch count $($launches.Count)"
        $hasWindow = if ($rollback) { $case.OldWindow } else { $case.NewWindow }
        $expectedEntry = if ($case.Entry -eq 'window' -and $hasWindow) { 'window' } else { 'console' }
        $lastLaunch = $launches[-1]
        Assert-True ($lastLaunch.entry -eq $expectedEntry) "$($case.Name): expected $expectedEntry, got $($lastLaunch.entry)"
        if ($expectedEntry -eq 'window') {
            $expectedName = if ($rollback) { $oldName } else { $newName }
            Assert-True ($lastLaunch.launcher -eq $expectedName) "$($case.Name): reopened wrong EXE name $($lastLaunch.launcher)"
        }
        Assert-True ($lastLaunch.version -eq $(if ($rollback) { '1.0.0' } else { '1.0.1' })) "$($case.Name): reopened wrong version"
        Assert-True ($launches[0].skip -and $launches[0].probe_only) "$($case.Name): readiness arguments were lost"
        if ($expectedEntry -eq 'window') { Assert-True ($lastLaunch.sta -eq 'STA') "$($case.Name): window did not receive STA" }
        $probeFiles = @(Get-ChildItem -LiteralPath (Join-Path $install '.update') -Filter 'relaunch-*.json' -File)
        Assert-True ($probeFiles.Count -eq 0) "$($case.Name): readiness probe was not cleaned"
        $results.Add([ordered]@{ name = $case.Name; status = 'passed'; rollback = $rollback; relaunched_entry = $lastLaunch.entry; preserved_files = $userFiles.Count; exit_code = $exitCode })
        Write-Output "PASS $($case.Name) ($($lastLaunch.entry))"
    }
    Write-Text (Join-Path $evidence 'results.json') ($results | ConvertTo-Json -Depth 6)
    Write-Output "Dual-entry OTA: $($results.Count) cases passed."
} finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    # Retain fixtures and logs as reviewable evidence; never touch a real install.
}
