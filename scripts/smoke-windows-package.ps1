[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PackageRoot,
    [string]$ExpectedVersion = '',
    [switch]$SkipDependencyCheck
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$PackageRoot = [System.IO.Path]::GetFullPath($PackageRoot)
$binaryPath = Join-Path $PackageRoot 'infr.exe'
$wizardPath = Join-Path $PackageRoot 'scripts\infr-wizard.ps1'
$launcherPath = Join-Path $PackageRoot 'Start-INFR-Wizard.cmd'

$requiredPaths = @(
    $binaryPath
    $wizardPath
    $launcherPath
    (Join-Path $PackageRoot 'README.md')
    (Join-Path $PackageRoot 'README_EN.md')
    (Join-Path $PackageRoot 'GETTING_STARTED.md')
    (Join-Path $PackageRoot 'LICENSE')
    (Join-Path $PackageRoot 'LICENSE-MIT')
    (Join-Path $PackageRoot 'NOTICE')
)
foreach ($requiredPath in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required package file is missing: $requiredPath"
    }
}

function Invoke-Infr {
    param([Parameter(Mandatory = $true)][string[]]$NativeArguments)
    $output = & $binaryPath @NativeArguments 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "infr.exe $($NativeArguments -join ' ') failed with exit code $exitCode`n$output"
    }
    return $output
}

$versionOutput = Invoke-Infr @('--version')
if ($ExpectedVersion -and $versionOutput -notmatch [regex]::Escape("infr $ExpectedVersion")) {
    throw "Expected infr $ExpectedVersion, got: $($versionOutput.Trim())"
}

[void](Invoke-Infr @('--help'))
foreach ($command in @('run', 'serve', 'bench')) {
    [void](Invoke-Infr @($command, '--help'))
}

# A release archive must not require users to install the Visual C++ runtime.
# Vulkan is loaded dynamically from the GPU driver and does not appear here.
if (-not $SkipDependencyCheck) {
    $dependencyOutput = ''
    $dumpbin = Get-Command 'dumpbin.exe' -ErrorAction SilentlyContinue
    if ($null -eq $dumpbin) {
        $programFilesX86 = [Environment]::GetFolderPath('ProgramFilesX86')
        $vswherePath = Join-Path $programFilesX86 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (Test-Path -LiteralPath $vswherePath -PathType Leaf) {
            $visualStudioPath = (& $vswherePath -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
            if ($visualStudioPath) {
                $dumpbin = Get-ChildItem -LiteralPath (Join-Path $visualStudioPath 'VC\Tools\MSVC') -Filter 'dumpbin.exe' -Recurse |
                    Where-Object { $_.FullName -match 'Hostx64[\\/]x64[\\/]dumpbin\.exe$' } |
                    Sort-Object FullName -Descending |
                    Select-Object -First 1
            }
        }
    }
    if ($null -ne $dumpbin) {
        $dumpbinPath = if ($dumpbin -is [System.IO.FileInfo]) { $dumpbin.FullName } else { $dumpbin.Source }
        $dependencyOutput = & $dumpbinPath /dependents $binaryPath 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { throw "dumpbin failed with exit code $LASTEXITCODE" }
    } else {
        $objdump = Get-Command 'objdump.exe' -ErrorAction SilentlyContinue
        if ($null -eq $objdump) {
            throw 'Neither dumpbin.exe nor objdump.exe is available to verify release dependencies.'
        }
        $dependencyOutput = & $objdump.Source -p $binaryPath 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { throw "objdump failed with exit code $LASTEXITCODE" }
    }
    if ($dependencyOutput -match '(?i)\b(?:vcruntime|msvcp|concrt)[^\s]*\.dll\b|\bapi-ms-win-crt-[^\s]*\.dll\b') {
        throw "Release executable still depends on the Visual C++ runtime.`n$dependencyOutput"
    }
}

# Parse with Windows PowerShell before exercising the CMD wrapper that end users double-click.
$wizardSource = Get-Content -LiteralPath $wizardPath -Raw -Encoding UTF8
[void][scriptblock]::Create($wizardSource)

$modelPath = Join-Path $PackageRoot 'ci smoke model.gguf'
[System.IO.File]::WriteAllBytes($modelPath, [byte[]]::new(0))
$dataDirectory = Join-Path $PackageRoot 'gui-data'
New-Item -ItemType Directory -Path $dataDirectory -Force | Out-Null

function Invoke-WizardDryRun {
    param(
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][string]$ExpectedCommand,
        [string]$ModelSelection = '',
        [switch]$PassModelArgument
    )

    $state = [ordered]@{
        launch_mode = $Mode
        setup_mode = 'quick'
        model = $modelPath
        think_mode = 'default'
        max_new = ''
        configure_sampling = $false
        server_addr = '127.0.0.1:8080'
        server_parallel = '1'
        server_auth = $false
        bench_kind = 'decode'
        gen_tokens = '1'
        depth_mode = 'none'
        reps = '1'
        json_output = $false
    }
    $statePath = Join-Path $dataDirectory 'wizard-state.json'
    $stateJson = $state | ConvertTo-Json -Depth 4
    [System.IO.File]::WriteAllText($statePath, $stateJson, [System.Text.UTF8Encoding]::new($false))

    $processInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $processInfo.FileName = 'cmd.exe'
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    $processInfo.RedirectStandardInput = $true
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $processInfo.WorkingDirectory = $PackageRoot
    $quotedLauncher = $launcherPath.Replace('"', '""')
    $launcherArguments = '-DryRun -SkipUpdateCheck'
    if ($PassModelArgument) {
        $quotedModelArgument = $modelPath.Replace('"', '""')
        $launcherArguments += " `"$quotedModelArgument`""
    }
    $processInfo.Arguments = "/d /s /c `"`"$quotedLauncher`" $launcherArguments`""

    $process = [System.Diagnostics.Process]::Start($processInfo)
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    # Keep the saved launch mode. Without a launcher argument, exercise the same Read-Host model
    # prompt used when a path is pasted into an already-open terminal.
    $process.StandardInput.WriteLine('')
    if (-not $PassModelArgument) {
        $process.StandardInput.WriteLine($ModelSelection)
    }
    for ($i = 0; $i -lt 30; $i++) {
        $process.StandardInput.WriteLine('')
    }
    $process.StandardInput.WriteLine('x')
    $process.StandardInput.Close()

    if (-not $process.WaitForExit(30000)) {
        $process.Kill()
        throw "Wizard $Mode DryRun timed out."
    }
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    if ($process.ExitCode -ne 0) {
        throw "Wizard $Mode DryRun failed with exit code $($process.ExitCode)`n$stdout`n$stderr"
    }
    if ($stdout -notmatch 'DryRun: command was not started\.') {
        throw "Wizard $Mode did not reach DryRun completion.`n$stdout`n$stderr"
    }
    if ($stdout -notmatch [regex]::Escape("MoE4All v$ExpectedVersion")) {
        throw "Wizard $Mode did not display the expected MoE4All version banner.`n$stdout"
    }
    if ($stdout -notmatch "\s$ExpectedCommand\s") {
        throw "Wizard $Mode did not generate the expected '$ExpectedCommand' command.`n$stdout"
    }
    if ($stdout -notmatch [regex]::Escape($modelPath)) {
        throw "Wizard $Mode did not preserve the model path.`n$stdout"
    }
    if ($PassModelArgument -and $stdout -notmatch 'Model selected from launcher argument') {
        throw "Wizard $Mode did not accept the model passed by CMD/drag-and-drop.`n$stdout"
    }
}

$quotedModelPath = '"' + $modelPath + '"'
$powerShellDrop = "& '$modelPath'"
Invoke-WizardDryRun -Mode 'chat' -ExpectedCommand 'run' -PassModelArgument
Invoke-WizardDryRun -Mode 'server' -ExpectedCommand 'serve' -ModelSelection $powerShellDrop
Invoke-WizardDryRun -Mode 'benchmark' -ExpectedCommand 'bench' -ModelSelection $quotedModelPath

Remove-Item -LiteralPath $modelPath -Force
Remove-Item -LiteralPath $dataDirectory -Recurse -Force
Write-Host "Windows package smoke test passed: $PackageRoot"
