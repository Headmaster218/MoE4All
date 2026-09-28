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
$updateCommonPath = Join-Path $PackageRoot 'scripts\moe4all-update-common.ps1'
$updateHelperPath = Join-Path $PackageRoot 'scripts\apply-engine-update.ps1'
$manifestPath = Join-Path $PackageRoot 'install-manifest.json'
$launcherPath = Join-Path $PackageRoot 'Start-INFR-Wizard.cmd'

$requiredPaths = @(
    $binaryPath
    $wizardPath
    $updateCommonPath
    $updateHelperPath
    $manifestPath
    $launcherPath
    (Join-Path $PackageRoot 'README.md')
    (Join-Path $PackageRoot 'README_EN.md')
    (Join-Path $PackageRoot 'LICENSE')
    (Join-Path $PackageRoot 'LICENSE-MIT')
    (Join-Path $PackageRoot 'NOTICE')
)
foreach ($requiredPath in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required package file is missing: $requiredPath"
    }
}

. $updateCommonPath
$manifest = Read-Moe4AllInstallManifest -ManifestPath $manifestPath -PackageRoot $PackageRoot -VerifyFiles
if ($ExpectedVersion -and [string]$manifest.version -ne $ExpectedVersion) {
    throw "Expected install manifest version $ExpectedVersion, got $($manifest.version)"
}
$requiredManagedPaths = @(
    'infr.exe'
    'Start-INFR-Wizard.cmd'
    'scripts/infr-wizard.ps1'
    'scripts/moe4all-update-common.ps1'
    'scripts/apply-engine-update.ps1'
)
$managedPaths = @($manifest.files | ForEach-Object { [string]$_.path })
foreach ($requiredManagedPath in $requiredManagedPaths) {
    if ($requiredManagedPath -notin $managedPaths) {
        throw "Install manifest does not own required file: $requiredManagedPath"
    }
}

$releaseFixture = @(
    [pscustomobject]@{
        tag_name = 'agent-9.0.0'; name = 'Agent'; html_url = 'https://example.test/agent'; body = 'agent';
        draft = $false; prerelease = $false; published_at = '2026-01-02T00:00:00Z'; assets = @()
    },
    [pscustomobject]@{
        tag_name = 'release-1.2.3'; name = 'Engine'; html_url = 'https://example.test/engine'; body = '# Engine notes';
        draft = $false; prerelease = $false; published_at = '2026-01-01T00:00:00Z'; assets = @(
            [pscustomobject]@{ name = 'MoE4All-Windows-x86_64-v1.2.3.zip'; browser_download_url = 'https://example.test/engine.zip'; size = 123 },
            [pscustomobject]@{ name = 'MoE4All-Windows-x86_64-v1.2.3.zip.sha256'; browser_download_url = 'https://example.test/engine.zip.sha256'; size = 64 }
        )
    }
)
$selectedFixture = Select-Moe4AllEngineRelease -Releases $releaseFixture
if ($null -eq $selectedFixture -or $selectedFixture.Tag -ne 'release-1.2.3') {
    throw 'Engine release selection did not isolate release-* from agent-* releases.'
}
if ((Compare-Moe4AllSemVer (ConvertTo-Moe4AllSemVer '1.2.3') (ConvertTo-Moe4AllSemVer '1.2.3-rc.1')) -le 0) {
    throw 'Semantic version comparison did not rank a stable release above its prerelease.'
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

$updateTestRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("moe4all-update-smoke-" + [guid]::NewGuid().ToString('N'))
try {
    New-Item -ItemType Directory -Path $updateTestRoot -Force | Out-Null
    $legacyPath = Join-Path $updateTestRoot 'legacy-managed.txt'
    [System.IO.File]::WriteAllText($legacyPath, 'old managed content', [System.Text.UTF8Encoding]::new($false))
    $legacyInfo = Get-Item -LiteralPath $legacyPath
    $oldManifest = [ordered]@{
        schema_version = 1
        updater_protocol = 1
        product = 'moe4all-engine'
        version = '0.0.1'
        tag = 'release-0.0.1'
        files = @([ordered]@{
            path = 'legacy-managed.txt'
            size = [long]$legacyInfo.Length
            sha256 = (Get-FileHash -LiteralPath $legacyPath -Algorithm SHA256).Hash.ToLowerInvariant()
        })
    }
    [System.IO.File]::WriteAllText(
        (Join-Path $updateTestRoot 'install-manifest.json'),
        (($oldManifest | ConvertTo-Json -Depth 5) + "`n"),
        [System.Text.UTF8Encoding]::new($false)
    )
    New-Item -ItemType Directory -Path (Join-Path $updateTestRoot 'gui-data') -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $updateTestRoot 'kv-sessions') -Force | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $updateTestRoot 'gui-data\wizard-state.json'), '{"custom":true}', [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::WriteAllText((Join-Path $updateTestRoot 'kv-sessions\keep.bin'), 'kv', [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::WriteAllText((Join-Path $updateTestRoot 'infr.toml'), 'custom = true', [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::WriteAllText((Join-Path $updateTestRoot 'custom-user-file.txt'), 'keep me', [System.Text.UTF8Encoding]::new($false))

    $targetVersion = [string]$manifest.version
    $nativeArgs = @(
        '-NoLogo'
        '-NoProfile'
        '-NonInteractive'
        '-File'
        $updateHelperPath
        '-InstallRoot'
        $updateTestRoot
        '-StagedPackageRoot'
        $PackageRoot
        '-ExpectedVersion'
        $targetVersion
        '-NoRelaunch'
    )
    $updateOutput = & powershell.exe @nativeArgs 2>&1 | Out-String
    $updateExitCode = $LASTEXITCODE
    if ($updateExitCode -ne 0) {
        throw "Engine update transaction smoke test failed with exit code $updateExitCode`n$updateOutput"
    }
    foreach ($preservedPath in @(
        'gui-data\wizard-state.json'
        'kv-sessions\keep.bin'
        'infr.toml'
        'custom-user-file.txt'
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $updateTestRoot $preservedPath) -PathType Leaf)) {
            throw "Engine update did not preserve user-owned path: $preservedPath"
        }
    }
    if (Test-Path -LiteralPath $legacyPath) {
        throw 'Engine update did not remove a product-owned file absent from the new manifest.'
    }
    [void](Read-Moe4AllInstallManifest -ManifestPath (Join-Path $updateTestRoot 'install-manifest.json') -PackageRoot $updateTestRoot -VerifyFiles)
    $backupLegacy = Get-ChildItem -LiteralPath (Join-Path $updateTestRoot '.update\backups') -Filter 'legacy-managed.txt' -File -Recurse | Select-Object -First 1
    if ($null -eq $backupLegacy) { throw 'Engine update did not retain a rollback copy of the old managed file.' }
} finally {
    if (Test-Path -LiteralPath $updateTestRoot) {
        Remove-Item -LiteralPath $updateTestRoot -Recurse -Force
    }
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
foreach ($scriptPath in @($wizardPath, $updateCommonPath, $updateHelperPath)) {
    $scriptSource = Get-Content -LiteralPath $scriptPath -Raw -Encoding UTF8
    [void][scriptblock]::Create($scriptSource)
}

$modelPath = Join-Path $PackageRoot 'ci smoke model.gguf'
[System.IO.File]::WriteAllBytes($modelPath, [byte[]]::new(0))
$embeddingDirectory = Join-Path $PackageRoot 'ci embedding model'
$embeddingPath = Join-Path $embeddingDirectory 'ci embed.gguf'
New-Item -ItemType Directory -Path $embeddingDirectory -Force | Out-Null
[System.IO.File]::WriteAllBytes($embeddingPath, [byte[]]::new(0))
$mtpPath = Join-Path $PackageRoot 'ci mtp.gguf'
$visionPath = Join-Path $PackageRoot 'mmproj-ci.gguf'
[System.IO.File]::WriteAllBytes($mtpPath, [byte[]]::new(0))
[System.IO.File]::WriteAllBytes($visionPath, [byte[]]::new(0))
$dataDirectory = Join-Path $PackageRoot 'gui-data'
New-Item -ItemType Directory -Path $dataDirectory -Force | Out-Null

function Invoke-WizardDryRun {
    param(
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][string]$ExpectedCommand,
        [string]$ModelSelection = '',
        [switch]$PassModelArgument,
        [switch]$NoSavedModel,
        [switch]$ExpectRecommendations,
        [switch]$EnableEmbedding,
        [switch]$EnableMtp,
        [switch]$EnableVision
    )

    $state = [ordered]@{
        launch_mode = $Mode
        setup_mode = 'quick'
        model = $(if ($NoSavedModel) { '' } else { $modelPath })
        think_mode = 'default'
        max_new = ''
        configure_sampling = $false
        server_addr = '127.0.0.1:8080'
        server_parallel = $(if ($EnableMtp) { '2' } else { '1' })
        server_session_cache = ($Mode -eq 'server')
        session_idle_secs = '120'
        session_cache_max = '5GiB'
        session_cache_ttl_hours = '24'
        server_auth = $false
        server_vision = [bool]$EnableVision
        vision_projector = $(if ($EnableVision) { $visionPath } else { '' })
        server_embedding = [bool]$EnableEmbedding
        embedding_model = $(if ($EnableEmbedding) { $embeddingDirectory } else { '' })
        embedding_idle_timeout = '17'
        mtp_enabled = [bool]$EnableMtp
        mtp_model = $(if ($EnableMtp) { $mtpPath } else { '' })
        mtp_verify_tokens = '4'
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
    if ($ExpectRecommendations) {
        foreach ($expectedText in @(
            'Official recommended models',
            'https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true',
            'https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64',
            'https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true',
            'https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf?download=true'
        )) {
            if ($stdout -notmatch [regex]::Escape($expectedText)) {
                throw "Wizard $Mode did not display the expected recommendation '$expectedText'.`n$stdout"
            }
        }
    }
    if ($Mode -eq 'server') {
        $expectedCacheDirectory = Join-Path $PackageRoot 'kv-sessions'
        if ($stdout -notmatch [regex]::Escape("kv.session_cache_dir=$expectedCacheDirectory")) {
            throw "Wizard server mode did not use the package-local KV cache directory.`n$stdout"
        }
    }
    if ($EnableEmbedding) {
        if ($stdout -notmatch [regex]::Escape("--embedding-model '$embeddingPath'")) {
            throw "Wizard $Mode did not resolve the embedding directory to its GGUF.`n$stdout"
        }
        if ($stdout -notmatch [regex]::Escape('--embedding-idle-timeout 17')) {
            throw "Wizard $Mode did not preserve the embedding idle timeout.`n$stdout"
        }
    }
    if ($EnableMtp) {
        foreach ($argument in @('--set spec.mtp=true', "--set 'spec.draft=$mtpPath'", '--set spec.k=4', '--parallel 2', '--temp 0')) {
            if ($stdout -notmatch [regex]::Escape($argument)) {
                throw "Wizard $Mode did not generate the opportunistic MTP argument '$argument'.`n$stdout"
            }
        }
        if ($stdout -notmatch 'Two active decodes automatically use ordinary batched decode') {
            throw "Wizard $Mode did not explain the opportunistic two-slot MTP policy.`n$stdout"
        }
    }
    if ($EnableVision -and $stdout -notmatch [regex]::Escape("--mmproj '$visionPath'")) {
        throw "Wizard $Mode did not preserve the vision projector while MTP was enabled.`n$stdout"
    }
}

$quotedModelPath = '"' + $modelPath + '"'
$powerShellDrop = "& '$modelPath'"
Invoke-WizardDryRun -Mode 'chat' -ExpectedCommand 'run' -PassModelArgument
Invoke-WizardDryRun -Mode 'server' -ExpectedCommand 'serve' -ModelSelection $powerShellDrop -NoSavedModel -ExpectRecommendations -EnableEmbedding -EnableMtp -EnableVision
Invoke-WizardDryRun -Mode 'benchmark' -ExpectedCommand 'bench' -ModelSelection $quotedModelPath
Invoke-WizardDryRun -Mode 'benchmark' -ExpectedCommand 'bench' -ModelSelection 'R' -ExpectRecommendations

Remove-Item -LiteralPath $modelPath -Force
Remove-Item -LiteralPath $embeddingDirectory -Recurse -Force
Remove-Item -LiteralPath $mtpPath -Force
Remove-Item -LiteralPath $visionPath -Force
Remove-Item -LiteralPath $dataDirectory -Recurse -Force
Write-Host "Windows package smoke test passed: $PackageRoot"
