[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^agent-v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$')]
    [string]$Version,
    [string]$OutputDirectory,
    [string]$EngineExecutable,
    [string]$PrebuiltHarnessRoot
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root 'artifacts\agent-release' }
if (-not $EngineExecutable) {
    $EngineExecutable = Join-Path (Split-Path -Parent $root) 'infr\target\release\infr.exe'
}
$output = [IO.Path]::GetFullPath($OutputDirectory)
$engineExe = [IO.Path]::GetFullPath($EngineExecutable)
$name = "MoE4All-Agent-Windows-x64-$Version"
$stage = Join-Path $output $name
$archive = Join-Path $output "$name.zip"
$checksum = "$archive.sha256"
$deploy = Join-Path $output "$name-dsh-deploy"
if ((Test-Path -LiteralPath $stage) -or (Test-Path -LiteralPath $archive) -or (Test-Path -LiteralPath $deploy)) {
    throw "Release output already exists: $name"
}
if (-not (Test-Path -LiteralPath $engineExe -PathType Leaf)) { throw "Engine executable not found: $engineExe" }
if ($Version -notmatch '^agent-v([0-9]+\.[0-9]+\.[0-9]+)') { throw 'Cannot derive the Engine version.' }
$engineVersion = $Matches[1]
$engineTag = "v$engineVersion"
$engineOutput = ((& $engineExe --version 2>$null) -join ' ').Trim()
if ($LASTEXITCODE -ne 0 -or $engineOutput -notmatch ("^infr\s+" + [regex]::Escape($engineVersion) + '$')) {
    throw "Expected infr $engineVersion, got: $engineOutput"
}

function Get-RelativePath([string]$base, [string]$path) {
    $basePath = [IO.Path]::GetFullPath($base).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $pathValue = [IO.Path]::GetFullPath($path)
    $baseUri = New-Object Uri($basePath)
    $pathUri = New-Object Uri($pathValue)
    return [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($pathUri).ToString()).Replace('/', '\')
}

$status = & git -C $root status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $status) { throw 'Commit tracked Agent changes before building a release.' }
$rootCommit = (& git -C $root rev-parse HEAD).Trim()
$paths = & git -C $root config --file (Join-Path $root '.gitmodules') --get-regexp '^submodule\..*\.path$'
if ($LASTEXITCODE -ne 0) { throw 'Cannot read Agent submodules.' }
$submodules = @($paths | ForEach-Object { ($_ -split '\s+', 2)[1] })
foreach ($relative in $submodules) {
    $source = Join-Path $root $relative
    $pin = (& git -C $root ls-tree HEAD $relative) -join ''
    if ($LASTEXITCODE -ne 0 -or $pin -notmatch '^160000 commit ([0-9a-f]{40})\s') {
        throw "Missing committed submodule pin: $relative"
    }
    $expected = $Matches[1]
    $actual = (& git -C $source rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actual -ne $expected) { throw "Submodule does not match its pin: $relative" }
    $dirty = & git -C $source status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0 -or $dirty) { throw "Commit tracked submodule changes: $relative" }
}

New-Item -ItemType Directory -Path $output, $stage -Force | Out-Null
$temp = Join-Path $output "$name-source.zip"
$agentPaths = @(
    'harness/distribution/profile',
    'harness/launcher/MoE4All.cs',
    'harness/scripts/AgentUpdateVersion.ps1',
    'harness/scripts/Build-Agent.ps1',
    'harness/scripts/Build-DevHarness.ps1',
    'harness/scripts/Check-AgentUpdate.ps1',
    'harness/scripts/Initialize-Agent.ps1',
    'harness/scripts/Prepare-BundledRuntime.ps1',
    'harness/scripts/Run-Agent.ps1'
)
& git -C $root archive '--format=zip' ("--output=$temp") HEAD @agentPaths
if ($LASTEXITCODE -ne 0) { throw 'Archiving Agent scripts failed.' }
Expand-Archive -LiteralPath $temp -DestinationPath $stage

foreach ($relative in $submodules) {
    $source = Join-Path $root $relative
    $subArchive = Join-Path $output ("$name-" + ($relative -replace '[\\/]', '-') + '.zip')
    & git -C $source archive '--format=zip' ("--output=$subArchive") HEAD
    if ($LASTEXITCODE -ne 0) { throw "Archiving $relative failed." }
    Expand-Archive -LiteralPath $subArchive -DestinationPath (Join-Path $stage $relative) -Force
    Remove-Item -LiteralPath $subArchive
}
Remove-Item -LiteralPath $temp

$stageHarness = Join-Path $stage 'harness'
$dshCommit = (& git -C (Join-Path $root 'harness\dsh') rev-parse HEAD).Trim()
$manifest = [ordered]@{
    schema = 2
    product = 'moe4all-agent'
    packageKind = 'integrated'
    version = $Version
    sourceCommit = $rootCommit
    dshCommit = $dshCommit
    engineVersion = $engineTag
}
$manifestPath = Join-Path $stageHarness 'distribution\agent-release.json'
[IO.File]::WriteAllText($manifestPath, (($manifest | ConvertTo-Json) + "`n"), (New-Object Text.UTF8Encoding($false)))
& (Join-Path $root 'harness\scripts\Build-AgentLauncher.ps1') -OutputPath (Join-Path $stage 'MoE4All.exe')
if ($LASTEXITCODE -ne 0) { throw 'Building the Agent EXE failed.' }

$buildHarness = $stageHarness
if ($PrebuiltHarnessRoot) {
    $buildHarness = [IO.Path]::GetFullPath($PrebuiltHarnessRoot)
    if (-not (Test-Path -LiteralPath (Join-Path $buildHarness 'dsh\apps\cli\lib\bin.js') -PathType Leaf)) {
        throw "Prebuilt DSH runtime is incomplete: $buildHarness"
    }
    $prebuiltManifestPath = Join-Path $buildHarness 'distribution\agent-release.json'
    if (-not (Test-Path -LiteralPath $prebuiltManifestPath -PathType Leaf)) {
        throw "Prebuilt release manifest is missing: $prebuiltManifestPath"
    }
    $prebuiltManifest = Get-Content -LiteralPath $prebuiltManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $builtCommit = [string]$prebuiltManifest.dshCommit
    if ($builtCommit -ne $dshCommit) {
        throw "Prebuilt DSH commit does not match the release pin: $builtCommit"
    }
    & (Join-Path $stageHarness 'scripts\Initialize-Agent.ps1') -StateRoot (Join-Path $stageHarness 'agent-state')
    $builtProfile = Join-Path $buildHarness 'agent-state\home\profiles\web'
    $stageProfile = Join-Path $stageHarness 'agent-state\home\profiles\web'
    if (-not (Test-Path -LiteralPath (Join-Path $builtProfile 'node_modules') -PathType Container)) {
        throw "Prebuilt Agent profile is incomplete: $builtProfile"
    }
    & robocopy (Join-Path $builtProfile 'node_modules') (Join-Path $stageProfile 'node_modules') /E /XJ /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying the prebuilt profile failed (robocopy $LASTEXITCODE)." }
    Copy-Item -LiteralPath (Join-Path $builtProfile 'pnpm-lock.yaml') -Destination (Join-Path $stageProfile 'pnpm-lock.yaml') -Force

    $builtNode = Get-ChildItem -LiteralPath (Join-Path $buildHarness 'runtime') -Filter node.exe -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match 'node-v[^\\]+-win-x64\\node\.exe$' } |
        Select-Object -First 1
    if ($null -eq $builtNode) { throw 'The prebuilt Node runtime is missing.' }
    $stageNodeRoot = Join-Path (Join-Path $stageHarness 'runtime') $builtNode.Directory.Name
    & robocopy $builtNode.Directory.FullName $stageNodeRoot /E /XJ /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying the prebuilt Node runtime failed (robocopy $LASTEXITCODE)." }
} else {
    & (Join-Path $stageHarness 'scripts\Build-Agent.ps1') -StateRoot (Join-Path $stageHarness 'agent-state')
}

$node = Get-ChildItem -LiteralPath (Join-Path $buildHarness 'runtime') -Filter node.exe -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match 'node-v[^\\]+-win-x64\\node\.exe$' } |
    Select-Object -First 1 -ExpandProperty FullName
$pnpmCli = Join-Path $buildHarness 'runtime\pnpm\node_modules\pnpm\bin\pnpm.cjs'
$pnpmStore = Join-Path $buildHarness 'runtime\pnpm-store'
if (-not $node -or -not (Test-Path -LiteralPath $pnpmCli -PathType Leaf)) { throw 'The release build runtime is incomplete.' }
Push-Location (Join-Path $buildHarness 'dsh')
try {
    & $node $pnpmCli ("--config.store-dir=$pnpmStore") --filter '@deepseek-ai/dsh' deploy --prod --legacy $deploy
    if ($LASTEXITCODE -ne 0) { throw 'Creating the production DSH deployment failed.' }
} finally {
    Pop-Location
}

$deployRoot = [IO.Path]::GetFullPath($deploy).TrimEnd([IO.Path]::DirectorySeparatorChar)
$links = @()
foreach ($entry in @(Get-ChildItem -LiteralPath $deployRoot -Recurse -Force -Attributes ReparsePoint)) {
    if (-not $entry.PSIsContainer) { throw "The DSH deployment contains a file symlink: $($entry.FullName)" }
    $targetValue = [string]@($entry.Target)[0]
    $target = [IO.Path]::GetFullPath((Join-Path $entry.DirectoryName $targetValue))
    if (-not $target.StartsWith($deployRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "The DSH deployment link escapes its root: $($entry.FullName)"
    }
    $links += [ordered]@{
        path = Get-RelativePath $deployRoot $entry.FullName
        target = Get-RelativePath $deployRoot $target
    }
}
$links = @($links | Sort-Object { $_.path.Length }, path)
$stageRuntime = Join-Path $stageHarness 'runtime'
$stageDsh = Join-Path $stageRuntime 'dsh'
New-Item -ItemType Directory -Path $stageDsh -Force | Out-Null
& robocopy $deployRoot $stageDsh /E /XJ /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Copying the production DSH deployment failed (robocopy $LASTEXITCODE)." }
$linksManifest = [ordered]@{ schema = 1; links = $links }
[IO.File]::WriteAllText((Join-Path $stageRuntime 'dsh-links.json'), (($linksManifest | ConvertTo-Json -Depth 6) + "`n"), (New-Object Text.UTF8Encoding($false)))

$engineRoot = Join-Path $stageRuntime "engine\$engineTag"
New-Item -ItemType Directory -Path $engineRoot -Force | Out-Null
$bundledEngine = Join-Path $engineRoot 'infr.exe'
Copy-Item -LiteralPath $engineExe -Destination $bundledEngine
$engineHash = (Get-FileHash -LiteralPath $bundledEngine -Algorithm SHA256).Hash.ToLowerInvariant()
$engineManifest = [ordered]@{
    schema = 1
    tag = $engineTag
    name = "MoE4All Engine $engineTag"
    executable = "$engineTag\infr.exe"
    sha256 = $engineHash
}
[IO.File]::WriteAllText((Join-Path $stageRuntime 'engine\bundled-engine.json'), (($engineManifest | ConvertTo-Json) + "`n"), (New-Object Text.UTF8Encoding($false)))

$seed = Join-Path $stageHarness 'distribution\prebuilt-state'
Move-Item -LiteralPath (Join-Path $stageHarness 'agent-state') -Destination $seed

$removePaths = @(
    (Join-Path $stageHarness 'dsh'),
    (Join-Path $stageHarness 'plugins'),
    (Join-Path $stageRuntime 'pnpm'),
    (Join-Path $stageRuntime 'pnpm-store'),
    (Join-Path $stageRuntime 'plugin-packs')
)
foreach ($path in $removePaths) {
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
}
Get-ChildItem -LiteralPath $stageRuntime -File -Filter 'node-v*-win-x64.zip' -ErrorAction SilentlyContinue |
    Remove-Item -Force
foreach ($scriptName in @('Build-Agent.ps1', 'Build-DevHarness.ps1', 'Initialize-Agent.ps1')) {
    $path = Join-Path $stageHarness "scripts\$scriptName"
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
}
Remove-Item -LiteralPath $deploy -Recurse -Force

Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($checksum, "$hash  $([IO.Path]::GetFileName($archive))`n", (New-Object Text.UTF8Encoding($false)))
Write-Host "Agent release: $archive"
Write-Host "SHA256: $hash"
