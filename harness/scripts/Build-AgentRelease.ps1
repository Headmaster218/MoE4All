[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^agent-v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$')]
    [string]$Version,
    [string]$OutputDirectory,
    [string]$EngineExecutable,
    [string]$PrebuiltHarnessRoot,
    [string]$BaseArchive
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

# A launcher-only hotfix can reuse a verified integrated runtime when all runtime source pins match.
if ($BaseArchive) {
    if ($PrebuiltHarnessRoot) { throw 'BaseArchive and PrebuiltHarnessRoot cannot be combined.' }
    $baseZip = [IO.Path]::GetFullPath($BaseArchive)
    $baseSum = "$baseZip.sha256"
    if (-not (Test-Path -LiteralPath $baseSum -PathType Leaf)) { throw 'The base archive checksum is missing.' }
    $sumPattern = '(?im)^([a-f0-9]{64})\s+\*?' + [regex]::Escape([IO.Path]::GetFileName($baseZip)) + '\s*$'
    if ((Get-Content -LiteralPath $baseSum -Raw -Encoding UTF8) -notmatch $sumPattern) {
        throw 'The base checksum does not name the expected archive.'
    }
    $expectedHash = $Matches[1]
    if ((Get-FileHash -LiteralPath $baseZip -Algorithm SHA256).Hash -ne $expectedHash) {
        throw 'The base archive SHA256 does not match.'
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $baseReader = [IO.Compression.ZipFile]::OpenRead($baseZip)
    try {
        $baseTop = [IO.Path]::GetFileNameWithoutExtension($baseZip)
        foreach ($entry in $baseReader.Entries) {
            $parts = $entry.FullName -split '[\\/]'
            if ($parts[0] -ne $baseTop -or $parts -contains '..' -or $entry.FullName.Contains(':')) {
                throw "Unexpected base archive path: $($entry.FullName)"
            }
        }
        $baseManifestEntry = $baseReader.GetEntry("$baseTop/harness/distribution/agent-release.json")
        if ($null -eq $baseManifestEntry) { throw 'The base release manifest is missing.' }
        $manifestReader = New-Object IO.StreamReader($baseManifestEntry.Open())
        try { $baseManifest = $manifestReader.ReadToEnd() | ConvertFrom-Json } finally { $manifestReader.Dispose() }
        $baseCommit = [string]$baseManifest.sourceCommit
        if ($baseManifest.schema -ne 2 -or $baseManifest.product -ne 'moe4all-agent' -or
            $baseManifest.packageKind -ne 'integrated' -or $baseManifest.engineVersion -ne $engineTag -or
            $baseCommit -notmatch '^[0-9a-f]{40}$' -or $baseTop -ne "MoE4All-Agent-Windows-x64-$($baseManifest.version)") {
            throw 'The base archive is not a matching integrated Agent release.'
        }
    } finally { $baseReader.Dispose() }
    . (Join-Path $PSScriptRoot 'AgentUpdateVersion.ps1')
    if (-not (Test-AgentVersionNewer $Version ([string]$baseManifest.version))) {
        throw 'The hotfix release must be newer than the base release.'
    }
    & git -C $root cat-file -e ($baseCommit + '^{commit}')
    if ($LASTEXITCODE -ne 0) { throw 'The base source commit is not available for verification.' }
    $runtimePaths = @($submodules) + @('harness/distribution/profile')
    & git -C $root diff --quiet $baseCommit HEAD -- @runtimePaths
    if ($LASTEXITCODE -ne 0) { throw 'Runtime submodules or profile changed; perform a full release build.' }

    $baseExtract = Join-Path $output "$name-base"
    if (Test-Path -LiteralPath $baseExtract) { throw "Base extraction already exists: $baseExtract" }
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    [IO.Compression.ZipFile]::ExtractToDirectory($baseZip, $baseExtract)
    Move-Item -LiteralPath (Join-Path $baseExtract $baseTop) -Destination $stage
    [IO.Directory]::Delete($baseExtract, $false)
    $stageHarness = Join-Path $stage 'harness'
    if (Test-Path -LiteralPath (Join-Path $stageHarness 'agent-state')) {
        throw 'The base archive contains user runtime state.'
    }
    $bundledEnginePath = Join-Path $stageHarness "runtime\engine\$engineTag\infr.exe"
    if ((Get-FileHash -LiteralPath $bundledEnginePath -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $engineExe -Algorithm SHA256).Hash) {
        throw 'The reused Engine differs from the selected release executable.'
    }
    foreach ($scriptName in @('AgentUpdateVersion.ps1', 'Check-AgentUpdate.ps1', 'Prepare-BundledRuntime.ps1', 'Run-Agent.ps1')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $scriptName) -Destination (Join-Path $stageHarness "scripts\$scriptName") -Force
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Prepare-BundledRuntime.cjs') -Destination (Join-Path $stageHarness 'scripts\Prepare-BundledRuntime.cjs') -Force
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'vendor') -Destination (Join-Path $stageHarness 'scripts\vendor') -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $root 'harness\launcher\MoE4All.cs') -Destination (Join-Path $stageHarness 'launcher\MoE4All.cs') -Force
    & (Join-Path $PSScriptRoot 'Build-AgentLauncher.ps1') -OutputPath (Join-Path $stage 'MoE4All.exe')
    if ($LASTEXITCODE -ne 0) { throw 'Building the hotfix launcher failed.' }
    $baseVersion = [string]$baseManifest.version
    $baseManifest.version = $Version
    $baseManifest.sourceCommit = $rootCommit
    $runtimeSourceCommit = $baseCommit
    if ($baseManifest.runtimeSourceCommit) { $runtimeSourceCommit = [string]$baseManifest.runtimeSourceCommit }
    $baseManifest | Add-Member -NotePropertyName runtimeSourceCommit -NotePropertyValue $runtimeSourceCommit -Force
    $baseManifest | Add-Member -NotePropertyName basePackageVersion -NotePropertyValue $baseVersion -Force
    [IO.File]::WriteAllText((Join-Path $stageHarness 'distribution\agent-release.json'),
        (($baseManifest | ConvertTo-Json) + "`n"), (New-Object Text.UTF8Encoding($false)))
    & (Join-Path $PSScriptRoot 'Pack-AgentRuntime.ps1') -HarnessRoot $stageHarness
    [IO.Compression.ZipFile]::CreateFromDirectory($stage, $archive, [IO.Compression.CompressionLevel]::Optimal, $true)
    $hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText($checksum, "$hash  $([IO.Path]::GetFileName($archive))`n", (New-Object Text.UTF8Encoding($false)))
    Write-Host "Agent release: $archive"
    Write-Host "SHA256: $hash"
    return
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
    'harness/scripts/Prepare-BundledRuntime.cjs',
    'harness/scripts/vendor',
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
$buildDshRoot = [IO.Path]::GetFullPath((Join-Path $buildHarness 'dsh')).TrimEnd([IO.Path]::DirectorySeparatorChar)
$deployLinks = @(Get-ChildItem -LiteralPath $deployRoot -Recurse -Force -Attributes ReparsePoint)
$externalTargets = @{}
foreach ($entry in $deployLinks) {
    if (-not $entry.PSIsContainer) { throw "The DSH deployment contains a file symlink: $($entry.FullName)" }
    $targetValue = [string]@($entry.Target)[0]
    $target = [IO.Path]::GetFullPath((Join-Path $entry.Parent.FullName $targetValue))
    if (-not $target.StartsWith($deployRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        $externalTargets[$target] = $true
    }
}
$materializedTargets = @{}
$materializedRoot = Join-Path $deployRoot '.moe4all-workspace'
foreach ($target in @($externalTargets.Keys | Sort-Object)) {
    if (-not $target.StartsWith($buildDshRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "The DSH deployment link escapes the source workspace: $target"
    }
    $sourceRelative = Get-RelativePath $buildDshRoot $target
    $safeName = $sourceRelative -replace '[^0-9A-Za-z._-]', '-'
    $materialized = Join-Path $materializedRoot $safeName
    New-Item -ItemType Directory -Path $materialized -Force | Out-Null
    & robocopy $target $materialized /E /XJ /XD node_modules .git /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Materializing DSH workspace package $sourceRelative failed (robocopy $LASTEXITCODE)." }
    $materializedTargets[$target] = $materialized
}

# pnpm deploy does not materialize workspace packages that are only peer
# dependencies. Build the production dependency closure explicitly so the
# offline runtime never falls back to the source checkout's hoisted modules.
$workspacePackages = @{}
$dshSourceRoot = Join-Path $root 'harness\dsh'
$workspaceManifestPaths = @(& git -C $dshSourceRoot ls-files '--' ':(glob)**/package.json')
if ($LASTEXITCODE -ne 0 -or $workspaceManifestPaths.Count -eq 0) {
    throw 'Cannot enumerate DSH workspace package manifests.'
}
foreach ($relativeManifestPath in $workspaceManifestPaths) {
    $manifestPath = Join-Path $buildDshRoot ($relativeManifestPath.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { continue }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        continue
    }
    $packageName = [string]$manifest.name
    if ($packageName -match '^@deepseek-ai/[0-9A-Za-z._-]+$' -and -not $workspacePackages.ContainsKey($packageName)) {
        $workspacePackages[$packageName] = Split-Path -Parent $manifestPath
    }
}

$runtimePackages = @{}
$runtimeQueue = New-Object 'Collections.Generic.Queue[string]'
function Register-RuntimePackage([string]$packagePath) {
    $manifestPath = Join-Path $packagePath 'package.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { return }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return
    }
    $packageName = [string]$manifest.name
    if ($packageName -notmatch '^@deepseek-ai/[0-9A-Za-z._-]+$' -or $runtimePackages.ContainsKey($packageName)) { return }
    $runtimePackages[$packageName] = $packagePath
    foreach ($section in @('dependencies', 'optionalDependencies', 'peerDependencies')) {
        $dependencies = $manifest.$section
        if ($null -eq $dependencies) { continue }
        foreach ($dependency in @($dependencies.PSObject.Properties.Name)) {
            if ($dependency -match '^@deepseek-ai/[0-9A-Za-z._-]+$') { $runtimeQueue.Enqueue($dependency) }
        }
    }
}

$deployPnpm = Join-Path $deployRoot 'node_modules\.pnpm'
foreach ($entry in Get-ChildItem -LiteralPath $deployPnpm -Directory) {
    $scope = Join-Path $entry.FullName 'node_modules\@deepseek-ai'
    if (-not (Test-Path -LiteralPath $scope -PathType Container)) { continue }
    foreach ($package in Get-ChildItem -LiteralPath $scope -Directory -Force) {
        if (($package.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
            Register-RuntimePackage $package.FullName
        }
    }
}
foreach ($materialized in @($materializedTargets.Values)) {
    Register-RuntimePackage ([string]$materialized)
}
while ($runtimeQueue.Count -gt 0) {
    $packageName = $runtimeQueue.Dequeue()
    if ($runtimePackages.ContainsKey($packageName)) { continue }
    if (-not $workspacePackages.ContainsKey($packageName)) {
        throw "The production DSH dependency closure is missing workspace package $packageName."
    }
    $source = [string]$workspacePackages[$packageName]
    $sourceRelative = Get-RelativePath $buildDshRoot $source
    $safeName = $sourceRelative -replace '[^0-9A-Za-z._-]', '-'
    $materialized = Join-Path $materializedRoot $safeName
    New-Item -ItemType Directory -Path $materialized -Force | Out-Null
    & robocopy $source $materialized /E /XJ /XD node_modules .git /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Materializing DSH workspace dependency $sourceRelative failed (robocopy $LASTEXITCODE)." }
    Register-RuntimePackage $materialized
}

$links = @()
foreach ($entry in $deployLinks) {
    $targetValue = [string]@($entry.Target)[0]
    $target = [IO.Path]::GetFullPath((Join-Path $entry.Parent.FullName $targetValue))
    if (-not $target.StartsWith($deployRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        $target = [string]$materializedTargets[$target]
        if (-not $target) { throw "The DSH deployment link was not materialized: $($entry.FullName)" }
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
    (Join-Path $stageHarness ([IO.Path]::GetFileName($deploy))),
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

& (Join-Path $PSScriptRoot 'Pack-AgentRuntime.ps1') -HarnessRoot $stageHarness
Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($checksum, "$hash  $([IO.Path]::GetFileName($archive))`n", (New-Object Text.UTF8Encoding($false)))
Write-Host "Agent release: $archive"
Write-Host "SHA256: $hash"
