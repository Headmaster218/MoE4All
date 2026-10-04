[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^agent-v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$')]
    [string]$Version,
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root 'artifacts\agent-release' }
$output = [IO.Path]::GetFullPath($OutputDirectory)
$name = "MoE4All-Agent-Windows-x64-$Version"
$stage = Join-Path $output $name
$archive = Join-Path $output "$name.zip"
$checksum = "$archive.sha256"
if ((Test-Path -LiteralPath $stage) -or (Test-Path -LiteralPath $archive)) {
    throw "Release output already exists: $name"
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
    $actual = (& git -C $source rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actual -ne $Matches[1]) { throw "Submodule does not match its pin: $relative" }
    $dirty = & git -C $source status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0 -or $dirty) { throw "Commit tracked submodule changes: $relative" }
}

New-Item -ItemType Directory -Path $output, $stage -Force | Out-Null
$temp = Join-Path $output "$name-source.zip"
& git -C $root archive '--format=zip' ("--output=$temp") HEAD harness
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

$dshCommit = (& git -C (Join-Path $root 'harness\dsh') rev-parse HEAD).Trim()
$manifest = [ordered]@{
    schema = 1
    product = 'moe4all-agent'
    version = $Version
    sourceCommit = $rootCommit
    dshCommit = $dshCommit
}
$manifestPath = Join-Path $stage 'harness\distribution\agent-release.json'
[IO.File]::WriteAllText($manifestPath, (($manifest | ConvertTo-Json) + "`n"), (New-Object Text.UTF8Encoding($false)))
& (Join-Path $PSScriptRoot 'Build-AgentLauncher.ps1') -OutputPath (Join-Path $stage 'MoE4All.exe')
if ($LASTEXITCODE -ne 0) { throw 'Building the Agent EXE failed.' }

Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($checksum, "$hash  $([IO.Path]::GetFileName($archive))`n", (New-Object Text.UTF8Encoding($false)))
Write-Host "Agent release: $archive"
Write-Host "SHA256: $hash"
