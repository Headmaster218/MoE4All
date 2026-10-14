[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$HarnessRoot)

$ErrorActionPreference = 'Stop'
$harness = [IO.Path]::GetFullPath($HarnessRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
if ((Get-Content -LiteralPath (Join-Path $harness 'distribution\agent-release.json') -Raw | ConvertFrom-Json).product -ne 'moe4all-agent') {
    throw 'Packing requires an Agent release staging directory.'
}
if (Test-Path -LiteralPath (Join-Path $harness 'agent-state')) { throw 'Do not pack an initialized Agent installation.' }
$manifestPath = Join-Path $harness 'runtime\runtime-payload.json'
$bootstrapRoot = Join-Path $harness 'bootstrap'
$bootstrapNode = Join-Path $bootstrapRoot 'node.exe'
if (-not (Test-Path -LiteralPath $bootstrapNode -PathType Leaf)) {
    $runtimeNode = Get-ChildItem -LiteralPath (Join-Path $harness 'runtime') -Directory -Filter 'node-v*-win-x64' |
        ForEach-Object { Join-Path $_.FullName 'node.exe' } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
    if (-not $runtimeNode) { throw 'The bundled Node executable is missing.' }
    New-Item -ItemType Directory -Path $bootstrapRoot -Force | Out-Null
    Move-Item -LiteralPath $runtimeNode -Destination $bootstrapNode
}
if (Test-Path -LiteralPath $manifestPath) {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    foreach ($entry in $manifest.archives) {
        $archive = [IO.Path]::GetFullPath((Join-Path $harness $entry.file))
        if (-not $archive.StartsWith($harness + '\', [StringComparison]::OrdinalIgnoreCase) -or
            (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
            throw 'The reused runtime payload is invalid.'
        }
    }
    return
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$nodeDirectory = Get-ChildItem -LiteralPath (Join-Path $harness 'runtime') -Directory -Filter 'node-v*-win-x64' | Select-Object -First 1
if (-not $nodeDirectory) { throw 'The bundled Node runtime is missing.' }
$groups = @(
    @{ root='runtime\dsh'; file='runtime\dsh-data.zip'; keep=$null },
    @{ root='distribution\prebuilt-state'; file='distribution\profile-data.zip'; keep=$null },
    @{ root=('runtime\' + $nodeDirectory.Name); file='runtime\node-data.zip'; keep=$null }
)
$archives = @()
foreach ($group in $groups) {
    $source = [IO.Path]::GetFullPath((Join-Path $harness $group.root))
    if (-not $source.StartsWith($harness + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Runtime source escapes staging.' }
    if (-not (Test-Path -LiteralPath $source -PathType Container)) { throw "Runtime source is missing: $source" }
    if (Test-Path -LiteralPath (Join-Path $source '.moe4all-links-ready')) { throw 'Do not pack restored runtime junctions.' }
    $destination = Join-Path $harness $group.file
    if (Test-Path -LiteralPath $destination) { throw "Payload already exists: $destination" }
    $writer = [IO.Compression.ZipFile]::Open($destination, [IO.Compression.ZipArchiveMode]::Create)
    $count = 0
    try {
        foreach ($filename in [IO.Directory]::EnumerateFiles($source, '*', [IO.SearchOption]::AllDirectories)) {
            $relative = $filename.Substring($source.Length + 1).Replace('\', '/')
            if ($group.keep -and $relative -eq $group.keep) { continue }
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($writer, $filename, $relative, [IO.Compression.CompressionLevel]::Optimal) | Out-Null
            $count++
        }
        # Preserve empty workspace/agents directories from the portable state seed.
        foreach ($directory in [IO.Directory]::EnumerateDirectories($source, '*', [IO.SearchOption]::AllDirectories)) {
            if (([IO.Directory]::GetFileSystemEntries($directory)).Count -eq 0) {
                $relative = $directory.Substring($source.Length + 1).Replace('\', '/') + '/'
                $writer.CreateEntry($relative) | Out-Null
            }
        }
    } finally { $writer.Dispose() }
    $archives += [ordered]@{
        file=$group.file
        root=$group.root
        files=$count
        sha256=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    foreach ($entry in Get-ChildItem -LiteralPath $source -Force) {
        if ($group.keep -and $entry.Name -eq $group.keep) { continue }
        $target = [IO.Path]::GetFullPath($entry.FullName)
        if (-not $target.StartsWith($source + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Cleanup path escapes staging.' }
        Remove-Item -LiteralPath $target -Recurse -Force
    }
    Write-Host "Packed $($group.root): $count files"
}
[IO.File]::WriteAllText($manifestPath, (([ordered]@{schema=1;archives=$archives} | ConvertTo-Json -Depth 5) + "`n"), (New-Object Text.UTF8Encoding($false)))
