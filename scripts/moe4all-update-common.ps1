function ConvertTo-Moe4AllSemVer {
    param([Parameter(Mandatory = $true)][string]$Value)

    $trimmed = $Value.Trim()
    if ($trimmed -notmatch '^(?:release-|v)?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?$') {
        return $null
    }
    return [pscustomobject]@{
        Major = [int]$Matches[1]
        Minor = [int]$Matches[2]
        Patch = [int]$Matches[3]
        PreRelease = [string]$Matches[4]
        BuildMetadata = [string]$Matches[5]
        Text = "$($Matches[1]).$($Matches[2]).$($Matches[3])$(if ($Matches[4]) { '-' + $Matches[4] } else { '' })$(if ($Matches[5]) { '+' + $Matches[5] } else { '' })"
    }
}

function Compare-Moe4AllSemVer {
    param(
        [Parameter(Mandatory = $true)]$Left,
        [Parameter(Mandatory = $true)]$Right
    )

    foreach ($part in @('Major', 'Minor', 'Patch')) {
        if ($Left.$part -lt $Right.$part) { return -1 }
        if ($Left.$part -gt $Right.$part) { return 1 }
    }
    if (-not $Left.PreRelease -and -not $Right.PreRelease) { return 0 }
    if (-not $Left.PreRelease) { return 1 }
    if (-not $Right.PreRelease) { return -1 }

    $leftParts = @($Left.PreRelease -split '\.')
    $rightParts = @($Right.PreRelease -split '\.')
    $count = [Math]::Max($leftParts.Count, $rightParts.Count)
    for ($index = 0; $index -lt $count; $index++) {
        if ($index -ge $leftParts.Count) { return -1 }
        if ($index -ge $rightParts.Count) { return 1 }
        $leftNumber = 0L
        $rightNumber = 0L
        $leftNumeric = [long]::TryParse($leftParts[$index], [ref]$leftNumber)
        $rightNumeric = [long]::TryParse($rightParts[$index], [ref]$rightNumber)
        if ($leftNumeric -and $rightNumeric) {
            if ($leftNumber -lt $rightNumber) { return -1 }
            if ($leftNumber -gt $rightNumber) { return 1 }
            continue
        }
        if ($leftNumeric -and -not $rightNumeric) { return -1 }
        if (-not $leftNumeric -and $rightNumeric) { return 1 }
        $comparison = [string]::CompareOrdinal($leftParts[$index], $rightParts[$index])
        if ($comparison -lt 0) { return -1 }
        if ($comparison -gt 0) { return 1 }
    }
    return 0
}

function Select-Moe4AllEngineRelease {
    param([Parameter(Mandatory = $true)][object[]]$Releases)

    $selected = $null
    foreach ($release in @($Releases)) {
        if ([bool]$release.draft -or [bool]$release.prerelease) { continue }
        $tag = [string]$release.tag_name
        if ($tag -notmatch '^release-(.+)$') { continue }
        $version = ConvertTo-Moe4AllSemVer $Matches[1]
        if ($null -eq $version) { continue }

        $archiveName = "MoE4All-Windows-x86_64-v$($version.Text).zip"
        $archive = @($release.assets) | Where-Object { [string]$_.name -eq $archiveName } | Select-Object -First 1
        $checksum = @($release.assets) | Where-Object { [string]$_.name -eq "$archiveName.sha256" } | Select-Object -First 1
        if ($null -eq $archive -or $null -eq $checksum) { continue }

        $candidate = [pscustomobject]@{
            Tag = $tag
            Version = $version.Text
            SemVer = $version
            Name = $(if ([string]::IsNullOrWhiteSpace([string]$release.name)) { $tag } else { [string]$release.name })
            Url = [string]$release.html_url
            Notes = [string]$release.body
            PublishedAt = [string]$release.published_at
            ArchiveName = $archiveName
            ArchiveUrl = [string]$archive.browser_download_url
            ArchiveSize = [long]$archive.size
            ChecksumName = "$archiveName.sha256"
            ChecksumUrl = [string]$checksum.browser_download_url
        }
        if ($null -eq $selected -or (Compare-Moe4AllSemVer $candidate.SemVer $selected.SemVer) -gt 0) {
            $selected = $candidate
        }
    }
    return $selected
}

function Resolve-Moe4AllManagedPath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    if ([string]::IsNullOrWhiteSpace($RelativePath) -or [System.IO.Path]::IsPathRooted($RelativePath)) {
        throw "Invalid managed path: $RelativePath"
    }
    $rootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
    $normalized = $RelativePath.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
    $fullPath = [System.IO.Path]::GetFullPath((Join-Path $rootPath $normalized))
    $prefix = $rootPath + [System.IO.Path]::DirectorySeparatorChar
    if (-not $fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Managed path escapes its root: $RelativePath"
    }
    return $fullPath
}

function Read-Moe4AllInstallManifest {
    param(
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [string]$PackageRoot = '',
        [switch]$VerifyFiles
    )

    $manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([int]$manifest.schema_version -ne 1 -or [string]$manifest.product -ne 'moe4all-engine') {
        throw "Unsupported MoE4All install manifest: $ManifestPath"
    }
    if ([int]$manifest.updater_protocol -ne 1) {
        throw "Unsupported updater protocol in $ManifestPath"
    }
    if ($null -eq (ConvertTo-Moe4AllSemVer ([string]$manifest.version))) {
        throw "Invalid manifest version in $ManifestPath"
    }

    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in @($manifest.files)) {
        $relative = [string]$entry.path
        if (-not $seen.Add($relative)) { throw "Duplicate managed path in manifest: $relative" }
        if ($entry.sha256 -notmatch '^[a-fA-F0-9]{64}$') { throw "Invalid SHA-256 for $relative" }
        if ([long]$entry.size -lt 0) { throw "Invalid file size for $relative" }
        if ($VerifyFiles) {
            if ([string]::IsNullOrWhiteSpace($PackageRoot)) { throw 'PackageRoot is required when VerifyFiles is set.' }
            $path = Resolve-Moe4AllManagedPath -Root $PackageRoot -RelativePath $relative
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Managed package file is missing: $relative" }
            $item = Get-Item -LiteralPath $path
            if ($item.Length -ne [long]$entry.size) { throw "Managed package file has the wrong size: $relative" }
            $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
            if (-not $actual.Equals([string]$entry.sha256, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Managed package file failed SHA-256 verification: $relative"
            }
        } else {
            [void](Resolve-Moe4AllManagedPath -Root $(if ($PackageRoot) { $PackageRoot } else { [System.IO.Path]::GetTempPath() }) -RelativePath $relative)
        }
    }
    return $manifest
}

function Format-Moe4AllReleaseNotes {
    param([AllowEmptyString()][string]$Notes)

    if ([string]::IsNullOrWhiteSpace($Notes)) {
        return 'No release notes were provided.'
    }
    $clean = $Notes -replace "`r`n", "`n"
    $clean = $clean -replace "`r", "`n"
    $clean = $clean -replace '\x1B\[[0-?]*[ -/]*[@-~]', ''
    $lines = foreach ($line in ($clean -split "`n")) {
        $value = $line -replace '^\s{0,3}#{1,6}\s*', ''
        $value = $value -replace '^\s*[*+]\s+', '- '
        $value = $value -replace '!\[[^\]]*\]\([^)]*\)', ''
        $value = $value -replace '\[([^\]]+)\]\((https?://[^)]+)\)', '$1 ($2)'
        $value -replace '[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ''
    }
    return ($lines -join [Environment]::NewLine).Trim()
}
