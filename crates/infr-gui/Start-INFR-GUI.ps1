[CmdletBinding()]
param(
    [string]$ListenAddress = '0.0.0.0:8180'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$dataDir = Join-Path $repoRoot 'gui-data'
$keyFile = Join-Path $dataDir 'admin.key'
$guiExe = Join-Path $repoRoot 'target\release\infr-gui.exe'
$infrExe = Join-Path $repoRoot 'target\release\infr.exe'

New-Item -ItemType Directory -Path $dataDir -Force | Out-Null

if (Test-Path -LiteralPath $keyFile) {
    $adminKey = (Get-Content -LiteralPath $keyFile -Raw).Trim()
}
else {
    $adminKey = [Guid]::NewGuid().ToString('N')
    Set-Content -LiteralPath $keyFile -Value $adminKey -Encoding Ascii
}
if ([string]::IsNullOrWhiteSpace($adminKey)) {
    throw "Management key file is empty: $keyFile"
}

$cargoCommand = Get-Command 'cargo.exe' -ErrorAction SilentlyContinue
if ($null -ne $cargoCommand) {
    $cargoExe = $cargoCommand.Source
}
else {
    $cargoExe = Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe'
}
if (-not (Test-Path -LiteralPath $cargoExe -PathType Leaf)) {
    throw "cargo.exe was not found. Install Rust or add cargo.exe to PATH."
}

Write-Host 'Building INFR worker and browser control plane...'
Push-Location $repoRoot
try {
    $buildArgs = @('build', '--release', '-p', 'infr-cli', '-p', 'infr-gui')
    & $cargoExe @buildArgs
    $buildExit = $LASTEXITCODE
    if ($buildExit -ne 0) {
        throw "cargo build failed with exit code $buildExit"
    }
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host "INFR GUI management key: $adminKey" -ForegroundColor Yellow
Write-Host "Local URL: http://127.0.0.1:$($ListenAddress.Split(':')[-1])"
try {
    $ztAddress = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
        Where-Object { $_.InterfaceAlias -like '*ZeroTier*' -and $_.IPAddress -notlike '169.254.*' } |
        Select-Object -First 1 -ExpandProperty IPAddress
    if ($ztAddress) {
        Write-Host "ZeroTier URL: http://${ztAddress}:$($ListenAddress.Split(':')[-1])" -ForegroundColor Cyan
    }
}
catch {
    Write-Verbose "Could not discover a ZeroTier IPv4 address: $($_.Exception.Message)"
}
Write-Host 'Press Ctrl+C to stop the GUI. Its managed worker will be drained first.'
Write-Host ''

$guiArgs = @(
    '--addr'
    $ListenAddress
    '--key-file'
    $keyFile
    '--infr'
    $infrExe
    '--data-dir'
    $dataDir
    '--workdir'
    $repoRoot
)
& $guiExe @guiArgs
$guiExit = $LASTEXITCODE
if ($guiExit -ne 0) {
    throw "infr-gui.exe failed with exit code $guiExit"
}
