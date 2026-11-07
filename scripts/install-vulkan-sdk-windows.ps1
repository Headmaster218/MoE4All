[CmdletBinding()]
param(
    [string]$Version = '1.4.357.0',
    [string]$Sha256 = '81f474711e9042f4cd22b31b2f7a8870db2e428b21586fb43dd80150be97310d',
    [string]$Destination = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $baseDirectory = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
    $Destination = Join-Path $baseDirectory "VulkanSDK\$Version"
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
$binDirectory = Join-Path $Destination 'Bin'
$glslcPath = Join-Path $binDirectory 'glslc.exe'

if (-not (Test-Path -LiteralPath $glslcPath -PathType Leaf)) {
    $downloadUrl = "https://sdk.lunarg.com/sdk/download/$Version/windows/vulkan-sdk.exe"
    $installerPath = Join-Path ([System.IO.Path]::GetTempPath()) "vulkansdk-windows-X64-$Version.exe"

    Write-Host "Downloading Vulkan SDK $Version from LunarG..."
    Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath

    $actualSha256 = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $Sha256.ToLowerInvariant()) {
        throw "Vulkan SDK checksum mismatch: expected $Sha256, got $actualSha256"
    }

    $installerArgs = @(
        '--root'
        $Destination
        '--accept-licenses'
        '--default-answer'
        '--confirm-command'
        'install'
        'copy_only=1'
    )
    & $installerPath @installerArgs
    $installerExitCode = $LASTEXITCODE
    Remove-Item -LiteralPath $installerPath -Force -ErrorAction SilentlyContinue
    if ($installerExitCode -ne 0) {
        throw "Vulkan SDK installer failed with exit code $installerExitCode"
    }
}

if (-not (Test-Path -LiteralPath $glslcPath -PathType Leaf)) {
    throw "glslc.exe was not installed at $glslcPath"
}

if ($env:GITHUB_ENV) {
    Add-Content -LiteralPath $env:GITHUB_ENV -Value "VULKAN_SDK=$Destination" -Encoding UTF8
}
if ($env:GITHUB_PATH) {
    Add-Content -LiteralPath $env:GITHUB_PATH -Value $binDirectory -Encoding UTF8
}

Write-Host "VULKAN_SDK=$Destination"
& $glslcPath --version
if ($LASTEXITCODE -ne 0) {
    throw "glslc --version failed with exit code $LASTEXITCODE"
}
