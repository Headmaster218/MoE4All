param(
    [string]$OutputDirectory = (Join-Path ([Environment]::GetFolderPath('ApplicationData')) 'infr-console\certs'),
    [int]$MinDays = 14,
    [switch]$Force
)

# Replaces the self-signed console certificate with the publicly trusted one Tailscale issues
# for this machine's MagicDNS name. Browsers let you click through an untrusted certificate;
# third-party API clients do not, and fail the TLS handshake with a network level error.
#
# `tailscale cert` reuses its cached certificate while it is still valid, so calling this on
# every launch costs almost nothing and keeps the 90-day certificate from expiring unnoticed.

$ErrorActionPreference = 'Stop'

function Find-Tailscale {
    $command = Get-Command tailscale.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $installed = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
    if (Test-Path -LiteralPath $installed -PathType Leaf) { return $installed }
    return $null
}

function Get-CertificateInfo {
    param([string]$Path)
    try {
        $certificate = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($Path)
        return [pscustomobject]@{
            Subject = $certificate.Subject
            Days    = [int][Math]::Floor(($certificate.NotAfter.ToUniversalTime() - [DateTime]::UtcNow).TotalDays)
        }
    }
    catch {
        return $null
    }
}

$tailscale = Find-Tailscale
if (-not $tailscale) {
    Write-Output 'Tailscale not found - keeping the current certificate.'
    exit 2
}

try {
    $raw = & $tailscale status --json 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) { throw "tailscale status exited with $LASTEXITCODE" }
    $status = $raw | ConvertFrom-Json
}
catch {
    Write-Output "Cannot read Tailscale status - keeping the current certificate. ($_)"
    exit 2
}

$domain = "$($status.Self.DNSName)".TrimEnd('.')
if ([string]::IsNullOrWhiteSpace($domain)) {
    Write-Output 'This machine has no MagicDNS name - keeping the current certificate.'
    exit 2
}

$certPath = Join-Path $OutputDirectory 'console-cert.pem'
$keyPath = Join-Path $OutputDirectory 'console-key.pem'

if (-not $Force -and (Test-Path -LiteralPath $certPath -PathType Leaf) -and (Test-Path -LiteralPath $keyPath -PathType Leaf)) {
    $info = Get-CertificateInfo $certPath
    # A self-signed certificate has no MagicDNS name in its subject, and a stale one is about to
    # expire; both cases fall through and get replaced.
    if ($info -and $info.Subject -like "*$domain*" -and $info.Days -ge $MinDays) {
        Write-Output "Certificate for $domain is valid for $($info.Days) more day(s); nothing to do."
        exit 0
    }
}

$temp = Join-Path ([IO.Path]::GetTempPath()) ('infr-ts-cert-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temp -Force | Out-Null

try {
    # `tailscale cert` writes <domain>.crt / <domain>.key into the working directory. Passing the
    # paths as flags trips argument parsing on some builds, so run it from a scratch directory.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        Push-Location $temp
        try {
            $output = & $tailscale cert $domain 2>&1
            $exitCode = $LASTEXITCODE
        }
        finally {
            Pop-Location
        }
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($exitCode -ne 0) {
        throw "tailscale cert failed with exit code $exitCode`n$($output | Out-String)"
    }

    $issuedCert = Join-Path $temp "$domain.crt"
    $issuedKey = Join-Path $temp "$domain.key"
    if (-not (Test-Path -LiteralPath $issuedCert -PathType Leaf) -or -not (Test-Path -LiteralPath $issuedKey -PathType Leaf)) {
        throw 'tailscale cert did not produce the expected files.'
    }

    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    Copy-Item -LiteralPath $issuedCert -Destination $certPath -Force
    Copy-Item -LiteralPath $issuedKey -Destination $keyPath -Force
}
catch {
    # Never leave a half-installed pair behind: HTTPS would fail to start on a mismatched pair.
    Write-Output "Could not install a Tailscale certificate - keeping the current one. ($_)"
    exit 1
}
finally {
    Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
}

$installed = Get-CertificateInfo $certPath
$remaining = if ($installed) { $installed.Days } else { -1 }
Write-Output "Installed the Tailscale certificate for $domain (valid for $remaining day(s))."
Write-Output "Restart the console to use it, then reach it at https://${domain}:8090"
