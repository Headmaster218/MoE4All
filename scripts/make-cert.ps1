param(
    [string]$OutputDirectory = (Join-Path ([Environment]::GetFolderPath('ApplicationData')) 'infr-console\certs'),
    [int]$Days = 825,
    [switch]$Trust,
    [string[]]$DnsNames = @(),
    [string[]]$IpAddresses = @()
)

$ErrorActionPreference = 'Stop'

function Find-OpenSsl {
    $command = Get-Command openssl.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidates = @(
        (Join-Path $env:ProgramFiles 'Git\usr\bin\openssl.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Git\usr\bin\openssl.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Git\usr\bin\openssl.exe')
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) { return $candidate }
    }
    return $null
}

$openssl = Find-OpenSsl
if (-not $openssl) {
    throw 'openssl.exe was not found. Install Git for Windows (it ships openssl) or OpenSSL, then run this again.'
}

if ($DnsNames.Count -eq 0) {
    $DnsNames = @('localhost', [System.Net.Dns]::GetHostName())
}
if ($IpAddresses.Count -eq 0) {
    $IpAddresses = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -ne '127.0.0.1' -and $_.IPAddress -notlike '169.254.*' } |
        Select-Object -ExpandProperty IPAddress -Unique)
}
# Loopback is always kept: scripts and health probes that talk to 127.0.0.1 must not fail
# hostname verification just because the machine also has LAN addresses.
if ($IpAddresses -notcontains '127.0.0.1') {
    $IpAddresses = @('127.0.0.1') + $IpAddresses
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$certPath = Join-Path $OutputDirectory 'console-cert.pem'
$keyPath = Join-Path $OutputDirectory 'console-key.pem'

$altLines = @()
$index = 1
foreach ($name in $DnsNames) {
    $altLines += "DNS.$index=$name"
    $index++
}
$index = 1
foreach ($ip in $IpAddresses) {
    $altLines += "IP.$index=$ip"
    $index++
}

$configLines = @(
    '[req]'
    'distinguished_name=dn'
    'x509_extensions=v3'
    'prompt=no'
    '[dn]'
    'CN=INFR Console'
    '[v3]'
    'basicConstraints=critical,CA:TRUE'
    'keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign'
    'extendedKeyUsage=serverAuth'
    'subjectAltName=@alt'
    '[alt]'
) + $altLines

$configPath = Join-Path ([System.IO.Path]::GetTempPath()) ("infr-console-cert-" + [guid]::NewGuid().ToString('N') + '.cnf')
[System.IO.File]::WriteAllText($configPath, ($configLines -join "`n"), [System.Text.UTF8Encoding]::new($false))

try {
    # openssl writes its key-generation progress to stderr; PowerShell 5.1 turns redirected stderr
    # into error records, so relax the preference for the call and judge by the exit code instead.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $opensslLog = & $openssl req -x509 -newkey rsa:2048 -sha256 -days $Days -nodes `
            -keyout $keyPath -out $certPath -config $configPath 2>&1
        $opensslExit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($opensslExit -ne 0) {
        throw "openssl failed with exit code $opensslExit`n$($opensslLog | Out-String)"
    }
}
finally {
    Remove-Item -LiteralPath $configPath -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath $certPath) -or -not (Test-Path -LiteralPath $keyPath)) {
    throw 'openssl did not produce the certificate files.'
}

Write-Output "Certificate: $certPath"
Write-Output "Private key: $keyPath"
Write-Output "Names: $($DnsNames -join ', ') / $($IpAddresses -join ', ')"
Write-Output "Valid for $Days days."

if ($Trust) {
    & certutil.exe -addstore -user Root $certPath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "certutil failed with exit code $LASTEXITCODE; the certificate was not trusted."
    }
    else {
        Write-Output 'Imported into the current user''s Trusted Root store.'
    }
}

Write-Output ''
Write-Output 'Done. Restart the console (start-webui.cmd or infr-console.exe) and it will use HTTPS automatically.'
Write-Output 'Pass --no-tls to force plain HTTP when a certificate is present.'
