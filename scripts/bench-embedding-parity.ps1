[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Model,

    [string]$Infr = '',
    [string]$EmbeddingRunner = '',
    [string]$Device = 'Vulkan0',
    [string]$Address = '127.0.0.1:18384',
    [ValidateRange(1, 100)]
    [int]$Repeat = 5,
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'

function Invoke-EmbeddingRequest {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$InputText,
        [Parameter(Mandatory = $true)]
        [string]$Endpoint,
        [Parameter(Mandatory = $true)]
        [string]$ApiKey
    )

    $payload = @{
        model = 'embedding-baseline'
        input = $InputText
        encoding_format = 'float'
    } | ConvertTo-Json -Depth 4 -Compress
    $headers = @{ Authorization = "Bearer $ApiKey" }
    Invoke-RestMethod -Method Post -Uri "$Endpoint/v1/embeddings" -Headers $headers `
        -ContentType 'application/json; charset=utf-8' -Body $payload -TimeoutSec 600
}

function ConvertFrom-Utf8Base64 {
    param([Parameter(Mandatory = $true)][string]$Value)
    [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($Value))
}

function ConvertTo-WindowsCommandLineArgument {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }
    # CommandLineToArgvW quoting: double backslashes before a quote and at the closing quote.
    $escaped = [System.Text.RegularExpressions.Regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [System.Text.RegularExpressions.Regex]::Replace($escaped, '(\\+)$', '$1$1')
    '"' + $escaped + '"'
}

if ([string]::IsNullOrWhiteSpace($Infr)) {
    $Infr = Join-Path $PSScriptRoot '..\target\release\infr.exe'
}
$infrPath = [System.IO.Path]::GetFullPath($Infr)
$modelPath = [System.IO.Path]::GetFullPath($Model)
if (-not (Test-Path -LiteralPath $infrPath -PathType Leaf)) {
    throw "infr executable was not found: $infrPath"
}
if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
    throw "embedding model was not found: $modelPath"
}

$endpoint = "http://$Address"
$apiKey = "embedding-benchmark-$([Guid]::NewGuid().ToString('N'))"
$stopFile = Join-Path ([System.IO.Path]::GetTempPath()) "infr-embedding-bench-$PID.stop"
if ([string]::IsNullOrWhiteSpace($Output)) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $Output = Join-Path $PSScriptRoot "..\target\perf\embedding-baseline-$stamp.json"
}
$outputPath = [System.IO.Path]::GetFullPath($Output)
$outputDir = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue

$nativeArgs = @(
    '--set'
    "device.dev=$Device"
    '--set'
    "serve.shutdown_file=$stopFile"
    'serve-embedding'
    $modelPath
    '--addr'
    $Address
    '--parallel'
    '1'
)
if (-not [string]::IsNullOrWhiteSpace($EmbeddingRunner)) {
    $nativeArgs += @('--embedding-runner', ([System.IO.Path]::GetFullPath($EmbeddingRunner)))
}

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $infrPath
$startInfo.WorkingDirectory = Split-Path -Parent $infrPath
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.Environment['INFR_API_KEY'] = $apiKey
$startInfo.Arguments = (($nativeArgs | ForEach-Object {
    ConvertTo-WindowsCommandLineArgument ([string]$_)
}) -join ' ')

$process = [System.Diagnostics.Process]::Start($startInfo)
try {
    $ready = $false
    $deadline = [DateTime]::UtcNow.AddMinutes(3)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($process.HasExited) {
            $stdout = $process.StandardOutput.ReadToEnd()
            $stderr = $process.StandardError.ReadToEnd()
            throw "embedding server exited during startup ($($process.ExitCode))`n$stdout`n$stderr"
        }
        try {
            $health = Invoke-WebRequest -Uri "$endpoint/health" -TimeoutSec 2 -UseBasicParsing
            if ($health.StatusCode -eq 200) {
                $ready = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 250
        }
    }
    if (-not $ready) {
        throw 'embedding server did not become ready within three minutes'
    }

    $zhShort = ConvertFrom-Utf8Base64 '5LuK5aSp5aSp5rCU5b6I5aW977yM5oiR5Lus5Y675YWs5Zut5pWj5q2l44CC'
    $zhQuestion = ConvertFrom-Utf8Base64 '5oCO5qC35o+Q6auY5pys5Zyw5aSn5qih5Z6L5o6o55CG6YCf5bqm77yf'
    $zhBatchTemplate = ConvertFrom-Utf8Base64 '55So5LqO5om56YeP5bWM5YWl5Z+65YeG55qE56ysIHtOfSDmnaHmlofmnKzjgII='
    $zhLong = ConvertFrom-Utf8Base64 '6ZW/5paH5pys6K+t5LmJ5qOA57Si5Z+65YeG44CC'
    $cases = @(
        @{ name = 'zh-short'; input = @($zhShort) },
        @{ name = 'en-short'; input = @('A quick brown fox jumps over the lazy dog.') },
        @{ name = 'semantic-pair'; input = @($zhQuestion, 'How can local LLM inference be accelerated?') },
        @{ name = 'batch-8'; input = 1..8 | ForEach-Object { $zhBatchTemplate.Replace('{N}', $_) } },
        # Keep the reference request below llama.cpp's default physical batch of 512 tokens.
        @{ name = 'long'; input = @(($zhLong * 40)) }
    )

    # Compile/warm the backend before collecting timings.
    $null = Invoke-EmbeddingRequest -InputText @('warmup') -Endpoint $endpoint -ApiKey $apiKey

    $results = foreach ($case in $cases) {
        $samples = @()
        $response = $null
        for ($iteration = 0; $iteration -lt $Repeat; $iteration++) {
            $watch = [System.Diagnostics.Stopwatch]::StartNew()
            $response = Invoke-EmbeddingRequest -InputText $case.input -Endpoint $endpoint -ApiKey $apiKey
            $watch.Stop()
            $samples += $watch.Elapsed.TotalMilliseconds
        }
        $ordered = @($samples | Sort-Object)
        $middle = [int][Math]::Floor($ordered.Count / 2)
        [pscustomobject]@{
            name = $case.name
            input = $case.input
            prompt_tokens = $response.usage.prompt_tokens
            dimensions = $response.data[0].embedding.Count
            elapsed_ms = $samples
            median_ms = $ordered[$middle]
            embeddings = @($response.data | Sort-Object index | ForEach-Object { $_.embedding })
        }
    }

    $report = [pscustomobject]@{
        schema = 1
        generated_at = (Get-Date).ToUniversalTime().ToString('o')
        implementation = 'infr external llama.cpp embedding oracle'
        model = $modelPath
        device = $Device
        repeat = $Repeat
        cases = @($results)
    }
    $report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $outputPath -Encoding utf8
    Write-Host "Embedding baseline written to $outputPath"
}
finally {
    Set-Content -LiteralPath $stopFile -Value 'stop' -Encoding ascii -ErrorAction SilentlyContinue
    if (-not $process.WaitForExit(30000)) {
        $process.Kill($true)
        $process.WaitForExit()
    }
    Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue
}
