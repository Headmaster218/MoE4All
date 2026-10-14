[CmdletBinding()]
param([string]$OutputDirectory)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root 'artifacts\agent-runtime-test' }
$fixture = Join-Path ([IO.Path]::GetFullPath($OutputDirectory)) ('runtime-test-' + [guid]::NewGuid().ToString('N'))
while ((Join-Path $fixture 'harness\bootstrap\node.exe').Length -le 260) {
    $fixture = Join-Path $fixture 'long'
}
$harness = Join-Path $fixture 'harness'
$runtime = Join-Path $harness 'runtime'
$nodeRoot = Join-Path $harness 'bootstrap'
$cli = Join-Path $runtime 'dsh\lib\bin.js'
$state = Join-Path $fixture 'data'
$workspace = Join-Path $state 'workspace'
$capture = Join-Path $fixture 'capture.json'

New-Item -ItemType Directory -Path (Join-Path $harness 'scripts'), $nodeRoot, (Split-Path -Parent $cli), (Join-Path $state 'home\profiles\web\node_modules'), $workspace -Force | Out-Null
$node = (Get-Command node -CommandType Application -ErrorAction Stop).Source
Copy-Item -LiteralPath $node -Destination (Join-Path $nodeRoot 'node.exe')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Prepare-BundledRuntime.ps1') -Destination (Join-Path $harness 'scripts\Prepare-BundledRuntime.ps1') -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Run-Agent.ps1') -Destination (Join-Path $harness 'scripts\Run-Agent.ps1') -Force
[IO.File]::WriteAllText((Join-Path $harness 'scripts\Prepare-BundledRuntime.cjs'), "process.exitCode = 0;`n",
    (New-Object Text.UTF8Encoding($false)))
[IO.File]::WriteAllText($cli, @'
const fs = require('node:fs');
fs.writeFileSync(process.env.MOE4ALL_TEST_CAPTURE, JSON.stringify({
  argv: process.argv.slice(2),
  root: process.env.MOE4ALL_ROOT,
  data: process.env.MOE4ALL_DATA_DIR,
  models: process.env.MOE4ALL_MODELS_DIR,
  cache: process.env.MOE4ALL_CACHE_DIR,
  bridge: process.env.MOE4ALL_DESKTOP_BRIDGE,
  token: process.env.MOE4ALL_DESKTOP_TOKEN,
}));
'@, (New-Object Text.UTF8Encoding($false)))

$oldCapture = $env:MOE4ALL_TEST_CAPTURE
$env:MOE4ALL_TEST_CAPTURE = $capture
try {
    $windowsPowerShell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    & $windowsPowerShell -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `
        (Join-Path $harness 'scripts\Prepare-BundledRuntime.ps1') -InstallRoot $fixture -StateRoot $state
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $fixture 'node.exe') -PathType Leaf)) {
        throw 'Long-path Node bootstrap did not materialize the root runtime.'
    }
    & (Join-Path $harness 'scripts\Run-Agent.ps1') -StateRoot $state -WorkspaceRoot $workspace `
        -HostAddress '127.0.0.1' -Port 18080 -DesktopBridge 'http://127.0.0.1:19090' -DesktopToken 'test-token'
    if ($LASTEXITCODE -ne 0) { throw "Run-Agent fixture exited with code $LASTEXITCODE" }
} finally {
    $env:MOE4ALL_TEST_CAPTURE = $oldCapture
}

$actual = Get-Content -LiteralPath $capture -Raw | ConvertFrom-Json
$expectedArgs = @('web', '--host', '127.0.0.1', '--port', '18080', '--no-open')
if (($actual.argv -join '|') -ne ($expectedArgs -join '|')) { throw "Unexpected Node arguments: $($actual.argv -join ' ')" }
if ($actual.root -ne $fixture -or $actual.data -ne $state -or $actual.models -ne (Join-Path $fixture 'models') -or
    $actual.cache -ne (Join-Path $fixture 'cache') -or $actual.bridge -ne 'http://127.0.0.1:19090' -or $actual.token -ne 'test-token') {
    throw "Managed Agent environment was not isolated under the product root: $($actual | ConvertTo-Json -Compress)"
}

Write-Host 'Agent runtime test passed: CLI arguments and managed product directories remain intact.'
Write-Host "Evidence: $fixture"
