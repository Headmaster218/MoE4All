[CmdletBinding()]
param(
    [string]$StateRoot,
    [string]$EngineEndpoint = ''
)

$ErrorActionPreference = 'Stop'
$harnessRoot = Split-Path -Parent $PSScriptRoot
if (-not $StateRoot) { $StateRoot = Join-Path $harnessRoot 'agent-state' }
$state = [System.IO.Path]::GetFullPath($StateRoot)
$agentHome = Join-Path $state 'home'
$profile = Join-Path $agentHome 'profiles\web'
$workspace = Join-Path $state 'workspace'
$sourceHome = [System.IO.Path]::GetFullPath((Join-Path $env:APPDATA 'dsh-desktop\harness'))
$legacyHome = [System.IO.Path]::GetFullPath((Join-Path $env:USERPROFILE '.dsh'))
$cursor = $state
while ($cursor) {
    if (Test-Path -LiteralPath $cursor) {
        $entry = Get-Item -LiteralPath $cursor -Force
        if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "MoE4All state cannot traverse a junction or symlink: $cursor"
        }
    }
    $parent = Split-Path -Parent $cursor
    if (-not $parent -or $parent -eq $cursor) { break }
    $cursor = $parent
}
$forbiddenHomes = @($sourceHome, $legacyHome)
if ($env:DSH_HOME) {
    $inheritedHome = [System.IO.Path]::GetFullPath($env:DSH_HOME)
    if (-not $inheritedHome.Equals($agentHome, [System.StringComparison]::OrdinalIgnoreCase)) {
        $forbiddenHomes += $inheritedHome
    }
}
foreach ($forbidden in $forbiddenHomes) {
    if ($state.Equals($forbidden, [System.StringComparison]::OrdinalIgnoreCase) -or
        $state.StartsWith($forbidden + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "MoE4All state cannot live inside a DSH home: $state"
    }
}

$endpoint = $null
if ($EngineEndpoint) {
    $endpoint = [Uri]$EngineEndpoint
    if ($endpoint.Scheme -notin @('http', 'https') -or $endpoint.UserInfo -or $endpoint.Query -or $endpoint.Fragment) {
        throw 'EngineEndpoint must be an http(s) URL without credentials, query, or fragment.'
    }
}
if ($null -eq $endpoint) { $endpoint = [Uri]'http://127.0.0.1:1234/v1' }

$manifestPath = Join-Path $profile 'package.json'
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    Write-Host "Existing MoE4All Agent profile preserved: $profile"
    return
}

New-Item -ItemType Directory -Path $profile, $workspace, (Join-Path $state 'agents') -Force | Out-Null
$sourceProfile = Join-Path $harnessRoot 'distribution\profile'
$manifest = Get-Content -LiteralPath (Join-Path $sourceProfile 'package.json') -Raw | ConvertFrom-Json
foreach ($dependency in $manifest.dependencies.PSObject.Properties) {
    $dependency.Value = "file:../../../../runtime/plugin-packs/$($dependency.Name).tgz"
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($manifestPath, (($manifest | ConvertTo-Json -Depth 12) + "`n"), $utf8)
[System.IO.File]::WriteAllText((Join-Path $profile 'cordis.yml'), "[]`n", $utf8)

$patch = @'
- id: agent-default-model
  config:
    provider: moe4all
    model: select-a-model

- id: llm-deepseek
  disabled: true

- id: web
  config:
    searchProvider: brave-search

- id: remote-web-ui-moe4all
  config:
    cookieName: moe4all_pair

- id: dsh-market-moe4all
  config:
    allowRestart: false

- id: moe4all-engine
  config:
    mode: connect
    endpoint: __ENGINE_ENDPOINT__
    allowRemoteEndpoint: true
'@
$patch = $patch.Replace('__ENGINE_ENDPOINT__', $endpoint.AbsoluteUri.TrimEnd('/'))
[System.IO.File]::WriteAllText((Join-Path $profile 'cordis.patch.yml'), "$patch`n", $utf8)

$workspaceYaml = @'
packages:
  - .

overrides:
  '@huggingface/transformers': link:./stubs/hf-transformers-stub

nodeLinker: hoisted
autoInstallPeers: false

allowBuilds:
  cloudflared: false
'@
[System.IO.File]::WriteAllText((Join-Path $profile 'pnpm-workspace.yaml'), "$workspaceYaml`n", $utf8)
$npmrc = @'
package-import-method=clone-or-copy
child-concurrency=4
side-effects-cache=false
store-dir=../../../../runtime/pnpm-store
ignore-scripts=true
'@
[System.IO.File]::WriteAllText((Join-Path $profile '.npmrc'), "$npmrc`n", $utf8)
Copy-Item -LiteralPath (Join-Path $sourceProfile 'stubs') -Destination $profile -Recurse

$settings = "ui-theme:`n  preference: dark`n"
try {
    $models = Invoke-RestMethod -Uri ($endpoint.AbsoluteUri.TrimEnd('/') + '/models') -TimeoutSec 5
    $model = $models.data | Where-Object { $_.id -and $_.id -notmatch '(?i)embed' } | Select-Object -First 1
    if ($null -ne $model) {
        $modelId = ConvertTo-Json -InputObject ([string]$model.id) -Compress
        $settings += "agent-default-model:`n  provider: moe4all`n  model: $modelId`n"
    }
} catch {
    Write-Warning 'Model discovery is unavailable; select a model in the MoE4All UI after startup.'
}
[System.IO.File]::WriteAllText((Join-Path $agentHome 'settings.yaml'), $settings, $utf8)

Write-Host "MoE4All state: $state"
Write-Host "MoE4All profile: $profile"
Write-Host "Isolated workspace: $workspace"
