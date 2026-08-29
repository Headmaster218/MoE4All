[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$InitialModelPath = '',
    [switch]$DryRun,
    [switch]$SkipUpdateCheck
)

$ErrorActionPreference = 'Stop'

try {
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = $utf8
    [Console]::OutputEncoding = $utf8
    $OutputEncoding = $utf8
} catch {
    # Some redirected hosts do not expose a mutable console encoding.
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$releaseInfrPath = Join-Path $repoRoot 'infr.exe'
$sourceInfrPath = Join-Path $repoRoot 'target\release\infr.exe'
$infrPath = if (Test-Path -LiteralPath $releaseInfrPath -PathType Leaf) {
    $releaseInfrPath
} else {
    $sourceInfrPath
}
$dataDir = Join-Path $repoRoot 'gui-data'
$statePath = Join-Path $dataDir 'wizard-state.json'
$guiStatePath = Join-Path $dataDir 'state.json'
$script:Saved = $null

if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    try {
        $script:Saved = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Warning "无法读取上次设置，将使用默认值。Could not read saved settings; defaults will be used."
    }
}

function Get-SavedValue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        $Default
    )
    if ($null -ne $script:Saved) {
        $property = $script:Saved.PSObject.Properties[$Name]
        if ($null -ne $property -and $null -ne $property.Value) {
            return $property.Value
        }
    }
    return $Default
}

function Read-TextValue {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [AllowEmptyString()][string]$Default = '',
        [switch]$Required
    )
    while ($true) {
        $suffix = if ([string]::IsNullOrWhiteSpace($Default)) { '' } else { " [$Default]" }
        $value = Read-Host "$Label$suffix"
        if ([string]::IsNullOrWhiteSpace($value)) {
            $value = $Default
        }
        $value = $value.Trim()
        if (-not $Required -or -not [string]::IsNullOrWhiteSpace($value)) {
            return $value
        }
        Write-Host '此项不能为空。This value is required.' -ForegroundColor Yellow
    }
}

function Read-IntegerValue {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [AllowEmptyString()][string]$Default = '',
        [long]$Minimum = 0,
        [switch]$AllowBlank
    )
    while ($true) {
        $value = Read-TextValue -Label $Label -Default $Default
        if ($AllowBlank -and [string]::IsNullOrWhiteSpace($value)) {
            return ''
        }
        $parsed = 0L
        if ([long]::TryParse($value, [ref]$parsed) -and $parsed -ge $Minimum) {
            return $parsed.ToString()
        }
        Write-Host "请输入不小于 $Minimum 的整数。Enter an integer greater than or equal to $Minimum." -ForegroundColor Yellow
    }
}

function Read-YesNo {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [bool]$Default = $true
    )
    $hint = if ($Default) { '[Y/n]' } else { '[y/N]' }
    while ($true) {
        $value = (Read-Host "$Label $hint").Trim().ToLowerInvariant()
        if ([string]::IsNullOrWhiteSpace($value)) {
            return $Default
        }
        if ($value -match '^(y|yes|1|true|是)$') {
            return $true
        }
        if ($value -match '^(n|no|0|false|否)$') {
            return $false
        }
        Write-Host '请输入 Y 或 N。Please enter Y or N.' -ForegroundColor Yellow
    }
}

function Read-SecretValue {
    param([Parameter(Mandatory = $true)][string]$Label)
    while ($true) {
        $secureValue = Read-Host $Label -AsSecureString
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureValue)
        try {
            $value = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        } finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            return $value
        }
        Write-Host '此项不能为空。This value is required.' -ForegroundColor Yellow
    }
}

function Read-Choice {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][object[]]$Options,
        [Parameter(Mandatory = $true)][string]$DefaultValue
    )
    Write-Host "`n$Label" -ForegroundColor Cyan
    foreach ($option in $Options) {
        $marker = if ($option.Value -eq $DefaultValue) { '*' } else { ' ' }
        Write-Host (" {0} [{1}] {2}" -f $marker, $option.Key, $option.Label)
    }
    while ($true) {
        $value = (Read-Host '选择 / Select').Trim()
        if ([string]::IsNullOrWhiteSpace($value)) {
            return $DefaultValue
        }
        $selected = $Options | Where-Object { $_.Key -eq $value -or $_.Value -eq $value } | Select-Object -First 1
        if ($null -ne $selected) {
            return [string]$selected.Value
        }
        Write-Host '选择无效。Invalid selection.' -ForegroundColor Yellow
    }
}

function ConvertTo-FullPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    $value = $Value.Trim()

    # PowerShell-aware terminals may paste a dropped file as `& 'C:\path\model.gguf'`.
    # Read-Host returns that as plain text, so remove the call operator and paired quotes before
    # treating it as a path. Repeating the quote pass also accepts a copied quoted path wrapped by
    # another pair of quotes.
    if ($value -match '^&\s*(.+)$') {
        $value = $Matches[1].Trim()
    }
    while ($value.Length -ge 2) {
        $first = $value[0]
        $last = $value[$value.Length - 1]
        if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
            $value = $value.Substring(1, $value.Length - 2).Trim()
            continue
        }
        break
    }

    $uri = $null
    if ($value.StartsWith('file:', [System.StringComparison]::OrdinalIgnoreCase) -and
        [System.Uri]::TryCreate($value, [System.UriKind]::Absolute, [ref]$uri) -and $uri.IsFile) {
        $value = $uri.LocalPath
    }
    if ([System.IO.Path]::IsPathRooted($value)) {
        return [System.IO.Path]::GetFullPath($value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $value))
}

function Select-ModelPath {
    param([AllowEmptyString()][string]$InitialPath = '')

    if (-not [string]::IsNullOrWhiteSpace($InitialPath)) {
        try {
            $modelPath = ConvertTo-FullPath $InitialPath
        } catch {
            throw "启动参数中的模型路径无效 / Invalid model path passed to the launcher: $InitialPath"
        }
        if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
            throw "找不到启动参数中的模型文件 / Model file passed to the launcher was not found: $modelPath"
        }
        if ([System.IO.Path]::GetExtension($modelPath) -ne '.gguf') {
            throw "启动参数必须指向 GGUF 文件 / The launcher argument must point to a GGUF file: $modelPath"
        }
        Write-Host "`n已从启动参数选择模型 / Model selected from launcher argument" -ForegroundColor Cyan
        Write-Host " $modelPath"
        return $modelPath
    }

    $models = [System.Collections.Generic.List[string]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

    $savedModel = [string](Get-SavedValue -Name 'model' -Default '')
    if (-not [string]::IsNullOrWhiteSpace($savedModel) -and (Test-Path -LiteralPath $savedModel -PathType Leaf)) {
        if ($seen.Add($savedModel)) { [void]$models.Add($savedModel) }
    }

    if (Test-Path -LiteralPath $guiStatePath -PathType Leaf) {
        try {
            $guiState = Get-Content -LiteralPath $guiStatePath -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($candidate in @($guiState.recent) + @($guiState.favorites) + @($guiState.profiles | ForEach-Object { $_.model_path })) {
                $path = [string]$candidate
                if (-not [string]::IsNullOrWhiteSpace($path) -and (Test-Path -LiteralPath $path -PathType Leaf)) {
                    if ($seen.Add($path)) { [void]$models.Add($path) }
                }
            }
        } catch {
            Write-Warning '无法读取 GUI 最近模型列表。Could not read the GUI recent-model list.'
        }
    }

    if ($models.Count -gt 0) {
        Write-Host "`n模型 / Model" -ForegroundColor Cyan
        for ($i = 0; $i -lt $models.Count; $i++) {
            Write-Host (" [{0}] {1}" -f ($i + 1), $models[$i])
        }
        Write-Host ' [N] 输入新路径，或直接粘贴路径 / Enter a new path, or paste it directly'
        while ($true) {
            $choice = (Read-Host '选择模型 [1] / Select model [1]').Trim()
            if ([string]::IsNullOrWhiteSpace($choice)) { return $models[0] }
            $index = 0
            if ([int]::TryParse($choice, [ref]$index) -and $index -ge 1 -and $index -le $models.Count) {
                return $models[$index - 1]
            }
            if ($choice -match '^(n|new|新)$') { break }
            try {
                $modelPath = ConvertTo-FullPath $choice
                if (Test-Path -LiteralPath $modelPath -PathType Leaf) {
                    return $modelPath
                }
            } catch {
                # Keep prompting with the common, actionable error below.
            }
            Write-Host '选择无效或找不到该模型文件。Invalid selection or model file not found.' -ForegroundColor Yellow
        }
    }

    while ($true) {
        $inputPath = Read-TextValue -Label '模型 GGUF 路径 / Model GGUF path' -Default $savedModel -Required
        try {
            $modelPath = ConvertTo-FullPath $inputPath
            if (Test-Path -LiteralPath $modelPath -PathType Leaf) {
                return $modelPath
            }
        } catch {
            # The common error below is more useful than Path.GetFullPath's exception.
        }
        Write-Host '找不到该模型文件。Model file not found.' -ForegroundColor Yellow
    }
}

function Add-SetArgument {
    param(
        [Parameter(Mandatory = $true)][System.Collections.Generic.List[string]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value
    )
    [void]$Arguments.Add('--set')
    [void]$Arguments.Add("$Path=$Value")
}

function Format-PowerShellArgument {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)
    if ($Value -match '^[A-Za-z0-9_./:=,+%-]+$') {
        return $Value
    }
    return "'" + $Value.Replace("'", "''") + "'"
}

function Test-IsLoopbackAddress {
    param([Parameter(Mandatory = $true)][string]$Address)
    return $Address.Trim() -match '^(127(?:\.\d{1,3}){3}|\[::1\]):\d+$'
}

function Read-ListenAddress {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Default
    )
    while ($true) {
        $value = Read-TextValue -Label $Label -Default $Default -Required
        $matched = if ($value -match '^\[([^\]]+)\]:(\d+)$') {
            $hostText = $Matches[1]
            $portText = $Matches[2]
            $true
        } elseif ($value -match '^([^:]+):(\d+)$') {
            $hostText = $Matches[1]
            $portText = $Matches[2]
            $true
        } else {
            $false
        }
        $addressValue = $null
        $port = 0
        if ($matched -and
            [System.Net.IPAddress]::TryParse($hostText, [ref]$addressValue) -and
            [int]::TryParse($portText, [ref]$port) -and
            $port -ge 1 -and $port -le 65535) {
            return $value
        }
        Write-Host '请输入有效的 IP:端口，例如 127.0.0.1:8080 或 [::1]:8080。' -ForegroundColor Yellow
        Write-Host 'Enter a valid IP:port, for example 127.0.0.1:8080 or [::1]:8080.' -ForegroundColor Yellow
    }
}

function Get-ClientAddress {
    param([Parameter(Mandatory = $true)][string]$ListenAddress)
    $value = $ListenAddress.Trim()
    if ($value -match '^0\.0\.0\.0:(\d+)$') {
        return "127.0.0.1:$($Matches[1])"
    }
    if ($value -match '^\[::\]:(\d+)$') {
        return "[::1]:$($Matches[1])"
    }
    return $value
}

function Get-EngineVersion {
    $output = (& $infrPath --version 2>$null | Out-String).Trim()
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0 -and $output -match '(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)') {
        return $Matches[1]
    }
    return 'unknown'
}

function ConvertTo-ComparableVersion {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value -match '(\d+)\.(\d+)\.(\d+)') {
        return [version]::new([int]$Matches[1], [int]$Matches[2], [int]$Matches[3])
    }
    return $null
}

function Show-UpdateStatus {
    param([Parameter(Mandatory = $true)][string]$CurrentVersion)

    $disabled = [Environment]::GetEnvironmentVariable('MOE4ALL_NO_UPDATE_CHECK', 'Process')
    if ($SkipUpdateCheck -or $disabled -match '^(1|true|yes|on)$') {
        return
    }
    $current = ConvertTo-ComparableVersion $CurrentVersion
    if ($null -eq $current) {
        return
    }

    try {
        $headers = @{
            Accept = 'application/vnd.github+json'
            'User-Agent' = "MoE4All-Wizard/$CurrentVersion"
        }
        $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/Headmaster218/MoE4All/releases/latest' -Headers $headers -TimeoutSec 3 -ErrorAction Stop
        $latestText = [string]$release.tag_name
        $latest = ConvertTo-ComparableVersion $latestText
        if ($null -ne $latest -and $latest -gt $current) {
            Write-Host "发现新版本 / Update available: $latestText" -ForegroundColor Yellow
            Write-Host ([string]$release.html_url) -ForegroundColor Cyan
        } elseif ($null -ne $latest) {
            Write-Host "更新检查 / Update check: v$CurrentVersion is current" -ForegroundColor DarkGray
        }
    } catch {
        Write-Host '更新检查不可用，继续离线启动。Update check unavailable; continuing offline.' -ForegroundColor DarkGray
    }
}

if (-not (Test-Path -LiteralPath $infrPath -PathType Leaf)) {
    throw "找不到 infr.exe。发布包请把 infr.exe 与 Start-INFR-Wizard.cmd 放在同一目录；源码构建请先运行 cargo build --release --locked -p infr-cli。`nExecutable not found beside the launcher or under target\release."
}

$productVersion = Get-EngineVersion
Clear-Host
Write-Host '============================================================' -ForegroundColor Green
$banner = @'
 __  __       _____ _  _     _   _ _
|  \/  | ___ | ____| || |   / \ | | |
| |\/| |/ _ \|  _| | || |_ / _ \| | |
| |  | | (_) | |___|__   _/ ___ \ | |___
|_|  |_|\___/|_____|  |_|/_/   \_\_|_____|
'@

Write-Host $banner -ForegroundColor Green
Write-Host "  MoE4All v$productVersion" -ForegroundColor Green
Write-Host '  Making huge MoE LLMs accessible to AMD users.'
Write-Host '  让 A 卡用户也能在本地运行大型 MoE AI！'
Write-Host '  John / Headmaster218  https://github.com/Headmaster218/MoE4All' -ForegroundColor DarkGray
Write-Host '============================================================' -ForegroundColor Green
Write-Host 'MoE4All 启动向导 / MoE4All Launch Wizard' -ForegroundColor Green
Write-Host '上次设置会作为默认值；直接回车即可复用。Press Enter to reuse the previous value.'
Write-Host "引擎 / Engine: $infrPath" -ForegroundColor DarkGray
Show-UpdateStatus -CurrentVersion $productVersion

$launchMode = Read-Choice -Label '你想做什么？/ What would you like to do?' -DefaultValue ([string](Get-SavedValue 'launch_mode' 'chat')) -Options @(
    [pscustomobject]@{ Key = '1'; Value = 'chat'; Label = '实时终端对话（推荐）/ Interactive terminal chat (recommended)' }
    [pscustomobject]@{ Key = '2'; Value = 'server'; Label = '启动 OpenAI 兼容 API / Start OpenAI-compatible API server' }
    [pscustomobject]@{ Key = '3'; Value = 'benchmark'; Label = '性能测试 / Benchmark' }
)
$modelPath = Select-ModelPath -InitialPath $InitialModelPath

$setupModeDefault = 'quick'
if ($null -ne $script:Saved) {
    $savedSetupMode = $script:Saved.PSObject.Properties['setup_mode']
    # States written by the older all-advanced wizard keep their previous behavior after upgrade.
    $setupModeDefault = if ($null -eq $savedSetupMode) { 'advanced' } else { [string]$savedSetupMode.Value }
}
$setupMode = Read-Choice -Label '配置方式 / Configuration' -DefaultValue $setupModeDefault -Options @(
    [pscustomobject]@{ Key = '1'; Value = 'quick'; Label = '自动配置（推荐）/ Automatic setup (recommended)' }
    [pscustomobject]@{ Key = '2'; Value = 'advanced'; Label = '高级设置 / Advanced settings' }
)

$device = [string](Get-SavedValue 'device' '')
$context = [string](Get-SavedValue 'context' '')
$ubatch = [string](Get-SavedValue 'ubatch' '')
$threads = [string](Get-SavedValue 'threads' '')
$configPath = [string](Get-SavedValue 'config_path' '')
$kvPreset = [string](Get-SavedValue 'kv_preset' 'auto')
$kvTypeK = [string](Get-SavedValue 'kv_type_k' 'q8_0')
$kvTypeV = [string](Get-SavedValue 'kv_type_v' 'q8_0')
$configureMemory = [bool](Get-SavedValue 'configure_memory' $false)

if ($setupMode -eq 'advanced') {
    Write-Host "`n高级通用设置 / Advanced common settings" -ForegroundColor Cyan
    Write-Host '各项留空即可继续使用引擎的硬件探测与自动预算。Leave values blank to keep engine auto-detection.' -ForegroundColor DarkGray
    $device = Read-TextValue -Label '设备，留空为自动 / Device, blank for auto' -Default $device
    $context = Read-TextValue -Label '上下文窗口，留空为自动 / Context window, blank for auto' -Default $context
    $ubatch = Read-IntegerValue -Label 'Ubatch，留空为自动 / Ubatch, blank for auto' -Default $ubatch -Minimum 1 -AllowBlank
    $threads = Read-IntegerValue -Label 'CPU 线程，留空为全部 / CPU threads, blank for all' -Default $threads -Minimum 1 -AllowBlank
    $configPath = Read-TextValue -Label '配置 TOML，留空使用默认查找 / Config TOML, blank for default lookup' -Default $configPath
    if (-not [string]::IsNullOrWhiteSpace($configPath)) {
        $configPath = ConvertTo-FullPath $configPath
        if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
            throw "找不到配置文件 / Config file not found: $configPath"
        }
    }

    $kvPreset = Read-Choice -Label 'KV Cache 类型 / KV cache type' -DefaultValue $kvPreset -Options @(
        [pscustomobject]@{ Key = '1'; Value = 'auto'; Label = '引擎自动（推荐）/ Engine default (recommended)' }
        [pscustomobject]@{ Key = '2'; Value = 'q8'; Label = 'Q8_0 K + Q8_0 V' }
        [pscustomobject]@{ Key = '3'; Value = 'f16'; Label = 'F16 K + F16 V' }
        [pscustomobject]@{ Key = '4'; Value = 'custom'; Label = '分别指定 / Custom K and V' }
    )
    switch ($kvPreset) {
        'q8' { $kvTypeK = 'q8_0'; $kvTypeV = 'q8_0' }
        'f16' { $kvTypeK = 'f16'; $kvTypeV = 'f16' }
        'custom' {
            $kvTypeK = Read-TextValue -Label 'K cache 类型 / K cache dtype' -Default $kvTypeK -Required
            $kvTypeV = Read-TextValue -Label 'V cache 类型 / V cache dtype' -Default $kvTypeV -Required
        }
    }

    $configureMemory = Read-YesNo -Label '设置显存、内存和分页参数？/ Configure memory and paging?' -Default $configureMemory
} else {
    Write-Host "`n将自动探测 GPU、上下文、显存和 RAM；不覆盖引擎默认值。" -ForegroundColor DarkGray
    Write-Host 'GPU, context, VRAM and RAM will be detected automatically; engine defaults stay intact.' -ForegroundColor DarkGray
}

$vramBudget = [string](Get-SavedValue 'vram_budget' '')
$vramReserve = [string](Get-SavedValue 'vram_reserve' '')
$expertCache = [string](Get-SavedValue 'expert_cache' '')
$ramBudget = [string](Get-SavedValue 'ram_budget' '')
if (-not $ramBudget) {
    # One-time compatibility with state written before the setting became a total-process budget.
    $ramBudget = [string](Get-SavedValue 'dram_cache' '')
}
$pagerRing = [string](Get-SavedValue 'pager_ring' '')
$pagerRingSlots = [string](Get-SavedValue 'pager_ring_slots' '')
$hostDma = [bool](Get-SavedValue 'host_dma' $true)
$kvOverflow = [bool](Get-SavedValue 'kv_overflow' $false)
$kvOverflowVram = [string](Get-SavedValue 'kv_overflow_vram_mb' '')
$kvOverflowReserve = [string](Get-SavedValue 'kv_overflow_reserve_mb' '')
if ($setupMode -eq 'advanced' -and $configureMemory) {
    Write-Host '大小可写 21g、512m、80%；留空表示自动。Sizes accept 21g, 512m or 80%; blank means auto.' -ForegroundColor DarkGray
    $vramBudget = Read-TextValue -Label '总显存预算 / Total VRAM budget' -Default $vramBudget
    $vramReserve = Read-TextValue -Label '额外显存保留 / Additional VRAM reserve' -Default $vramReserve
    $expertCache = Read-TextValue -Label 'GPU 专家缓存 / GPU expert cache' -Default $expertCache
    $ramBudget = Read-TextValue -Label '进程总 RAM 预算，0 为禁用 RAM tier / Total process RAM budget, 0 disables RAM tier' -Default $ramBudget
    $hostDma = Read-YesNo -Label '启用 RAM 到 VRAM Host DMA？/ Enable RAM-to-VRAM Host DMA?' -Default $hostDma
    $pagerRing = Read-TextValue -Label 'Pager staging ring，留空为自动 / Pager staging ring, blank for auto' -Default $pagerRing
    $pagerRingSlots = Read-IntegerValue -Label 'Pager ring slots，留空为默认 / Pager ring slots, blank for default' -Default $pagerRingSlots -Minimum 2 -AllowBlank
    $kvOverflow = Read-YesNo -Label '允许 KV 溢出到 RAM？/ Allow KV overflow to RAM?' -Default $kvOverflow
    if ($kvOverflow) {
        $kvOverflowVram = Read-IntegerValue -Label 'KV 显存上限 MiB，留空为自动 / KV VRAM ceiling MiB, blank for auto' -Default $kvOverflowVram -Minimum 1 -AllowBlank
        $kvOverflowReserve = Read-IntegerValue -Label 'KV 之外保留 MiB，留空为自动 / Non-KV reserve MiB, blank for auto' -Default $kvOverflowReserve -Minimum 1 -AllowBlank
    }
}

$submitMode = [string](Get-SavedValue 'submit_mode' 'auto')
$submitCap = [string](Get-SavedValue 'submit_cap' '64')
if ($setupMode -eq 'advanced') {
    $submitMode = Read-Choice -Label 'Submit splitter' -DefaultValue $submitMode -Options @(
        [pscustomobject]@{ Key = '1'; Value = 'auto'; Label = '自动反馈 / Automatic feedback' }
        [pscustomobject]@{ Key = '2'; Value = 'disabled'; Label = '禁用，no-split / Disabled, no-split' }
        [pscustomobject]@{ Key = '3'; Value = 'fixed'; Label = '固定 cap / Fixed cap' }
    )
    if ($submitMode -eq 'fixed') {
        $submitCap = Read-IntegerValue -Label '固定 dispatch cap / Fixed dispatch cap' -Default $submitCap -Minimum 1
    }
}

$configureDiagnostics = [bool](Get-SavedValue 'configure_diagnostics' $false)
$pagerStats = [bool](Get-SavedValue 'pager_stats' $false)
$pagerProfile = [bool](Get-SavedValue 'pager_profile' $false)
$stageProfile = [bool](Get-SavedValue 'stage_profile' $false)
$vramProfile = [bool](Get-SavedValue 'vram_profile' $false)
if ($setupMode -eq 'advanced') {
    $configureDiagnostics = Read-YesNo -Label '设置统计或 profiler？/ Configure statistics or profilers?' -Default $configureDiagnostics
}
if ($setupMode -eq 'advanced' -and $configureDiagnostics) {
    $pagerStats = Read-YesNo -Label '输出 pager 命中统计？/ Print pager hit statistics?' -Default $pagerStats
    $pagerProfile = Read-YesNo -Label '启用聚合 pager profiler？/ Enable aggregate pager profiler?' -Default $pagerProfile
    $stageProfile = Read-YesNo -Label '启用阶段计时？/ Enable stage timings?' -Default $stageProfile
    $vramProfile = Read-YesNo -Label '输出实时显存信息？/ Print live VRAM information?' -Default $vramProfile
}

$benchKind = [string](Get-SavedValue 'bench_kind' 'decode')
$promptTokens = [string](Get-SavedValue 'prompt_tokens' '1024')
$genTokens = [string](Get-SavedValue 'gen_tokens' '128')
$depthMode = [string](Get-SavedValue 'depth_mode' 'none')
$depthTokens = [string](Get-SavedValue 'depth_tokens' '0')
$reps = [string](Get-SavedValue 'reps' '1')
$jsonOutput = [bool](Get-SavedValue 'json_output' $false)
$thinkMode = [string](Get-SavedValue 'think_mode' 'default')
$maxNew = [string](Get-SavedValue 'max_new' '')
$configureSampling = [bool](Get-SavedValue 'configure_sampling' $false)
$temperature = [string](Get-SavedValue 'temperature' '')
$topK = [string](Get-SavedValue 'top_k' '')
$topP = [string](Get-SavedValue 'top_p' '')
$seed = [string](Get-SavedValue 'seed' '')
$serverAddr = [string](Get-SavedValue 'server_addr' '127.0.0.1:8080')
$serverParallel = [string](Get-SavedValue 'server_parallel' '1')
$serverAuth = [bool](Get-SavedValue 'server_auth' $false)
$serverApiKey = ''

if ($launchMode -eq 'benchmark') {
    $benchKind = Read-Choice -Label '测试类型 / Benchmark type' -DefaultValue $benchKind -Options @(
        [pscustomobject]@{ Key = '1'; Value = 'decode'; Label = 'Decode：-p 0 -n N' }
        [pscustomobject]@{ Key = '2'; Value = 'prefill'; Label = 'Prefill：-p N -n 0' }
        [pscustomobject]@{ Key = '3'; Value = 'mixed'; Label = '组合轮次 / Combined turn：--pg P,G' }
        [pscustomobject]@{ Key = '4'; Value = 'custom'; Label = '自定义 -p/-n / Custom -p/-n' }
    )
    switch ($benchKind) {
        'decode' {
            $promptTokens = '0'
            $genTokens = Read-IntegerValue -Label 'Decode token 数 / Decode tokens' -Default $genTokens -Minimum 1
        }
        'prefill' {
            $promptTokens = Read-IntegerValue -Label 'Prefill token 数 / Prefill tokens' -Default $promptTokens -Minimum 1
            $genTokens = '0'
        }
        'mixed' {
            $promptTokens = Read-IntegerValue -Label '本轮 prompt token 数 / Turn prompt tokens' -Default $promptTokens -Minimum 1
            $genTokens = Read-IntegerValue -Label '本轮生成 token 数 / Turn generation tokens' -Default $genTokens -Minimum 1
        }
        'custom' {
            $promptTokens = Read-IntegerValue -Label 'Prompt token 数 / Prompt tokens' -Default $promptTokens -Minimum 0
            $genTokens = Read-IntegerValue -Label 'Generation token 数 / Generation tokens' -Default $genTokens -Minimum 0
        }
    }
    $depthMode = Read-Choice -Label '测量前的上下文深度 / Context depth before measurement' -DefaultValue $depthMode -Options @(
        [pscustomobject]@{ Key = '1'; Value = 'none'; Label = '无 / None' }
        [pscustomobject]@{ Key = '2'; Value = 'real'; Label = '真实 warmup：-d / Real warmup: -d' }
        [pscustomobject]@{ Key = '3'; Value = 'synthetic'; Label = '快速 synthetic depth / Fast synthetic depth' }
    )
    if ($depthMode -ne 'none') {
        $depthTokens = Read-IntegerValue -Label '上下文深度 token 数 / Context depth tokens' -Default $depthTokens -Minimum 1
    } else {
        $depthTokens = '0'
    }
    $reps = Read-IntegerValue -Label '重复次数 / Repetitions' -Default $reps -Minimum 1
    $jsonOutput = Read-YesNo -Label '输出 JSON？/ Emit JSON?' -Default $jsonOutput
} else {
    $thinkMode = Read-Choice -Label '思考模式 / Reasoning mode' -DefaultValue $thinkMode -Options @(
        [pscustomobject]@{ Key = '1'; Value = 'default'; Label = '模型默认 / Model default' }
        [pscustomobject]@{ Key = '2'; Value = 'think'; Label = '强制开启思考 / Force reasoning on' }
        [pscustomobject]@{ Key = '3'; Value = 'no-think'; Label = '关闭思考 / Disable reasoning' }
    )
    $maxNew = Read-IntegerValue -Label '每轮最大生成 token，留空为模型默认 / Max new tokens per reply, blank for model default' -Default $maxNew -Minimum 1 -AllowBlank
    $configureSampling = Read-YesNo -Label '设置采样参数？/ Configure sampling?' -Default $configureSampling
    if ($configureSampling) {
        $temperature = Read-TextValue -Label 'Temperature，留空为模型默认 / blank for model default' -Default $temperature
        $topK = Read-IntegerValue -Label 'Top-K，留空为模型默认 / blank for model default' -Default $topK -Minimum 0 -AllowBlank
        $topP = Read-TextValue -Label 'Top-P，留空为模型默认 / blank for model default' -Default $topP
        $seed = Read-IntegerValue -Label '随机种子，留空为随机 / Seed, blank for random' -Default $seed -Minimum 0 -AllowBlank
    }

    if ($launchMode -eq 'server') {
        Write-Host "`nAPI 服务器 / API server" -ForegroundColor Cyan
        Write-Host '本机使用 127.0.0.1；局域网访问可用 0.0.0.0，但应启用 API key。' -ForegroundColor DarkGray
        Write-Host 'Use 127.0.0.1 locally. For LAN access use 0.0.0.0 and enable an API key.' -ForegroundColor DarkGray
        $serverAddr = Read-ListenAddress -Label '监听地址（IP:端口）/ Listen address (IP:port)' -Default $serverAddr
        $serverParallel = Read-IntegerValue -Label '并发会话数（每个会话有独立 KV）/ Concurrent slots (one KV cache each)' -Default $serverParallel -Minimum 1
        $serverAuth = Read-YesNo -Label '启用 Bearer API key 鉴权？/ Enable Bearer API-key authentication?' -Default $serverAuth
        if ($serverAuth) {
            $serverApiKey = Read-SecretValue -Label 'API key（隐藏输入且不会保存）/ API key (hidden and not saved)'
        } elseif (-not (Test-IsLoopbackAddress $serverAddr)) {
            Write-Host '警告：该监听地址可能被其他设备访问，且当前未启用鉴权。' -ForegroundColor Yellow
            Write-Host 'Warning: this address may be reachable by other devices and authentication is disabled.' -ForegroundColor Yellow
            if (-not (Read-YesNo -Label '仍然继续？/ Continue anyway?' -Default $false)) {
                exit 0
            }
        }
    }
}

$customSets = [string](Get-SavedValue 'custom_sets' '')
if ($setupMode -eq 'advanced') {
    $customSets = Read-TextValue -Label '额外 --set，以分号分隔，留空为无 / Extra --set entries separated by semicolons, blank for none' -Default $customSets
}

$nativeArgs = [System.Collections.Generic.List[string]]::new()
[void]$nativeArgs.Add($(switch ($launchMode) {
    'benchmark' { 'bench' }
    'server' { 'serve' }
    default { 'run' }
}))
if ($setupMode -eq 'advanced') {
    if (-not [string]::IsNullOrWhiteSpace($configPath)) { [void]$nativeArgs.Add('--config'); [void]$nativeArgs.Add($configPath) }
    if (-not [string]::IsNullOrWhiteSpace($device)) { [void]$nativeArgs.Add('--dev'); [void]$nativeArgs.Add($device) }
    if (-not [string]::IsNullOrWhiteSpace($context)) { [void]$nativeArgs.Add('--ctx'); [void]$nativeArgs.Add($context) }
    if (-not [string]::IsNullOrWhiteSpace($ubatch)) { [void]$nativeArgs.Add('--ubatch'); [void]$nativeArgs.Add($ubatch) }
    if (-not [string]::IsNullOrWhiteSpace($threads)) { [void]$nativeArgs.Add('--threads'); [void]$nativeArgs.Add($threads) }
}

if ($setupMode -eq 'advanced' -and $kvPreset -ne 'auto') {
    Add-SetArgument $nativeArgs 'kv.type_k' $kvTypeK
    Add-SetArgument $nativeArgs 'kv.type_v' $kvTypeV
}
if ($setupMode -eq 'advanced' -and $configureMemory) {
    if ($vramBudget) { Add-SetArgument $nativeArgs 'device.vram_budget' $vramBudget }
    if ($vramReserve) { Add-SetArgument $nativeArgs 'device.vram_reserve' $vramReserve }
    if ($expertCache) { Add-SetArgument $nativeArgs 'paging.cache' $expertCache }
    if ($ramBudget) { Add-SetArgument $nativeArgs 'device.ram_budget' $ramBudget }
    Add-SetArgument $nativeArgs 'paging.host_dma' $hostDma.ToString().ToLowerInvariant()
    if ($pagerRing) { Add-SetArgument $nativeArgs 'paging.ring' $pagerRing }
    if ($pagerRingSlots) { Add-SetArgument $nativeArgs 'paging.ring_slots' $pagerRingSlots }
    Add-SetArgument $nativeArgs 'kv.overflow' $kvOverflow.ToString().ToLowerInvariant()
    if ($kvOverflow -and $kvOverflowVram) { Add-SetArgument $nativeArgs 'kv.overflow_vram_mb' $kvOverflowVram }
    if ($kvOverflow -and $kvOverflowReserve) { Add-SetArgument $nativeArgs 'kv.overflow_reserve_mb' $kvOverflowReserve }
}
if ($setupMode -eq 'advanced') {
    switch ($submitMode) {
        'disabled' { Add-SetArgument $nativeArgs 'device.submit_dispatches' '0' }
        'fixed' { Add-SetArgument $nativeArgs 'device.submit_dispatches' $submitCap }
    }
}
if ($setupMode -eq 'advanced' -and $configureDiagnostics) {
    Add-SetArgument $nativeArgs 'paging.stats' $pagerStats.ToString().ToLowerInvariant()
    Add-SetArgument $nativeArgs 'prof.pager_profile' $pagerProfile.ToString().ToLowerInvariant()
    Add-SetArgument $nativeArgs 'prof.stages' $stageProfile.ToString().ToLowerInvariant()
    Add-SetArgument $nativeArgs 'prof.vram' $vramProfile.ToString().ToLowerInvariant()
}
if ($setupMode -eq 'advanced' -and -not [string]::IsNullOrWhiteSpace($customSets)) {
    foreach ($entry in $customSets.Split(';')) {
        $entry = $entry.Trim()
        if (-not $entry) { continue }
        $equals = $entry.IndexOf('=')
        if ($equals -le 0) {
            throw "额外配置缺少 path=value / Invalid extra setting: $entry"
        }
        $settingPath = $entry.Substring(0, $equals).Trim()
        if ($launchMode -eq 'server' -and $settingPath -eq 'serve.api_key') {
            throw '服务器 API key 请使用专用提示输入，以免密钥出现在命令和历史记录中。Use the server API-key prompt so the secret is not exposed in commands or history.'
        }
        Add-SetArgument $nativeArgs $settingPath $entry.Substring($equals + 1).Trim()
    }
}

if ($launchMode -eq 'benchmark') {
    if ($benchKind -eq 'mixed') {
        [void]$nativeArgs.Add('--pg'); [void]$nativeArgs.Add("$promptTokens,$genTokens")
    } else {
        [void]$nativeArgs.Add('-p'); [void]$nativeArgs.Add($promptTokens)
        [void]$nativeArgs.Add('-n'); [void]$nativeArgs.Add($genTokens)
    }
    switch ($depthMode) {
        'real' { [void]$nativeArgs.Add('-d'); [void]$nativeArgs.Add($depthTokens) }
        'synthetic' { [void]$nativeArgs.Add('--synthetic-depth'); [void]$nativeArgs.Add($depthTokens) }
    }
    [void]$nativeArgs.Add('-r'); [void]$nativeArgs.Add($reps)
    if ($jsonOutput) { [void]$nativeArgs.Add('--json') }
} else {
    switch ($thinkMode) {
        'think' { [void]$nativeArgs.Add('--think') }
        'no-think' { [void]$nativeArgs.Add('--no-think') }
    }
    if ($maxNew) { [void]$nativeArgs.Add('--max-new'); [void]$nativeArgs.Add($maxNew) }
    if ($configureSampling) {
        if ($temperature) { [void]$nativeArgs.Add('--temp'); [void]$nativeArgs.Add($temperature) }
        if ($topK) { [void]$nativeArgs.Add('--top-k'); [void]$nativeArgs.Add($topK) }
        if ($topP) { [void]$nativeArgs.Add('--top-p'); [void]$nativeArgs.Add($topP) }
        if ($seed) { [void]$nativeArgs.Add('--seed'); [void]$nativeArgs.Add($seed) }
    }
}
if ($launchMode -eq 'server') {
    if (-not $serverAuth) {
        # The explicit empty CLI layer also disables a key inherited from infr.toml or INFR_API_KEY.
        Add-SetArgument $nativeArgs 'serve.api_key' ''
    }
    [void]$nativeArgs.Add('--addr'); [void]$nativeArgs.Add($serverAddr)
    [void]$nativeArgs.Add('--parallel'); [void]$nativeArgs.Add($serverParallel)
}
[void]$nativeArgs.Add($modelPath)

$commandText = '& ' + (Format-PowerShellArgument $infrPath) + ' ' + (($nativeArgs | ForEach-Object { Format-PowerShellArgument $_ }) -join ' ')
$state = [ordered]@{
    launch_mode = $launchMode; setup_mode = $setupMode; model = $modelPath; device = $device; context = $context
    ubatch = $ubatch; threads = $threads; config_path = $configPath
    kv_preset = $kvPreset; kv_type_k = $kvTypeK; kv_type_v = $kvTypeV
    configure_memory = $configureMemory; vram_budget = $vramBudget; vram_reserve = $vramReserve
    expert_cache = $expertCache; ram_budget = $ramBudget; host_dma = $hostDma
    pager_ring = $pagerRing; pager_ring_slots = $pagerRingSlots
    kv_overflow = $kvOverflow; kv_overflow_vram_mb = $kvOverflowVram; kv_overflow_reserve_mb = $kvOverflowReserve
    submit_mode = $submitMode; submit_cap = $submitCap
    configure_diagnostics = $configureDiagnostics; pager_stats = $pagerStats
    pager_profile = $pagerProfile; stage_profile = $stageProfile; vram_profile = $vramProfile
    bench_kind = $benchKind; prompt_tokens = $promptTokens; gen_tokens = $genTokens
    depth_mode = $depthMode; depth_tokens = $depthTokens; reps = $reps; json_output = $jsonOutput
    think_mode = $thinkMode; max_new = $maxNew; configure_sampling = $configureSampling
    temperature = $temperature; top_k = $topK; top_p = $topP; seed = $seed
    server_addr = $serverAddr; server_parallel = $serverParallel; server_auth = $serverAuth
    custom_sets = $customSets; last_command = $commandText
}
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statePath -Encoding UTF8

Write-Host "`n最终命令 / Final command" -ForegroundColor Green
Write-Host $commandText -ForegroundColor White
if ($launchMode -eq 'server' -and $serverAuth) {
    Write-Host 'API key 将通过当前子进程环境传入，未显示在命令中，也不会保存。' -ForegroundColor DarkGray
    Write-Host 'The API key is passed through the child-process environment; it is hidden above and not saved.' -ForegroundColor DarkGray
}
Write-Host "`n设置已保存 / Settings saved: $statePath" -ForegroundColor DarkGray
if ($DryRun) {
    Write-Host 'DryRun：未启动。DryRun: command was not started.' -ForegroundColor Yellow
    exit 0
}
$startLabel = if ($launchMode -eq 'server') { '现在启动服务器？/ Start the server now?' } else { '现在启动？/ Start now?' }
if (-not (Read-YesNo -Label $startLabel -Default $true)) {
    exit 0
}

if ($launchMode -eq 'server') {
    $clientAddress = Get-ClientAddress $serverAddr
    Write-Host "`nAPI 地址 / API base URL: http://$clientAddress/v1" -ForegroundColor Cyan
    Write-Host "健康检查 / Health check: http://$clientAddress/health"
    Write-Host '兼容 OpenAI 客户端时，将 Base URL 设为上面的 /v1 地址。' -ForegroundColor DarkGray
    Write-Host 'For OpenAI clients, use the /v1 address above as the Base URL.' -ForegroundColor DarkGray
    Write-Host '按 Ctrl+C 停止服务器。Press Ctrl+C to stop the server.' -ForegroundColor Yellow
} elseif ($launchMode -eq 'chat') {
    Write-Host "`n模型加载完成后，在 > 提示符输入消息；输入 exit、quit 或 :q 退出。" -ForegroundColor Cyan
    Write-Host 'After the model loads, type at the > prompt; use exit, quit or :q to leave.' -ForegroundColor DarkGray
}

Write-Host "`n启动中 / Starting..." -ForegroundColor Green
$previousApiKey = [Environment]::GetEnvironmentVariable('INFR_API_KEY', 'Process')
$hadApiKey = $null -ne $previousApiKey
if ($launchMode -eq 'server' -and $serverAuth) {
    [Environment]::SetEnvironmentVariable('INFR_API_KEY', $serverApiKey, 'Process')
}
Push-Location $repoRoot
try {
    & $infrPath @nativeArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
    if ($launchMode -eq 'server' -and $serverAuth) {
        $restoreApiKey = if ($hadApiKey) { $previousApiKey } else { $null }
        [Environment]::SetEnvironmentVariable('INFR_API_KEY', $restoreApiKey, 'Process')
    }
}
if ($null -eq $exitCode) { $exitCode = 0 }
Write-Host "`nMoE4All / infr 退出码 / exit code: $exitCode" -ForegroundColor $(if ($exitCode -eq 0) { 'Green' } else { 'Red' })
exit $exitCode
