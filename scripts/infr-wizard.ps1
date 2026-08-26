[CmdletBinding()]
param(
    [switch]$DryRun
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
$infrPath = Join-Path $repoRoot 'target\release\infr.exe'
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
    $value = $Value.Trim().Trim('"').Trim("'")
    if ([System.IO.Path]::IsPathRooted($value)) {
        return [System.IO.Path]::GetFullPath($value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $value))
}

function Select-ModelPath {
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
        Write-Host ' [N] 输入新路径 / Enter a new path'
        while ($true) {
            $choice = (Read-Host '选择模型 [1] / Select model [1]').Trim()
            if ([string]::IsNullOrWhiteSpace($choice)) { return $models[0] }
            $index = 0
            if ([int]::TryParse($choice, [ref]$index) -and $index -ge 1 -and $index -le $models.Count) {
                return $models[$index - 1]
            }
            if ($choice -match '^(n|new|新)$') { break }
            Write-Host '选择无效。Invalid selection.' -ForegroundColor Yellow
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

Clear-Host
Write-Host 'INFR 启动向导 / INFR Launch Wizard' -ForegroundColor Green
Write-Host '上次设置会作为默认值；直接回车即可复用。Press Enter to reuse the previous value.'

if (-not (Test-Path -LiteralPath $infrPath -PathType Leaf)) {
    throw "找不到 infr.exe，请先运行 cargo build --release --locked -p infr-cli。`nExecutable not found: $infrPath"
}

$launchMode = Read-Choice -Label '运行模式 / Launch mode' -DefaultValue ([string](Get-SavedValue 'launch_mode' 'benchmark')) -Options @(
    [pscustomobject]@{ Key = '1'; Value = 'benchmark'; Label = '性能测试 / Benchmark' }
    [pscustomobject]@{ Key = '2'; Value = 'chat'; Label = '实时终端对话 / Interactive terminal chat' }
)
$modelPath = Select-ModelPath

Write-Host "`n通用设置 / Common settings" -ForegroundColor Cyan
$device = Read-TextValue -Label '设备 / Device' -Default ([string](Get-SavedValue 'device' 'Vulkan0')) -Required
$context = Read-TextValue -Label '上下文窗口，留空为自动 / Context window, blank for auto' -Default ([string](Get-SavedValue 'context' ''))
$ubatch = Read-IntegerValue -Label 'Ubatch，留空为自动 / Ubatch, blank for auto' -Default ([string](Get-SavedValue 'ubatch' '512')) -Minimum 1 -AllowBlank
$threads = Read-IntegerValue -Label 'CPU 线程，留空为全部 / CPU threads, blank for all' -Default ([string](Get-SavedValue 'threads' '')) -Minimum 1 -AllowBlank
$configPath = Read-TextValue -Label '配置 TOML，留空使用默认查找 / Config TOML, blank for default lookup' -Default ([string](Get-SavedValue 'config_path' ''))
if (-not [string]::IsNullOrWhiteSpace($configPath)) {
    $configPath = ConvertTo-FullPath $configPath
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "找不到配置文件 / Config file not found: $configPath"
    }
}

$kvPreset = Read-Choice -Label 'KV Cache 类型 / KV cache type' -DefaultValue ([string](Get-SavedValue 'kv_preset' 'q8')) -Options @(
    [pscustomobject]@{ Key = '1'; Value = 'q8'; Label = 'Q8_0 K + Q8_0 V' }
    [pscustomobject]@{ Key = '2'; Value = 'f16'; Label = 'F16 K + F16 V' }
    [pscustomobject]@{ Key = '3'; Value = 'auto'; Label = '引擎自动 / Engine default' }
    [pscustomobject]@{ Key = '4'; Value = 'custom'; Label = '分别指定 / Custom K and V' }
)
$kvTypeK = [string](Get-SavedValue 'kv_type_k' 'q8_0')
$kvTypeV = [string](Get-SavedValue 'kv_type_v' 'q8_0')
switch ($kvPreset) {
    'q8' { $kvTypeK = 'q8_0'; $kvTypeV = 'q8_0' }
    'f16' { $kvTypeK = 'f16'; $kvTypeV = 'f16' }
    'custom' {
        $kvTypeK = Read-TextValue -Label 'K cache 类型 / K cache dtype' -Default $kvTypeK -Required
        $kvTypeV = Read-TextValue -Label 'V cache 类型 / V cache dtype' -Default $kvTypeV -Required
    }
}

$configureMemory = Read-YesNo -Label '设置显存、内存和分页参数？/ Configure memory and paging?' -Default ([bool](Get-SavedValue 'configure_memory' $true))
$vramBudget = [string](Get-SavedValue 'vram_budget' '')
$vramReserve = [string](Get-SavedValue 'vram_reserve' '')
$expertCache = [string](Get-SavedValue 'expert_cache' '')
$dramCache = [string](Get-SavedValue 'dram_cache' '')
$pagerRing = [string](Get-SavedValue 'pager_ring' '')
$pagerRingSlots = [string](Get-SavedValue 'pager_ring_slots' '')
$hostDma = [bool](Get-SavedValue 'host_dma' $true)
$kvOverflow = [bool](Get-SavedValue 'kv_overflow' $false)
$kvOverflowVram = [string](Get-SavedValue 'kv_overflow_vram_mb' '')
$kvOverflowReserve = [string](Get-SavedValue 'kv_overflow_reserve_mb' '')
if ($configureMemory) {
    Write-Host '大小可写 21g、512m、80%；留空表示自动。Sizes accept 21g, 512m or 80%; blank means auto.' -ForegroundColor DarkGray
    $vramBudget = Read-TextValue -Label '总显存预算 / Total VRAM budget' -Default $vramBudget
    $vramReserve = Read-TextValue -Label '额外显存保留 / Additional VRAM reserve' -Default $vramReserve
    $expertCache = Read-TextValue -Label 'GPU 专家缓存 / GPU expert cache' -Default $expertCache
    $dramCache = Read-TextValue -Label 'RAM 专家缓存，0 为禁用 / RAM expert cache, 0 disables' -Default $dramCache
    $hostDma = Read-YesNo -Label '启用 RAM 到 VRAM Host DMA？/ Enable RAM-to-VRAM Host DMA?' -Default $hostDma
    $pagerRing = Read-TextValue -Label 'Pager staging ring，留空为自动 / Pager staging ring, blank for auto' -Default $pagerRing
    $pagerRingSlots = Read-IntegerValue -Label 'Pager ring slots，留空为默认 / Pager ring slots, blank for default' -Default $pagerRingSlots -Minimum 2 -AllowBlank
    $kvOverflow = Read-YesNo -Label '允许 KV 溢出到 RAM？/ Allow KV overflow to RAM?' -Default $kvOverflow
    if ($kvOverflow) {
        $kvOverflowVram = Read-IntegerValue -Label 'KV 显存上限 MiB，留空为自动 / KV VRAM ceiling MiB, blank for auto' -Default $kvOverflowVram -Minimum 1 -AllowBlank
        $kvOverflowReserve = Read-IntegerValue -Label 'KV 之外保留 MiB，留空为自动 / Non-KV reserve MiB, blank for auto' -Default $kvOverflowReserve -Minimum 1 -AllowBlank
    }
}

$submitMode = Read-Choice -Label 'Submit splitter' -DefaultValue ([string](Get-SavedValue 'submit_mode' 'auto')) -Options @(
    [pscustomobject]@{ Key = '1'; Value = 'auto'; Label = '自动反馈 / Automatic feedback' }
    [pscustomobject]@{ Key = '2'; Value = 'disabled'; Label = '禁用，no-split / Disabled, no-split' }
    [pscustomobject]@{ Key = '3'; Value = 'fixed'; Label = '固定 cap / Fixed cap' }
)
$submitCap = [string](Get-SavedValue 'submit_cap' '64')
if ($submitMode -eq 'fixed') {
    $submitCap = Read-IntegerValue -Label '固定 dispatch cap / Fixed dispatch cap' -Default $submitCap -Minimum 1
}

$configureDiagnostics = Read-YesNo -Label '设置统计或 profiler？/ Configure statistics or profilers?' -Default ([bool](Get-SavedValue 'configure_diagnostics' $false))
$pagerStats = [bool](Get-SavedValue 'pager_stats' $false)
$pagerProfile = [bool](Get-SavedValue 'pager_profile' $false)
$stageProfile = [bool](Get-SavedValue 'stage_profile' $false)
$vramProfile = [bool](Get-SavedValue 'vram_profile' $false)
if ($configureDiagnostics) {
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
}

$customSets = Read-TextValue -Label '额外 --set，以分号分隔，留空为无 / Extra --set entries separated by semicolons, blank for none' -Default ([string](Get-SavedValue 'custom_sets' ''))

$nativeArgs = [System.Collections.Generic.List[string]]::new()
[void]$nativeArgs.Add($(if ($launchMode -eq 'benchmark') { 'bench' } else { 'run' }))
if (-not [string]::IsNullOrWhiteSpace($configPath)) { [void]$nativeArgs.Add('--config'); [void]$nativeArgs.Add($configPath) }
[void]$nativeArgs.Add('--dev'); [void]$nativeArgs.Add($device)
if (-not [string]::IsNullOrWhiteSpace($context)) { [void]$nativeArgs.Add('--ctx'); [void]$nativeArgs.Add($context) }
if (-not [string]::IsNullOrWhiteSpace($ubatch)) { [void]$nativeArgs.Add('--ubatch'); [void]$nativeArgs.Add($ubatch) }
if (-not [string]::IsNullOrWhiteSpace($threads)) { [void]$nativeArgs.Add('--threads'); [void]$nativeArgs.Add($threads) }

if ($kvPreset -ne 'auto') {
    Add-SetArgument $nativeArgs 'kv.type_k' $kvTypeK
    Add-SetArgument $nativeArgs 'kv.type_v' $kvTypeV
}
if ($configureMemory) {
    if ($vramBudget) { Add-SetArgument $nativeArgs 'device.vram_budget' $vramBudget }
    if ($vramReserve) { Add-SetArgument $nativeArgs 'device.vram_reserve' $vramReserve }
    if ($expertCache) { Add-SetArgument $nativeArgs 'paging.cache' $expertCache }
    if ($dramCache) { Add-SetArgument $nativeArgs 'paging.dram' $dramCache }
    Add-SetArgument $nativeArgs 'paging.host_dma' $hostDma.ToString().ToLowerInvariant()
    if ($pagerRing) { Add-SetArgument $nativeArgs 'paging.ring' $pagerRing }
    if ($pagerRingSlots) { Add-SetArgument $nativeArgs 'paging.ring_slots' $pagerRingSlots }
    Add-SetArgument $nativeArgs 'kv.overflow' $kvOverflow.ToString().ToLowerInvariant()
    if ($kvOverflow -and $kvOverflowVram) { Add-SetArgument $nativeArgs 'kv.overflow_vram_mb' $kvOverflowVram }
    if ($kvOverflow -and $kvOverflowReserve) { Add-SetArgument $nativeArgs 'kv.overflow_reserve_mb' $kvOverflowReserve }
}
switch ($submitMode) {
    'disabled' { Add-SetArgument $nativeArgs 'device.submit_dispatches' '0' }
    'fixed' { Add-SetArgument $nativeArgs 'device.submit_dispatches' $submitCap }
}
if ($configureDiagnostics) {
    Add-SetArgument $nativeArgs 'paging.stats' $pagerStats.ToString().ToLowerInvariant()
    Add-SetArgument $nativeArgs 'prof.pager_profile' $pagerProfile.ToString().ToLowerInvariant()
    Add-SetArgument $nativeArgs 'prof.stages' $stageProfile.ToString().ToLowerInvariant()
    Add-SetArgument $nativeArgs 'prof.vram' $vramProfile.ToString().ToLowerInvariant()
}
if (-not [string]::IsNullOrWhiteSpace($customSets)) {
    foreach ($entry in $customSets.Split(';')) {
        $entry = $entry.Trim()
        if (-not $entry) { continue }
        $equals = $entry.IndexOf('=')
        if ($equals -le 0) {
            throw "额外配置缺少 path=value / Invalid extra setting: $entry"
        }
        Add-SetArgument $nativeArgs $entry.Substring(0, $equals).Trim() $entry.Substring($equals + 1).Trim()
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
[void]$nativeArgs.Add($modelPath)

$commandText = '& ' + (Format-PowerShellArgument $infrPath) + ' ' + (($nativeArgs | ForEach-Object { Format-PowerShellArgument $_ }) -join ' ')
$state = [ordered]@{
    launch_mode = $launchMode; model = $modelPath; device = $device; context = $context
    ubatch = $ubatch; threads = $threads; config_path = $configPath
    kv_preset = $kvPreset; kv_type_k = $kvTypeK; kv_type_v = $kvTypeV
    configure_memory = $configureMemory; vram_budget = $vramBudget; vram_reserve = $vramReserve
    expert_cache = $expertCache; dram_cache = $dramCache; host_dma = $hostDma
    pager_ring = $pagerRing; pager_ring_slots = $pagerRingSlots
    kv_overflow = $kvOverflow; kv_overflow_vram_mb = $kvOverflowVram; kv_overflow_reserve_mb = $kvOverflowReserve
    submit_mode = $submitMode; submit_cap = $submitCap
    configure_diagnostics = $configureDiagnostics; pager_stats = $pagerStats
    pager_profile = $pagerProfile; stage_profile = $stageProfile; vram_profile = $vramProfile
    bench_kind = $benchKind; prompt_tokens = $promptTokens; gen_tokens = $genTokens
    depth_mode = $depthMode; depth_tokens = $depthTokens; reps = $reps; json_output = $jsonOutput
    think_mode = $thinkMode; max_new = $maxNew; configure_sampling = $configureSampling
    temperature = $temperature; top_k = $topK; top_p = $topP; seed = $seed
    custom_sets = $customSets; last_command = $commandText
}
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statePath -Encoding UTF8

Write-Host "`n最终命令 / Final command" -ForegroundColor Green
Write-Host $commandText -ForegroundColor White
Write-Host "`n设置已保存 / Settings saved: $statePath" -ForegroundColor DarkGray
if ($DryRun) {
    Write-Host 'DryRun：未启动。DryRun: command was not started.' -ForegroundColor Yellow
    exit 0
}
if (-not (Read-YesNo -Label '现在启动？/ Start now?' -Default $true)) {
    exit 0
}

Write-Host "`n启动中 / Starting..." -ForegroundColor Green
Push-Location $repoRoot
try {
    & $infrPath @nativeArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($null -eq $exitCode) { $exitCode = 0 }
Write-Host "`nINFR 退出码 / exit code: $exitCode" -ForegroundColor $(if ($exitCode -eq 0) { 'Green' } else { 'Red' })
exit $exitCode
