# Dashboard state and UI metadata only. Dot-sourcing this file performs no I/O,
# model loading, state saving, probing or changes to the legacy wizard's globals.

function Get-DashboardInitialState {
    param([AllowNull()]$Saved)
    $installRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    $state = [ordered]@{
        launch_mode = 'chat'; setup_mode = 'conservative'; model = ''; model_history = @()
        wizard_language = ''; wizard_theme = 'system'; download_dir = ''
        device = ''; context = ''; ubatch = ''; threads = ''; config_path = ''
        kv_preset = 'auto'; kv_type_k = ''; kv_type_v = ''; configure_memory = $false
        vram_budget = ''; vram_reserve = ''; expert_cache = ''; ram_budget = ''; host_dma = $true
        pager_ring = ''; pager_ring_slots = ''; kv_overflow = $false
        kv_overflow_vram_mb = ''; kv_overflow_reserve_mb = ''
        submit_mode = 'auto'; submit_cap = '64'; configure_diagnostics = $false
        pager_stats = $false; pager_profile = $false; stage_profile = $false; vram_profile = $false
        bench_kind = 'quick'; bench_parallel = '1'; manual_bench_kind = 'decode'
        prompt_tokens = '6144'; gen_tokens = '512'; depth_mode = 'none'; depth_tokens = '0'; reps = '1'; json_output = $false
        benchmark_upload = $false; benchmark_upload_has_github = $true; benchmark_upload_gh_auth = $true
        benchmark_upload_fallback = 'folder'; submission_browser_action = 'later'; submission_failure_action = 'later'
        submission_pending_retry = $true; error_upload = $false
        think_mode = 'default'; configure_thinking = $false; reasoning_effort = 'default'; max_new = '65536'
        reasoning_probe_model = ''; reasoning_probe_length = -1; reasoning_probe_mtime_ticks = -1; reasoning_probe_efforts = @()
        configure_sampling = $false; temperature = ''; top_k = ''; top_p = ''; seed = ''
        server_addr = '127.0.0.1:8080'; server_parallel = '1'; server_auth = $false
        server_session_cache = $true; session_cache_dir = (Join-Path $installRoot 'kv-sessions')
        session_idle_secs = '120'; session_cache_max = '5GiB'; session_cache_ttl_hours = '24'
        server_llm_enabled = $true; server_vision = $false; vision_projector = ''
        server_embedding = $false; embedding_model = ''; embedding_idle_timeout = '300'
        server_transcription = $false; transcription_model = ''; transcription_threads = '4'; transcription_idle_timeout = '300'
        transcription_upload_dir = (Join-Path $installRoot 'gui-data\audio-uploads')
        server_speech = $false; speech_model = ''; speech_lexicon = ''; speech_threads = '4'
        speech_idle_timeout = '300'; speech_max_output_seconds = '30'
        cpu_miss_enabled = $false; cpu_miss_max = '1'; cpu_miss_cores = ''; custom_sets = ''
    }
    $inputValues = @{}
    if ($null -ne $Saved) {
        if ($Saved -is [Collections.IDictionary]) {
            foreach ($key in $Saved.Keys) { $inputValues[[string]$key] = $Saved[$key] }
        } else {
            foreach ($property in $Saved.PSObject.Properties) { $inputValues[$property.Name] = $property.Value }
        }
    }
    $boolKeys = @($state.Keys | Where-Object { $state[$_] -is [bool] })
    foreach ($key in $inputValues.Keys) {
        if ($key -match '^(?i:(?:serve\.)?(?:server_)?api_key|INFR_API_KEY|last_command|command_text|argv|env)$') { continue }
        if ($null -ne $inputValues[$key]) { $state[$key] = $inputValues[$key] }
    }
    # Unix mode aliases and older Windows profile names are accepted explicitly.
    if (-not $inputValues.ContainsKey('launch_mode') -and $inputValues.ContainsKey('mode')) {
        if ([string]$inputValues.mode -in @('run', 'serve', 'bench', 'chat', 'server', 'benchmark')) { $state.launch_mode = $inputValues.mode }
    }
    switch ([string]$state.launch_mode) { 'run' { $state.launch_mode = 'chat' }; 'serve' { $state.launch_mode = 'server' }; 'bench' { $state.launch_mode = 'benchmark' } }
    if (-not $inputValues.ContainsKey('setup_mode')) {
        if ([string]$inputValues.mode -in @('quick', 'advanced')) { $state.setup_mode = $inputValues.mode }
        elseif ($null -ne $Saved) { $state.setup_mode = 'manual' } # Old all-advanced wizard behavior.
    }
    switch ([string]$state.setup_mode) { 'quick' { $state.setup_mode = 'conservative' }; 'advanced' { $state.setup_mode = 'manual' } }
    foreach ($key in $boolKeys) {
        # Avoid [bool]'false' == $true. Leave invalid strings visible to validation.
        if ($state[$key] -is [bool]) { continue }
        switch -Regex ([string]$state[$key]) {
            '^(?i:true|1|yes|on)$' { $state[$key] = $true }
            '^(?i:false|0|no|off)$' { $state[$key] = $false }
        }
    }
    if ([string]::IsNullOrWhiteSpace([string]$state.ram_budget) -and $inputValues.ContainsKey('dram_cache')) { $state.ram_budget = $inputValues.dram_cache }
    if (-not $inputValues.ContainsKey('configure_thinking')) { $state.configure_thinking = ($state.think_mode -ne 'default' -or $state.reasoning_effort -ne 'default') }
    if ($state.submit_mode -eq 'off') { $state.submit_mode = 'disabled' }
    switch ([string]$state.manual_bench_kind) { 'p' { $state.manual_bench_kind = 'prefill' }; 'n' { $state.manual_bench_kind = 'decode' }; 'pg' { $state.manual_bench_kind = 'mixed' } }
    if ($state.bench_kind -in @('prefill', 'decode', 'mixed')) { $state.manual_bench_kind = $state.bench_kind; $state.bench_kind = 'manual' }
    if ($state.benchmark_upload_fallback -notin @('folder', 'clipboard', 'skip')) { $state.benchmark_upload_fallback = 'folder' }
    # Old extra settings may embed a key. Never expose it in an edit field or export.
    $safeSets = @()
    foreach ($entry in ([string]$state.custom_sets).Split(';')) {
        $equals = $entry.IndexOf('=')
        $setting = if ($equals -gt 0) { $entry.Substring(0, $equals).Trim() } else { $entry.Trim() }
        if ($setting -match '^(?i:(?:serve\.)?(?:server_)?api_key|INFR_API_KEY)$') { continue }
        if (-not [string]::IsNullOrWhiteSpace($entry)) { $safeSets += $entry }
    }
    $state.custom_sets = $safeSets -join ';'
    $legacyCache = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'MoE4All\kv-sessions'
    if ([string]::Equals([string]$state.session_cache_dir, $legacyCache, [StringComparison]::OrdinalIgnoreCase)) { $state.session_cache_dir = Join-Path $installRoot 'kv-sessions' }
    # Return a fresh top-level collection; never mutate Saved or $script:Saved.
    $state.model_history = @($state.model_history)
    $state.reasoning_probe_efforts = @($state.reasoning_probe_efforts)
    return [pscustomobject]$state
}

function Get-DashboardSchema {
    param([AllowNull()]$State, [object[]]$DeviceOptions = @())
    $current = Get-DashboardInitialState -Saved $State
    $pages = @(
        [pscustomobject]@{ id = 'home'; en = 'Start'; label = '开始 / Start'; description = '选择任务与设备，确认后直接运行。Choose a task and launch.' }
        [pscustomobject]@{ id = 'models'; en = 'Models'; label = '模型 / Models'; description = '独立选择 LLM、视觉、Embedding 与 CPU 语音服务。Choose independent modules.' }
        [pscustomobject]@{ id = 'runtime'; en = 'Settings'; label = '运行配置 / Settings'; description = '资源、生成与高级设置。Resources and generation.' }
        [pscustomobject]@{ id = 'api'; en = 'API'; label = '服务 / API'; description = '监听、鉴权及 SSD 会话缓存。Listen, authentication and session cache.' }
        [pscustomobject]@{ id = 'benchmark'; en = 'Benchmark'; label = '测速 / Benchmark'; description = '真实 Prefill，Decode 分前中后三段统计。Real Prefill and three Decode windows.' }
        [pscustomobject]@{ id = 'reports'; en = 'Reports'; label = '报告反馈 / Reports'; description = '本地报告与待发记录。Reports and pending submissions.' }
        [pscustomobject]@{ id = 'maintenance'; en = 'Maintenance'; label = '维护 / Maintenance'; description = '保存、导入导出与更新。Configuration and updates.' }
    )
    $fields = [Collections.Generic.List[object]]::new()
    $add = {
        param($page, $key, $label, $kind = 'text', $help = '', $when = @{}, $choices = @(), $action = '', $group = '')
        $options = @($choices | ForEach-Object {
            if ($_ -is [string]) {
                $parts = $_ -split '\|', 2
                [pscustomobject]@{ value = $parts[0]; label = $(if ($parts.Count -gt 1) { $parts[1] } else { $parts[0] }) }
            } else { $_ }
        })
        $field = [ordered]@{ page = $page; key = $key; label = $label; kind = $kind; help = $help; options = $options; when = $when; group = $group }
        if ($action) { $field.action = $action }
        [void]$fields.Add([pscustomobject]$field)
    }
    $deviceChoices = @([pscustomobject]@{ value = ''; label = '自动选择 / Automatic' })
    foreach ($device in $DeviceOptions) {
        if ($device -is [string]) { $deviceChoices += [pscustomobject]@{ value = $device; label = $device }; continue }
        $value = [string]$device.value; $label = [string]$device.label
        if (-not $value) { $value = [string]$device.id }
        if (-not $label) { $label = [string]$device.name }
        if ($value) { $deviceChoices += [pscustomobject]@{ value = $value; label = $(if ($label) { $label } else { $value }) } }
    }
    if ('cpu' -notin @($deviceChoices.value)) { $deviceChoices += [pscustomobject]@{ value = 'cpu'; label = 'CPU' } }
    if ($current.device -and $current.device -notin @($deviceChoices.value)) { $deviceChoices += [pscustomobject]@{ value = [string]$current.device; label = ('{0}（已保存，待探测 / saved）' -f $current.device) } }
    $llm = @{ _llm = 'true' }; $api = @{ _api = 'true' }; $bench = @{ _bench = 'true' }; $manual = @{ _llm = 'true'; _manual = 'true' }
    $normal = @{ _llm = 'true'; launch_mode = @('chat', 'server') }
    $profiles = @('conservative|自动保守（日常使用）/ Conservative', 'aggressive|自动激进（优先推理）/ Aggressive', 'manual|手动设置 / Manual')
    $sampling = @{ _llm = 'true'; launch_mode = @('chat', 'server'); configure_sampling = 'true' }
    $session = @{ _api = 'true'; _llm = 'true'; server_session_cache = 'true' }
    $bm = @{ _bench = 'true'; bench_kind = 'manual' }
    $customKv = @{ _llm = 'true'; _manual = 'true'; kv_preset = 'custom' }
    $memory = @{ _llm = 'true'; _manual = 'true'; configure_memory = 'true' }
    $overflow = @{ _llm = 'true'; _manual = 'true'; configure_memory = 'true'; kv_overflow = 'true' }
    $diagnostics = @{ _llm = 'true'; _manual = 'true'; configure_diagnostics = 'true' }
    $efforts = @('default|模板默认 / Template default')
    $supportedEfforts = @($current.reasoning_probe_efforts)
    if (-not $current.reasoning_probe_model -or $current.reasoning_probe_model -ne $current.model) { $supportedEfforts = @('low', 'medium', 'high', 'xhigh', 'max') }
    foreach ($effort in $supportedEfforts) { if ([string]$effort -and [string]$effort -ne 'default') { $efforts += [string]$effort } }
    if ($current.reasoning_effort -ne 'default' -and $current.reasoning_effort -notin $supportedEfforts) { $efforts += [string]$current.reasoning_effort }

    $groups = @(
        [pscustomobject]@{ id = 'home_main'; page = 'home'; label = '本次启动 / Launch'; open = $true }
        [pscustomobject]@{ id = 'model_main'; page = 'models'; label = '语言模型 / Language model'; open = $true; summary_key = 'server_llm_enabled' }
        [pscustomobject]@{ id = 'vision'; page = 'models'; label = '图片理解 / Vision'; open = $false; summary_key = 'server_vision' }
        [pscustomobject]@{ id = 'embedding'; page = 'models'; label = '向量服务 / Embedding'; open = $false; summary_key = 'server_embedding' }
        [pscustomobject]@{ id = 'audio'; page = 'models'; label = '语音转录 · CPU / Transcription'; open = $false; summary_key = 'server_transcription' }
        [pscustomobject]@{ id = 'speech'; page = 'models'; label = '语音合成 · CPU / Speech synthesis'; open = $false; summary_key = 'server_speech' }
        [pscustomobject]@{ id = 'basic'; page = 'runtime'; label = '基础资源 / Basics'; open = $true; summary_key = 'setup_mode' }
        [pscustomobject]@{ id = 'generation'; page = 'runtime'; label = '生成设置 / Generation'; open = $true; summary_key = 'think_mode' }
        [pscustomobject]@{ id = 'sampling'; page = 'runtime'; label = '采样参数 / Sampling'; open = $false; summary_key = 'configure_sampling' }
        [pscustomobject]@{ id = 'memory'; page = 'runtime'; label = '手动资源 / Manual resources'; open = $false; summary_key = 'configure_memory' }
        [pscustomobject]@{ id = 'diagnostics'; page = 'runtime'; label = '高级与诊断 / Advanced'; open = $false; summary_key = 'configure_diagnostics' }
        [pscustomobject]@{ id = 'connection'; page = 'api'; label = '连接设置 / Connection'; open = $true; summary_key = 'server_addr' }
        [pscustomobject]@{ id = 'session'; page = 'api'; label = '会话缓存 / Session cache'; open = $false; summary_key = 'server_session_cache' }
        [pscustomobject]@{ id = 'plan'; page = 'benchmark'; label = '测试计划 / Test plan'; open = $true; summary_key = 'bench_kind' }
        [pscustomobject]@{ id = 'parallel'; page = 'benchmark'; label = '测试并发 / Concurrency'; open = $true; summary_key = 'bench_parallel' }
        [pscustomobject]@{ id = 'manual'; page = 'benchmark'; label = '手动测试 / Manual test'; open = $true; summary_key = 'manual_bench_kind' }
        [pscustomobject]@{ id = 'benchmark_actions'; page = 'benchmark'; label = '开始测试 / Run'; open = $true }
        [pscustomobject]@{ id = 'report_actions'; page = 'reports'; label = '报告 / Reports'; open = $true }
        [pscustomobject]@{ id = 'report_files'; page = 'reports'; label = '本地报告 / Local files'; open = $false }
        [pscustomobject]@{ id = 'submission_preferences'; page = 'reports'; label = '提交偏好 / Submission preferences'; open = $false }
        [pscustomobject]@{ id = 'maintenance'; page = 'maintenance'; label = '配置与更新 / Configuration'; open = $true }
    )

    & $add home launch_mode '用途 / Task' choice '切换用途会保留其他配置。Other settings are retained.' @{} @('chat|终端对话 / Chat', 'server|API 服务 / API server', 'benchmark|性能测试 / Benchmark') -group 'home_main'
    & $add home model '语言模型 / LLM model' path 'GGUF 文件或模型目录。GGUF file or model directory.' $llm -group 'home_main'
    & $add home device '计算设备 / Device' choice '选择运行模型的显卡；转录固定使用 CPU。' @{ _device = 'true' } $deviceChoices -group 'home_main'
    & $add home setup_mode '资源策略 / Profile' choice '日常使用选保守，专注推理选激进。' $llm @('conservative|自动保守（日常使用）/ Conservative', 'aggressive|自动激进（优先推理）/ Aggressive', 'manual|手动设置 / Manual') -group 'home_main'
    & $add home context '上下文窗口 / Context' text '不确定可留空，使用模型默认长度。' $normal -group 'home_main'
    & $add home server_parallel '并发槽位 / Slots' text '同时处理的会话数。Concurrent sessions.' @{ _api = 'true'; _llm = 'true' } -group 'home_main'
    & $add home action.modules '服务模块 / Modules' action '选择语言、图片、向量、转录和合成。Choose service modules.' $api @() page.models -group home_main
    & $add home action.benchmark '测试计划 / Test plan' action '选择方案与并发。Choose a plan and concurrency.' $bench @() page.benchmark -group home_main
    & $add home action.start '开始运行 / Launch' action '先校验配置，再显示启动摘要。Validate and review before running.' @{} @() start -group 'home_main'

    & $add models server_llm_enabled '语言模型 / LLM' toggle 'API 模块可以独立启用。API modules can run independently.' $api -group 'model_main'
    & $add models model '语言模型 / LLM model' path 'GGUF 文件或目录；F2 可选历史，F4 可选文件。GGUF path; F2 history, F4 file picker.' $llm -group 'model_main'
    & $add models device '计算设备 / Device' choice '转录和合成固定使用 CPU。Transcription and speech synthesis use CPU.' @{ _device = 'true' } $deviceChoices -group 'model_main'
    & $add models action.download '下载推荐模型 / Download models' action '选择一个或多个官方推荐项目。Choose recommended model bundles.' @{} @() download -group 'model_main'
    & $add models server_vision '图片理解 / Vision' toggle '理解图片，需要配套视觉模型文件。' @{ _api = 'true'; _llm = 'true' } -group 'vision'
    & $add models vision_projector '视觉投影文件 / Vision projector' path '与语言模型匹配的 mmproj GGUF。Matching mmproj GGUF.' @{ _api = 'true'; _llm = 'true'; server_vision = 'true' } -group 'vision'
    & $add models server_embedding '向量服务 / Embedding' toggle '为知识库等应用生成文本向量。' $api -group 'embedding'
    & $add models embedding_model 'Embedding 模型 / Model' path 'GGUF 文件或目录。GGUF file or directory.' @{ _api = 'true'; server_embedding = 'true' } -group 'embedding'
    & $add models embedding_idle_timeout 'Embedding 空闲释放秒数 / Idle seconds' text '0 表示常驻。0 keeps the loaded instance resident.' @{ _api = 'true'; server_embedding = 'true' } -group 'embedding'
    & $add models server_transcription '语音转录 / Transcription' toggle '将音频转成文字，使用 CPU。' $api -group 'audio'
    & $add models transcription_model '语音 GGUF / Transcription model' path 'SenseVoice 模型，首次请求载入。SenseVoice; loads on the first request.' @{ _api = 'true'; _audio = 'true' } -group 'audio'
    & $add models transcription_threads '语音 CPU 线程 / CPU threads' text '1–256，默认 4。Default 4.' @{ _api = 'true'; _audio = 'true' } -group 'audio'
    & $add models transcription_idle_timeout '语音空闲释放秒数 / Idle seconds' text '0 表示常驻。0 keeps resident.' @{ _api = 'true'; _audio = 'true' } -group 'audio'
    & $add models transcription_upload_dir '音频上传目录 / Audio upload directory' path '仅保存用户上传的音频。Directory for uploaded audio.' @{ _api = 'true'; _audio = 'true' } -group 'audio'
    & $add models server_speech '语音合成 / Speech synthesis' toggle '独立 CPU 服务，可不加载 LLM；需要支持 TTS 的引擎。Independent CPU service; requires a TTS-enabled build.' $api -group 'speech'
    & $add models speech_model '合成 GGUF / Speech model' path '原生 Melo GGUF，与资源目录配套；下载入口提供本版本安装说明和转换脚本。Matching native Melo assets; Downloads opens setup instructions and the conversion script for this version.' @{ _api = 'true'; server_speech = 'true' } -group 'speech'
    & $add models speech_lexicon '合成资源目录 / Speech assets' path '目录中需要 lexicon.txt、tokens.txt、config.json。Directory containing lexicon.txt, tokens.txt and config.json.' @{ _api = 'true'; server_speech = 'true' } -group 'speech'
    & $add models speech_threads '合成 CPU 线程 / Speech CPU threads' text '1–256，默认 4。Default 4.' @{ _api = 'true'; server_speech = 'true' } -group 'speech'
    & $add models speech_idle_timeout '合成空闲释放秒数 / Speech idle seconds' text '默认 300；0 表示常驻，首次请求加载。Default 300; 0 keeps resident; loads on first request.' @{ _api = 'true'; server_speech = 'true' } -group 'speech'
    & $add models speech_max_output_seconds '合成时长上限 / Speech duration cap' text '每段 1–120 秒，默认 30；输出 WAV/PCM，完整片段返回。1–120 seconds per clip, default 30; WAV/PCM, complete clips.' @{ _api = 'true'; server_speech = 'true' } -group 'speech'

    & $add runtime setup_mode '资源策略 / Profile' choice '日常使用选保守，专注推理选激进。' $llm @('conservative|自动保守（日常使用）/ Conservative', 'aggressive|自动激进（优先推理）/ Aggressive', 'manual|手动设置 / Manual') -group 'basic'
    & $add runtime device '计算设备 / Device' choice '选择运行模型的显卡；转录固定使用 CPU。' @{ _device = 'true' } $deviceChoices -group 'basic'
    & $add runtime context '上下文窗口 / Context' text '不确定可留空，使用模型默认长度。' $normal -group 'basic'
    & $add api server_parallel '并发槽位 / Slots' text '同时处理的会话数。Concurrent sessions.' @{ _api = 'true'; _llm = 'true' } -group 'connection'
    & $add runtime kv_preset 'KV 类型 / KV format' choice '自动档由引擎选择；MLA 自动回退 F16。Engine chooses a supported format.' $manual @('auto|自动 / Automatic', 'q8|Q8_0 K + V', 'f16|F16 K + V', 'custom|分别指定 / Custom') -group 'basic'
    & $add runtime kv_type_k 'K cache 类型 / K dtype' text '' $customKv -group 'basic'
    & $add runtime kv_type_v 'V cache 类型 / V dtype' text '' $customKv -group 'basic'
    & $add runtime think_mode '思考模式 / Reasoning mode' choice '' @{ _llm = 'true'; launch_mode = @('chat', 'server') } @('default|模型默认 / Model default', 'think|开启 / On', 'no-think|关闭 / Off') -group 'generation'
    & $add runtime reasoning_effort '思考强度 / Reasoning effort' choice '按模型支持的档位选择。Choose a supported level.' @{ _llm = 'true'; launch_mode = @('chat', 'server'); think_mode = @('default', 'think') } $efforts -group 'generation'
    & $add runtime max_new '每轮生成上限 / Max new tokens' text '包含思考与正文，默认 65536。Includes reasoning and answer.' $normal -group 'generation'
    & $add runtime configure_sampling '设置采样 / Configure sampling' toggle '留空使用模型默认值。Blank values keep model defaults.' $normal -group 'sampling'
    & $add runtime temperature '温度 / Temperature' text '留空使用默认。Blank for default.' $sampling -group 'sampling'
    & $add runtime top_k 'Top-K' text '非负整数；留空为默认。Nonnegative integer or blank.' $sampling -group 'sampling'
    & $add runtime top_p 'Top-P' text '0–1；留空为默认。0–1 or blank.' $sampling -group 'sampling'
    & $add runtime seed '随机种子 / Seed' text '非负整数；留空随机。Nonnegative integer; blank for random.' $sampling -group 'sampling'
    & $add runtime ubatch 'Ubatch' text '留空为引擎自动。Blank for auto.' $manual -group 'memory'
    & $add runtime threads 'CPU 线程 / Threads' text '留空使用全部。Blank uses all.' $manual -group 'memory'
    & $add runtime config_path '配置 TOML / Config file' path '留空按引擎默认查找。Blank uses engine lookup.' $manual -group 'memory'
    & $add runtime configure_memory '手动内存与分页 / Memory and paging' toggle '自动档忽略这些已保存值。Automatic profiles ignore saved manual values.' $manual -group 'memory'
    & $add runtime vram_budget 'VRAM 总预算 / VRAM budget' text '如 21GiB、80%；空白为自动。Blank for auto.' $memory -group 'memory'
    & $add runtime vram_reserve '额外显存保留 / VRAM reserve' text '如 512MiB；空白为自动。Blank for auto.' $memory -group 'memory'
    & $add runtime expert_cache '专家缓存 / Expert cache' text '绝对大小或百分比。Absolute size or percentage.' $memory -group 'memory'
    & $add runtime ram_budget '进程总 RAM 预算 / RAM budget' text '0 禁用 RAM tier；并非额外缓存大小。0 disables RAM tier; total process budget.' $memory -group 'memory'
    & $add runtime host_dma 'Host DMA' toggle '控制 RAM→VRAM Host DMA。Controls host DMA uploads.' $memory -group 'memory'
    & $add runtime pager_ring 'Pager staging ring' text '大小；留空自动。Size; blank for auto.' $memory -group 'memory'
    & $add runtime pager_ring_slots 'Pager ring slots' text '至少 2；留空默认。At least 2; blank for default.' $memory -group 'memory'
    & $add runtime kv_overflow 'KV 溢出到 RAM / KV overflow' toggle '' $memory -group 'memory'
    & $add runtime kv_overflow_vram_mb 'KV VRAM 上限 MiB / Ceiling' text '正整数；留空自动。Positive integer or blank.' $overflow -group 'memory'
    & $add runtime kv_overflow_reserve_mb '非 KV 保留 MiB / Reserve' text '正整数；留空自动。Positive integer or blank.' $overflow -group 'memory'
    & $add runtime submit_mode 'Submit splitter' choice '' $manual @('auto|自动反馈 / Automatic', 'disabled|禁用 / No split', 'fixed|固定 cap / Fixed cap') -group 'memory'
    & $add runtime submit_cap '固定 dispatch cap / Fixed cap' text '正整数。Positive integer.' @{ _llm = 'true'; _manual = 'true'; submit_mode = 'fixed' } -group 'memory'
    & $add runtime cpu_miss_enabled '实验性 CPU miss / CPU expert miss' toggle '要求 AVX2/FMA3、支持的量化与完整 RAM 专家缓存。Unsupported cases use GPU.' $llm -group 'diagnostics'
    & $add runtime cpu_miss_max '最多 CPU miss 数 / Maximum misses' choice '' @{ _llm = 'true'; cpu_miss_enabled = 'true' } @('1', '2', '3') -group 'diagnostics'
    & $add runtime cpu_miss_cores 'CPU 计算核心 / Compute cores' text '正整数，建议物理核数减 2。Positive integer; normally physical cores minus two.' @{ _llm = 'true'; cpu_miss_enabled = 'true' } -group 'diagnostics'
    & $add runtime configure_diagnostics '统计与 profiler / Diagnostics' toggle '' $manual -group 'diagnostics'
    & $add runtime pager_stats 'Pager 命中统计 / Hit statistics' toggle '' $diagnostics -group 'diagnostics'
    & $add runtime pager_profile 'Pager profiler' toggle '' $diagnostics -group 'diagnostics'
    & $add runtime stage_profile '阶段计时 / Stage timings' toggle '' $diagnostics -group 'diagnostics'
    & $add runtime vram_profile '实时显存信息 / VRAM information' toggle '' $diagnostics -group 'diagnostics'
    & $add runtime custom_sets '额外 --set / Extra settings' text 'path=value；多项以分号分隔。密钥必须使用专用输入。Semicolon-separated; no API keys.' $manual -group 'diagnostics'

    & $add api action.use_api '切换到 API 服务 / Use API' action '保留现有配置。Keep current settings.' @{ _api = 'false' } @() task.server -group connection
    & $add api server_addr '监听地址 / Listen address' text 'IP:端口；IPv6 使用 [::1]:8080。IP:port; bracket IPv6.' $api -group 'connection'
    & $add api server_auth 'API key 鉴权 / Authentication' toggle '密钥仅传给本次子进程，不保存。Key is passed only to the child process.' $api -group 'connection'
    & $add api server_api_key '临时 API key / Ephemeral API key' secret '隐藏输入，不进摘要、配置或命令。Never in summary, saved state or command.' @{ _api = 'true'; server_auth = 'true' } -group 'connection'
    & $add api server_session_cache 'SSD 会话缓存 / Session cache' toggle '动态分段 Q8 KV；闲置后保存并释放显存。Dynamic segmented Q8 KV only.' @{ _api = 'true'; _llm = 'true' } -group 'session'
    & $add api session_cache_dir '会话缓存目录 / Cache directory' path '默认安装目录下 kv-sessions。Default: kv-sessions beside the launcher.' $session -group 'session'
    & $add api session_idle_secs '闲置写盘秒数 / Spill idle seconds' text '非负整数。Nonnegative integer.' $session -group 'session'
    & $add api session_cache_max 'SSD 缓存上限 / Cache size limit' text '绝对大小，如 5GiB；不接受百分比。Absolute size, no percentages.' $session -group 'session'
    & $add api session_cache_ttl_hours '保留小时数 / TTL hours' text '0 为不按时间清理。0 disables age expiry.' $session -group 'session'

    & $add benchmark action.use_benchmark '切换到性能测试 / Use benchmark' action '保留现有配置。Keep current settings.' @{ _bench = 'false' } @() task.benchmark -group plan
    & $add benchmark bench_kind '测试方案 / Plan' choice '所有 Decode 报告均分前、中、后三段及均值。Three Decode windows and overall.' $bench @('quick|快速：6K，单次 / Quick: 6K, once', 'standard|标准：6K + 30K，单次 / Standard: once', 'full|完整：6K + 30K + 150K，各 1 次 / Full: once each', 'manual|单项手动 / Manual') -group 'plan'
    & $add benchmark bench_parallel '最大并发 / Maximum lanes' choice '依次测 x1 至 xN；快速默认 1，标准 2，完整 4。Tests x1 through xN.' @{ _bench = 'true'; bench_kind = @('quick', 'standard', 'full') } @('1|x1', '2|x1–x2', '3|x1–x3', '4|x1–x4') -group 'parallel'
    & $add benchmark manual_bench_kind '工作类型 / Workload' choice '按所选工作设置输入、生成与深度。Set prompt, generation and depth.' $bm @('prefill|只测 Prefill / Prefill only', 'decode|只测 Decode / Decode only', 'mixed|Prefill + Decode') -group 'manual'
    & $add benchmark prompt_tokens 'Prefill token 数 / Prompt tokens' text '只在 Prefill 或组合测试生效。Used by Prefill and mixed tests.' @{ _bench = 'true'; bench_kind = 'manual'; manual_bench_kind = @('prefill', 'mixed') } -group 'manual'
    & $add benchmark gen_tokens 'Decode token 数 / Generated tokens' text '只在 Decode 或组合测试生效。Used by Decode and mixed tests.' @{ _bench = 'true'; bench_kind = 'manual'; manual_bench_kind = @('decode', 'mixed') } -group 'manual'
    & $add benchmark depth_mode '测量前上下文 / Initial depth' choice '真实：先 Prefill；synthetic：仅构造 KV 形状。Real Prefill or synthetic KV shape.' @{ _bench = 'true'; bench_kind = 'manual'; manual_bench_kind = @('decode', 'mixed') } @('none|无 / None', 'real|真实上下文 / Real', 'synthetic|Synthetic KV') -group 'manual'
    & $add benchmark depth_tokens '已有上下文 token 数 / Depth tokens' text '' @{ _bench = 'true'; bench_kind = 'manual'; manual_bench_kind = @('decode', 'mixed'); depth_mode = @('real', 'synthetic') } -group 'manual'
    & $add benchmark reps '重复次数 / Repetitions' text '正整数；手动结果自动生成 JSON。Positive integer; JSON is always saved.' $bm -group 'manual'
    & $add benchmark action.start '开始测试 / Run benchmark' action '先校验方案，测试结束显示表格并询问是否上传。Review results and choose whether to submit.' $bench @() start -group 'benchmark_actions'

    & $add reports action.report '报告运行问题 / Report a problem' action '整理启动及错误附近日志，描述可返回编辑，提交前确认。Review before submission.' @{} @() report -group 'report_actions'
    & $add reports action.pending '待发报告 / Pending reports' action '保留同一 ID，询问后再重试。Retry with the same ID after consent.' @{} @() pending -group 'report_actions'
    & $add reports action.benchmark_files '性能报告文件夹 / Benchmark files' action '' @{} @() reports.benchmark -group 'report_files'
    & $add reports action.error_files '错误报告文件夹 / Error files' action '' @{} @() reports.errors -group 'report_files'
    & $add reports benchmark_upload '测速后默认愿意上传 / Benchmark upload preference' toggle '仅为询问默认值，不代替最终确认。A prompt default, never automatic consent.' -group 'submission_preferences'
    & $add reports benchmark_upload_has_github '有 GitHub 账号 / GitHub account' toggle '' -group 'submission_preferences'
    & $add reports benchmark_upload_gh_auth '愿意授权 gh CLI / Authorize gh CLI' toggle '上传时仍需确认。Consent is still requested during upload.' -group 'submission_preferences'
    & $add reports benchmark_upload_fallback '本地报告处理 / Local report fallback' choice '' @{} @('folder|打开文件夹 / Open folder', 'clipboard|复制内容 / Clipboard', 'skip|以后再说 / Later') -group 'submission_preferences'
    & $add reports submission_browser_action '网页提交默认操作 / Browser default action' choice '每份报告单独确认。Confirm each report separately.' @{} @('later|稍后处理 / Later', 'cloudflare|Cloudflare 中转 / Relay', 'reopen|重新打开网页 / Reopen browser') -group 'submission_preferences'
    & $add reports submission_failure_action '上传失败后 / After upload failure' choice '' @{} @('later|保存待发 / Keep pending', 'retry|重试 Cloudflare / Retry', 'channel|返回渠道选择 / Choose channel') -group 'submission_preferences'
    & $add reports submission_pending_retry '下次询问重试 / Ask to retry pending' toggle '' -group 'submission_preferences'
    & $add reports error_upload '报错后默认愿意上传 / Error upload preference' toggle '仍需编辑描述与确认公开提交。Description and public consent still required.' -group 'submission_preferences'

    & $add maintenance action.save '保存配置 / Save' action '保存当前设置，不启动模型。Save without launching.' @{} @() save -group maintenance
    & $add maintenance action.import '导入配置 / Import configuration' action '导入后先预览，不直接启动。Review imported settings before launch.' @{} @() import -group 'maintenance'
    & $add maintenance action.export '导出配置 / Export configuration' action '不包含 API key 或历史完整命令。Excludes API key and last command.' @{} @() export -group 'maintenance'
    & $add maintenance action.update '检查更新 / Check updates' action '复用现有自更新流程，保留配置与缓存。Preserve configuration and caches.' @{} @() update -group 'maintenance'
    & $add maintenance action.cancel '退出向导 / Exit' action '不启动模型。Do not start a model.' @{} @() cancel -group 'maintenance'
    $catalog = @()
    if (Get-Command Get-RecommendedModelCatalog -CommandType Function -ErrorAction SilentlyContinue) { $catalog = @(Get-RecommendedModelCatalog) }
    return [pscustomobject]@{ pages = $pages; groups = $groups; fields = @($fields.ToArray()); catalog = $catalog }
}

function Test-DashboardState {
    param([Parameter(Mandatory = $true)]$State)
    $rawCustomSets = ''
    if ($State -is [Collections.IDictionary]) { $rawCustomSets = [string]$State['custom_sets'] }
    else { $property = $State.PSObject.Properties['custom_sets']; if ($null -ne $property) { $rawCustomSets = [string]$property.Value } }
    $normalized = Get-DashboardInitialState -Saved $State
    $s = @{}
    foreach ($property in $normalized.PSObject.Properties) { $s[$property.Name] = $property.Value }
    $errors = [Collections.Generic.List[object]]::new()
    $fail = { param($field, $message) [void]$errors.Add([pscustomobject]@{ field = $field; message = $message }) }
    $integer = {
        param($key, [long]$minimum, [long]$maximum = [int]::MaxValue, [bool]$optional = $false)
        $v = [string]$s[$key]; if ($optional -and [string]::IsNullOrWhiteSpace($v)) { return }
        $parsed = [long]0
        if ($v -notmatch '^\d+$' -or -not [long]::TryParse($v, [ref]$parsed) -or $parsed -lt $minimum -or $parsed -gt $maximum) { & $fail $key ('请输入 {0}–{1} 的整数 / Enter an integer in range.' -f $minimum, $maximum) }
    }
    $choice = { param($key, $values) if ([string]$s[$key] -notin $values) { & $fail $key '请选择有效选项 / Choose a valid option.' } }
    $text = {
        param($key, [bool]$required = $false)
        if ([string]$s[$key] -match '[\x00-\x1f\x7f]' -or ($required -and [string]::IsNullOrWhiteSpace([string]$s[$key]))) { & $fail $key '请输入有效内容，不含控制字符 / Enter a value without control characters.' }
    }
    $path = {
        param($key, [bool]$required = $true, [bool]$fileOnly = $false)
        $v = [string]$s[$key]; & $text $key $required
        if ([string]::IsNullOrWhiteSpace($v)) { return }
        if ($v -match '[\x00-\x1f\x7f]') { return }
        if ($v.StartsWith('-')) { & $fail $key '路径以 - 开头时请加目录前缀 / Prefix paths starting with -.'; return }
        try {
            if ($fileOnly) { if (-not (Test-Path -LiteralPath $v -PathType Leaf)) { & $fail $key '文件不存在 / File not found.' } }
            elseif (-not (Test-Path -LiteralPath $v)) { & $fail $key '模型文件或目录不存在 / Model file or directory not found.' }
            elseif ((Test-Path -LiteralPath $v -PathType Container) -and -not (Get-ChildItem -LiteralPath $v -Filter '*.gguf' -File -ErrorAction SilentlyContinue | Select-Object -First 1)) { & $fail $key '目录内没有 GGUF 文件 / No GGUF file in directory.' }
        } catch { & $fail $key '路径无效 / Invalid path.' }
    }
    $size = {
        param($key, [bool]$percent = $true, [bool]$required = $false)
        $v = [string]$s[$key]; if (-not $required -and [string]::IsNullOrWhiteSpace($v)) { return }
        $m = [regex]::Match($v.Trim(), '^(?i:([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*([kmgt](?:i?b)?|b|%)?)$')
        $n = [double]0
        if (-not $m.Success -or -not [double]::TryParse($m.Groups[1].Value, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$n) -or [double]::IsInfinity($n) -or [double]::IsNaN($n) -or $n -lt 0 -or ($m.Groups[2].Value -eq '%' -and (-not $percent -or $n -le 0))) { & $fail $key '请输入有效大小 / Enter a valid size.' }
    }
    $flag = { param($key) if ($s[$key] -isnot [bool]) { & $fail $key '请选择开启或关闭 / Choose on or off.' } }
    & $choice launch_mode @('chat', 'server', 'benchmark')
    $isApi = $s.launch_mode -eq 'server'; $isBench = $s.launch_mode -eq 'benchmark'
    if ($isApi) {
        foreach ($key in @('server_llm_enabled', 'server_embedding', 'server_transcription', 'server_speech', 'server_auth')) { & $flag $key }
        if (-not ($s.server_llm_enabled -eq $true -or $s.server_embedding -eq $true -or $s.server_transcription -eq $true -or $s.server_speech -eq $true)) { & $fail server_llm_enabled '至少启用一个 API 模块 / Enable at least one API module.' }
        $m = [regex]::Match([string]$s.server_addr, '^(?:\[([^\]]+)\]|([^:]+)):(\d+)$')
        $ip = $null; $port = 0
        $hostPart = if ($m.Groups[1].Success) { $m.Groups[1].Value } else { $m.Groups[2].Value }
        if (-not $m.Success -or -not [Net.IPAddress]::TryParse($hostPart, [ref]$ip) -or -not [int]::TryParse($m.Groups[3].Value, [ref]$port) -or $port -lt 1 -or $port -gt 65535) { & $fail server_addr '请输入 IP:端口 / Enter an IP:port.' }
        # Secret presence/consent is checked by the host against its ephemeral store.
        if ($s.server_embedding -eq $true) { & $path embedding_model; & $integer embedding_idle_timeout 0 }
        if ($s.server_transcription -eq $true) {
            & $path transcription_model
            & $integer transcription_threads 1 256
            & $integer transcription_idle_timeout 0 999999999
            & $text transcription_upload_dir $true
        }
        if ($s.server_speech -eq $true) {
            & $path speech_model $true $true
            if ($s.speech_model -and [string]$s.speech_model -notmatch '(?i)\.gguf$') { & $fail speech_model '需要原生 Melo GGUF 文件 / A native Melo GGUF file is required.' }
            & $text speech_lexicon $true
            try {
                if (-not (Test-Path -LiteralPath ([string]$s.speech_lexicon) -PathType Container)) { throw 'Missing asset directory' }
                foreach ($asset in @('lexicon.txt', 'tokens.txt', 'config.json')) {
                    if (-not (Test-Path -LiteralPath (Join-Path ([string]$s.speech_lexicon) $asset) -PathType Leaf)) { throw "Missing $asset" }
                }
            } catch { & $fail speech_lexicon '资源目录需要 lexicon.txt、tokens.txt、config.json / Asset directory must contain lexicon.txt, tokens.txt and config.json.' }
            & $integer speech_threads 1 256
            $idle = [UInt64]0
            if ([string]$s.speech_idle_timeout -notmatch '^\d+$' -or -not [UInt64]::TryParse([string]$s.speech_idle_timeout, [ref]$idle)) { & $fail speech_idle_timeout '请输入非负整数秒数 / Enter unsigned whole seconds.' }
            $seconds = [single]0
            if (-not [single]::TryParse([string]$s.speech_max_output_seconds, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$seconds) -or [single]::IsNaN($seconds) -or [single]::IsInfinity($seconds) -or $seconds -lt 1 -or $seconds -gt 120) { & $fail speech_max_output_seconds '合成时长须为 1–120 秒 / Speech duration must be 1–120 seconds.' }
        }
    }
    $llm = -not $isApi -or $s.server_llm_enabled -eq $true
    if ($llm -or ($isApi -and $s.server_embedding -eq $true)) { & $text device }
    if ($llm) {
        & $path model
        & $choice setup_mode @('conservative', 'aggressive', 'manual')
        if (-not $isBench) { & $size context }
        if ($s.setup_mode -eq 'manual') {
            & $integer ubatch 1 ([int]::MaxValue) $true; & $integer threads 1 ([int]::MaxValue) $true
            & $path config_path $false $true
            & $choice kv_preset @('auto', 'q8', 'f16', 'custom')
            if ($s.kv_preset -eq 'custom') { & $text kv_type_k $true; & $text kv_type_v $true }
            & $flag configure_memory
            if ($s.configure_memory -eq $true) {
                foreach ($key in @('vram_budget', 'vram_reserve', 'expert_cache', 'ram_budget', 'pager_ring')) { & $size $key }
                & $flag host_dma; & $flag kv_overflow
                & $integer pager_ring_slots 2 ([int]::MaxValue) $true
                if ($s.kv_overflow -eq $true) { & $integer kv_overflow_vram_mb 1 ([int]::MaxValue) $true; & $integer kv_overflow_reserve_mb 1 ([int]::MaxValue) $true }
            }
            & $choice submit_mode @('auto', 'disabled', 'fixed')
            if ($s.submit_mode -eq 'fixed') { & $integer submit_cap 1 }
            & $flag configure_diagnostics
            if ($s.configure_diagnostics -eq $true) { foreach ($key in @('pager_stats', 'pager_profile', 'stage_profile', 'vram_profile')) { & $flag $key } }
            & $text custom_sets
            foreach ($entry in $rawCustomSets.Split(';')) {
                if ([string]::IsNullOrWhiteSpace($entry)) { continue }
                if ($entry.IndexOf('=') -le 0) { & $fail custom_sets '额外配置须为 path=value / Use path=value.'; continue }
                $setting = $entry.Substring(0, $entry.IndexOf('=')).Trim()
                if ($setting -match '^(?i:(?:serve\.)?(?:server_)?api_key|INFR_API_KEY)$') { & $fail custom_sets '密钥须使用专用输入，不写入配置 / Use the dedicated key input.' }
            }
        }
        & $flag cpu_miss_enabled
        if ($s.cpu_miss_enabled -eq $true) { & $integer cpu_miss_max 1 3; & $integer cpu_miss_cores 1 }
        if (-not $isBench) {
            & $choice think_mode @('default', 'think', 'no-think')
            & $integer max_new 1 ([int]::MaxValue) $true
            & $flag configure_sampling
            if ($s.configure_sampling -eq $true) {
                & $integer top_k 0 ([int]::MaxValue) $true; & $integer seed 0 ([long]::MaxValue) $true
                foreach ($key in @('temperature', 'top_p')) {
                    if ([string]::IsNullOrWhiteSpace([string]$s[$key])) { continue }
                    $n = [double]0
                    if (-not [double]::TryParse([string]$s[$key], [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$n) -or [double]::IsInfinity($n) -or [double]::IsNaN($n) -or $n -lt 0 -or ($key -eq 'top_p' -and $n -gt 1)) { & $fail $key '请输入有效采样数值 / Enter a valid sampling value.' }
                }
            }
        }
        if ($isApi) {
            & $integer server_parallel 1
            & $flag server_vision; & $flag server_session_cache
            if ($s.server_vision -eq $true) { & $path vision_projector }
            if ($s.server_session_cache -eq $true) {
                & $text session_cache_dir $true; & $integer session_idle_secs 0
                & $size session_cache_max $false $true; & $integer session_cache_ttl_hours 0
            }
        }
    }
    if ($isBench) {
        & $choice bench_kind @('quick', 'standard', 'full', 'manual')
        if ($s.bench_kind -eq 'manual') {
            & $choice manual_bench_kind @('prefill', 'decode', 'mixed')
            if ($s.manual_bench_kind -in @('prefill', 'mixed')) { & $integer prompt_tokens 1 }
            if ($s.manual_bench_kind -in @('decode', 'mixed')) {
                & $integer gen_tokens 1
                & $choice depth_mode @('none', 'real', 'synthetic')
                if ($s.depth_mode -in @('real', 'synthetic')) { & $integer depth_tokens 1 }
            }
            & $integer reps 1
        } else { & $integer bench_parallel 1 4 }
    }
    return $errors.ToArray()
}
