# Windows native launcher and terminal fallback. Inference configuration/argv remain owned by infr-wizard.ps1.
function Initialize-DashboardRuntime {
    if ('MoE4All.Launcher.Dashboard' -as [type]) { [MoE4All.Launcher.WizardDpi]::Initialize(); return }
    $source = Join-Path $PSScriptRoot 'wizard-console.cs'
    $sources = @($source)
    foreach ($name in @('wizard-theme.cs','wizard-desktop.cs','wizard-prompts.cs','wizard-runtime.cs','wizard-downloads.cs','wizard-markdown.cs','wizard-workspaces.cs','wizard-api-status.cs')) {
        $candidate=Join-Path $PSScriptRoot $name
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $sources += $candidate }
    }
    $icon = Join-Path $PSScriptRoot 'moe4all.ico'
    if (-not (Test-Path -LiteralPath $icon -PathType Leaf)) { throw 'Required launcher icon is missing: moe4all.ico' }
    $hashAlgorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $sourceBytes = [byte[]]@()
        foreach ($sourceFile in $sources) { $sourceBytes += [IO.File]::ReadAllBytes($sourceFile) }
        $sourceBytes += [IO.File]::ReadAllBytes($icon)
        $identity = ([BitConverter]::ToString($hashAlgorithm.ComputeHash($sourceBytes))).Replace('-','').Substring(0,20).ToLowerInvariant()
    } finally { $hashAlgorithm.Dispose() }
    if ($env:MOE4ALL_LAUNCHER_CACHE) { $cache = $env:MOE4ALL_LAUNCHER_CACHE }
    elseif ($script:IsPackagedInstall) { $cache = Join-Path $dataDir 'cache\launcher' }
    else {
        $generatedRoot = if ($env:MOE4ALL_WORK_ROOT) { $env:MOE4ALL_WORK_ROOT } else { Join-Path $repoRoot 'local-work' }
        $cache = Join-Path $generatedRoot 'build-cache\agent\wizard-netfx'
    }
    [void][IO.Directory]::CreateDirectory($cache)
    $assemblyPath = Join-Path $cache ("wizard-$identity.dll")
    if (-not (Test-Path -LiteralPath $assemblyPath -PathType Leaf)) {
        $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
        if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe' }
        if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { throw '系统 .NET Framework 编译器不可用。请使用 -ClassicWizard。Windows .NET Framework is unavailable; use -ClassicWizard.' }
        $temporary = Join-Path $cache ("wizard-$identity-$([guid]::NewGuid().ToString('N')).dll")
        $processInfo = [Diagnostics.ProcessStartInfo]::new()
        $processInfo.FileName = $compiler
        $processInfo.Arguments = '/nologo /optimize+ /target:library /utf8output /r:System.Web.Extensions.dll /r:System.Windows.Forms.dll /r:System.Drawing.dll /out:"' + $temporary + '"'
        foreach ($sourceFile in $sources) { $processInfo.Arguments += ' "' + $sourceFile + '"' }
        $processInfo.Arguments += ' /resource:"' + $icon + '",MoE4All.Icon'
        $processInfo.UseShellExecute = $false; $processInfo.CreateNoWindow = $true
        $processInfo.RedirectStandardOutput = $true; $processInfo.RedirectStandardError = $true
        $processInfo.EnvironmentVariables['TEMP'] = $cache
        $processInfo.EnvironmentVariables['TMP'] = $cache
        $process = [Diagnostics.Process]::new(); $process.StartInfo = $processInfo
        try {
            [void]$process.Start()
            $output = $process.StandardOutput.ReadToEndAsync(); $errorOutput = $process.StandardError.ReadToEndAsync()
            $process.WaitForExit()
            if ($process.ExitCode -ne 0) { throw ($output.Result + $errorOutput.Result) }
            # Concurrent launches may compile the same hash; keep the first complete result.
            try { [IO.File]::Move($temporary, $assemblyPath) } catch { if (-not (Test-Path -LiteralPath $assemblyPath -PathType Leaf)) { throw } }
        } finally { $process.Dispose() }
    }
    Add-Type -Path $assemblyPath
    [MoE4All.Launcher.WizardDpi]::Initialize()
}

function Invoke-DashboardHelper {
    param([string]$Action, [object]$State, [string]$Target='')
    $savedBefore = $script:Saved
    $preferenceMap = @{
        BenchmarkUploadDefault='benchmark_upload'; BenchmarkUploadHasGithubDefault='benchmark_upload_has_github'
        BenchmarkUploadGhAuthDefault='benchmark_upload_gh_auth'; BenchmarkUploadFallbackDefault='benchmark_upload_fallback'
        SubmissionBrowserActionDefault='submission_browser_action'; SubmissionFailureActionDefault='submission_failure_action'
        SubmissionPendingRetryDefault='submission_pending_retry'; ErrorUploadDefault='error_upload'
    }
    $script:Saved=$State
    foreach ($name in $preferenceMap.Keys) { Set-Variable -Scope Script -Name $name -Value $State.($preferenceMap[$name]) }
    $nativePromptsBefore=$script:NativePromptsEnabled
    try {
        if($Action -eq 'update' -and ('MoE4All.Launcher.NativeStartupPrompts' -as [type])) {
            Show-UpdateStatus -CurrentVersion $productVersion -Window -Force
            return
        }
        if ('MoE4All.Launcher.NativePrompt' -as [type]) { $script:NativePromptsEnabled=$true; [MoE4All.Launcher.NativePrompt]::BeginHelper('MoE4All') }
        Clear-Host
        switch ($Action) {
            'download' {
                $kind=switch($Target){'model'{'llm'} 'vision_projector'{'vision'} 'embedding_model'{'embedding'} 'transcription_model'{'audio'} 'speech_model'{'speech'} 'speech_lexicon'{'speech'} default{'all'}}
                $downloaded=@(Select-RecommendedDownloads -Kind $kind)
                if($kind -ne 'all' -and $downloaded.Count -gt 0){$State.$Target=[string]$downloaded[0]}
            }
            'report' { [void](Complete-ErrorReport) }
            'pending' { Offer-PendingSubmissions -ExplicitAction }
            'update' { Show-UpdateStatus -CurrentVersion $productVersion }
            default { throw "Unknown dashboard action: $Action" }
        }
        [void](Read-Host '按回车返回工作台 / Enter to return')
    } catch {
        if (-not (Test-WizardPromptCancellation $_.Exception)) { throw }
    } finally {
        if ('MoE4All.Launcher.NativePrompt' -as [type]) { [MoE4All.Launcher.NativePrompt]::EndHelper() }
        $script:NativePromptsEnabled=$nativePromptsBefore
        foreach ($name in $preferenceMap.Keys) { $State.($preferenceMap[$name]) = (Get-Variable -Scope Script -Name $name).Value }
        $script:Saved=$savedBefore
    }
}

function Test-WizardPromptCancellation {
    param([Exception]$Exception)
    while ($null -ne $Exception) { if ($Exception.GetType().FullName -eq 'MoE4All.Launcher.NativePromptCanceledException') { return $true }; $Exception=$Exception.InnerException }
    return $false
}

function Assert-WizardHelperActive {
    if ($script:NativePromptsEnabled -and ('MoE4All.Launcher.NativePrompt' -as [type])) { [MoE4All.Launcher.NativePrompt]::CheckCancellation() }
}

function Write-Warning {
    [CmdletBinding()]
    param([Parameter(Mandatory=$true,Position=0)][string]$Message)
    if ($script:NativePromptsEnabled -and ('MoE4All.Launcher.NativePrompt' -as [type])) { [MoE4All.Launcher.NativePrompt]::Append(('警告 / Warning: '+$Message+"`n")); return }
    Microsoft.PowerShell.Utility\Write-Warning @PSBoundParameters
}

function Write-Progress {
    [CmdletBinding()]
    param([Parameter(Mandatory=$true,Position=0)][string]$Activity,[Parameter(Position=1)][string]$Status='',[int]$Id=0,[int]$PercentComplete=-1,[int]$SecondsRemaining=-1,[string]$CurrentOperation='',[int]$ParentId=-1,[switch]$Completed,[int]$SourceId=0)
    if ($script:NativePromptsEnabled -and ('MoE4All.Launcher.NativePrompt' -as [type])) {
        $label=if($Completed){$Activity+' ✓'}else{$Activity+'  '+$Status};[MoE4All.Launcher.NativePrompt]::Progress($label);return
    }
    Microsoft.PowerShell.Utility\Write-Progress @PSBoundParameters
}

function Read-Host {
    [CmdletBinding()]
    param([Parameter(Position=0)][object]$Prompt='', [switch]$AsSecureString)
    if ($script:NativePromptsEnabled -and ('MoE4All.Launcher.NativePrompt' -as [type])) {
        $value=[MoE4All.Launcher.NativePrompt]::Read([string]$Prompt,[bool]$AsSecureString)
        if ($AsSecureString) { if ($value.Length -eq 0) { return [Security.SecureString]::new() }; return (ConvertTo-SecureString -String $value -AsPlainText -Force) }
        return $value
    }
    Microsoft.PowerShell.Utility\Read-Host @PSBoundParameters
}

function Write-Host {
    [CmdletBinding()]
    param([Parameter(Position=0,ValueFromRemainingArguments=$true)][object[]]$Object, [string]$Separator=' ', [switch]$NoNewLine, [ConsoleColor]$ForegroundColor='Gray', [ConsoleColor]$BackgroundColor='Black')
    if ($script:NativePromptsEnabled -and ('MoE4All.Launcher.NativePrompt' -as [type])) {
        $text=($Object | ForEach-Object {[string]$_}) -join $Separator
        if (-not $NoNewLine) { $text+="`n" }
        [MoE4All.Launcher.NativePrompt]::Append($text)
        return
    }
    Microsoft.PowerShell.Utility\Write-Host @PSBoundParameters
}

function Resolve-DashboardModelFile {
    param([string]$Value)
    $path = ConvertTo-FullPath $Value
    if (Test-Path -LiteralPath $path -PathType Container) {
        $candidates = @(Get-ChildItem -LiteralPath $path -File -Filter '*.gguf' | Sort-Object Name)
        if ($candidates.Count -eq 1) { return $candidates[0].FullName }
        $firstShards = @($candidates | Where-Object { $_.Name -match '-00001-of-\d+\.gguf$' -and $_.Name -notmatch '^mmproj' })
        if ($firstShards.Count -eq 1) { return $firstShards[0].FullName }
        throw '目录包含多个模型或没有 GGUF，请选择具体文件。Directory contains multiple models or no GGUF; select a file.'
    }
    if ([IO.Path]::GetExtension($path) -ine '.gguf') { throw '请选择 GGUF 文件。Select a GGUF file.' }
    return $path
}

function Update-DashboardModelMetadata {
    param([object]$State)
    if ($State.launch_mode -eq 'server' -and -not $State.server_llm_enabled) { return }
    if (-not $State.model -or -not (Test-Path -LiteralPath $State.model -PathType Leaf)) { return }
    $previousTemp = $env:TEMP; $previousTmp = $env:TMP
    $probeTemp = if ($env:MOE4ALL_WIZARD_TEST_TMP) { $env:MOE4ALL_WIZARD_TEST_TMP }
        elseif ($script:IsPackagedInstall) { Join-Path $dataDir 'cache\launcher\metadata-tmp' }
        else {
            $generatedRoot = if ($env:MOE4ALL_WORK_ROOT) { $env:MOE4ALL_WORK_ROOT } else { Join-Path $repoRoot 'local-work' }
            Join-Path $generatedRoot 'runtime\moe4all\launcher\tmp'
        }
    [void][IO.Directory]::CreateDirectory($probeTemp)
    try {
        $env:TEMP=$probeTemp; $env:TMP=$probeTemp
        $efforts = @(Get-ModelReasoningEfforts -ModelPath $State.model)
        $State.reasoning_probe_efforts = $efforts
        $identity = Get-ModelProbeIdentity -ModelPath $State.model
        $State.reasoning_probe_model = $identity.Path
        $State.reasoning_probe_length = $identity.Length
        $State.reasoning_probe_mtime_ticks = $identity.LastWriteTicks
        if ($State.reasoning_effort -ne 'default' -and $State.reasoning_effort -notin $efforts) { $State.reasoning_effort = 'default' }
    } finally { $env:TEMP=$previousTemp; $env:TMP=$previousTmp }
}

function Resolve-DashboardLaunchPaths {
    param([object]$State)
    $llm = $State.launch_mode -ne 'server' -or $State.server_llm_enabled
    if ($llm -and $State.setup_mode -eq 'manual') {
        switch ([string]$State.kv_preset) {
            'q8' { $State.kv_type_k='q8_0'; $State.kv_type_v='q8_0' }
            'f16' { $State.kv_type_k='f16'; $State.kv_type_v='f16' }
            'auto' { $State.kv_type_k=''; $State.kv_type_v='' }
        }
    }
    $modelKeys = @()
    if ($llm) { $modelKeys += 'model' }
    if ($State.launch_mode -eq 'server') {
        if ($llm -and $State.server_vision) { $modelKeys += 'vision_projector' }
        if ($State.server_embedding) { $modelKeys += 'embedding_model' }
        if ($State.server_transcription) { $modelKeys += 'transcription_model' }
    }
    foreach ($key in $modelKeys) { if ($State.$key) { $State.$key = Resolve-DashboardModelFile ([string]$State.$key) } }
    if ($llm -and $State.setup_mode -eq 'manual' -and $State.config_path) { $State.config_path = ConvertTo-FullPath $State.config_path }
    if ($State.launch_mode -eq 'server') {
        # Speech accepts an exact GGUF file plus a separate asset directory.
        # Never resolve its assets as a GGUF directory or touch disabled speech paths.
        if ($State.server_speech) {
            if ($State.speech_model) { $State.speech_model = ConvertTo-FullPath $State.speech_model }
            if ($State.speech_lexicon) { $State.speech_lexicon = ConvertTo-FullPath $State.speech_lexicon }
        }
        if ($llm -and $State.server_session_cache -and $State.session_cache_dir) { $State.session_cache_dir = ConvertTo-FullPath $State.session_cache_dir }
        if ($State.server_transcription -and $State.transcription_upload_dir) { $State.transcription_upload_dir = ConvertTo-FullPath $State.transcription_upload_dir }
    }
}

function Save-DashboardDraft {
    param([Parameter(Mandatory=$true)][object]$State)
    $clean = Get-DashboardInitialState -Saved $State
    $destination = if ($statePath) { $statePath } else { Join-Path $dataDir 'wizard-state.json' }
    $parent = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($destination))
    [void][IO.Directory]::CreateDirectory($parent)
    $temporary = Join-Path $parent ('wizard-state.new-'+[guid]::NewGuid().ToString('N'))
    try {
        [IO.File]::WriteAllText($temporary,(($clean|ConvertTo-Json -Depth 20)+"`n"),[Text.UTF8Encoding]::new($true))
        if (Test-Path -LiteralPath $destination -PathType Leaf) { [IO.File]::Replace($temporary,$destination,[NullString]::Value) }
        else { [IO.File]::Move($temporary,$destination) }
    } finally { if (Test-Path -LiteralPath $temporary -PathType Leaf) { [IO.File]::Delete($temporary) } }
}

function Invoke-WizardDashboard {
    param([AllowNull()][object]$Saved, [switch]$Demo)
    . (Join-Path $PSScriptRoot 'infr-dashboard-state.ps1')
    if(-not $Demo){Start-EngineUpdateProbe -CurrentVersion $productVersion}
    Initialize-DashboardRuntime
    $working = Get-DashboardInitialState -Saved $Saved
    if ($InitialModelPath) { $working.model = $InitialModelPath }
    if ((Test-WizardConsoleInput) -and -not ('MoE4All.Launcher.NativeWizard' -as [type])) {
        try {
            $window = $Host.UI.RawUI.WindowSize; $maximum = $Host.UI.RawUI.MaxPhysicalWindowSize
            $buffer = $Host.UI.RawUI.BufferSize
            $window.Width = [Math]::Min($maximum.Width, [Math]::Max($window.Width, 136))
            $window.Height = [Math]::Min($maximum.Height, [Math]::Max($window.Height, 44))
            $buffer.Width = [Math]::Max($buffer.Width, $window.Width); $buffer.Height = [Math]::Max($buffer.Height, $window.Height)
            $Host.UI.RawUI.BufferSize = $buffer; $Host.UI.RawUI.WindowSize = $window
        } catch { }
    }
    $message = ''; $notification = ''; $resumePage = 'home'; $resumeField = ''; $resumeSettingsPage=''; $resumeSettingsField=''; $resumeSettingsGroups=''; $ephemeralKey = ''; $dirty=$false; $resumeGroups=$null
    $deviceOptions = @()
    if (-not $Demo) {
        # Only enumeration, no model allocation. Existing probe is shared with the classic wizard.
        try { $deviceOptions = @(Get-VulkanDeviceOptions) } catch { $message = '设备探测不可用；可手动填写设备。Device probe unavailable; enter a device manually.' }
        try { Update-DashboardModelMetadata -State $working } catch { $message = '模型模板探测不可用，可继续调整配置。Template probe unavailable; you may continue configuring.' }
        [MoE4All.Launcher.WizardTheme]::SetPreference([string]$working.wizard_theme)
        Show-UpdateStatus -CurrentVersion $productVersion -Window
        $script:DashboardBenchmarkGuideActive = Invoke-BenchmarkStartupGuide -Interactive $true -Window
        if($script:DashboardBenchmarkGuideActive){$working.launch_mode='benchmark';$working.bench_kind='quick';$working.bench_parallel='1';$resumePage='benchmark'}
    }
    while ($true) {
        if ($ephemeralKey) { $working | Add-Member -NotePropertyName server_api_key -NotePropertyValue $ephemeralKey -Force }
        $schema = Get-DashboardSchema -State $working -DeviceOptions $deviceOptions
        $schema | Add-Member -NotePropertyName download_catalog -NotePropertyValue @(Get-RecommendedModelCatalog) -Force
        $downloadRoot = if ($working.download_dir) { ConvertTo-FullPath $working.download_dir } else { Join-Path $repoRoot 'models' }
        $schema | Add-Member -NotePropertyName download_root -NotePropertyValue $downloadRoot -Force
        $schema | Add-Member -NotePropertyName resume_page -NotePropertyValue $resumePage -Force
        $schema | Add-Member -NotePropertyName resume_field -NotePropertyValue $resumeField -Force
        $schema | Add-Member -NotePropertyName resume_dirty -NotePropertyValue $dirty -Force
        $schema | Add-Member -NotePropertyName notice -NotePropertyValue $notification -Force
        $schema | Add-Member -NotePropertyName native_settings_page -NotePropertyValue $resumeSettingsPage -Force
        $schema | Add-Member -NotePropertyName native_settings_field -NotePropertyValue $resumeSettingsField -Force
        $schema | Add-Member -NotePropertyName native_settings_groups -NotePropertyValue $resumeSettingsGroups -Force
        if ($null -ne $resumeGroups) { $schema | Add-Member -NotePropertyName resume_groups -NotePropertyValue @($resumeGroups) -Force }
        $uiVersion = if ($Demo) { 'PREVIEW' } else { $productVersion }
        if ('MoE4All.Launcher.NativeWizard' -as [type]) {
            $response = [MoE4All.Launcher.NativeWizard]::Run(($working | ConvertTo-Json -Depth 20 -Compress), ($schema | ConvertTo-Json -Depth 20 -Compress), $uiVersion, [bool]$Demo, $message) | ConvertFrom-Json
        } else {
            $response = [MoE4All.Launcher.Dashboard]::Run(($working | ConvertTo-Json -Depth 20 -Compress), ($schema | ConvertTo-Json -Depth 20 -Compress), $uiVersion, [bool]$Demo, $message) | ConvertFrom-Json
        }
        $working = $response.state; $resumePage = [string]$response.page; $resumeField=[string]$response.focus; $message = ''; $notification=''
        $dirty=[bool]$response.dirty; $resumeGroups=@($response.groups)
        $resumeSettingsPage=if($response.PSObject.Properties['settings_page']){[string]$response.settings_page}else{''}
        $resumeSettingsField=if($response.PSObject.Properties['settings_field']){[string]$response.settings_field}else{''}
        $resumeSettingsGroups=if($response.PSObject.Properties['settings_groups']){[string]$response.settings_groups}else{''}
        $ephemeralKey = [string]$response.secrets.server_api_key
        switch ([string]$response.action) {
            'cancel' { return $null }
            { $_ -in @('save','save_exit') } {
                if ($Demo) {
                    if ($response.action -eq 'save_exit') { return $null }
                    $notification='演示模式，不保存到本机配置'; continue
                }
                try {
                    Save-DashboardDraft -State $working
                    $dirty=$false
                    if ($response.action -eq 'save_exit') { return $null }
                    $notification='已保存'
                } catch { $message='保存失败：'+$_.Exception.Message }
            }
            'start' {
                try { Resolve-DashboardLaunchPaths -State $working } catch { $message=$_.Exception.Message; continue }
                if ($working.launch_mode -ne 'server' -or $working.server_llm_enabled) {
                    try { Update-DashboardModelMetadata -State $working } catch { $message=$_.Exception.Message; continue }
                }
                if (-not $working.configure_thinking) { $working.think_mode='default'; $working.reasoning_effort='default' }
                if ($working.launch_mode -eq 'benchmark' -and $working.bench_kind -eq 'manual') {
                    $working.json_output = $true
                    if ($working.manual_bench_kind -eq 'decode') { $working.prompt_tokens = '0' }
                    if ($working.manual_bench_kind -eq 'prefill') { $working.gen_tokens = '0'; $working.depth_mode = 'none'; $working.depth_tokens = '0' }
                }
                $problems = @(Test-DashboardState -State $working)
                if ($working.launch_mode -eq 'server' -and $working.server_auth -and -not $ephemeralKey) {
                    $problems += [pscustomobject]@{field='server_api_key';message='启用了鉴权，请填写 API key。Authentication requires an API key.'}
                }
                if ($problems.Count) {
                    $message = ($problems | ForEach-Object {$_.message}) -join '；'
                    $resumeField=[string]$problems[0].field
                    $problemField = @($schema.fields | Where-Object {$_.key -eq $resumeField}) | Select-Object -First 1
                    if ($problemField) { $resumePage=[string]$problemField.page }
                    continue
                }
                $script:DashboardBenchmarkGuideActive=[bool]$script:DashboardBenchmarkGuideActive -and $working.launch_mode-eq'benchmark' -and $working.bench_kind-eq'quick' -and [string]$working.bench_parallel-eq'1'
                return [pscustomobject]@{ State=$working; ApiKey=$ephemeralKey }
            }
            'model' {
                if (-not $Demo) {
                    try {
                        $working.model=Resolve-DashboardModelFile ([string]$working.model)
                        if (Test-Path -LiteralPath $working.model -PathType Leaf) {
                            $history=[Collections.Generic.List[string]]::new();$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
                            foreach($candidate in @($working.model)+@($working.model_history)){if($candidate -and $seen.Add([string]$candidate)){$history.Add([string]$candidate)}}
                            $working.model_history=$history.ToArray()
                        }
                        Update-DashboardModelMetadata -State $working
                    } catch { $message=$_.Exception.Message }
                }
            }
            { $_ -in @('reports.benchmark','reports.errors') } {
                if ($Demo) { $notification='演示模式，不打开本机文件夹'; continue }
                $folder = if ($response.action -eq 'reports.benchmark') {
                    if ($working.benchmark_result_dir) { ConvertTo-FullPath ([string]$working.benchmark_result_dir) } else { Join-Path $repoRoot 'benchmark-database' }
                } else { Join-Path $repoRoot 'diagnostics\reports' }
                if (-not (Test-Path -LiteralPath $folder -PathType Container)) { $notification='还没有报告；运行后会自动保存'; continue }
                try { $open=[Diagnostics.ProcessStartInfo]::new();$open.FileName=$folder;$open.UseShellExecute=$true;[void][Diagnostics.Process]::Start($open) } catch { $message='无法打开文件夹：'+$_.Exception.Message }
            }
            { $_ -in @('download','report','pending','update') } {
                if ($Demo) { $message = '演示模式不下载、不上传、不调用网络服务。Preview does not download, upload or call network services.'; continue }
                try {
                    $beforeHelperSettings = $working | ConvertTo-Json -Depth 20 -Compress
                    $downloadTarget=if($response.PSObject.Properties['download_target']){[string]$response.download_target}else{''}
                    Invoke-DashboardHelper -Action $response.action -State $working -Target $downloadTarget
                    if ($response.action -eq 'download') {
                        $history = [Collections.Generic.List[string]]::new()
                        foreach ($path in @($working.model_history)) { if ($path -and -not $history.Contains([string]$path)) { $history.Add([string]$path) } }
                        foreach ($bundle in @(Get-RecommendedModelCatalog)) {
                            if (-not $bundle.Primary) { continue }
                            $candidate = Join-Path $downloadRoot $bundle.Primary
                            if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
                            if ($bundle.Kind -eq 'llm') { if (-not $history.Contains($candidate)) { $history.Add($candidate) }; if (-not $working.model) { $working.model=$candidate } }
                            if ($bundle.Kind -eq 'audio' -and -not $working.transcription_model) { $working.transcription_model=$candidate }
                            if ($bundle.Kind -eq 'embedding' -and -not $working.embedding_model) { $working.embedding_model=$candidate }
                        }
                        $working.model_history = $history.ToArray()
                    }
                    if (($working | ConvertTo-Json -Depth 20 -Compress) -ne $beforeHelperSettings) { $dirty=$true }
                } catch { $message = $_.Exception.Message }
            }
            'import' {
                try {
                    Add-Type -AssemblyName System.Windows.Forms
                    $dialog = [Windows.Forms.OpenFileDialog]::new(); $dialog.Filter = 'Wizard settings (*.json)|*.json'; $dialog.Title = '导入向导配置 / Import wizard settings'
                    try {
                        if ($dialog.ShowDialog() -eq [Windows.Forms.DialogResult]::OK) {
                            if ((Get-Item -LiteralPath $dialog.FileName).Length -gt 1048576) { throw '配置文件超过 1 MiB / Settings file exceeds 1 MiB' }
                            $imported = [IO.File]::ReadAllText($dialog.FileName) | ConvertFrom-Json
                            if ($null -eq $imported -or $imported -is [Array] -or $imported -isnot [pscustomobject]) { throw '配置必须为 JSON 对象 / Settings must be a JSON object' }
                            $working = Get-DashboardInitialState -Saved $imported
                            $dirty=$true; $notification='已导入，启动前可继续修改'
                        }
                    } finally { $dialog.Dispose() }
                } catch { $message = '导入失败 / Import failed: ' + $_.Exception.Message }
            }
            'export' {
                try {
                    Add-Type -AssemblyName System.Windows.Forms
                    $dialog = [Windows.Forms.SaveFileDialog]::new(); $dialog.Filter = 'Wizard settings (*.json)|*.json'; $dialog.FileName = 'wizard-state.json'; $dialog.Title = '导出配置（不含密钥）/ Export without secrets'
                    try {
                        if ($dialog.ShowDialog() -eq [Windows.Forms.DialogResult]::OK) { $exportable=Get-DashboardInitialState -Saved $working; [IO.File]::WriteAllText($dialog.FileName, (($exportable | ConvertTo-Json -Depth 20) + "`n"), [Text.UTF8Encoding]::new($true)); $notification = '已导出，不包含密钥' }
                    } finally { $dialog.Dispose() }
                } catch { $message = '导出失败 / Export failed: ' + $_.Exception.Message }
            }
            default { $message = '这个操作尚未接入。This action is unavailable.' }
        }
        # The next frame receives the ephemeral key separately from the exportable response state.
    }
}
