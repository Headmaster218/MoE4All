[CmdletBinding()]
param()

# Pure fixtures: no engine, network, user state, probe, model or output files.
$ErrorActionPreference = 'Stop'
$stateScript = Join-Path $PSScriptRoot 'infr-dashboard-state.ps1'
$wizardScript = Join-Path $PSScriptRoot 'infr-wizard.ps1'
. $stateScript
$script:DashboardChecks = 0
function Assert-DashboardCheck {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw ('Dashboard check failed: ' + $Message) }
    $script:DashboardChecks++
}
function New-DashboardFixture {
    param([hashtable]$Changes = @{})
    $s = Get-DashboardInitialState
    # Existing source is only an existence fixture, never opened as a model.
    $s.model = $stateScript
    foreach ($key in $Changes.Keys) { $s | Add-Member -NotePropertyName $key -NotePropertyValue $Changes[$key] -Force }
    return $s
}
function Assert-DashboardErrors {
    param($State, [string[]]$Expected = @())
    $actual = @(Test-DashboardState -State $State | ForEach-Object { $_.field } | Sort-Object -Unique)
    $wanted = @($Expected | Sort-Object -Unique)
    Assert-DashboardCheck (($actual -join '|') -ceq ($wanted -join '|')) ("Expected [$($wanted -join ',')], got [$($actual -join ',')]")
}
function Get-FixtureVisibleFields {
    param($State, [string]$Page)
    $values = @{}
    foreach ($property in $State.PSObject.Properties) { $values[$property.Name] = $property.Value }
    $values._llm = ($State.launch_mode -ne 'server' -or $State.server_llm_enabled)
    $values._api = $State.launch_mode -eq 'server'
    $values._bench = $State.launch_mode -eq 'benchmark'
    $values._manual = $State.setup_mode -eq 'manual'
    $values._device = $values._llm -or $State.server_embedding
    $values._audio = $State.server_transcription
    foreach ($field in (Get-DashboardSchema -State $State).fields) {
        if ($field.page -ne $Page) { continue }
        $visible = $true
        foreach ($key in $field.when.Keys) {
            $v = $values[$key]; if ($v -is [bool]) { $v = $v.ToString().ToLowerInvariant() }
            if ([string]$v -notin @($field.when[$key])) { $visible = $false; break }
        }
        if ($visible) { $field }
    }
}

$fresh = Get-DashboardInitialState
Assert-DashboardCheck ($fresh -is [pscustomobject]) 'Initial state is a PSCustomObject'
$fresh | Add-Member -NotePropertyName resume_page -NotePropertyValue 'home'
Assert-DashboardCheck ($fresh.resume_page -eq 'home') 'Caller can add resume_page'
foreach ($entry in @{
    launch_mode='chat'; setup_mode='conservative'; device=''; context=''; submit_cap='64'
    bench_kind='quick'; bench_parallel='1'; gen_tokens='512'; prompt_tokens='6144'
    max_new='65536'; server_addr='127.0.0.1:8080'; server_parallel='1'
    session_idle_secs='120'; session_cache_max='5GiB'; session_cache_ttl_hours='24'
    embedding_idle_timeout='300'; transcription_threads='4'; transcription_idle_timeout='300'
}.GetEnumerator()) { Assert-DashboardCheck ([string]$fresh.($entry.Key) -ceq [string]$entry.Value) ('Legacy default: '+$entry.Key) }
foreach ($key in @('host_dma','server_llm_enabled','server_session_cache','benchmark_upload_has_github','submission_pending_retry')) { Assert-DashboardCheck ($fresh.$key -is [bool] -and $fresh.$key) ('True boolean: '+$key) }
foreach ($key in @('server_auth','server_vision','server_embedding','server_transcription','configure_memory','configure_sampling','cpu_miss_enabled')) { Assert-DashboardCheck ($fresh.$key -is [bool] -and -not $fresh.$key) ('False boolean: '+$key) }
Assert-DashboardCheck ($fresh.session_cache_dir -eq (Join-Path (Split-Path $PSScriptRoot) 'kv-sessions')) 'KV directory remains installation-local'
Assert-DashboardCheck ($fresh.transcription_upload_dir -eq (Join-Path (Split-Path $PSScriptRoot) 'gui-data\audio-uploads')) 'Audio directory remains installation-local'
Assert-DashboardCheck (@($fresh.model_history).Count -eq 0) 'Fresh history is an empty array'

$saved = [pscustomobject]@{
    launch_mode='serve'; setup_mode='advanced'; model_history=@('fixture-b.gguf','fixture-a.gguf')
    unknown_setting=[pscustomobject]@{version=7}; server_transcription='false'; server_embedding='true'
    server_api_key='fixture-secret'; api_key='fixture-secret'; last_command='fixture-secret'
    custom_sets='paging.foo=2; serve.api_key=fixture-secret;foo.bar=3;API_KEY=fixture-secret'
}
$before = $saved | ConvertTo-Json -Depth 10 -Compress
$migrated = Get-DashboardInitialState -Saved $saved
Assert-DashboardCheck ($migrated.launch_mode -eq 'server' -and $migrated.setup_mode -eq 'manual') 'Migrate native aliases'
Assert-DashboardCheck ($migrated.unknown_setting.version -eq 7) 'Unknown fields preserved'
Assert-DashboardCheck (($migrated.model_history -join '|') -eq 'fixture-b.gguf|fixture-a.gguf') 'History content and ordering preserved'
Assert-DashboardCheck ($migrated.server_transcription -is [bool] -and -not $migrated.server_transcription) 'String false is not truthy'
Assert-DashboardCheck ($migrated.server_embedding -is [bool] -and $migrated.server_embedding) 'String true normalized'
Assert-DashboardCheck (($migrated | ConvertTo-Json -Depth 10 -Compress) -notmatch 'fixture-secret') 'Initial state removes dedicated and extra-setting keys'
Assert-DashboardCheck ($migrated.custom_sets -eq 'paging.foo=2;foo.bar=3') 'Other extra-setting entries preserved'
Assert-DashboardCheck (($saved | ConvertTo-Json -Depth 10 -Compress) -ceq $before) 'Input object unmodified'
Assert-DashboardCheck ((Get-DashboardInitialState -Saved ([pscustomobject]@{})).setup_mode -eq 'manual') 'Old all-advanced state keeps manual semantics'
foreach ($pair in @(@('run','chat'),@('serve','server'),@('bench','benchmark'))) {
    Assert-DashboardCheck ((Get-DashboardInitialState -Saved @{mode=$pair[0];setup_mode='quick'}).launch_mode -eq $pair[1]) ('Mode alias '+$pair[0])
}
Assert-DashboardCheck ((Get-DashboardInitialState -Saved @{setup_mode='quick'}).setup_mode -eq 'conservative') 'Quick profile alias'
Assert-DashboardCheck ((Get-DashboardInitialState -Saved @{mode='advanced'}).setup_mode -eq 'manual') 'Legacy mode used as profile'
Assert-DashboardCheck ((Get-DashboardInitialState -Saved @{dram_cache='36g'}).ram_budget -eq '36g') 'Legacy dram_cache is total-process RAM budget'
Assert-DashboardCheck ((Get-DashboardInitialState -Saved @{dram_cache='36g';ram_budget='24g'}).ram_budget -eq '24g') 'Explicit RAM budget wins'
Assert-DashboardCheck ((Get-DashboardInitialState -Saved @{think_mode='think'}).configure_thinking -eq $true) 'Legacy reasoning config remains editable'
Assert-DashboardCheck ((Get-DashboardInitialState -Saved @{submit_mode='off'}).submit_mode -eq 'disabled') 'No-split alias'
foreach ($kind in @('prefill','decode','mixed')) {
    $s = Get-DashboardInitialState -Saved @{bench_kind=$kind}
    Assert-DashboardCheck ($s.bench_kind -eq 'manual' -and $s.manual_bench_kind -eq $kind) ('Legacy benchmark kind '+$kind)
}
foreach ($pair in @(@('p','prefill'),@('n','decode'),@('pg','mixed'))) { Assert-DashboardCheck ((Get-DashboardInitialState -Saved @{manual_bench_kind=$pair[0]}).manual_bench_kind -eq $pair[1]) ('Benchmark work alias '+$pair[0]) }
foreach ($flagValue in @('false','False','0','off','no')) { Assert-DashboardCheck ((Get-DashboardInitialState -Saved @{server_embedding=$flagValue}).server_embedding -ceq $false) ('False spelling '+$flagValue) }

$schema = Get-DashboardSchema -State $fresh -DeviceOptions @([pscustomobject]@{Value='Vulkan1';Label='Fixture GPU'}, 'Vulkan0')
Assert-DashboardCheck ($schema -is [pscustomobject]) 'Schema is a PSCustomObject'
Assert-DashboardCheck (($schema.pages.id -join '/') -eq 'home/models/runtime/api/benchmark/reports/maintenance') 'Seven pages in required order'
foreach ($page in $schema.pages) { Assert-DashboardCheck (-not [string]::IsNullOrWhiteSpace($page.en)) ('English page label '+$page.id) }
foreach ($field in $schema.fields) {
    Assert-DashboardCheck ($field.kind -in @('choice','toggle','text','path','secret','action')) ('Valid field kind '+$field.key)
    Assert-DashboardCheck ($field.page -in $schema.pages.id) ('Valid field page '+$field.key)
    Assert-DashboardCheck ($field.group -in @($schema.groups.id)) ('Field has a declared group '+$field.key)
    $group = @($schema.groups | Where-Object { $_.id -eq $field.group })[0]
    Assert-DashboardCheck ($group.page -eq $field.page) ('Field and group share page '+$field.key)
    Assert-DashboardCheck ($field.help -notmatch '[\r\n]') ('Help remains one line '+$field.key)
    if ($field.kind -eq 'action') { Assert-DashboardCheck ($field.action -in @('start','download','report','pending','update','import','export','cancel','save','page.models','page.benchmark','task.server','task.benchmark','reports.benchmark','reports.errors')) ('Bridge action '+$field.key) }
}
$secret = @($schema.fields | Where-Object { $_.kind -eq 'secret' })
Assert-DashboardCheck ($secret.Count -eq 1 -and $secret[0].key -eq 'server_api_key') 'Ephemeral key uses dedicated secret field'
$device = @($schema.fields | Where-Object { $_.key -eq 'device' })[0]
Assert-DashboardCheck ('cpu' -in @($device.options.value) -and 'Vulkan1' -in @($device.options.value)) 'Device enumeration and CPU present'
Assert-DashboardCheck (@($device.options | Where-Object {$_.value -eq 'Vulkan1'})[0].label -eq 'Fixture GPU') 'Device display name retained'
$unknownDeviceSchema = Get-DashboardSchema -State (New-DashboardFixture @{device='Vulkan19'})
Assert-DashboardCheck ('Vulkan19' -in @($unknownDeviceSchema.fields | Where-Object { $_.key -eq 'device' } | Select-Object -First 1).options.value) 'Unavailable saved device remains selectable'
foreach ($duplicate in @($schema.fields | Group-Object key | Where-Object {$_.Count -gt 1})) { Assert-DashboardCheck ($duplicate.Name -in @('model','device','setup_mode','context','server_parallel','action.start')) ('Only intentional duplicate key '+$duplicate.Name) }
$normal = New-DashboardFixture
Assert-DashboardCheck ('model' -in @(Get-FixtureVisibleFields $normal models).key) 'Chat Models page includes main model'
Assert-DashboardCheck ('setup_mode' -in @(Get-FixtureVisibleFields $normal runtime).key) 'Automatic profile can switch to manual in Settings'
$benchmark = New-DashboardFixture @{launch_mode='benchmark'}
Assert-DashboardCheck ('action.start' -in @(Get-FixtureVisibleFields $benchmark benchmark).key) 'Benchmark page has an explicit start action'
Assert-DashboardCheck ('context' -notin @(Get-FixtureVisibleFields $benchmark home).key) 'Benchmark hides general context'
Assert-DashboardCheck ('gen_tokens' -notin @(Get-FixtureVisibleFields $benchmark benchmark).key) 'Automatic benchmark hides manual tokens'
$manualPrefill = New-DashboardFixture @{launch_mode='benchmark';bench_kind='manual';manual_bench_kind='prefill'}
$visible = @(Get-FixtureVisibleFields $manualPrefill benchmark).key
Assert-DashboardCheck ('prompt_tokens' -in $visible -and 'gen_tokens' -notin $visible -and 'depth_mode' -notin $visible) 'Prefill-only has no irrelevant Decode fields'
$manualDecode = New-DashboardFixture @{launch_mode='benchmark';bench_kind='manual';manual_bench_kind='decode';depth_mode='real'}
$visible = @(Get-FixtureVisibleFields $manualDecode benchmark).key
Assert-DashboardCheck ('gen_tokens' -in $visible -and 'depth_tokens' -in $visible -and 'prompt_tokens' -notin $visible) 'Decode-only shows generation and selected depth'
$audio = New-DashboardFixture @{launch_mode='server';server_llm_enabled=$false;server_transcription=$true}
$visible = @(Get-FixtureVisibleFields $audio home).key
Assert-DashboardCheck ('model' -notin $visible -and 'device' -notin $visible -and 'setup_mode' -notin $visible -and 'server_parallel' -notin $visible) 'Pure audio hides all LLM and GPU settings'
Assert-DashboardCheck ('transcription_threads' -in @(Get-FixtureVisibleFields $audio models).key) 'Pure audio exposes CPU settings'

# Group order is also rendering order: no group may resume after another group.
Assert-DashboardCheck (@($schema.groups.id | Sort-Object -Unique).Count -eq $schema.groups.Count) 'Unique group IDs'
foreach ($page in $schema.pages) {
    $seen = @(); $previous = ''
    foreach ($field in @($schema.fields | Where-Object { $_.page -eq $page.id })) {
        if ($field.group -eq $previous) { continue }
        Assert-DashboardCheck ($field.group -notin $seen) ('Contiguous fields in group '+$field.group)
        $seen += $field.group; $previous = $field.group
    }
    $declared = @($schema.groups | Where-Object { $_.page -eq $page.id }).id
    Assert-DashboardCheck (($seen -join '/') -ceq ($declared -join '/')) ('Group declarations follow first field order '+$page.id)
}
foreach ($group in $schema.groups) {
    Assert-DashboardCheck ($group.open -is [bool]) ('Group default is boolean '+$group.id)
    Assert-DashboardCheck ($group.page -in $schema.pages.id) ('Valid group page '+$group.id)
    if ($group.PSObject.Properties['summary_key']) {
        Assert-DashboardCheck ($null -ne $fresh.PSObject.Properties[$group.summary_key]) ('Group summary uses a state key '+$group.id)
    }
}
foreach ($id in @('home_main','model_main','basic','generation','connection','plan','parallel','report_actions')) {
    Assert-DashboardCheck (@($schema.groups | Where-Object { $_.id -eq $id })[0].open) ('Common group opens by default '+$id)
}
foreach ($id in @('vision','embedding','audio','sampling','memory','diagnostics','session','submission_preferences')) {
    Assert-DashboardCheck (-not @($schema.groups | Where-Object { $_.id -eq $id })[0].open) ('Optional group folds by default '+$id)
}
Assert-DashboardCheck ('advanced' -notin $schema.pages.id) 'No separate Advanced page'
Assert-DashboardCheck ('configure_thinking' -notin $schema.fields.key) 'No duplicate reasoning switch in UI'
Assert-DashboardCheck ($null -ne $fresh.PSObject.Properties['configure_thinking']) 'Legacy reasoning state retained'
Assert-DashboardCheck ('think_mode' -in @(Get-FixtureVisibleFields $normal runtime).key) 'Reasoning selection available without extra switch'
Assert-DashboardCheck ('reasoning_effort' -in @(Get-FixtureVisibleFields $normal runtime).key) 'Effort available without extra switch'
$noThinking = New-DashboardFixture @{think_mode='no-think'}
Assert-DashboardCheck ('reasoning_effort' -notin @(Get-FixtureVisibleFields $noThinking runtime).key) 'Disabled reasoning hides effort'
Assert-DashboardCheck ('think_mode' -notin @(Get-FixtureVisibleFields $benchmark runtime).key) 'Benchmark hides generation settings'
Assert-DashboardCheck ('setup_mode' -in @(Get-FixtureVisibleFields $normal runtime).key -and 'ubatch' -notin @(Get-FixtureVisibleFields $normal runtime).key) 'Profile accessible while manual resources inactive'
$runtimeFields = @($schema.fields | Where-Object { $_.page -eq 'runtime' })
Assert-DashboardCheck ($runtimeFields[0].key -eq 'setup_mode') 'Profile is first in runtime page'
$activeManual = New-DashboardFixture @{launch_mode='server';setup_mode='manual';configure_memory=$true;kv_preset='custom';kv_overflow=$true;submit_mode='fixed';configure_sampling=$true;configure_diagnostics=$true;cpu_miss_enabled=$true;server_embedding=$true;server_transcription=$true;server_vision=$true;server_auth=$true}
Assert-DashboardCheck ('ubatch' -in @(Get-FixtureVisibleFields $activeManual runtime).key -and 'submit_cap' -in @(Get-FixtureVisibleFields $activeManual runtime).key) 'Manual controls reachable under manual profile'
Assert-DashboardCheck ('kv_preset' -notin @(Get-FixtureVisibleFields $normal runtime).key) 'Automatic KV semantics unchanged'
foreach ($field in @($schema.fields | Where-Object { $_.group -eq 'memory' })) {
    Assert-DashboardCheck ($field.when._manual -eq 'true') ('Manual resource remains manual only '+$field.key)
}
foreach ($page in @('api','benchmark')) {
    $visible = @(Get-FixtureVisibleFields $normal $page)
    $key = if ($page -eq 'api') { 'action.use_api' } else { 'action.use_benchmark' }
    $target = if ($page -eq 'api') { 'task.server' } else { 'task.benchmark' }
    Assert-DashboardCheck ($visible.Count -eq 1 -and $visible[0].key -eq $key -and $visible[0].action -eq $target) ('Inactive page offers explicit task switch only '+$page)
}
Assert-DashboardCheck ('action.use_api' -notin @(Get-FixtureVisibleFields $activeManual api).key) 'Active API hides task switch'
Assert-DashboardCheck ('action.use_benchmark' -notin @(Get-FixtureVisibleFields $benchmark benchmark).key) 'Active benchmark hides task switch'
Assert-DashboardCheck ('action.modules' -notin @(Get-FixtureVisibleFields $normal home).key -and 'action.modules' -in @(Get-FixtureVisibleFields $activeManual home).key) 'Home modules shortcut only for API'
Assert-DashboardCheck ('action.benchmark' -notin @(Get-FixtureVisibleFields $normal home).key -and 'action.benchmark' -in @(Get-FixtureVisibleFields $benchmark home).key) 'Home plan shortcut only for benchmark'
Assert-DashboardCheck (@($schema.fields | Where-Object { $_.key -eq 'action.modules' })[0].action -eq 'page.models') 'Modules action navigates without changing task'
Assert-DashboardCheck (@($schema.fields | Where-Object { $_.key -eq 'action.benchmark' })[0].action -eq 'page.benchmark') 'Plan action navigates without changing task'
Assert-DashboardCheck (@($schema.fields | Where-Object { $_.key -eq 'action.download' })[0].page -eq 'models') 'Model download belongs to Models'
Assert-DashboardCheck ((@($schema.fields | Where-Object { $_.page -eq 'maintenance' }).action -join '/') -eq 'save/import/export/update/cancel') 'Maintenance limited to configuration and update actions'
Assert-DashboardCheck ((@($schema.fields | Where-Object { $_.group -eq 'report_actions' }).action -join '/') -eq 'report/pending') 'Reports page exposes report and pending actions'
$beforeSchema = $activeManual | ConvertTo-Json -Depth 12 -Compress
foreach ($unused in 1..2) { $null = Get-DashboardSchema -State $activeManual }
Assert-DashboardCheck (($activeManual | ConvertTo-Json -Depth 12 -Compress) -ceq $beforeSchema) 'Building schema never changes task or settings'

# All controls from the original dashboard remain reachable; only the duplicate
# configure_thinking toggle is intentionally represented by think_mode/effort.
$legacyEditableKeys = @(
    'launch_mode','model','device','setup_mode','context','server_parallel','server_llm_enabled',
    'server_vision','vision_projector','server_embedding','embedding_model','embedding_idle_timeout',
    'server_transcription','transcription_model','transcription_threads','transcription_idle_timeout','transcription_upload_dir',
    'think_mode','reasoning_effort','max_new','configure_sampling','temperature','top_k','top_p','seed',
    'server_addr','server_auth','server_api_key','server_session_cache','session_cache_dir','session_idle_secs','session_cache_max','session_cache_ttl_hours',
    'bench_kind','bench_parallel','manual_bench_kind','prompt_tokens','gen_tokens','depth_mode','depth_tokens','reps',
    'ubatch','threads','config_path','kv_preset','kv_type_k','kv_type_v','configure_memory','vram_budget','vram_reserve','expert_cache','ram_budget','host_dma',
    'pager_ring','pager_ring_slots','kv_overflow','kv_overflow_vram_mb','kv_overflow_reserve_mb','submit_mode','submit_cap',
    'cpu_miss_enabled','cpu_miss_max','cpu_miss_cores','configure_diagnostics','pager_stats','pager_profile','stage_profile','vram_profile','custom_sets',
    'benchmark_upload','benchmark_upload_has_github','benchmark_upload_gh_auth','benchmark_upload_fallback','submission_browser_action','submission_failure_action','submission_pending_retry','error_upload'
)
$reachable = @()
foreach ($fixture in @($normal,$activeManual,$benchmark,$manualPrefill,$manualDecode)) {
    foreach ($page in $schema.pages) { $reachable += @(Get-FixtureVisibleFields $fixture $page.id).key }
}
foreach ($key in $legacyEditableKeys) {
    Assert-DashboardCheck ($key -in $schema.fields.key) ('Original editable setting retained '+$key)
    Assert-DashboardCheck ($key -in $reachable) ('Original editable setting reachable '+$key)
}
Assert-DashboardCheck ('device' -notin @(Get-FixtureVisibleFields $audio runtime).key -and 'model' -notin @(Get-FixtureVisibleFields $audio runtime).key) 'Pure transcription has no runtime GPU settings'
Assert-DashboardCheck ('server_session_cache' -notin @(Get-FixtureVisibleFields $audio api).key) 'Pure transcription has no LLM session cache'
$metadataJson = $schema | ConvertTo-Json -Depth 16 -Compress
Assert-DashboardCheck ($metadataJson -notmatch 'fixture-secret|alias-secret') 'Schema never embeds fixture keys'

# Import the single pure catalog function; never dot-source the live launcher.
$tokens = $null; $parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($wizardScript, [ref]$tokens, [ref]$parseErrors)
Assert-DashboardCheck ($parseErrors.Count -eq 0) 'Legacy source parses'
$catalogDefinition = $ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-RecommendedModelCatalog'}, $true) | Select-Object -First 1
. ([scriptblock]::Create($catalogDefinition.Extent.Text))
$catalog = (Get-DashboardSchema -State $normal).catalog
Assert-DashboardCheck (@($catalog).Count -eq @(Get-RecommendedModelCatalog).Count) 'Use the existing pure recommended catalog'
Assert-DashboardCheck (@($catalog | Where-Object {$_.Kind -eq 'audio'}).Count -gt 0) 'Audio download catalog is available'
$saveAssignment = $ast.EndBlock.Statements | Where-Object { $_.Extent.Text -match '^\$state\s*=\s*\[ordered\]' } | Select-Object -Last 1
Assert-DashboardCheck ($null -ne $saveAssignment) 'Find the real saved-state schema'
$saveTable = $saveAssignment.FindAll({param($node) $node -is [Management.Automation.Language.HashtableAst]}, $true) | Select-Object -First 1
foreach ($pair in $saveTable.KeyValuePairs) {
    $key = [string]$pair.Item1.SafeGetValue()
    if ($key -in @('last_command','api_key','server_api_key')) { continue }
    Assert-DashboardCheck ($null -ne $fresh.PSObject.Properties[$key]) ('Every legacy saved field has a default: '+$key)
}

Assert-DashboardErrors (New-DashboardFixture)
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';server_parallel='64'})
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';server_parallel='0'}) @('server_parallel')
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';server_auth=$true}) # Host owns secret validation.
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';server_addr='[::1]:8080'})
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';server_addr='127.0.0.1:70000'}) @('server_addr')
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';server_addr='example.invalid:80'}) @('server_addr')
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';server_llm_enabled=$false}) @('server_llm_enabled')
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';server_llm_enabled=$false;server_embedding=$true;embedding_model=$stateScript;context='not-used';ubatch='not-used';cpu_miss_enabled=$true;server_parallel='not-used';configure_memory=$true;ram_budget='not-used'})
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';server_llm_enabled=$false;server_transcription=$true;transcription_model=$stateScript;device="`nbad";context='not-used';setup_mode='not-used';server_parallel='not-used'})
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';server_llm_enabled=$false;server_transcription=$true;transcription_model=$stateScript;transcription_threads='257'}) @('transcription_threads')
Assert-DashboardErrors (New-DashboardFixture @{server_embedding=$true;embedding_model='missing-unused.gguf';server_vision=$true;vision_projector='missing-unused.gguf';server_auth='not-used'})
foreach ($profile in @('conservative','aggressive')) {
    Assert-DashboardErrors (New-DashboardFixture @{setup_mode=$profile;ubatch='bad';threads='bad';configure_memory='bad';ram_budget='bad';kv_preset='bad';submit_cap='bad';custom_sets='ignored invalid saved extra'})
}
Assert-DashboardErrors (New-DashboardFixture @{setup_mode='manual';ubatch='0';threads='0';kv_preset='bad';submit_mode='fixed';submit_cap='0'}) @('ubatch','threads','kv_preset','submit_cap')
Assert-DashboardErrors (New-DashboardFixture @{setup_mode='manual';configure_memory=$true;vram_budget='21GiB';vram_reserve='512MiB';ram_budget='0';expert_cache='80%';pager_ring='1.5GiB';pager_ring_slots='2'})
Assert-DashboardErrors (New-DashboardFixture @{setup_mode='manual';configure_memory=$true;pager_ring_slots='1';ram_budget='bad'}) @('pager_ring_slots','ram_budget')
Assert-DashboardErrors (New-DashboardFixture @{setup_mode='manual';kv_overflow_vram_mb='bad';configure_memory=$true;kv_overflow=$false})
Assert-DashboardErrors (New-DashboardFixture @{setup_mode='manual';kv_overflow_vram_mb='bad';configure_memory=$true;kv_overflow=$true}) @('kv_overflow_vram_mb')
Assert-DashboardErrors (New-DashboardFixture @{setup_mode='manual';kv_preset='custom';kv_type_k='';kv_type_v=''}) @('kv_type_k','kv_type_v')
Assert-DashboardErrors (New-DashboardFixture @{setup_mode='manual';custom_sets='missing-equals'}) @('custom_sets')
Assert-DashboardErrors (New-DashboardFixture @{setup_mode='manual';custom_sets='serve.api_key=fixture-only'}) @('custom_sets')
Assert-DashboardErrors (New-DashboardFixture @{setup_mode='manual';custom_sets='paging.expert_prefetch=false;spec.mtp=false'})
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';server_session_cache=$false;session_cache_max='bad';session_idle_secs='bad'})
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='server';session_cache_max='80%'}) @('session_cache_max')
Assert-DashboardErrors (New-DashboardFixture @{configure_sampling=$false;temperature='bad';top_k='bad';top_p='bad'})
Assert-DashboardErrors (New-DashboardFixture @{configure_sampling=$true;temperature='NaN';top_k='-1';top_p='1.5'}) @('temperature','top_k','top_p')
Assert-DashboardErrors (New-DashboardFixture @{configure_sampling=$true;temperature='0';top_k='0';top_p='0.95';seed='0'})
Assert-DashboardErrors (New-DashboardFixture @{cpu_miss_enabled=$true;cpu_miss_cores='2';cpu_miss_max='3'})
Assert-DashboardErrors (New-DashboardFixture @{cpu_miss_enabled=$true;cpu_miss_cores='0';cpu_miss_max='4'}) @('cpu_miss_cores','cpu_miss_max')
foreach ($ctx in @('6k','150k','1.5MiB','50%','1e4')) { Assert-DashboardErrors (New-DashboardFixture @{context=$ctx}) }
Assert-DashboardErrors (New-DashboardFixture @{context='not-a-count'}) @('context')
foreach ($plan in @('quick','standard','full')) {
    Assert-DashboardErrors (New-DashboardFixture @{launch_mode='benchmark';bench_kind=$plan;context='stale-unused';max_new='bad';gen_tokens='bad';reps='bad';bench_parallel='4'})
}
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='benchmark';bench_parallel='8'}) @('bench_parallel')
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='benchmark';bench_kind='manual';manual_bench_kind='prefill';gen_tokens='not-used';depth_mode='not-used';depth_tokens='not-used'})
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='benchmark';bench_kind='manual';manual_bench_kind='decode';prompt_tokens='not-used';gen_tokens='1024';depth_mode='real';depth_tokens='6k'}) @('depth_tokens')
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='benchmark';bench_kind='manual';manual_bench_kind='decode';prompt_tokens='not-used';gen_tokens='1024';depth_mode='synthetic';depth_tokens='6144'})
Assert-DashboardErrors (New-DashboardFixture @{launch_mode='benchmark';bench_kind='manual';manual_bench_kind='mixed';prompt_tokens='6144';gen_tokens='1024';reps='0'}) @('reps')
Assert-DashboardErrors (New-DashboardFixture @{model=(Join-Path $PSScriptRoot 'missing-dashboard-fixture.gguf')}) @('model')
Assert-DashboardErrors (New-DashboardFixture @{model="`tinvalid"}) @('model')

$aliases=Get-DashboardInitialState -Saved ([pscustomobject]@{INFR_API_KEY='alias-secret';'serve.api_key'='alias-secret';api_key='alias-secret';server_api_key='alias-secret';unknown_future='kept'})
Assert-DashboardCheck (($aliases|ConvertTo-Json -Depth 10 -Compress) -notmatch 'alias-secret') 'Known top-level key aliases cannot be re-exported'
Assert-DashboardCheck ($aliases.unknown_future -eq 'kept') 'Non-secret future settings survive alias cleanup'
$noEffort=New-DashboardFixture @{configure_thinking=$true;reasoning_probe_model=$stateScript;reasoning_probe_efforts=@()}
$effortField=(Get-DashboardSchema -State $noEffort).fields|Where-Object{$_.key -eq 'reasoning_effort'}|Select-Object -First 1
Assert-DashboardCheck (@($effortField.options).Count -eq 1 -and $effortField.options[0].value -eq 'default') 'A known empty capability probe exposes only template default'
Write-Host ("Dashboard state/schema: {0} checks passed; no engine, user state or generated files." -f $script:DashboardChecks)
