[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
if (-not $env:MOE4ALL_WORK_ROOT -or -not $env:MOE4ALL_WIZARD_TEST_TMP) { throw 'Set MOE4ALL_WORK_ROOT and MOE4ALL_WIZARD_TEST_TMP under local-work first.' }
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$dataDir=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'never-real-user-data'
$script:IsPackagedInstall=$false
. (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
. (Join-Path $PSScriptRoot 'infr-dashboard-state.ps1')
Initialize-DashboardRuntime
$checks=0
function Check([bool]$Condition,[string]$Message){if(-not $Condition){throw $Message};$script:checks++}
function Json($Object){ConvertTo-Json -InputObject $Object -Depth 20 -Compress}
$state=Get-DashboardInitialState -Saved ([pscustomobject]@{launch_mode='server';model='fixtures/中文 model.gguf';device='Vulkan1';setup_mode='aggressive';context='30k';server_parallel='8';server_auth=$true;server_embedding=$true;embedding_model='fixtures/embedding.gguf'})
$state|Add-Member -NotePropertyName server_api_key -NotePropertyValue 'DO_NOT_RENDER_THIS_SECRET'
$schema=Get-DashboardSchema -State $state -DeviceOptions @([pscustomobject]@{Value='Vulkan1';Label='Fixture GPU'})
Check ($schema.pages.Count -eq 7) 'Seven working pages exist'
$before=Json $state
foreach($page in $schema.pages){foreach($geometry in @(@(80,24),@(100,32),@(132,42),@(160,48))){
 $snapshot=[MoE4All.Launcher.Dashboard]::Snapshot((Json $state),(Json $schema),$page.id,$geometry[0],$geometry[1],$false)
 $rows=@($snapshot.TrimEnd([char]10).Split([char]10))
 Check ($rows.Count -eq $geometry[1]) ($page.id+' height')
 foreach($row in $rows){Check ([MoE4All.Launcher.Canvas]::Columns($row) -eq $geometry[0]) ($page.id+' column bounds')}
 Check ($snapshot -notmatch 'DO_NOT_RENDER_THIS_SECRET') ($page.id+' no secret rendering')
 Check ($snapshot -notmatch '[\x00-\x09\x0b-\x1f\x7f]') ($page.id+' no control injection')
 Check ($snapshot -match 'F5') ($page.id+' start shortcut is discoverable')
 Check ($snapshot -match 'F6') ($page.id+' save shortcut avoids terminal flow-control keys')
}}
Check ((Json $state) -eq $before) 'Rendering does not mutate state'
$ui=[MoE4All.Launcher.Dashboard]::new((Json $state),(Json $schema),'1.2.0',$false)
$private=[Reflection.BindingFlags]'Instance,NonPublic'
$change=$ui.GetType().GetMethod('Change',$private)
$resultMethod=$ui.GetType().GetMethod('Result',$private)
$setPage=$ui.GetType().GetMethod('SetPage',$private)
$current=$ui.GetType().GetMethod('Current',$private)
$rowsMethod=$ui.GetType().GetMethod('Rows',$private)
$activate=$ui.GetType().GetMethod('Activate',$private)
$null=$setPage.Invoke($ui,[object[]]@([int]4))
$result=$resultMethod.Invoke($ui,[object[]]@('start'))|ConvertFrom-Json
Check ($result.state.launch_mode -eq 'server') 'Browsing benchmark must not change current task'
$switch=@($current.Invoke($ui,@())|Where-Object{$_['action'] -eq 'task.benchmark'})[0]
$null=$activate.Invoke($ui,[object[]]@($switch.PSObject.BaseObject))
$null=$change.Invoke($ui,[object[]]@('bench_kind','standard'))
$result=$resultMethod.Invoke($ui,[object[]]@('start'))|ConvertFrom-Json
Check ($result.state.launch_mode -eq 'benchmark' -and $result.state.bench_parallel -eq '2') 'Explicit benchmark action activates benchmark and standard defaults'
$null=$change.Invoke($ui,[object[]]@('bench_parallel','3'))
$null=$change.Invoke($ui,[object[]]@('bench_kind','standard'))
$result=$resultMethod.Invoke($ui,[object[]]@('start'))|ConvertFrom-Json
Check ($result.state.bench_parallel -eq '3') 'Re-confirming same plan preserves custom concurrency'
$null=$setPage.Invoke($ui,[object[]]@([int]3))
$result=$resultMethod.Invoke($ui,[object[]]@('start'))|ConvertFrom-Json
Check ($result.state.launch_mode -eq 'benchmark') 'Browsing API must not change task'
$switch=@($current.Invoke($ui,@())|Where-Object{$_['action'] -eq 'task.server'})[0]
$null=$activate.Invoke($ui,[object[]]@($switch.PSObject.BaseObject))
Check ($result.state.PSObject.Properties.Name -notcontains 'server_api_key') 'Exportable state excludes secret'
Check ($result.secrets.server_api_key -eq 'DO_NOT_RENDER_THIS_SECRET') 'Key stays in ephemeral channel'
$null=$change.Invoke($ui,[object[]]@('server_llm_enabled',$false))
$null=$change.Invoke($ui,[object[]]@('server_embedding',$false))
$null=$change.Invoke($ui,[object[]]@('server_transcription',$true))
$null=$setPage.Invoke($ui,[object[]]@([int]0))
$visible=@($current.Invoke($ui,@())|ForEach-Object{$_['key']})
Check ('model' -notin $visible -and 'device' -notin $visible -and 'setup_mode' -notin $visible) 'CPU audio service hides inactive LLM and GPU controls'
$null=$setPage.Invoke($ui,[object[]]@([int]4));$null=$setPage.Invoke($ui,[object[]]@([int]2))
$result=$resultMethod.Invoke($ui,[object[]]@('start'))|ConvertFrom-Json
Check ($result.state.launch_mode -eq 'server') 'Browsing runtime keeps the selected task'
$foldUi=[MoE4All.Launcher.Dashboard]::new((Json $state),(Json $schema),'1.2.0',$false)
$null=$setPage.Invoke($foldUi,[object[]]@([int]1))
$shown=@($rowsMethod.Invoke($foldUi,@()))
Check ('embedding_model' -notin @($shown|ForEach-Object{$_['key']})) 'Optional module settings start collapsed'
$header=@($shown|Where-Object{$_['group_id'] -eq 'models/embedding'})[0]
$null=$activate.Invoke($foldUi,[object[]]@($header.PSObject.BaseObject))
$shown=@($rowsMethod.Invoke($foldUi,@()))
Check ('embedding_model' -in @($shown|ForEach-Object{$_['key']})) 'Expanding group reveals active settings'
$foldResult=$resultMethod.Invoke($foldUi,[object[]]@('save'))|ConvertFrom-Json
Check (-not $foldResult.dirty) 'Folding and navigation do not dirty configuration'
Check ('models/embedding' -in $foldResult.groups) 'Group expansion state survives callbacks'
$clip=[MoE4All.Launcher.Canvas]::Clip('中文文件名🙂.gguf',9)
Check ([MoE4All.Launcher.Canvas]::Columns($clip) -le 9) 'Unicode clipping is bounded'
Check ([MoE4All.Launcher.Canvas]::Safe("x`e[31m`ny") -notmatch '[\x00-\x1f]') 'Terminal control strings cannot escape canvas'
Check ([MoE4All.Launcher.Canvas]::ElementWidth([string][char]0xd83d) -eq 1) 'Partially delivered surrogate does not crash editor'

# Path resolution reads only tiny test fixtures below the caller-owned task directory.
$fixture=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP ('dashboard-fixtures-'+[guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($fixture)
function ConvertTo-FullPath {param([string]$Value) if([IO.Path]::IsPathRooted($Value)){return [IO.Path]::GetFullPath($Value)};return [IO.Path]::GetFullPath((Join-Path $fixture $Value))}
$one=Join-Path $fixture 'single';[void][IO.Directory]::CreateDirectory($one)
[IO.File]::WriteAllText((Join-Path $one 'model.gguf'),'fixture')
Check ((Resolve-DashboardModelFile $one) -eq (Join-Path $one 'model.gguf')) 'Single-GGUF directory resolves to a concrete file'
$shards=Join-Path $fixture 'shards';[void][IO.Directory]::CreateDirectory($shards)
foreach($name in @('model-00001-of-00002.gguf','model-00002-of-00002.gguf','mmproj.gguf')){[IO.File]::WriteAllText((Join-Path $shards $name),'fixture')}
Check ((Resolve-DashboardModelFile $shards) -eq (Join-Path $shards 'model-00001-of-00002.gguf')) 'Shard directory resolves first shard instead of mmproj'
[IO.File]::WriteAllText((Join-Path $shards 'other-00001-of-00002.gguf'),'fixture')
$rejected=$false;try{$null=Resolve-DashboardModelFile $shards}catch{$rejected=$true}
Check $rejected 'Ambiguous model groups require selection'
$standalone=Get-DashboardInitialState -Saved ([pscustomobject]@{launch_mode='server';server_llm_enabled=$false;server_embedding=$false;server_transcription=$true;model='invalid-unused-path';transcription_model=(Join-Path $one 'model.gguf')})
Resolve-DashboardLaunchPaths $standalone
Check ($standalone.model -eq 'invalid-unused-path') 'Inactive LLM path is not resolved or touched'
foreach($preset in @('q8','f16','auto','custom')){
 $kv=Get-DashboardInitialState -Saved @{launch_mode='chat';setup_mode='manual';model=(Join-Path $one 'model.gguf');kv_preset=$preset;kv_type_k='q4_0';kv_type_v='q5_0'}
 Resolve-DashboardLaunchPaths $kv
 $expectedK=switch($preset){q8{'q8_0'} f16{'f16'} auto{''} custom{'q4_0'}}
 $expectedV=switch($preset){q8{'q8_0'} f16{'f16'} auto{''} custom{'q5_0'}}
 Check ($kv.kv_type_k -eq $expectedK -and $kv.kv_type_v -eq $expectedV) ('Manual KV preset resolves for '+$preset)
}
$statePath=Join-Path $fixture 'draft.json'
$draft=Get-DashboardInitialState -Saved @{launch_mode='server';server_auth=$true;model='unfinished';custom_sets='serve.api_key=PRIVATE-TO-DROP;paging.foo=1';future_option='kept'}
$draft|Add-Member -NotePropertyName server_api_key -NotePropertyValue 'PRIVATE-TO-DROP'
Save-DashboardDraft -State $draft
$savedText=[IO.File]::ReadAllText($statePath);$savedDraft=$savedText|ConvertFrom-Json
Check ($savedText -notmatch 'PRIVATE-TO-DROP') 'Saving draft removes keys including extra settings'
Check ($savedDraft.model -eq 'unfinished' -and $savedDraft.future_option -eq 'kept') 'Incomplete draft and future fields can be saved without launch'
$draft.model='edited';Save-DashboardDraft -State $draft
Check (([IO.File]::ReadAllText($statePath)|ConvertFrom-Json).model -eq 'edited') 'Saving over existing state is atomic'
$statePath=$one;$saveFailed=$false;try{Save-DashboardDraft -State $draft}catch{$saveFailed=$true}
Check ($saveFailed -and (Test-Path -LiteralPath $one -PathType Container)) 'Save failure preserves existing destination'
Check (@(Get-ChildItem -LiteralPath $fixture -Filter 'wizard-state.new-*').Count -eq 0) 'Save failure cleans only its own temporary file'
Write-Host "Dashboard renderer/state bridge: $checks checks passed. No engine, GPU, network or user-state access."
