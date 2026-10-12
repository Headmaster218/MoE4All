[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$wizardPath = Join-Path $PSScriptRoot 'infr-wizard.ps1'
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($wizardPath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
$functions = @('Get-SavedValue', 'Read-TextValue', 'Read-IntegerValue', 'Read-YesNo', 'Get-CpuMissTopology', 'Add-SetArgument')
foreach ($name in $functions) {
    $definition = $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true) | Select-Object -First 1
    if ($null -eq $definition) { throw "Missing wizard function: $name" }
    . ([scriptblock]::Create($definition.Extent.Text))
}

$actual = Get-CpuMissTopology
if ($actual.Physical -lt 1 -or $actual.Preferred -lt 1 -or $actual.Preferred -gt $actual.Physical) {
    throw 'Invalid live CPU topology counts'
}
Write-Host ("Live CPU topology: physical={0}, preferred={1}, detected={2}" -f $actual.Physical, $actual.Preferred, $actual.Detected)

function Get-CpuMissTopology { return $script:Topology }
function Read-Host {
    param([string]$Prompt)
    if ($script:Answers.Count -eq 0) { throw "Unexpected prompt: $Prompt" }
    return $script:Answers.Dequeue()
}

$statements = @($ast.EndBlock.Statements)
$start = $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and $node.Extent.Text.StartsWith('$cpuMissEnabled = Read-YesNo') }, $true) | Select-Object -First 1
if ($null -eq $start) { throw 'Missing CPU wizard prompts' }
$promptStatements = @($start.Parent.Statements)
$promptIndex = [array]::IndexOf($promptStatements, $start)
$prompts = [scriptblock]::Create($start.Extent.Text + "`n" + $promptStatements[$promptIndex + 1].Extent.Text)
$argumentStart = $statements | Where-Object { $_.Extent.Text.StartsWith('Add-SetArgument $nativeArgs ''kernels.vulkan.cpu_miss_threads''') } | Select-Object -First 1
$argumentIndex = [array]::IndexOf($statements, $argumentStart)
if ($argumentIndex -lt 0) { throw 'Missing CPU command arguments' }
$arguments = [scriptblock]::Create($argumentStart.Extent.Text + "`n" + $statements[$argumentIndex + 1].Extent.Text)
$customStatement = $statements | Where-Object { $_.Extent.Text.StartsWith('if ($setupMode -eq ''manual'' -and -not') -and $_.Extent.Text.Contains('$customSets') } | Select-Object -First 1
if ($null -eq $customStatement) { throw 'Missing extra settings block' }
$customArguments = [scriptblock]::Create($customStatement.Extent.Text)
$stateStatement = $statements | Where-Object { $_.Extent.Text.StartsWith('$state = [ordered]') } | Select-Object -First 1
$save = [scriptblock]::Create($stateStatement.Extent.Text)

function Test-Case {
    param([string]$Name, $Saved, [int]$Physical, [int]$Preferred, [string[]]$Inputs, [int]$ExpectedCores, [int]$ExpectedMax, [string]$ExtraSets = '')
    $script:Saved = $Saved
    $script:Topology = [pscustomobject]@{ Physical = $Physical; Preferred = $Preferred; Detected = $true }
    $script:Answers = [System.Collections.Generic.Queue[string]]::new()
    foreach ($inputValue in $Inputs) { $script:Answers.Enqueue($inputValue) }
    $cpuMissMax = [string](Get-SavedValue 'cpu_miss_max' '1')
    $cpuMissCores = [string](Get-SavedValue 'cpu_miss_cores' '')
    . $prompts
    if ($script:Answers.Count) { throw "$Name did not consume its input" }
    $nativeArgs = [System.Collections.Generic.List[string]]::new()
    [void]$nativeArgs.Add('run')
    . $arguments
    $setupMode = 'manual'
    $launchMode = 'chat'
    $customSets = $ExtraSets
    . $customArguments
    $sets = @($nativeArgs | Where-Object { $_ -ne '--set' -and $_ -ne 'run' })
    $paths = @($sets | ForEach-Object { ($_ -split '=', 2)[0] })
    if (@($paths | Sort-Object -Unique).Count -ne $paths.Count) { throw "$Name emitted duplicate setting paths" }
    if ("kernels.vulkan.cpu_miss_threads=$ExpectedCores" -notin $sets) { throw "$Name core argument mismatch: $sets" }
    if ($ExpectedCores -eq 0) {
        if ($sets.Count -ne 1) { throw "$Name should emit only explicit disable" }
    } else {
        foreach ($setting in @(
            "kernels.vulkan.cpu_miss_max=$ExpectedMax",
            'kernels.vulkan.cpu_miss_push=true',
            'kernels.vulkan.cpu_miss_host_result=true',
            'kernels.vulkan.cpu_miss_token_park=true',
            'kernels.vulkan.cpu_miss_spin=262144'
        )) { if ($setting -notin $sets) { throw "$Name missing $setting" } }
    }
    . $save
    if ($state.cpu_miss_enabled -ne ($ExpectedCores -gt 0)) { throw "$Name saved enable mismatch" }
    if ($ExpectedCores -gt 0 -and ([int]$state.cpu_miss_max -ne $ExpectedMax -or [int]$state.cpu_miss_cores -ne $ExpectedCores)) {
        throw "$Name saved limits mismatch"
    }
    Write-Host "PASS $Name"
}

Test-Case -Name 'disabled by default' -Physical 6 -Preferred 6 -Inputs @('') -ExpectedCores 0
Test-Case -Name 'homogeneous default' -Physical 6 -Preferred 6 -Inputs @('y', '', '') -ExpectedCores 4 -ExpectedMax 1
Test-Case -Name 'hybrid reserves two performance cores by default' -Physical 24 -Preferred 8 -Inputs @('y', '', '') -ExpectedCores 22 -ExpectedMax 1
Test-Case -Name 'hybrid all-core override' -Physical 24 -Preferred 8 -Inputs @('y', '', '24') -ExpectedCores 24 -ExpectedMax 1
Test-Case -Name 'custom three misses and seven cores' -Physical 12 -Preferred 8 -Inputs @('y', '3', '7') -ExpectedCores 7 -ExpectedMax 3
Test-Case -Name 'small CPU retains a usable minimum' -Physical 2 -Preferred 2 -Inputs @('y', '', '') -ExpectedCores 1 -ExpectedMax 1
Test-Case -Name 'saved settings' -Saved ([pscustomobject]@{ cpu_miss_enabled = $true; cpu_miss_max = '2'; cpu_miss_cores = '3' }) -Physical 8 -Preferred 8 -Inputs @('', '', '') -ExpectedCores 3 -ExpectedMax 2
Test-Case -Name 'invalid saved limits reset' -Saved ([pscustomobject]@{ cpu_miss_enabled = $true; cpu_miss_max = '9'; cpu_miss_cores = '99' }) -Physical 8 -Preferred 8 -Inputs @('', '', '') -ExpectedCores 6 -ExpectedMax 1
Test-Case -Name 'disable saved enable' -Saved ([pscustomobject]@{ cpu_miss_enabled = $true; cpu_miss_max = '3'; cpu_miss_cores = '4' }) -Physical 6 -Preferred 6 -Inputs @('n') -ExpectedCores 0
Test-Case -Name 'range checks reprompt' -Physical 6 -Preferred 6 -Inputs @('y', '0', '4', '2', '99', '0', '1') -ExpectedCores 1 -ExpectedMax 2
Test-Case -Name 'old extra enable cannot override disable' -Physical 6 -Preferred 6 -Inputs @('') -ExpectedCores 0 -ExtraSets 'kernels.vulkan.cpu_miss_threads=4'
Test-Case -Name 'new controls override old extra limits' -Physical 6 -Preferred 6 -Inputs @('y', '', '') -ExpectedCores 4 -ExpectedMax 1 -ExtraSets 'kernels.vulkan.cpu_miss_threads=6;kernels.vulkan.cpu_miss_max=3;kernels.vulkan.cpu_miss_push=false;kernels.vulkan.cpu_miss_spin=1'
Write-Host 'CPU-miss wizard tests passed; no engine or GPU was started.'
