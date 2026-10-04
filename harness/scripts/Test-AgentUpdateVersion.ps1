$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'AgentUpdateVersion.ps1')
$cases = @(
    @('agent-v0.1.0-preview.2', 'agent-v0.1.0-preview.1', $true),
    @('agent-v0.1.0', 'agent-v0.1.0-preview.2', $true),
    @('agent-v0.1.0-preview.3', 'agent-v0.1.0', $false),
    @('agent-v0.2.0', 'agent-v0.1.0', $true),
    @('agent-v0.1.0', 'agent-v0.2.0', $false),
    @('agent-v0.2.0-preview.1', 'agent-v0.1.0', $false),
    @('invalid', 'agent-v0.1.0', $false)
)
foreach ($case in $cases) {
    $actual = Test-AgentVersionNewer $case[0] $case[1]
    if ($actual -ne $case[2]) { throw "Version ordering failed: $($case[0]) vs $($case[1])" }
}
Write-Host "Agent version comparisons passed: $($cases.Count)"
