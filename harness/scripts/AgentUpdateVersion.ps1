function Get-AgentVersionParts([string]$Tag) {
    if ($Tag -notmatch '^agent-v([0-9]+)\.([0-9]+)\.([0-9]+)(?:-preview\.([0-9]+))?$') { return $null }
    $preview = if ($Matches.ContainsKey(4)) { [int]$Matches[4] } else { [int]::MaxValue }
    return @([version]::new([int]$Matches[1], [int]$Matches[2], [int]$Matches[3]), $preview)
}

function Test-AgentVersionNewer([string]$Candidate, [string]$Current) {
    $next = Get-AgentVersionParts $Candidate
    $old = Get-AgentVersionParts $Current
    if ($null -eq $next -or $null -eq $old) { return $false }
    if (-not $Current.Contains('-preview.') -and $Candidate.Contains('-preview.')) { return $false }
    if ($next[0] -gt $old[0]) { return $true }
    return $next[0] -eq $old[0] -and $next[1] -gt $old[1]
}
