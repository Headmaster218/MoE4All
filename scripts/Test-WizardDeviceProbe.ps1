[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$WorkDir, [string]$WizardPath = '')

$ErrorActionPreference = 'Stop'
if (-not $WizardPath) { $WizardPath = Join-Path $PSScriptRoot 'infr-wizard.ps1' }
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
if (-not $WorkDir.StartsWith('D:\AIinfr\infr\local-work\tmp\moe4all\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Use local-work/tmp/moe4all/<task>.' }
$ancestor = $WorkDir
while ($ancestor) {
    if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Fixture paths cannot traverse reparse points.' }
    $ancestor = [IO.Path]::GetDirectoryName($ancestor)
}
[void][IO.Directory]::CreateDirectory($WorkDir)
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($WizardPath, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
foreach ($fn in $ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -in @('Start-VulkanDeviceProbe','Get-VulkanDeviceOptions')}, $true)) { . ([scriptblock]::Create($fn.Extent.Text)) }
$fixtureDir = Join-Path $WorkDir ([guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($fixtureDir)
$source = Join-Path $fixtureDir 'probe.cs'; $infrPath = Join-Path $fixtureDir 'ProbeFixture.exe'
[IO.File]::WriteAllText($source, 'using System;using System.Threading;class Fixture{static void Main(){if(Environment.GetEnvironmentVariable("MOE4ALL_PROBE_FIXTURE_MODE")=="stall"){Thread.Sleep(12000);return;}Console.Error.Write(new string(''x'',262144));Console.Out.WriteLine("Vulkan0: Fixture GPU [discrete, 2.0 GiB device-local]  <- default");}}', [Text.UTF8Encoding]::new($false))
$oldMode = $env:MOE4ALL_PROBE_FIXTURE_MODE
$oldTemp = $env:TEMP; $oldTmp = $env:TMP
try {
    $env:TEMP=$fixtureDir; $env:TMP=$fixtureDir; $env:MOE4ALL_PROBE_FIXTURE_MODE=$null
    & (Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe') /nologo /target:exe ('/out:'+$infrPath) $source
    if ($LASTEXITCODE -ne 0) { throw 'Probe fixture compilation failed' }
    $script:VulkanDeviceProbe=$null; $script:VulkanDeviceProbeError=$null
    $watch=[Diagnostics.Stopwatch]::StartNew(); $devices=@(Get-VulkanDeviceOptions)
    if ($devices.Count -ne 1 -or $devices[0].Value -ne 'Vulkan0' -or -not $devices[0].IsDefault -or $watch.Elapsed.TotalSeconds -ge 6) { throw 'Probe did not drain full stderr concurrently' }
    $env:MOE4ALL_PROBE_FIXTURE_MODE='stall'; $script:VulkanDeviceProbe=$null; $script:VulkanDeviceProbeError=$null
    $watch.Restart(); $devices=@(Get-VulkanDeviceOptions)
    if ($devices.Count -ne 0 -or $watch.Elapsed.TotalSeconds -lt 5 -or $watch.Elapsed.TotalSeconds -ge 9) { throw 'Stalled probe did not return within its bounded wait' }
    Write-Output 'Device probe: stderr backpressure and stalled-child cases passed; own fixtures only, no GPU or network.'
} finally {
    $env:MOE4ALL_PROBE_FIXTURE_MODE=$oldMode; $env:TEMP=$oldTemp; $env:TMP=$oldTmp
}
