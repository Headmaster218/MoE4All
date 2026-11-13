[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$InitialModelPath = '',
    [switch]$DryRun,
    [switch]$SkipUpdateCheck,
    [switch]$PreviewUi,
    [switch]$ClassicWizard,
    [switch]$WindowWizard,
    [string]$DesktopExecutionPlanPath = '',
    [string]$PostUpdateProbePath = '',
    [string]$ExpectedPostUpdateVersion = '',
    [switch]$ExitAfterPostUpdateProbe
)

$ErrorActionPreference = 'Stop'
if ($ClassicWizard -and $WindowWizard) { throw 'Choose either the console entry or the window entry.' }

function Test-WizardOperationCancelled {
    param([Exception]$Exception)
    while ($null -ne $Exception) { if ($Exception.GetType().FullName -eq 'MoE4All.Launcher.NativePromptCanceledException') { return $true }; $Exception=$Exception.InnerException }
    return $false
}

function Assert-WizardOperationActive {
    if ($script:NativePromptsEnabled -and ('MoE4All.Launcher.NativePrompt' -as [type])) { [MoE4All.Launcher.NativePrompt]::CheckCancellation() }
}

function Wait-WizardOperation {
    param([int]$Milliseconds)
    if (-not $script:NativePromptsEnabled) { Start-Sleep -Milliseconds $Milliseconds; return }
    for ($remaining=$Milliseconds; $remaining -gt 0; $remaining-=100) { Assert-WizardOperationActive; Start-Sleep -Milliseconds ([Math]::Min(100,$remaining)) }
}

trap {
    $wizardException = $_.Exception
    while ($null -ne $wizardException) {
        if ($wizardException.GetType().FullName -eq 'MoE4All.Launcher.NativePromptCanceledException') {
            if ('MoE4All.Launcher.NativeWizard' -as [type]) { [MoE4All.Launcher.NativeWizard]::RestoreConsole() }
            exit 44
        }
        $wizardException = $wizardException.InnerException
    }
    if ($script:NativePromptsEnabled -and ('MoE4All.Launcher.NativeWizard' -as [type])) {
        if ('MoE4All.Launcher.NativePrompt' -as [type]) { [MoE4All.Launcher.NativePrompt]::ShowError($_.Exception.Message); [MoE4All.Launcher.NativeWizard]::RestoreConsole(); exit 46 }
        [MoE4All.Launcher.NativeWizard]::RestoreConsole()
    }
    throw
}

try {
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = $utf8
    [Console]::OutputEncoding = $utf8
    $OutputEncoding = $utf8
} catch {
    # Some redirected hosts do not expose a mutable console encoding.
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$updateCommonPath = Join-Path $PSScriptRoot 'moe4all-update-common.ps1'
if (-not (Test-Path -LiteralPath $updateCommonPath -PathType Leaf)) {
    throw "Missing update support file: $updateCommonPath"
}
. $updateCommonPath
$releaseInfrPath = Join-Path $repoRoot 'infr.exe'
$sourceInfrPath = Join-Path $repoRoot 'target\release\infr.exe'
$script:IsPackagedInstall = Test-Path -LiteralPath $releaseInfrPath -PathType Leaf
$infrPath = if ($script:IsPackagedInstall) {
    $releaseInfrPath
} else {
    $sourceInfrPath
}
$dataDir = Join-Path $repoRoot 'gui-data'
$statePath = Join-Path $dataDir 'wizard-state.json'
$guiStatePath = Join-Path $dataDir 'state.json'
$pendingSubmissionDir = Join-Path $dataDir 'private\pending-submissions'
$script:Saved = $null

if (-not $DesktopExecutionPlanPath -and -not $PreviewUi -and (Test-Path -LiteralPath $statePath -PathType Leaf)) {
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



function Test-WizardConsoleInput {
    if ($script:NativePromptsEnabled -or $env:MOE4ALL_DESKTOP_IO -eq '1') { return $false }
    # ReadKey is an enhancement for a real console, never for scripts or redirected input.
    try {
        return ($Host.Name -eq 'ConsoleHost' -and -not [Console]::IsInputRedirected -and
            -not [Console]::IsOutputRedirected -and [Environment]::UserInteractive -and -not $env:CI)
    } catch { return $false }
}



























function Get-BenchmarkGuideNow {
    return [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
}

function Test-BenchmarkGuideInteractive {
    if ($DryRun -or [Console]::IsInputRedirected -or [Console]::IsOutputRedirected -or $env:CI) { return $false }
    if ([Environment]::GetCommandLineArgs() -match '^-NonI(nteractive)?$') { return $false }
    return [Environment]::UserInteractive
}

function Get-BenchmarkGuideState {
    $result = [pscustomobject]@{ submitted = $false; next_remind_at = [long]0; last_prompt_version = ''; last_submitted_version = ''; reminder_reason = 'first_use' }
    $path = Join-Path $dataDir 'private\benchmark-first-submission.json'
    try {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $value = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($value.submitted -is [bool] -and $value.submitted) { $result.submitted = $true }
            $next = [long]0
            if ([long]::TryParse([string]$value.next_remind_at, [ref]$next) -and $next -ge 0) { $result.next_remind_at = $next }
            foreach ($key in @('last_prompt_version','last_submitted_version')) {
                $property=$value.PSObject.Properties[$key]
                if ($property -and [string]$property.Value -match '^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$') { $result.$key=[string]$property.Value }
            }
            if ($value.PSObject.Properties['reminder_reason'] -and $value.reminder_reason -eq 'new_version') { $result.reminder_reason='new_version' }
        }
    } catch { } # A damaged reminder must not stop normal startup.
    return $result
}

function Save-BenchmarkGuideState {
    param([bool]$Submitted = $false, [long]$NextRemindAt = 0, [string]$PromptVersion = '', [string]$SubmittedVersion = '', [ValidateSet('','first_use','new_version')][string]$ReminderReason = '')
    if ($DryRun) { return }
    $temporary = $null
    try {
        $previous = Get-BenchmarkGuideState
        if($previous.last_submitted_version -eq (Get-BenchmarkGuideVersion) -and $SubmittedVersion -ne $previous.last_submitted_version){$SubmittedVersion=$previous.last_submitted_version}
        $value = [ordered]@{
            schema_version = 2
            submitted = ($Submitted -or $previous.submitted)
            next_remind_at = $(if ($PSBoundParameters.ContainsKey('NextRemindAt')) { [Math]::Max(0,$NextRemindAt) } else { $previous.next_remind_at })
            last_prompt_version = $(if ($PromptVersion) { $PromptVersion } else { $previous.last_prompt_version })
            last_submitted_version = $(if ($SubmittedVersion) { $SubmittedVersion } else { $previous.last_submitted_version })
            reminder_reason = $(if ($ReminderReason) { $ReminderReason } else { $previous.reminder_reason })
        }
        $directory = Join-Path $dataDir 'private'
        [void][IO.Directory]::CreateDirectory($directory)
        $path = Join-Path $directory 'benchmark-first-submission.json'
        $temporary = Join-Path $directory ('.benchmark-guide-' + [guid]::NewGuid().ToString('N'))
        [IO.File]::WriteAllText($temporary, (($value | ConvertTo-Json) + "`n"), [Text.UTF8Encoding]::new($false))
        if ([IO.File]::Exists($path)) { [IO.File]::Replace($temporary, $path, [System.Management.Automation.Language.NullString]::Value) }
        else { [IO.File]::Move($temporary, $path) }
    } catch { Write-Warning 'Could not save the benchmark reminder; normal startup will continue.' }
    finally { if ($temporary -and [IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) } }
}

function Set-BenchmarkSubmissionSucceeded {
    param([string]$Kind, [string]$ReportVersion = '')
    if ($Kind -ne 'benchmark' -or $DryRun) { return }
    if (-not $PSBoundParameters.ContainsKey('ReportVersion')) { $ReportVersion = Get-BenchmarkGuideVersion }
    if ($ReportVersion -notmatch '^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$') { $ReportVersion='' }
    Save-BenchmarkGuideState -Submitted $true -SubmittedVersion $ReportVersion
}

function Get-BenchmarkGuideVersion {
    $version = [string](Get-Variable -Name productVersion -Scope Script -ValueOnly -ErrorAction SilentlyContinue)
    if ($version -match '^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$') { return $version }
    return ''
}

function Test-BenchmarkBodyReceipt {
    param($Receipt)
    $received = [DateTimeOffset]::MinValue
    return ($Receipt.api_version -eq 1 -and [string]$Receipt.kind -eq 'benchmark' -and
        [string]$Receipt.receipt_id -cmatch '^[A-Za-z0-9_-]+$' -and
        -not [string]::IsNullOrWhiteSpace([string]$Receipt.report_id) -and
        [string]$Receipt.file_sha256 -cmatch '^[0-9a-f]{64}$' -and
        [string]$Receipt.payload_sha256 -cmatch '^[0-9a-f]{64}$' -and
        [string]$Receipt.status -in @('forwarding', 'pending_review', 'published', 'rejected', 'delivery_failed') -and
        [DateTimeOffset]::TryParse([string]$Receipt.received_at, [ref]$received))
}

function Invoke-BenchmarkStartupGuide {
    param([bool]$Interactive = $false, [bool]$JsonAutomation = $false, [switch]$Window)
    if (-not $Interactive -or $DryRun -or $JsonAutomation) { return $false }
    $version=Get-BenchmarkGuideVersion
    if (-not $version) { return $false }
    $guide = Get-BenchmarkGuideState
    if ($guide.last_submitted_version -eq $version) { return $false }
    # Only stored body-received metadata is historical proof; a saved browser choice is not.
    $receipts = Join-Path $dataDir 'private\submission-receipts'
    foreach ($file in @($(if (-not $guide.submitted) { Get-ChildItem -LiteralPath $receipts -Filter '*.private-receipt.json' -File -ErrorAction SilentlyContinue }))) {
        try {
            $receipt = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
            if (Test-BenchmarkBodyReceipt $receipt) {
                $receiptVersion=if($receipt.PSObject.Properties['engine_version']){[string]$receipt.engine_version}else{''}
                Set-BenchmarkSubmissionSucceeded benchmark -ReportVersion $receiptVersion
                $guide=Get-BenchmarkGuideState
                if ($guide.last_submitted_version -eq $version) { return $false }
                break
            }
        } catch { } # Never print private receipt contents or parser errors.
    }
    $now = Get-BenchmarkGuideNow
    $previousVersion=if($guide.last_prompt_version){$guide.last_prompt_version}else{$guide.last_submitted_version}
    $changed=($previousVersion -and $previousVersion -ne $version) -or ($guide.submitted -and -not $previousVersion)
    if (-not $changed -and $now -lt $guide.next_remind_at) { return $false }
    $newVersion=$changed -or ($guide.last_prompt_version -eq $version -and $guide.reminder_reason -eq 'new_version')
    $reason=if($newVersion){'new_version'}else{'first_use'}
    $delay=if($newVersion){604800}else{172800}
    # Persist before asking so EOF, cancellation and failed benchmarks also get a cooldown.
    Save-BenchmarkGuideState -NextRemindAt ($now + $delay) -PromptVersion $version -ReminderReason $reason
    if ($Window) {
        $language=[string](Get-SavedValue 'wizard_language' '')
        $english=$language -eq 'en' -or (-not $language -and [Globalization.CultureInfo]::CurrentUICulture.TwoLetterISOLanguageName -ne 'zh')
        return ([MoE4All.Launcher.NativeStartupPrompts]::Benchmark($version,[bool]$newVersion,[bool]$english) -eq 'try')
    }
    if ($newVersion) {
        Write-Host "`n已更新到 v$version，要不要再测一次 Benchmark，看看这次的表现？" -ForegroundColor Cyan
        Write-Host "Updated to v$version. Run another benchmark and see how this version performs?"
    } else {
        Write-Host "`n欢迎使用 MoE4All！要不要先跑一次快速 Benchmark，看看这台设备的表现？" -ForegroundColor Cyan
        Write-Host 'Welcome to MoE4All! Try a quick benchmark and see how your device performs.'
    }
    Write-Host '上传前仍会征求同意。You will be asked before uploading.'
    $answer = Read-Choice -Label '来试试吗？/ Give it a try?' -DefaultValue 'later' -Options @(
        [pscustomobject]@{ Key = '1'; Value = 'try'; Label = '试试！ / Try it!' }
        [pscustomobject]@{ Key = '2'; Value = 'later'; Label = '后面再说~ / Maybe later~' }
    )
    return ($answer -eq 'try')
}

function Test-PagedKvEligible {
    param(
        [Parameter(Mandatory = $true)][string]$Preset,
        [AllowEmptyString()][string]$TypeK = '',
        [AllowEmptyString()][string]$TypeV = '',
        [bool]$Overflow = $false
    )
    # The wizard's mirror of the engine's `segmented_kv_wanted`: every paged-KV tier IS the dynamic
    # segmented layout, and that layout only exists over a Q8_0/Q8_0 cache with no RAM overflow.
    # `auto` counts as eligible because the Vulkan runner resolves it to Q8 for these models.
    if ($Overflow) {
        return $false
    }
    if ($Preset -in @('auto', 'q8')) {
        return $true
    }
    if ($Preset -eq 'custom') {
        return ($TypeK -eq 'q8_0' -and $TypeV -eq 'q8_0')
    }
    return $false
}

function Read-TextValue {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [AllowEmptyString()][string]$Default = '',
        [switch]$Required
    )
    if ($DesktopExecutionPlanPath -and $env:MOE4ALL_DESKTOP_IO -eq '1') {
        do { $value = Read-WizardRuntimeResponse -Prompt $Label -InputType 'text' -DefaultValue $Default } while ($Required -and [string]::IsNullOrWhiteSpace($value))
        return $value.Trim()
    }
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
        [long]$Maximum = [long]::MaxValue,
        [switch]$AllowBlank
    )
    while ($true) {
        $value = Read-TextValue -Label $Label -Default $Default
        if ($AllowBlank -and [string]::IsNullOrWhiteSpace($value)) {
            return ''
        }
        $parsed = 0L
        if ([long]::TryParse($value, [ref]$parsed) -and $parsed -ge $Minimum -and $parsed -le $Maximum) {
            return $parsed.ToString()
        }
        if ($Maximum -eq [long]::MaxValue) {
            Write-Host "请输入不小于 $Minimum 的整数。Enter an integer greater than or equal to $Minimum." -ForegroundColor Yellow
        } else {
            Write-Host "请输入 $Minimum 到 $Maximum 之间的整数。Enter an integer between $Minimum and $Maximum." -ForegroundColor Yellow
        }
    }
}

function Get-CpuMissTopology {
    try {
        if (-not ('Moe4All.WizardCpuTopology' -as [type])) {
            Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
namespace Moe4All {
    public static class WizardCpuTopology {
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool GetLogicalProcessorInformationEx(int relation, IntPtr buffer, ref uint length);
        public static int[] Counts() {
            if (IntPtr.Size != 8) throw new NotSupportedException("64-bit topology query required");
            uint length = 0;
            GetLogicalProcessorInformationEx(0, IntPtr.Zero, ref length);
            if (length == 0) throw new InvalidOperationException("CPU topology unavailable");
            IntPtr buffer = Marshal.AllocHGlobal(checked((int)length));
            try {
                uint capacity = length;
                if (!GetLogicalProcessorInformationEx(0, buffer, ref length) || length > capacity)
                    throw new InvalidOperationException("CPU topology query failed");
                var classes = new List<byte>();
                for (int offset = 0; offset < length;) {
                    if ((long)offset + 8 > length) throw new InvalidOperationException("Truncated CPU topology");
                    int size = Marshal.ReadInt32(buffer, offset + 4);
                    if (size < 8 || (long)offset + size > length) throw new InvalidOperationException("Invalid CPU topology");
                    if (Marshal.ReadInt32(buffer, offset) == 0) {
                        if (size < 48 || Marshal.ReadInt16(buffer, offset + 30) != 1 || Marshal.ReadInt64(buffer, offset + 32) == 0)
                            throw new InvalidOperationException("Invalid physical core");
                        classes.Add(Marshal.ReadByte(buffer, offset + 9));
                    }
                    offset += size;
                }
                if (classes.Count == 0) throw new InvalidOperationException("No physical cores");
                byte fastest = 0;
                foreach (byte value in classes) fastest = Math.Max(fastest, value);
                int preferred = 0;
                foreach (byte value in classes) if (value == fastest) preferred++;
                return new int[] { classes.Count, preferred };
            } finally { Marshal.FreeHGlobal(buffer); }
        }
    }
}
'@
        }
        $counts = [Moe4All.WizardCpuTopology]::Counts()
        return [pscustomobject]@{ Physical = $counts[0]; Preferred = $counts[1]; Detected = $true }
    } catch {
        $physical = 1
        try {
            $physical = [Math]::Max(1, [int]((Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum))
        } catch { }
        return [pscustomobject]@{ Physical = $physical; Preferred = $physical; Detected = $false }
    }
}

function Read-YesNo {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [bool]$Default = $true
    )
    if ($DesktopExecutionPlanPath -and $env:MOE4ALL_DESKTOP_IO -eq '1') {
        $value = Read-WizardRuntimeResponse -Prompt $Label -InputType 'confirm' -DefaultValue $(if ($Default) { 'y' } else { 'n' })
        return $value -eq 'y'
    }
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
    if ($DesktopExecutionPlanPath -and $env:MOE4ALL_DESKTOP_IO -eq '1' -and $launchMode -eq 'benchmark') {
        $value = Read-WizardRuntimeResponse -Prompt $Label -InputType 'choice' -DefaultValue $DefaultValue -Options $Options
        $selected = $Options | Where-Object { $_.Key -eq $value -or $_.Value -eq $value } | Select-Object -First 1
        if ($null -eq $selected) { throw 'Unknown window selection.' }
        return [string]$selected.Value
    }
    Write-Host "`n$Label" -ForegroundColor Cyan
    foreach ($option in $Options) {
        $marker = if ($option.Value -eq $DefaultValue) { '*' } else { ' ' }
        Write-Host (" {0} [{1}] {2}" -f $marker, $option.Key, $option.Label)
        if ($option.PSObject.Properties.Name -contains 'Detail' -and $option.Detail) {
            foreach ($detailLine in @($option.Detail)) {
                Write-Host ("       {0}" -f $detailLine) -ForegroundColor DarkGray
            }
        }
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

function Start-VulkanDeviceProbe {
    if ($null -ne $script:VulkanDeviceProbe -or $script:VulkanDeviceProbeError) { return }
    $process = $null
    try {
        $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = $infrPath
        $startInfo.Arguments = 'devices'
        $startInfo.WorkingDirectory = [System.IO.Path]::GetDirectoryName($infrPath)
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $startInfo
        if (-not $process.Start()) {
            throw 'process did not start'
        }
        $script:VulkanDeviceProbe = $process
    } catch {
        if ($null -ne $process) { $process.Dispose() }
        $script:VulkanDeviceProbeError = $_.Exception.Message
    }
}

function Get-VulkanDeviceOptions {
    if ($null -eq $script:VulkanDeviceProbe -and -not $script:VulkanDeviceProbeError) {
        Start-VulkanDeviceProbe
    }
    if ($script:VulkanDeviceProbeError) {
        Write-Warning "无法枚举 Vulkan 设备，请检查显卡驱动：$($script:VulkanDeviceProbeError) / Could not enumerate Vulkan devices; check the GPU driver."
        return @()
    }

    $process = $script:VulkanDeviceProbe
    $script:VulkanDeviceProbe = $null
    try {
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(6000)) {
            try { $process.Kill() } catch { }
            throw '设备探测超过 6 秒；可手动选择设备。Device probe timed out; choose a device manually.'
        }
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        $exitCode = $process.ExitCode
    } catch {
        Write-Warning "无法枚举 Vulkan 设备，请检查显卡驱动：$($_.Exception.Message) / Could not enumerate Vulkan devices; check the GPU driver."
        return @()
    } finally {
        if ($null -ne $process) {
            $process.Dispose()
        }
    }
    if ($exitCode -ne 0) {
        $detail = $stderr.Trim()
        if ($detail) { $detail = ": $detail" }
        Write-Warning "Vulkan 设备枚举失败（退出码 $exitCode）$detail。请检查显卡驱动。Vulkan device enumeration failed (exit code $exitCode); check the GPU driver."
        return @()
    }

    $options = [System.Collections.Generic.List[object]]::new()
    $lines = @($stdout -split '\r?\n')
    foreach ($line in $lines) {
        $text = [string]$line
        if ($text -notmatch '^\s*(Vulkan\d+):\s+(.+?)\s+\[([^\]]+)\](\s+<-\s+default)?\s*$') {
            continue
        }
        $value = $Matches[1]
        $name = $Matches[2]
        $details = $Matches[3]
        $isDefault = -not [string]::IsNullOrWhiteSpace($Matches[4])
        [void]$options.Add([pscustomobject]@{
            Key = ($options.Count + 1).ToString()
            Value = $value
            Label = ('{0}: {1} [{2}]' -f $value, $name, $details)
            IsDefault = $isDefault
        })
    }
    return $options.ToArray()
}

function Read-ComputeDevice {
    param([AllowEmptyString()][string]$Default = '')

    $options = @(Get-VulkanDeviceOptions)
    if ($options.Count -eq 0) {
        throw '未发现可用的 Vulkan GPU，请检查显卡驱动。No Vulkan GPU was found; check the GPU driver.'
    }
    $selected = $options | Where-Object { $_.Value -eq $Default } | Select-Object -First 1
    if ($null -eq $selected) {
        $selected = $options | Where-Object { $_.IsDefault } | Select-Object -First 1
    }
    if ($null -eq $selected) { $selected = $options[0] }
    return Read-Choice -Label '运行设备 / Compute device' -DefaultValue ([string]$selected.Value) -Options $options
}

function Read-ModelReasoningEfforts {
    param([Parameter(Mandatory = $true)][string]$ModelPath)

    try {
        if (-not ('Moe4All.WizardModelMetadata' -as [type])) {
            Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Text;
namespace Moe4All {
    public static class WizardModelMetadata {
        private static string Text(BinaryReader reader) {
            ulong length = reader.ReadUInt64();
            if (length > 16 * 1024 * 1024 || length > (ulong)(reader.BaseStream.Length - reader.BaseStream.Position))
                throw new InvalidDataException("Invalid GGUF string");
            return Encoding.UTF8.GetString(reader.ReadBytes((int)length));
        }
        private static void Skip(BinaryReader reader, uint type, int depth) {
            if (depth > 8) throw new InvalidDataException("Nested GGUF array");
            int size;
            switch (type) {
                case 0: case 1: case 7: size = 1; break;
                case 2: case 3: size = 2; break;
                case 4: case 5: case 6: size = 4; break;
                case 10: case 11: case 12: size = 8; break;
                case 8:
                    ulong length = reader.ReadUInt64();
                    if (length > (ulong)(reader.BaseStream.Length - reader.BaseStream.Position))
                        throw new InvalidDataException("Invalid GGUF string");
                    reader.BaseStream.Seek((long)length, SeekOrigin.Current);
                    return;
                case 9:
                    uint element = reader.ReadUInt32();
                    ulong count = reader.ReadUInt64();
                    if (count > 1000000) throw new InvalidDataException("Oversized GGUF array");
                    for (ulong i = 0; i < count; i++) Skip(reader, element, depth + 1);
                    return;
                default: throw new InvalidDataException("Unknown GGUF type");
            }
            if (size > reader.BaseStream.Length - reader.BaseStream.Position)
                throw new EndOfStreamException();
            reader.BaseStream.Seek(size, SeekOrigin.Current);
        }
        public static string[] Read(string path) {
            using (var stream = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var reader = new BinaryReader(stream)) {
                if (reader.ReadUInt32() != 0x46554747) throw new InvalidDataException("Not GGUF");
                uint version = reader.ReadUInt32();
                if (version < 2 || version > 3) throw new InvalidDataException("Unknown GGUF version");
                reader.ReadUInt64();
                ulong count = reader.ReadUInt64();
                if (count > 1000000) throw new InvalidDataException("Oversized GGUF metadata");
                string arch = "", template = "";
                for (ulong i = 0; i < count; i++) {
                    string key = Text(reader);
                    uint type = reader.ReadUInt32();
                    if (type == 8 && (key == "general.architecture" || key == "tokenizer.chat_template")) {
                        string value = Text(reader);
                        if (key == "general.architecture") arch = value; else template = value;
                    } else Skip(reader, type, 0);
                }
                return new string[] { arch, template };
            }
        }
    }
}
'@
        }
        $metadata = [Moe4All.WizardModelMetadata]::Read($ModelPath)
        if ($metadata[0] -in @('qwen35', 'qwen35moe') -and $metadata[1] -match '\benable_thinking\b') {
            return @('low', 'medium', 'high', 'max')
        }
        if ($metadata[1] -notmatch '\breasoning_effort\b') { return @() }
        if ($metadata[0] -eq 'qwen4exp') { return @('low', 'medium', 'xhigh') }
        return @('low', 'medium', 'high', 'max')
    } catch {
        Write-Warning '无法读取模型的思考强度能力，保留模板默认。Could not read reasoning-effort support; keeping the template default.'
        return @()
    }
}

function Get-ModelProbeIdentity {
    param([Parameter(Mandatory = $true)][string]$ModelPath)
    $item = Get-Item -LiteralPath $ModelPath -ErrorAction Stop
    return [pscustomobject]@{
        Path = $item.FullName
        Length = [int64]$item.Length
        LastWriteTicks = [int64]$item.LastWriteTimeUtc.Ticks
    }
}

function Set-ModelReasoningProbeResult {
    param($Identity, [string[]]$Efforts)
    $script:ReasoningProbeModel = [string]$Identity.Path
    $script:ReasoningProbeLength = [int64]$Identity.Length
    $script:ReasoningProbeMtimeTicks = [int64]$Identity.LastWriteTicks
    $script:ReasoningProbeEfforts = @($Efforts)
}

function Get-CachedModelReasoningEfforts {
    param($Identity)
    $cachedPath = [string](Get-SavedValue 'reasoning_probe_model' '')
    $cachedLength = [int64](Get-SavedValue 'reasoning_probe_length' -1)
    $cachedTicks = [int64](Get-SavedValue 'reasoning_probe_mtime_ticks' -1)
    if ([string]::Equals($cachedPath, [string]$Identity.Path, [System.StringComparison]::OrdinalIgnoreCase) -and
        $cachedLength -eq [int64]$Identity.Length -and $cachedTicks -eq [int64]$Identity.LastWriteTicks) {
        return [pscustomobject]@{ Found = $true; Efforts = @((Get-SavedValue 'reasoning_probe_efforts' @())) }
    }
    return [pscustomobject]@{ Found = $false; Efforts = @() }
}

function Start-ModelReasoningProbe {
    param([Parameter(Mandatory = $true)][string]$ModelPath)
    $worker = $null
    try {
        $identity = Get-ModelProbeIdentity -ModelPath $ModelPath
        $cached = Get-CachedModelReasoningEfforts -Identity $identity
        if ($cached.Found) {
            Set-ModelReasoningProbeResult -Identity $identity -Efforts @($cached.Efforts)
            return
        }
        if ($null -ne $script:ModelReasoningProbe) { return }

        $worker = [PowerShell]::Create()
        [void]$worker.AddScript(${function:Read-ModelReasoningEfforts}.ToString()).AddArgument($identity.Path)
        $script:ModelReasoningProbe = [pscustomobject]@{
            Identity = $identity
            Worker = $worker
            Async = $worker.BeginInvoke()
        }
    } catch {
        if ($null -ne $worker) { $worker.Dispose() }
        $script:ModelReasoningProbe = $null
    }
}

function Get-ModelReasoningEfforts {
    param([Parameter(Mandatory = $true)][string]$ModelPath)

    $identity = Get-ModelProbeIdentity -ModelPath $ModelPath
    if ([string]::Equals($script:ReasoningProbeModel, $identity.Path, [System.StringComparison]::OrdinalIgnoreCase) -and
        $script:ReasoningProbeLength -eq $identity.Length -and $script:ReasoningProbeMtimeTicks -eq $identity.LastWriteTicks) {
        return @($script:ReasoningProbeEfforts)
    }

    $cached = Get-CachedModelReasoningEfforts -Identity $identity
    if ($cached.Found) {
        Set-ModelReasoningProbeResult -Identity $identity -Efforts @($cached.Efforts)
        return @($cached.Efforts)
    }

    $efforts = @()
    $pending = $script:ModelReasoningProbe
    if ($null -ne $pending -and
        [string]::Equals($pending.Identity.Path, $identity.Path, [System.StringComparison]::OrdinalIgnoreCase) -and
        $pending.Identity.Length -eq $identity.Length -and $pending.Identity.LastWriteTicks -eq $identity.LastWriteTicks) {
        try {
            $efforts = @($pending.Worker.EndInvoke($pending.Async) | ForEach-Object { [string]$_ })
            foreach ($warning in $pending.Worker.Streams.Warning) {
                Write-Warning ([string]$warning.Message)
            }
        } finally {
            $pending.Worker.Dispose()
            $script:ModelReasoningProbe = $null
        }
    } else {
        $efforts = @(Read-ModelReasoningEfforts -ModelPath $identity.Path)
    }
    Set-ModelReasoningProbeResult -Identity $identity -Efforts $efforts
    return $efforts
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

function Get-RecommendedModelCatalog {
    $notesVersion = [string](Get-Variable -Name productVersion -Scope Script -ValueOnly -ErrorAction SilentlyContinue)
    $notesUrl = 'https://github.com/Headmaster218/MoE4All/releases/latest'
    if ($notesVersion -match '^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$') { $notesUrl = 'https://github.com/Headmaster218/MoE4All/releases/tag/release-' + $notesVersion }
    $flashDirectory = 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64'
    # Same weights, upstream re-split into three files; the old 33 shards now return 403.
    $flashRevision = 'f5b0b790eebb89033c48091c26abf74c1b385940'
    $flashSizes = @(44866868384,11258338528,38400184512)
    $flashFiles = @()
    for ($i = 1; $i -le 3; $i++) {
        $fileName = 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-{0:D5}-of-00003.gguf' -f $i
        $flashFiles += [pscustomobject]@{
            Url = "https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/$flashRevision/$fileName`?download=true"
            RelativePath = Join-Path $flashDirectory $fileName
            Bytes = [long]$flashSizes[$i - 1]
        }
    }

    return @(
        [pscustomobject]@{
            Id = 'qwen35-balanced'
            Kind = 'llm'
            Name = 'Qwen3.6-35B-A3B-APEX-I-Balanced'
            InfoUrl = 'https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/blob/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf'
            Primary = 'Qwen3.6-35B-A3B-APEX-I-Balanced\Qwen3.6-35B-A3B-APEX-I-Balanced.gguf'
            Files = @([pscustomobject]@{
                Url = 'https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true'
                RelativePath = 'Qwen3.6-35B-A3B-APEX-I-Balanced\Qwen3.6-35B-A3B-APEX-I-Balanced.gguf'
                Bytes = 25624957632
            })
        }
        [pscustomobject]@{
            Id = 'flashnext-427'
            Kind = 'llm'
            Name = 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64'
            InfoUrl = "https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/$flashRevision"
            Primary = "$flashDirectory\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00003.gguf"
            Files = $flashFiles
        }
        [pscustomobject]@{
            Id = 'flashnext-vision'
            Kind = 'vision'
            Name = 'mmproj-Qwen3.8-Flash-Next-F16'
            InfoUrl = 'https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/blob/main/mmproj-Qwen3.8-Flash-Next-F16.gguf'
            Primary = "$flashDirectory\mmproj-Qwen3.8-Flash-Next-F16.gguf"
            Files = @([pscustomobject]@{
                Url = 'https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true'
                RelativePath = "$flashDirectory\mmproj-Qwen3.8-Flash-Next-F16.gguf"
                Bytes = 904003840
            })
        }
        [pscustomobject]@{
            Id = 'qwen3-embedding-06b-q8'
            Kind = 'embedding'
            Name = 'Qwen3-Embedding-0.6B-Q8_0'
            InfoUrl = 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/blob/main/Qwen3-Embedding-0.6B-Q8_0.gguf'
            Primary = 'Qwen3-Embedding-0.6B-Q8_0\Qwen3-Embedding-0.6B-Q8_0.gguf'
            Files = @([pscustomobject]@{
                Url = 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true'
                RelativePath = 'Qwen3-Embedding-0.6B-Q8_0\Qwen3-Embedding-0.6B-Q8_0.gguf'
                Bytes = 639150592
            })
        }
        [pscustomobject]@{
            Id = 'sensevoice-small-q8'
            Kind = 'audio'
            Name = 'SenseVoice-Small-Q8_0'
            InfoUrl = 'https://huggingface.co/lovemefan/sense-voice-gguf/blob/main/sense-voice-small-q8_0.gguf'
            Primary = 'SenseVoice-Small-Q8_0\sense-voice-small-q8_0.gguf'
            Files = @([pscustomobject]@{
                Url = 'https://huggingface.co/lovemefan/sense-voice-gguf/resolve/main/sense-voice-small-q8_0.gguf?download=true'
                RelativePath = 'SenseVoice-Small-Q8_0\sense-voice-small-q8_0.gguf'
                Bytes = 291985952
            })
        }
        [pscustomobject]@{
            Id = 'qwen35-vision'
            Kind = 'vision'
            Name = 'Qwen3.6-35B-A3B · mmproj.gguf'
            Description = '35B 配套视觉投影文件 / Matching Qwen3.6-35B vision projector'
            InfoUrl = 'https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/blob/main/mmproj.gguf'
            Primary = 'Qwen3.6-35B-A3B-APEX-I-Balanced\mmproj.gguf'
            Files = @([pscustomobject]@{
                Url = 'https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/mmproj.gguf?download=true'
                RelativePath = 'Qwen3.6-35B-A3B-APEX-I-Balanced\mmproj.gguf'
                Bytes = 902822624
            })
        }
        [pscustomobject]@{
            Id = 'melotts-chinese-native-f32'
            Kind = 'speech'
            Name = 'MeloTTS-Chinese-Native-F32'
            Downloadable = $false
            UnavailableReason = '下载与转换步骤见本版本发行说明；提供配套转换脚本，不自动安装依赖。See this version''s release notes for setup and the conversion script; dependencies are not installed automatically.'
            InstructionsUrl = $notesUrl
            LocalPackageReady = $true
            PackageFileName = 'MeloTTS-Chinese-Native-F32-v1.zip'
            PackageBytes = 155780558
            PackageScript = 'scripts/package-native-speech.ps1'
            ModelSha256 = '6d0d6fffe43cf6dea169f584c07bbb1a43756f37958ba9f9b78e27e8f9b32cc2'
            ModelBytes = 164882176
            SourceCommit = '66a4434a518a3b6d124756952977f0b10f0e9a76'
            InfoUrl = $notesUrl
            # These are conversion inputs, deliberately not downloadable runtime files.
            SourceUrls = @(
                'https://huggingface.co/myshell-ai/MeloTTS-Chinese/tree/af5d207a364ea4208c6f589c89f57f88414bdd16'
                'https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/tree/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6'
            )
            ConversionScript = 'scripts/convert_melotts_chinese.py'
            RequiredAssets = @('lexicon.txt', 'tokens.txt', 'config.json')
            RuntimeAssets = @('melotts-chinese-f32.gguf', 'lexicon.txt', 'tokens.txt', 'config.json', 'frontend.json', 'custom-lexicon.txt')
            Primary = ''
            Files = @()
        }
    )
}

function Show-RecommendedModelDownloads {
    param([ValidateSet('all', 'llm', 'vision', 'embedding', 'audio', 'speech')][string]$Kind = 'all')

    Write-Host "`n官方推荐的模型 / Official recommended models" -ForegroundColor Cyan
    $matching = @(Get-RecommendedModelCatalog | Where-Object { $Kind -eq 'all' -or $_.Kind -eq $Kind })
    foreach ($download in $matching) {
        Write-Host " $($download.Name)"
        $downloadBytes = if ($download.PSObject.Properties['PackageBytes']) { [long]$download.PackageBytes } else { [long](($download.Files | Measure-Object -Property Bytes -Sum).Sum) }
        if ($downloadBytes -gt 0) { Write-Host "   $(Format-DownloadBytes $downloadBytes)" -ForegroundColor DarkGray }
        Write-Host "   $($download.InfoUrl)" -ForegroundColor DarkGray
        if ($download.PSObject.Properties['Downloadable'] -and -not $download.Downloadable) {
            Write-Host "   $($download.UnavailableReason)" -ForegroundColor Yellow
        }
    }
    if (@($matching | Where-Object { -not $_.PSObject.Properties['Downloadable'] -or $_.Downloadable }).Count -eq 0) { return }
    Write-Host '在模型列表选择 D，可选择下载项目和保存目录；默认是安装目录下的 models。' -ForegroundColor DarkGray
    Write-Host 'Choose D to select downloads and a destination; the default is models beside the launcher.' -ForegroundColor DarkGray
}

function Format-DownloadBytes {
    param([long]$Bytes)
    if ($Bytes -ge 1GB) { return ('{0:N2} GiB' -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ('{0:N1} MiB' -f ($Bytes / 1MB)) }
    if ($Bytes -ge 1KB) { return ('{0:N1} KiB' -f ($Bytes / 1KB)) }
    return "$Bytes B"
}

function Invoke-FileDownloadWithProgress {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$Label,
        [int]$MaximumAttempts = 3,
        [long]$ExpectedBytes = 0,
        [int]$ProgressId = 71,
        [int]$ParentProgressId = -1,
        [string]$UserAgent = 'MoE4All-Wizard/1.0',
        [int]$AttemptTimeoutMs = 30000,
        [switch]$ReplaceExisting
    )

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $existingFile = Get-Item -LiteralPath $Destination
        if (-not $ReplaceExisting -and $existingFile.Length -gt 0) {
            if ($ExpectedBytes -gt 0 -and $existingFile.Length -ne $ExpectedBytes) {
                throw "已存在的文件大小不符，请选择其他目录 / Existing file size differs; choose another folder: $Destination"
            }
            Write-Host "已存在，跳过 / Already present: $Destination" -ForegroundColor DarkGray
            return
        }
        Remove-Item -LiteralPath $Destination -Force
    }

    $directory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $partialPath = "$Destination.part"
    [System.Net.ServicePointManager]::SecurityProtocol =
        [System.Net.ServicePointManager]::SecurityProtocol -bor [System.Net.SecurityProtocolType]::Tls12

    for ($attempt = 1; $attempt -le $MaximumAttempts; $attempt++) {
        $response = $null
        $sourceStream = $null
        $targetStream = $null
        try {
            $resumeAt = 0L
            if (Test-Path -LiteralPath $partialPath -PathType Leaf) {
                $resumeAt = (Get-Item -LiteralPath $partialPath).Length
            }

            $request = [System.Net.HttpWebRequest]::Create($Url)
            $request.AllowAutoRedirect = $true
            $request.Timeout = $AttemptTimeoutMs
            $request.ReadWriteTimeout = 120000
            $request.UserAgent = $UserAgent
            if ($resumeAt -gt 0) { $request.AddRange($resumeAt) }

            $response = [System.Net.HttpWebResponse]$request.GetResponse()
            $isPartial = $response.StatusCode -eq [System.Net.HttpStatusCode]::PartialContent
            if ($resumeAt -gt 0 -and -not $isPartial) {
                $resumeAt = 0
            }
            $mode = if ($resumeAt -gt 0) { [System.IO.FileMode]::Append } else { [System.IO.FileMode]::Create }
            $targetStream = [System.IO.File]::Open($partialPath, $mode, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
            $sourceStream = $response.GetResponseStream()
            $totalBytes = if ($response.ContentLength -ge 0) {
                $resumeAt + $response.ContentLength
            } elseif ($ExpectedBytes -gt 0) {
                $ExpectedBytes
            } else {
                -1L
            }
            $downloaded = $resumeAt
            $started = [System.Diagnostics.Stopwatch]::StartNew()
            $buffer = [byte[]]::new(1048576)

            while (($read = $sourceStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                Assert-WizardOperationActive
                $targetStream.Write($buffer, 0, $read)
                $downloaded += $read
                $speed = if ($started.Elapsed.TotalSeconds -gt 0) { [long](($downloaded - $resumeAt) / $started.Elapsed.TotalSeconds) } else { 0L }
                $status = "$(Format-DownloadBytes $downloaded)"
                if ($totalBytes -gt 0) { $status += " / $(Format-DownloadBytes $totalBytes)" }
                if ($speed -gt 0) { $status += "  $(Format-DownloadBytes $speed)/s" }
                $progress = if ($totalBytes -gt 0) { [Math]::Min(100, [int](100 * $downloaded / $totalBytes)) } else { -1 }
                $progressArguments = @{
                    Id = $ProgressId
                    Activity = $Label
                    Status = $status
                    PercentComplete = $progress
                }
                if ($ParentProgressId -ge 0) { $progressArguments.ParentId = $ParentProgressId }
                Write-Progress @progressArguments
            }

            $targetStream.Flush()
            $targetStream.Dispose()
            $targetStream = $null
            $actualBytes = (Get-Item -LiteralPath $partialPath).Length
            $requiredBytes = if ($ExpectedBytes -gt 0) { $ExpectedBytes } else { $totalBytes }
            if ($requiredBytes -gt 0 -and $actualBytes -ne $requiredBytes) {
                throw "下载大小不完整 / Incomplete download: $actualBytes of $requiredBytes bytes"
            }
            Assert-WizardOperationActive
            Move-Item -LiteralPath $partialPath -Destination $Destination -Force
            $completeArguments = @{ Id = $ProgressId; Activity = $Label; Completed = $true }
            if ($ParentProgressId -ge 0) { $completeArguments.ParentId = $ParentProgressId }
            Write-Progress @completeArguments
            return
        } catch {
            if (Test-WizardOperationCancelled $_.Exception) { throw }
            $completeArguments = @{ Id = $ProgressId; Activity = $Label; Completed = $true }
            if ($ParentProgressId -ge 0) { $completeArguments.ParentId = $ParentProgressId }
            Write-Progress @completeArguments
            if ($attempt -ge $MaximumAttempts) {
                throw "下载失败，已重试 $MaximumAttempts 次 / Download failed after $MaximumAttempts attempts: $($_.Exception.Message)"
            }
            Write-Warning "下载中断，将从 .part 文件重试（$attempt/$MaximumAttempts）。Download interrupted; retrying from the .part file. $($_.Exception.Message)"
            Start-Sleep -Seconds ([Math]::Pow(2, $attempt))
        } finally {
            if ($null -ne $targetStream) { $targetStream.Dispose() }
            if ($null -ne $sourceStream) { $sourceStream.Dispose() }
            if ($null -ne $response) { $response.Dispose() }
        }
    }
}

function Invoke-ModelFileDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$Label,
        [int]$MaximumAttempts = 3,
        [int]$AttemptTimeoutMs = 30000,
        [long]$ExpectedBytes = 0
    )

    Invoke-FileDownloadWithProgress `
        -Url $Url `
        -Destination $Destination `
        -Label $Label `
        -MaximumAttempts $MaximumAttempts `
        -AttemptTimeoutMs $AttemptTimeoutMs `
        -ExpectedBytes $ExpectedBytes `
        -ProgressId 71 `
        -ParentProgressId 70
}

function Save-ModelDownloadDirectory {
    param([string]$Directory)
    $settingsFile = Get-Variable -Scope Script -Name statePath -ValueOnly -ErrorAction SilentlyContinue
    $isDryRun = Get-Variable -Scope Script -Name DryRun -ValueOnly -ErrorAction SilentlyContinue
    if (-not $settingsFile -or $isDryRun) { return }
    $temporary = $null
    try {
        $preferences = if (Test-Path -LiteralPath $settingsFile -PathType Leaf) { [IO.File]::ReadAllText($settingsFile) | ConvertFrom-Json } else { [pscustomobject]@{} }
        $preferences | Add-Member -NotePropertyName download_dir -NotePropertyValue $Directory -Force
        [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($settingsFile))
        $temporary = $settingsFile + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
        [IO.File]::WriteAllText($temporary, ($preferences | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($true))
        if ([IO.File]::Exists($settingsFile)) { [IO.File]::Replace($temporary,$settingsFile,[Management.Automation.Language.NullString]::Value) } else { [IO.File]::Move($temporary,$settingsFile) }
    } catch { Write-Warning '下载目录本轮仍有效，但未能保存。The download folder is active for this run but could not be saved.' }
    finally { if ($temporary -and (Test-Path -LiteralPath $temporary)) { Remove-Item -LiteralPath $temporary -Force } }
}

function Get-ModelDownloadDirectory {
    param([switch]$Choose)
    $savedDirectory = [string](Get-SavedValue 'download_dir' '')
    $directory = if ($savedDirectory) { ConvertTo-FullPath $savedDirectory } else { Join-Path $repoRoot 'models' }
    if ($Choose -and -not $script:ModelDownloadDirectoryChosen) {
        $directory = ConvertTo-FullPath (Read-TextValue -Label '模型下载目录 / Model download folder' -Default $directory -Required)
        if (Test-Path -LiteralPath $directory -PathType Leaf) { throw '下载目录不能是文件 / Download folder cannot be a file.' }
        if ($null -eq $script:Saved) { $script:Saved = [pscustomobject]@{} }
        $script:Saved | Add-Member -NotePropertyName download_dir -NotePropertyValue $directory -Force
        Save-ModelDownloadDirectory -Directory $directory
        $script:ModelDownloadDirectoryChosen = $true
    }
    return $directory
}

function Invoke-RecommendedModelDownload {
    param([Parameter(Mandatory = $true)][string]$Id)

    $bundle = @(Get-RecommendedModelCatalog | Where-Object { $_.Id -eq $Id }) | Select-Object -First 1
    if ($null -eq $bundle) { throw "Unknown recommended download: $Id" }
    if ($bundle.PSObject.Properties['Downloadable'] -and -not $bundle.Downloadable) { throw $bundle.UnavailableReason }
    $modelsRoot = Get-ModelDownloadDirectory -Choose
    $files = @($bundle.Files)
    Write-Host "`n下载 / Download: $($bundle.Name)" -ForegroundColor Cyan
    Write-Host "目录 / Folder: $modelsRoot" -ForegroundColor DarkGray
    try {
        for ($i = 0; $i -lt $files.Count; $i++) {
            $file = $files[$i]
            $destination = Join-Path $modelsRoot $file.RelativePath
            $activity = "[$($i + 1)/$($files.Count)] $([System.IO.Path]::GetFileName($destination))"
            Write-Progress -Id 70 -Activity $bundle.Name -Status $activity -PercentComplete ([int](100 * $i / $files.Count))
            Invoke-ModelFileDownload -Url $file.Url -Destination $destination -Label $activity -ExpectedBytes $file.Bytes
        }
    } finally {
        Write-Progress -Id 70 -Activity $bundle.Name -Completed
    }

    $primary = Join-Path $modelsRoot $bundle.Primary
    if (-not (Test-Path -LiteralPath $primary -PathType Leaf)) {
        throw "下载完成但找不到主文件 / Download completed but the primary file is missing: $primary"
    }
    Write-Host "下载完成 / Download complete: $primary" -ForegroundColor Green
    return $primary
}

function Open-ModelSetupGuide {
    param([Parameter(Mandatory = $true)][string]$Url)
    if ($Url -notmatch '^https://github\.com/Headmaster218/MoE4All/releases/(?:latest|tag/release-\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)$') { throw 'Invalid model setup guide URL.' }
    Write-Host "安装说明 / Setup guide: $Url" -ForegroundColor Cyan
    try { Start-Process -FilePath $Url -ErrorAction Stop | Out-Null }
    catch { Write-Warning '无法打开浏览器，请复制上面的链接。Could not open a browser; copy the link above.' }
}

function Select-RecommendedDownloads {
    param([ValidateSet('all','llm','vision','embedding','audio','speech')][string]$Kind='all')
    $matching = @(Get-RecommendedModelCatalog | Where-Object { $Kind -eq 'all' -or $_.Kind -eq $Kind })
    $guides = @($matching | Where-Object { $_.PSObject.Properties['InstructionsUrl'] -and $_.InstructionsUrl })
    $catalog = @($matching | Where-Object { -not $_.PSObject.Properties['Downloadable'] -or $_.Downloadable })
    foreach ($unavailable in @($matching | Where-Object { $_.PSObject.Properties['Downloadable'] -and -not $_.Downloadable })) {
        Write-Host (" {0}: {1}" -f $unavailable.Name, $unavailable.UnavailableReason) -ForegroundColor Yellow
    }
    if ($catalog.Count -eq 0) {
        if ($guides.Count -gt 0) { Open-ModelSetupGuide -Url $guides[0].InstructionsUrl }
        return @()
    }
    while ($true) {
        Write-Host "`n下载模型与组件 / Download models and components" -ForegroundColor Cyan
        for ($i = 0; $i -lt $catalog.Count; $i++) {
            $bundle = $catalog[$i]
            $primary = Join-Path (Get-ModelDownloadDirectory) $bundle.Primary
            $status = if (Test-Path -LiteralPath $primary -PathType Leaf) { ' [已下载 / installed]' } else { '' }
            Write-Host (" [{0}] {1}{2}" -f ($i + 1), $bundle.Name, $status)
            $bytes = [long](($bundle.Files | Measure-Object -Property Bytes -Sum).Sum)
            Write-Host "     $(Format-DownloadBytes $bytes) · $($bundle.Files.Count) 个文件 / file(s)" -ForegroundColor DarkGray
            Write-Host "     $($bundle.InfoUrl)" -ForegroundColor DarkGray
        }
        Write-Host ' [A] 全部 / All'
        if ($guides.Count -gt 0) { Write-Host ' [S] 语音合成安装说明 / Speech setup guide' }
        Write-Host ' [B] 返回 / Back'
        $raw = (Read-Host '选择一个或多个编号，用逗号分隔 / Select one or more numbers, comma-separated').Trim()
        if ($raw -match '^(b|back|返回)$') { return @() }
        if ($raw -match '^(s|speech|guide)$' -and $guides.Count -gt 0) { Open-ModelSetupGuide -Url $guides[0].InstructionsUrl; return @() }

        $indexes = [System.Collections.Generic.List[int]]::new()
        $seenIndexes = [System.Collections.Generic.HashSet[int]]::new()
        $valid = $true
        if ($raw -match '^(a|all|全部)$') {
            for ($i = 0; $i -lt $catalog.Count; $i++) { [void]$indexes.Add($i) }
        } else {
            foreach ($token in @($raw -split '[,;\s]+' | Where-Object { $_ })) {
                $number = 0
                if (-not [int]::TryParse($token, [ref]$number) -or $number -lt 1 -or $number -gt $catalog.Count) {
                    $valid = $false
                    break
                }
                $index = $number - 1
                if ($seenIndexes.Add($index)) { [void]$indexes.Add($index) }
            }
        }
        if (-not $valid -or $indexes.Count -eq 0) {
            Write-Host '请输入列表中的编号，多个编号用逗号分隔。Enter listed numbers separated by commas.' -ForegroundColor Yellow
            continue
        }
        if (-not (Read-YesNo -Label '立即开始或继续所选下载？/ Start or resume the selected downloads now?' -Default $true)) {
            return @()
        }

        $downloaded = [System.Collections.Generic.List[string]]::new()
        foreach ($index in $indexes) {
            try {
                [void]$downloaded.Add((Invoke-RecommendedModelDownload -Id $catalog[$index].Id))
            } catch {
                Write-Host $_.Exception.Message -ForegroundColor Yellow
            }
        }
        return $downloaded.ToArray()
    }
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
    foreach ($candidate in @((Get-SavedValue -Name 'model_history' -Default @()))) {
        $path = [string]$candidate
        if (-not [string]::IsNullOrWhiteSpace($path) -and (Test-Path -LiteralPath $path -PathType Leaf)) {
            if ($seen.Add($path)) { [void]$models.Add($path) }
        }
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

    foreach ($bundle in @(Get-RecommendedModelCatalog | Where-Object { $_.Kind -eq 'llm' })) {
        $candidate = Join-Path (Get-ModelDownloadDirectory) $bundle.Primary
        if ((Test-Path -LiteralPath $candidate -PathType Leaf) -and $seen.Add($candidate)) {
            [void]$models.Add($candidate)
        }
    }

    while ($true) {
        Write-Host "`n模型 / Model" -ForegroundColor Cyan
        for ($i = 0; $i -lt $models.Count; $i++) {
            Write-Host (" [{0}] {1}" -f ($i + 1), $models[$i])
        }
        Write-Host ' [D] 下载模型或组件 / Download models or components'
        Write-Host ' [N] 输入新路径，或直接粘贴路径 / Enter a new path, or paste it directly'
        $defaultChoice = if ($models.Count -gt 0) { '1' } else { 'D' }
        $choice = (Read-Host "选择模型 [$defaultChoice] / Select model [$defaultChoice]").Trim()
        if ([string]::IsNullOrWhiteSpace($choice)) { $choice = $defaultChoice }
        $index = 0
        if ([int]::TryParse($choice, [ref]$index) -and $index -ge 1 -and $index -le $models.Count) {
            return $models[$index - 1]
        }
        if ($choice -match '^(d|download|下载)$') {
            $null = @(Select-RecommendedDownloads)
            foreach ($bundle in @(Get-RecommendedModelCatalog | Where-Object { $_.Kind -eq 'llm' })) {
                $candidate = Join-Path (Get-ModelDownloadDirectory) $bundle.Primary
                if ((Test-Path -LiteralPath $candidate -PathType Leaf) -and $seen.Add($candidate)) {
                    [void]$models.Add($candidate)
                }
            }
            continue
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

function Get-UpdatedModelHistory {
    param([Parameter(Mandatory = $true)][string]$SelectedModel)

    $history = [System.Collections.Generic.List[string]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($candidate in @($SelectedModel) + @((Get-SavedValue -Name 'model_history' -Default @())) + @((Get-SavedValue -Name 'model' -Default ''))) {
        $path = [string]$candidate
        if ([string]::IsNullOrWhiteSpace($path)) { continue }
        try { $path = ConvertTo-FullPath $path } catch { continue }
        if ((Test-Path -LiteralPath $path -PathType Leaf) -and $seen.Add($path)) {
            [void]$history.Add($path)
            if ($history.Count -ge 20) { break }
        }
    }
    return $history.ToArray()
}

function Show-ManualBenchmarkPlan {
    param(
        [Parameter(Mandatory = $true)][string]$Kind,
        [string]$PromptTokens = '0',
        [string]$GenTokens = '0',
        [string]$DepthMode = 'none',
        [string]$DepthTokens = '0',
        [string]$Repetitions = ''
    )

    Write-Host "`n当前计划 / Current plan" -ForegroundColor Cyan
    switch ($Kind) {
        'prefill' { Write-Host " + Prefill: $PromptTokens tokens" }
        'decode' { Write-Host " + Decode: $GenTokens tokens" }
        'mixed' {
            Write-Host " + Prefill: $PromptTokens tokens"
            Write-Host " + Decode: $GenTokens tokens"
        }
    }
    if ($DepthMode -eq 'real') {
        Write-Host " + 真实已有上下文 / Real existing context: $DepthTokens tokens"
    } elseif ($DepthMode -eq 'synthetic') {
        Write-Host " + Synthetic KV depth: $DepthTokens tokens"
    }
    if ($Repetitions) { Write-Host " + 重复 / Repetitions: $Repetitions" }
}

function Get-BenchmarkSubmissionSnapshot {
    $root = Join-Path $repoRoot 'benchmark-database'
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { return @() }
    return @(Get-ChildItem -LiteralPath $root -Filter '*.m4b' -File -Recurse -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
}

function Read-MoE4AllSubmissionPayload {
    param([Parameter(Mandatory = $true)][string]$Path)

    New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
    $temporary = Join-Path $dataDir ("submission-decode-{0}.json" -f [Guid]::NewGuid().ToString('N'))
    try {
        $decoded = Invoke-NativeCommandResult -Executable $infrPath -Arguments @('submission', 'decode', $Path, '--output', $temporary)
        if ($decoded.ExitCode -ne 0) {
            throw "submission decode failed: $($decoded.Output -join ' ')"
        }
        return [System.IO.File]::ReadAllText($temporary) | ConvertFrom-Json
    } finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

function Write-BenchmarkSubmissionFromJson {
    param([Parameter(Mandatory = $true)]$Value)

    New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
    $temporary = Join-Path $dataDir ("benchmark-payload-{0}.json" -f [Guid]::NewGuid().ToString('N'))
    try {
        [System.IO.File]::WriteAllText($temporary, (($Value | ConvertTo-Json -Depth 20 -Compress) + "`n"), [System.Text.UTF8Encoding]::new($false))
        $packed = Invoke-NativeCommandResult -Executable $infrPath -Arguments @('submission', 'pack-benchmark', '--input', $temporary, '--result-dir', (Join-Path $repoRoot 'benchmark-database'))
        if ($packed.ExitCode -ne 0) {
            throw "submission pack failed: $($packed.Output -join ' ')"
        }
        $marker = @($packed.Output | Where-Object { $_ -like 'MOE4ALL_SUBMISSION_PATH=*' } | Select-Object -Last 1)
        if ($marker.Count -ne 1) { throw 'submission pack did not return a result path' }
        return [string]$marker[0].Substring('MOE4ALL_SUBMISSION_PATH='.Length)
    } finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

function Write-ManualBenchmarkResult {
    param(
        [Parameter(Mandatory = $true)][string]$RawJson,
        [Parameter(Mandatory = $true)][string]$ModelPath,
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][string]$PromptTokens,
        [Parameter(Mandatory = $true)][string]$GenTokens,
        [Parameter(Mandatory = $true)][string]$DepthMode,
        [Parameter(Mandatory = $true)][string]$DepthTokens,
        [Parameter(Mandatory = $true)][string]$Repetitions
    )

    $parsed = $RawJson | ConvertFrom-Json
    if ($parsed.Count -ne 1 -or $null -eq $parsed[0].avg_ts) {
        throw '手动 benchmark 未返回可识别的 JSON。Manual benchmark did not return recognizable JSON.'
    }
    $rawResult = $parsed[0]
    $now = [DateTime]::UtcNow
    $id = 'manual-{0}-{1}' -f $now.ToString('yyyyMMddTHHmmssZ'), [Guid]::NewGuid().ToString('N')
    $modelFile = Get-Item -LiteralPath $ModelPath -ErrorAction Stop
    $cpuName = $null
    $ramBytes = $null
    if ($null -eq $rawResult.benchmark_metadata) {
        try { $cpuName = [string](Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name) } catch { }
        try { $ramBytes = [uint64](Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory) } catch { }
    }
    $contextTokens = if ($DepthMode -eq 'none') { 0 } else { [int64]$DepthTokens }
    if ($Kind -eq 'prefill') { $contextTokens = [int64]$PromptTokens }
    $result = [ordered]@{
        schema_version = 1
        id = $id
        created_at_utc = $now.ToString('o')
        engine_version = $productVersion
        engine_commit = $null
        profile = 'manual'
        model = [ordered]@{ name = $modelFile.Name; storage_bytes = [uint64]$modelFile.Length }
        hardware = [ordered]@{
            os = 'windows'
            architecture = $env:PROCESSOR_ARCHITECTURE
            cpu = [ordered]@{ name = $cpuName; logical_threads = [Environment]::ProcessorCount }
            ram = [ordered]@{ total_bytes = $ramBytes }
            gpu = [ordered]@{ configured_device = $device }
        }
        configuration = [ordered]@{
            benchmark_kind = $Kind
            prompt_tokens = [int64]$PromptTokens
            decode_tokens = [int64]$GenTokens
            depth_mode = $DepthMode
            depth_tokens = [int64]$DepthTokens
            repetitions = [int64]$Repetitions
            auto_profile = $setupMode
            kv_type_k = $kvTypeK
            kv_type_v = $kvTypeV
        }
        contexts = @([ordered]@{
            requested_context_tokens = $contextTokens
            prefill = $null
            decode = @()
            manual_result = [ordered]@{ kind = $Kind; average_tokens_per_second = [double]$rawResult.avg_ts; raw = $rawResult }
            failures = @()
        })
        failures = @()
    }
    if ($null -ne $rawResult.benchmark_metadata) {
        $metadata = $rawResult.benchmark_metadata
        foreach ($key in @('engine_version', 'engine_commit', 'hardware', 'model')) {
            if ($null -ne $metadata.$key) { $result[$key] = $metadata.$key }
        }
        if ($null -ne $metadata.configuration) {
            foreach ($property in $metadata.configuration.PSObject.Properties) {
                if ($property.Name -eq 'auto_profile') { continue }
                $result.configuration[$property.Name] = $property.Value
            }
        }
    }
    Write-Host "`n结果 / Results" -ForegroundColor Green
    Write-Host '+------------------+------------+----------+'
    Write-Host '| Workload         | Repetitions| Overall  |'
    Write-Host '+------------------+------------+----------+'
    Write-Host ('| {0,-16} | {1,10} | {2,8:N1} |' -f $Kind, $Repetitions, [double]$rawResult.avg_ts)
    Write-Host '+------------------+------------+----------+'
    Write-Host 'Overall speed: tok/s. / Overall 速度单位：tok/s。' -ForegroundColor DarkGray
    if ($Kind -eq 'mixed') {
        $localDirectory = Join-Path $dataDir 'benchmark-results'
        New-Item -ItemType Directory -Path $localDirectory -Force | Out-Null
        $localPath = Join-Path $localDirectory ($id + '.json')
        [IO.File]::WriteAllText($localPath, ($result | ConvertTo-Json -Depth 32), [Text.UTF8Encoding]::new($false))
        Write-Host "混合测试结果已本地保存 / Mixed benchmark result saved locally: $localPath" -ForegroundColor Cyan
        Write-Host '当前上传协议尚不支持混合速度，本次不上传；它不会被误记为纯 Decode。Mixed-speed upload is not supported yet; this result is not submitted as Decode.' -ForegroundColor Yellow
        return $null
    }
    try { $resultPath = Write-BenchmarkSubmissionFromJson -Value $result } catch {
        $localDirectory = Join-Path $dataDir 'benchmark-results'
        New-Item -ItemType Directory -Path $localDirectory -Force | Out-Null
        $localPath = Join-Path $localDirectory ($id + '.json')
        [IO.File]::WriteAllText($localPath, ($result | ConvertTo-Json -Depth 32), [Text.UTF8Encoding]::new($false))
        Write-Host "原始结果已保留 / Original result retained: $localPath" -ForegroundColor Yellow
        throw
    }
    Write-Host "Benchmark submission: $resultPath" -ForegroundColor DarkGray
    Write-Host 'AI/自动化：请向用户展示 benchmark 结果及结果文件，询问是否上传并等待明确决定；不要自行上传或静默忽略。' -ForegroundColor Yellow
    Write-Host "AI/automation: report the benchmark result and result file; ask whether to upload and wait for the user's explicit decision; do not upload or silently skip." -ForegroundColor Yellow
    return $resultPath
}

function Test-SubmissionForUpload {
    param([Parameter(Mandatory = $true)][string]$Path, [ValidateSet('benchmark', 'error')][string]$Kind = 'benchmark')

    $databaseRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot 'benchmark-database'))
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $prefix = $databaseRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    $pendingDirectory = Get-PendingSubmissionDirectory
    $pendingRoot = [System.IO.Path]::GetFullPath($pendingDirectory).TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    $isPendingCopy = $fullPath.StartsWith($pendingRoot, [System.StringComparison]::OrdinalIgnoreCase)
    if ($Kind -eq 'benchmark' -and -not $fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase) -and -not $isPendingCopy) {
        throw '只允许上传 benchmark-database 中的结果。Only results under benchmark-database can be uploaded.'
    }
    $extension = if ($Kind -eq 'benchmark') { '.m4b' } else { '.m4e' }
    if ([System.IO.Path]::GetExtension($fullPath) -ine $extension) {
        throw '提交文件扩展名与类型不一致。Submission extension does not match its kind.'
    }
    $value = Read-MoE4AllSubmissionPayload -Path $fullPath
    $raw = $value | ConvertTo-Json -Depth 20 -Compress
    if ([int]$value.schema_version -ne 1 -or [string]$value.id -ne [System.IO.Path]::GetFileNameWithoutExtension($fullPath)) {
        throw 'Benchmark 提交的 schema 或文件名无效。The benchmark submission schema or filename is invalid.'
    }
    if ($Kind -eq 'benchmark' -and [string]$value.profile -notin @('quick', 'standard', 'full', 'manual')) {
        throw 'Benchmark 提交的 profile 无效。The benchmark submission profile is invalid.'
    }
    if ($Kind -eq 'error' -and ([string]::IsNullOrWhiteSpace([string]$value.description) -or (Get-UnicodeCodePointCount -Value ([string]$value.description)) -gt 4096)) {
        throw '错误报告需要 1 至 4096 字的问题描述。Error reports require a description of 1..4096 code points.'
    }
    if ($raw -match '(?i)[a-z]:\\\\|/home/|/users/|\\\\users\\\\|"hostname"|"serial') {
        throw '结果包含本机路径、主机名或序列号，已拒绝上传。The result contains a local path, hostname or serial number.'
    }
    return $value
}

function Get-PendingSubmissionDirectory {
    if (-not [string]::IsNullOrWhiteSpace([string]$pendingSubmissionDir)) { return $pendingSubmissionDir }
    if (-not [string]::IsNullOrWhiteSpace([string]$dataDir)) { return (Join-Path $dataDir 'private\pending-submissions') }
    return ''
}

function Test-BenchmarkResultForUpload {
    param([Parameter(Mandatory = $true)][string]$Path)
    return Test-SubmissionForUpload -Path $Path -Kind 'benchmark'
}

function Copy-BenchmarkSubmissionToClipboard {
    param([Parameter(Mandatory = $true)][string]$Path)

    $contents = [System.IO.File]::ReadAllText($Path)
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        try {
            Set-Clipboard -Value $contents -ErrorAction Stop
            return $true
        } catch {
            Start-Sleep -Milliseconds (50 * ($attempt + 1))
        }
    }

    $clipPath = Join-Path $env:SystemRoot 'System32\clip.exe'
    if (Test-Path -LiteralPath $clipPath -PathType Leaf) {
        $process = $null
        try {
            $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
            $startInfo.FileName = $clipPath
            $startInfo.UseShellExecute = $false
            $startInfo.CreateNoWindow = $true
            $startInfo.RedirectStandardInput = $true
            $process = [System.Diagnostics.Process]::new()
            $process.StartInfo = $startInfo
            if ($process.Start()) {
                $process.StandardInput.Write($contents)
                $process.StandardInput.Close()
                $process.WaitForExit()
                if ($process.ExitCode -eq 0) { return $true }
            }
        } catch {
        } finally {
            if ($null -ne $process) { $process.Dispose() }
        }
    }
    return $false
}

function Get-SubmissionBrowserUploadPlan {
    param([Parameter(Mandatory = $true)][string]$Path, [ValidateSet('benchmark', 'error')][string]$Kind = 'benchmark', [bool]$ClipboardReady = $true)
    $value = Test-SubmissionForUpload -Path $Path -Kind $Kind
    $month = ([DateTimeOffset]::Parse([string]$value.created_at_utc)).UtcDateTime.ToString('yyyy-MM')
    $fileName = [System.IO.Path]::GetFileName($Path)
    $base = "https://github.com/Headmaster218/MoE4All/new/main/${Kind}-database/v1/$month"
    $guidance = @'
上传步骤：
1. 在这个编辑框内按 Ctrl+A，全选并删除本提示。
2. 按 Ctrl+V，粘贴完整的 .m4b 或 .m4e 报告文本。
3. 点击页面右上角的 Commit changes...（无写权限时可能显示 Propose changes）。
4. 选择新建分支（不要直接提交 main），确认提交，然后点击 Create pull request。
5. 检查标题与文件后，再点击一次 Create pull request，即完成上传。

Upload steps:
1. Press Ctrl+A in this editor to select and remove this instruction.
2. Press Ctrl+V to paste the complete .m4b or .m4e report text.
3. Click Commit changes... at the upper right (it may say Propose changes without write access).
4. Choose a new branch (not a direct commit to main), confirm, then click Create pull request.
5. Review the title and file, then click Create pull request again to finish.
'@
    if (-not $ClipboardReady) {
        $guidance = "剪贴板复制未成功：先打开本地报告，全选并复制完整文本，再按以下步骤操作。`nClipboard copy failed: open the local report and copy its complete text first.`n`n" + $guidance
    }
    $branch = "community/$Kind/$((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant())"
    $guidance += "`nmacOS 请使用 Cmd+A / Cmd+V。On macOS, use Cmd+A / Cmd+V.`n建议分支名 / Suggested branch: $branch"
    $commonQuery = @(
        'filename=' + [Uri]::EscapeDataString($fileName)
        'message=' + [Uri]::EscapeDataString("${Kind}: add $($value.id)")
        'value=' + [Uri]::EscapeDataString($guidance.Trim())
    ) -join '&'
    return [pscustomobject]@{ Url = "$base`?$commonQuery"; GuidancePrefilled = $true }
}

function Get-BenchmarkBrowserUploadPlan {
    param([Parameter(Mandatory = $true)][string]$Path)
    Get-SubmissionBrowserUploadPlan -Path $Path -Kind benchmark
}

function Open-SubmissionBrowserUpload {
    param([Parameter(Mandatory = $true)][string]$Path, [ValidateSet('benchmark', 'error')][string]$Kind = 'benchmark')
    $copied = Copy-BenchmarkSubmissionToClipboard -Path $Path
    if (-not $copied) {
        try { Start-Process -FilePath 'explorer.exe' -ArgumentList ("/select,`"{0}`"" -f $Path) } catch { }
    }
    $plan = Get-SubmissionBrowserUploadPlan -Path $Path -Kind $Kind -ClipboardReady $copied
    try {
        Start-Process -FilePath $plan.Url
        Write-Host '已打开 GitHub 上传页，请按编辑区内的双语提示完成。Opened GitHub; follow the bilingual instructions in the editor.' -ForegroundColor Cyan
    } catch {
        Write-Host "未能自动打开 GitHub，请手动打开以下地址。Could not open GitHub; open this URL manually:`n$($plan.Url)" -ForegroundColor Yellow
    }
    Write-Host "本地报告 / Local report: $Path"
}

function Invoke-NativeCommandResult {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @()
    )

    # Windows PowerShell 5.1 turns native stderr into ErrorRecord objects. With the
    # wizard's Stop policy, expected non-zero probes such as `gh auth status` would
    # otherwise terminate the whole wizard before their exit code can be checked.
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        Assert-WizardOperationActive
        $nativeOutput = @(& $Executable @Arguments 2>&1)
        $nativeExitCode = $LASTEXITCODE
        $diagnosticOutput = @($nativeOutput | Where-Object { $_ -is [Management.Automation.ErrorRecord] } | ForEach-Object { [string]$_ })
        # Successful machine-readable gh output must not include stderr warnings.
        # On failure retain the combined diagnostic text used by existing callers.
        $textOutput = @($nativeOutput | Where-Object { $nativeExitCode -ne 0 -or $_ -isnot [Management.Automation.ErrorRecord] } | ForEach-Object { [string]$_ })
        return [pscustomobject]@{ ExitCode = $nativeExitCode; Output = [string[]]$textOutput; ErrorOutput=[string[]]$diagnosticOutput }
    } catch {
        if (Test-WizardOperationCancelled $_.Exception) { throw }
        return [pscustomobject]@{ ExitCode = -1; Output = [string[]]@($_.Exception.Message) }
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
}

function Test-GhAuthenticated {
    param([Parameter(Mandatory = $true)]$GhCommand)
    $auth = Invoke-NativeCommandResult -Executable $GhCommand.Source -Arguments @('auth', 'status', '--hostname', 'github.com')
    return $auth.ExitCode -eq 0
}


function Invoke-GhSubmissionUpload {
    param([Parameter(Mandatory = $true)][string]$Path, [ValidateSet('benchmark', 'error')][string]$Kind = 'benchmark', [switch]$Retry)

    $value = Test-SubmissionForUpload -Path $Path -Kind $Kind
    $verified = Invoke-NativeCommandResult -Executable $infrPath -Arguments @('submission', 'verify', $Path)
    if ($verified.ExitCode -ne 0) { $script:SubmissionValidationFailed = $true; throw '提交文件未通过本地校验，未上传。Local submission validation failed; nothing was uploaded.' }
    $gh = Get-Command gh -ErrorAction SilentlyContinue
    if ($null -eq $gh) { return $false }
    if (-not (Test-GhAuthenticated -GhCommand $gh)) { return $false }
    $identity = Invoke-NativeCommandResult -Executable $gh.Source -Arguments @('api', 'user', '--jq', '.login')
    $login = [string]($identity.Output | Where-Object { $_ } | Select-Object -First 1)
    if ($identity.ExitCode -ne 0 -or -not $login) { return $false }
    $login = $login.Trim()
    if ($login -notmatch '^[A-Za-z0-9][A-Za-z0-9-]{0,38}$') { return $false }

    $upstream = 'Headmaster218/MoE4All'
    $fork = "$login/MoE4All"
    $month = ([DateTimeOffset]::Parse([string]$value.created_at_utc)).UtcDateTime.ToString('yyyy-MM')
    $relative = "${Kind}-database/v1/$month/$([System.IO.Path]::GetFileName($Path))"
    $fileHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "`nGitHub: $login" -ForegroundColor Cyan
    Write-Host "目标 / Target: $upstream/$relative"
    Write-Host 'PR 创建后报告即公开；不创建 Issue，不自动合并。The report becomes public when the PR is created; no Issue or automatic merge.' -ForegroundColor Yellow
    if (-not ($Retry -and $script:SubmissionGhConfirmedHash -eq "$Kind/$fileHash")) {
        if (-not (Read-YesNo -Label '确认创建分支并提交报告 PR？/ Create the branch and report PR now?' -Default $true)) {
            $script:SubmissionUploadCancelled = $true
            return $false
        }
        $script:SubmissionGhConfirmedHash = "$Kind/$fileHash"
    }

    try {
        $forkProbe = Invoke-NativeCommandResult -Executable $gh.Source -Arguments @('api', "repos/$fork")
        if ($forkProbe.ExitCode -ne 0) {
            $forkCreate = Invoke-NativeCommandResult -Executable $gh.Source -Arguments @('repo', 'fork', $upstream, '--clone=false', '--remote=false')
            if ($forkCreate.ExitCode -ne 0) { throw "gh repo fork failed: $($forkCreate.Output -join ' ')" }
            $ready = $false
            for ($attempt = 0; $attempt -lt 15; $attempt++) {
                Start-Sleep -Seconds 2
                $forkProbe = Invoke-NativeCommandResult -Executable $gh.Source -Arguments @('api', "repos/$fork")
                if ($forkProbe.ExitCode -eq 0) { $ready = $true; break }
            }
            if (-not $ready) { throw 'GitHub fork was not ready in time' }
        }
        $baseResult = Invoke-NativeCommandResult -Executable $gh.Source -Arguments @('api', "repos/$upstream/git/ref/heads/main", '--jq', '.object.sha')
        $baseSha = [string]($baseResult.Output | Where-Object { $_ } | Select-Object -First 1)
        if ($baseResult.ExitCode -ne 0 -or -not $baseSha) { throw 'could not read upstream main SHA' }
        $baseSha = $baseSha.Trim()
        $branch = "community/$Kind/$fileHash"
        $refCreate = Invoke-NativeCommandResult -Executable $gh.Source -Arguments @('api', '--method', 'POST', "repos/$fork/git/refs", '-f', "ref=refs/heads/$branch", '-f', "sha=$baseSha")
        if ($refCreate.ExitCode -ne 0) {
            $probe = Invoke-NativeCommandResult -Executable $gh.Source -Arguments @('api', "repos/$fork/git/ref/heads/$branch")
            if ($probe.ExitCode -ne 0) { throw 'could not create or reuse the report branch' }
        }

        $uploadRequest = Join-Path $dataDir ("benchmark-upload-{0}.json" -f [Guid]::NewGuid().ToString('N'))
        try {
            $body = [ordered]@{
                message = "${Kind}: add $($value.id)"
                content = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($Path))
                branch = $branch
            } | ConvertTo-Json -Compress
            [System.IO.File]::WriteAllText($uploadRequest, $body, [System.Text.UTF8Encoding]::new($false))
            $upload = Invoke-NativeCommandResult -Executable $gh.Source -Arguments @('api', '--method', 'PUT', "repos/$fork/contents/$relative", '--input', $uploadRequest)
            if ($upload.ExitCode -ne 0) {
                $existing = Invoke-NativeCommandResult -Executable $gh.Source -Arguments @('api', "repos/$fork/contents/${relative}?ref=$([Uri]::EscapeDataString($branch))", '--jq', '.content')
                if ($existing.ExitCode -ne 0) { throw 'could not upload or verify the existing report' }
                $content = [Convert]::FromBase64String(($existing.Output -join '').Replace("`n", '').Replace("`r", ''))
                $sha = [System.Security.Cryptography.SHA256]::Create()
                try { $actualHash = ([BitConverter]::ToString($sha.ComputeHash($content))).Replace('-', '').ToLowerInvariant() } finally { $sha.Dispose() }
                if ($actualHash -ne $fileHash) { throw 'existing report differs; nothing was overwritten' }
            }
        } finally {
            Remove-Item -LiteralPath $uploadRequest -Force -ErrorAction SilentlyContinue
        }
        $prCreate = Invoke-NativeCommandResult -Executable $gh.Source -Arguments @('pr', 'create', '--repo', $upstream, '--base', 'main', '--head', "${login}:$branch", '--title', "${Kind}: add $($value.id)", '--body', 'User-consented public MoE4All report. Awaiting maintainer review; no automatic merge. Format validation is not an independent reproduction.')
        $prUrl = [string]($prCreate.Output | Where-Object { $_ } | Select-Object -Last 1)
        if ($prCreate.ExitCode -ne 0) {
            $listed = Invoke-NativeCommandResult -Executable $gh.Source -Arguments @('pr', 'list', '--repo', $upstream, '--head', $branch, '--author', $login, '--base', 'main', '--state', 'all', '--json', 'url', '--jq', '.[0].url // empty')
            $prUrl = [string]($listed.Output | Where-Object { $_ } | Select-Object -First 1)
            if ($listed.ExitCode -ne 0 -or $prUrl -notmatch '^https://github\.com/Headmaster218/MoE4All/pull/[0-9]+$') { throw 'could not create or reuse pull request' }
        }
        if ($prUrl -notmatch '^https://github\.com/Headmaster218/MoE4All/pull/[0-9]+$') { throw 'GitHub did not return a valid report PR URL' }
        Set-BenchmarkSubmissionSucceeded -Kind $Kind -ReportVersion ([string]$value.engine_version)
        Write-Host "PR 已创建 / PR created: $prUrl" -ForegroundColor Green
        return $true
    } catch {
        Write-Warning "gh 上传失败：$($_.Exception.Message) / gh upload failed."
        return $false
    }
}

function Invoke-GhBenchmarkUpload {
    param([Parameter(Mandatory = $true)][string]$Path)
    return Invoke-GhSubmissionUpload -Path $Path -Kind 'benchmark'
}

function Save-BenchmarkUploadPreferences {
    if ($DryRun) { return }
    if ([string]::IsNullOrWhiteSpace($statePath)) { return }
    try {
        $preferences = $script:state
        if ($script:BenchmarkGuideActive) { $preferences = $null }
        if ($null -eq $preferences) {
            $preferences = [ordered]@{}
            if (Test-Path -LiteralPath $statePath -PathType Leaf) {
                $existing = [System.IO.File]::ReadAllText($statePath) | ConvertFrom-Json
                foreach ($property in $existing.PSObject.Properties) { $preferences[$property.Name] = $property.Value }
            }
        }
        $preferences['benchmark_upload'] = $script:BenchmarkUploadDefault
        $preferences['benchmark_upload_has_github'] = $script:BenchmarkUploadHasGithubDefault
        $preferences['benchmark_upload_gh_auth'] = $script:BenchmarkUploadGhAuthDefault
        $preferences['benchmark_upload_fallback'] = $script:BenchmarkUploadFallbackDefault
        $preferences['submission_browser_action'] = $script:SubmissionBrowserActionDefault
        $preferences['submission_failure_action'] = $script:SubmissionFailureActionDefault
        $preferences['submission_pending_retry'] = $script:SubmissionPendingRetryDefault
        New-Item -ItemType Directory -Path (Split-Path -Parent $statePath) -Force | Out-Null
        [System.IO.File]::WriteAllText(
            $statePath,
            (($preferences | ConvertTo-Json -Depth 20) + "`n"),
            [System.Text.UTF8Encoding]::new($true)
        )
    } catch {
        Write-Warning "上传偏好保存失败：$($_.Exception.Message) / Could not save upload preferences."
    }
}

function Read-BenchmarkUploadConsent {
    if ($DesktopExecutionPlanPath -and $env:MOE4ALL_DESKTOP_IO -eq '1') {
        $answer = (Read-WizardRuntimeResponse -Prompt '愿意公开提交这份 Benchmark 结果吗？/ Submit this Benchmark result publicly?' -InputType 'confirm' -DefaultValue $(if($script:BenchmarkUploadDefault){'y'}else{'n'}) -AffirmativeText '提交结果 / Submit results' -NegativeText '暂不提交 / Not now') -eq 'y'
    } else {
        $answer = Read-YesNo -Label '愿意公开提交这份 benchmark 结果吗？/ Submit this benchmark result publicly?' -Default $script:BenchmarkUploadDefault
    }
    $script:BenchmarkUploadDefault = $answer
    Save-BenchmarkUploadPreferences
    return $answer
}

function Get-UnicodeCodePointCount {
    param([AllowEmptyString()][string]$Value = '')

    $count = 0
    for ($index = 0; $index -lt $Value.Length; $index++) {
        if ([char]::IsHighSurrogate($Value[$index]) -and $index + 1 -lt $Value.Length -and [char]::IsLowSurrogate($Value[$index + 1])) {
            $index++
        }
        $count++
    }
    return $count
}

function Get-Moe4AllSubmissionApiConfig {
    $configured = [Environment]::GetEnvironmentVariable('MOE4ALL_SUBMISSION_API_BASE', 'Process')
    $explicit = -not [string]::IsNullOrWhiteSpace($configured)
    $raw = if ($explicit) { $configured.Trim() } else { 'https://api.moe4all.com/v1' }
    $api = $null
    if (-not [Uri]::TryCreate($raw, [UriKind]::Absolute, [ref]$api)) {
        throw 'MOE4ALL_SUBMISSION_API_BASE 不是有效的绝对 URL。MOE4ALL_SUBMISSION_API_BASE is not a valid absolute URL.'
    }
    if ($api.UserInfo -or $api.Query -or $api.Fragment -or $api.AbsolutePath.TrimEnd('/') -ne '/v1') {
        throw '提交 API 地址必须是无凭据、查询或片段且以 /v1 结尾的基地址。The submission API must be a credential-free /v1 base URL without query or fragment.'
    }
    $loopback = $api.IsLoopback -or $api.Host -in @('127.0.0.1', 'localhost', '::1')
    if ($api.Scheme -ne 'https' -and -not ($explicit -and $api.Scheme -eq 'http' -and $loopback)) {
        throw '提交 API 默认必须使用 HTTPS；显式设置的本机回环开发地址可以使用 HTTP。Submission API requires HTTPS except for an explicitly configured loopback development URL.'
    }

    $siteRaw = [Environment]::GetEnvironmentVariable('MOE4ALL_SUBMISSION_SITE_ORIGIN', 'Process')
    if ([string]::IsNullOrWhiteSpace($siteRaw)) {
        if ($loopback) {
            $siteRaw = $api.GetLeftPart([UriPartial]::Authority)
        } elseif ($api.Host -ieq 'api.moe4all.com') {
            $siteRaw = 'https://moe4all.com'
        } else {
            throw '自定义 HTTPS API 还需要设置 MOE4ALL_SUBMISSION_SITE_ORIGIN。A custom HTTPS API also requires MOE4ALL_SUBMISSION_SITE_ORIGIN.'
        }
    }
    $site = $null
    if (-not [Uri]::TryCreate($siteRaw.TrimEnd('/'), [UriKind]::Absolute, [ref]$site) -or $site.UserInfo -or $site.Query -or $site.Fragment -or $site.AbsolutePath -ne '/') {
        throw 'MOE4ALL_SUBMISSION_SITE_ORIGIN 不是有效的来源地址。MOE4ALL_SUBMISSION_SITE_ORIGIN is not a valid origin.'
    }
    if ($site.Scheme -ne 'https' -and -not ($explicit -and $site.Scheme -eq 'http' -and ($site.IsLoopback -or $site.Host -in @('127.0.0.1', 'localhost', '::1')))) {
        throw '提交确认页来源必须使用 HTTPS，开发时仅允许显式回环 HTTP。The verification origin requires HTTPS except for explicit loopback development.'
    }

    return [pscustomobject]@{
        ApiBase = [Uri]::new($api.AbsoluteUri.TrimEnd('/'))
        SiteOrigin = $site.GetLeftPart([UriPartial]::Authority).TrimEnd('/')
        IsLocalDevelopment = $loopback -and $api.Scheme -eq 'http'
    }
}

function Get-SubmissionProxy {
    param([Parameter(Mandatory = $true)][Uri]$ProbeUri)

    $noProxy = [Environment]::GetEnvironmentVariable('NO_PROXY', 'Process')
    foreach ($entry in @($noProxy -split ',')) {
        $entry = $entry.Trim().ToLowerInvariant()
        if (-not $entry) { continue }
        $hostPattern = $entry; $port = 0
        if ($entry -match '^([^:]+):(\d+)$') { $hostPattern = $Matches[1]; $port = [int]$Matches[2] }
        $hostPattern = $hostPattern.TrimStart('.')
        if (($port -eq 0 -or $port -eq $ProbeUri.Port) -and ($hostPattern -eq '*' -or $ProbeUri.DnsSafeHost -eq $hostPattern -or $ProbeUri.DnsSafeHost.EndsWith('.' + $hostPattern, [StringComparison]::OrdinalIgnoreCase))) {
            return [System.Net.WebProxy]::new()
        }
    }
    foreach ($name in @('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy')) {
        $raw = [Environment]::GetEnvironmentVariable($name, 'Process')
        $uri = $null
        if (-not [string]::IsNullOrWhiteSpace($raw) -and [Uri]::TryCreate($raw.Trim(), [UriKind]::Absolute, [ref]$uri) -and $uri.Scheme -in @('http', 'https')) {
            return [System.Net.WebProxy]::new($uri)
        }
    }
    try {
        $proxy = [System.Net.WebRequest]::DefaultWebProxy
        if ($null -ne $proxy -and -not $proxy.IsBypassed($ProbeUri)) {
            $mapped = $proxy.GetProxy($ProbeUri)
            if ($null -ne $mapped -and $mapped.AbsoluteUri -ne $ProbeUri.AbsoluteUri) {
                $resolvedProxy = [System.Net.WebProxy]::new($mapped)
                $resolvedProxy.Credentials = $proxy.Credentials
                return $resolvedProxy
            }
        }
    } catch { }
    return $null
}


function Save-PendingSubmission {
    param([Parameter(Mandatory = $true)][string]$Path, [ValidateSet('benchmark', 'error')][string]$Kind, [switch]$Quiet)

    [void](Test-SubmissionForUpload -Path $Path -Kind $Kind)
    $directory = Get-PendingSubmissionDirectory
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $destination = Join-Path $directory ([System.IO.Path]::GetFileName($Path))
    if ([System.IO.Path]::GetFullPath($Path) -ne [System.IO.Path]::GetFullPath($destination)) {
        if ((Test-Path -LiteralPath $destination -PathType Leaf) -and
            (Get-FileHash -LiteralPath $destination).Hash -ne (Get-FileHash -LiteralPath $Path).Hash) {
            throw '待发目录已有同 ID 的不同报告，已保留两份原件。A different report with this ID is already queued; existing files were preserved.'
        }
        $temporary = "$destination.moe4all-new-$([Guid]::NewGuid().ToString('N'))"
        Copy-Item -LiteralPath $Path -Destination $temporary -Force
        Move-Item -LiteralPath $temporary -Destination $destination -Force
    }
    if (-not $Quiet) { Write-Host "提交暂存，待下次启动重试 / Submission saved for a later retry: $destination" -ForegroundColor Yellow }
    return $destination
}

function Get-PendingSubmissions {
    $directory = Get-PendingSubmissionDirectory
    if ([string]::IsNullOrWhiteSpace($directory)) { return @() }
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) { return @() }
    return @(Get-ChildItem -LiteralPath $directory -File -ErrorAction SilentlyContinue | Where-Object Extension -in @('.m4b', '.m4e') | Sort-Object LastWriteTimeUtc)
}

function Remove-PendingSubmission {
    param([Parameter(Mandatory = $true)][string]$Path)

    $root = [System.IO.Path]::GetFullPath((Get-PendingSubmissionDirectory)).TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if ($fullPath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $fullPath -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-WithSubmissionGithubRoute {
    param([Parameter(Mandatory = $true)][scriptblock]$Action)

    $names = @('HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'NO_PROXY')
    $saved = @{}
    foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    try {
        # gh honors environment proxies, but does not discover the Windows system proxy.
        if (-not ($saved['HTTPS_PROXY'] -or $saved['HTTP_PROXY'] -or $saved['ALL_PROXY'])) {
            $proxy = Get-SubmissionProxy -ProbeUri ([Uri]'https://api.github.com')
            if ($null -ne $proxy -and -not $proxy.IsBypassed([Uri]'https://api.github.com')) {
                [Environment]::SetEnvironmentVariable('HTTPS_PROXY', $proxy.GetProxy([Uri]'https://api.github.com').AbsoluteUri, 'Process')
            }
        }
        & $Action
    } finally {
        foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
    }
}

function New-Moe4AllSubmissionApiException {
    param(
        [int]$StatusCode,
        [string]$Code,
        [string]$Message
    )
    $safeCode = if ([string]::IsNullOrWhiteSpace($Code)) { 'http_error' } else { $Code }
    $safeMessage = if ([string]::IsNullOrWhiteSpace($Message)) { "HTTP $StatusCode" } else { $Message }
    $exception = [System.Exception]::new("$safeCode`: $safeMessage")
    $exception.Data['StatusCode'] = $StatusCode
    $exception.Data['ApiCode'] = $safeCode
    return $exception
}

function Invoke-Moe4AllSubmissionTransfer {
    param(
        [Parameter(Mandatory = $true)]$Client,
        [Parameter(Mandatory = $true)]$Request,
        [ValidateRange(1, 300)][int]$TimeoutSeconds = 30,
        [int]$Attempt = 1,
        [int]$TotalAttempts = 1
    )

    # Keep the deadline and cancellation checks in the wizard, independently of
    # HttpClient's timer. ResponseContentRead includes the complete response body.
    $cancellation = [System.Threading.CancellationTokenSource]::new()
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $pending = $null
    $completed = $false
    $nextNotice = 5
    try {
        $cancellation.CancelAfter([TimeSpan]::FromSeconds($TimeoutSeconds))
        $pending = $Client.SendAsync($Request, [System.Net.Http.HttpCompletionOption]::ResponseContentRead, $cancellation.Token)
        while (-not $pending.IsCompleted) {
            Assert-WizardOperationActive
            if ($timer.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
                throw [System.TimeoutException]::new('提交服务请求超时；本地文件仍保留。Submission request timed out; the local file is preserved.')
            }
            if ($timer.Elapsed.TotalSeconds -ge $nextNotice) {
                Write-Host ("正在等待上传服务：第 {0}/{1} 次，已等 {2} 秒，上限 {3} 秒。 / Waiting for submission service: attempt {0}/{1}, {2}s elapsed, {3}s limit." -f $Attempt, $TotalAttempts, [int]$timer.Elapsed.TotalSeconds, $TimeoutSeconds) -ForegroundColor DarkGray
                $nextNotice += 5
            }
            Start-Sleep -Milliseconds 100
        }
        Assert-WizardOperationActive
        try {
            $response = $pending.GetAwaiter().GetResult()
        } catch {
            if ($cancellation.IsCancellationRequested) {
                throw [System.TimeoutException]::new('提交服务请求超时；本地文件仍保留。Submission request timed out; the local file is preserved.')
            }
            throw
        }
        $completed = $true
        return $response
    } finally {
        if (-not $completed) {
            $cancellation.Cancel()
            # Also release a response that won the race with cancellation.
            if ($null -ne $pending -and $pending.Status -eq [System.Threading.Tasks.TaskStatus]::RanToCompletion) {
                $pending.GetAwaiter().GetResult().Dispose()
            }
        }
        $timer.Stop()
        $cancellation.Dispose()
    }
}

function Invoke-Moe4AllSubmissionApi {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('GET', 'POST')][string]$Method,
        [Parameter(Mandatory = $true)][Uri]$Uri,
        [AllowEmptyString()][string]$Token = '',
        $JsonBody = $null,
        [byte[]]$FileBytes = $null,
        [ValidateRange(1, 300)][int]$TimeoutSeconds = 30,
        [ValidateRange(0, 3)][int]$MaximumRetries = 3
    )

    Add-Type -AssemblyName System.Net.Http
    for ($attempt = 0; $attempt -le $MaximumRetries; $attempt++) {
        Assert-WizardOperationActive
        $handler = [System.Net.Http.HttpClientHandler]::new()
        $handler.AllowAutoRedirect = $false
        $proxy = Get-SubmissionProxy -ProbeUri $Uri
        if ($null -ne $proxy) { $handler.Proxy = $proxy }
        $client = [System.Net.Http.HttpClient]::new($handler)
        $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
        $client.MaxResponseContentBufferSize = 1048576
        $request = $null
        $response = $null
        $delaySeconds = 0
        $retryReason = ''
        try {
            $httpMethod = if ($Method -eq 'GET') { [System.Net.Http.HttpMethod]::Get } else { [System.Net.Http.HttpMethod]::Post }
            $request = [System.Net.Http.HttpRequestMessage]::new($httpMethod, $Uri)
            [void]$request.Headers.Accept.ParseAdd('application/json')
            [void]$request.Headers.UserAgent.ParseAdd("MoE4All-Wizard/$productVersion")
            if ($Token) { [void]$request.Headers.TryAddWithoutValidation('Authorization', "Bearer $Token") }
            if ($null -ne $FileBytes) {
                $request.Content = [System.Net.Http.ByteArrayContent]::new($FileBytes)
                $request.Content.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse('text/plain; charset=utf-8')
            } elseif ($null -ne $JsonBody) {
                $json = $JsonBody | ConvertTo-Json -Depth 8 -Compress
                $request.Content = [System.Net.Http.StringContent]::new($json, [System.Text.Encoding]::UTF8, 'application/json')
            }
            $response = Invoke-Moe4AllSubmissionTransfer -Client $client -Request $request -TimeoutSeconds $TimeoutSeconds -Attempt ($attempt + 1) -TotalAttempts ($MaximumRetries + 1)
            $status = [int]$response.StatusCode
            $bodyText = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            if ($bodyText.Length -gt 1048576) { throw [System.Exception]::new('提交服务响应超过 1 MiB，已拒绝处理。Submission service response exceeded 1 MiB.') }
            $value = $null
            if (-not [string]::IsNullOrWhiteSpace($bodyText)) {
                try { $value = $bodyText | ConvertFrom-Json } catch {
                    if ($status -ge 200 -and $status -lt 300) { throw (New-Moe4AllSubmissionApiException -StatusCode $status -Code 'invalid_response' -Message '提交服务返回了无效 JSON。The submission service returned invalid JSON.') }
                }
            }
            if ($status -ge 200 -and $status -lt 300) {
                if ($null -eq $value) { throw (New-Moe4AllSubmissionApiException -StatusCode $status -Code 'invalid_response' -Message '提交服务返回了空响应。The submission service returned an empty response.') }
                return [pscustomobject]@{ StatusCode = $status; Value = $value }
            }

            $code = ''
            $message = ''
            $retryable = $status -eq 429 -or $status -ge 500
            if ($null -ne $value -and $null -ne $value.error) {
                $code = [string]$value.error.code
                $message = [string]$value.error.message
                # Format/privacy/conflict 4xx responses must never replay a POST.
                if ($status -ge 500 -and $value.error.PSObject.Properties.Name -contains 'retryable') { $retryable = [bool]$value.error.retryable }
            }
            if (-not $retryable -or $attempt -eq $MaximumRetries) {
                throw (New-Moe4AllSubmissionApiException -StatusCode $status -Code $code -Message $message)
            }
            $delaySeconds = [Math]::Pow(2, $attempt + 1)
            if ($null -ne $response.Headers.RetryAfter) {
                if ($null -ne $response.Headers.RetryAfter.Delta) {
                    $delaySeconds = [Math]::Max(0, $response.Headers.RetryAfter.Delta.TotalSeconds)
                } elseif ($null -ne $response.Headers.RetryAfter.Date) {
                    $delaySeconds = [Math]::Max(0, ($response.Headers.RetryAfter.Date.UtcDateTime - [DateTime]::UtcNow).TotalSeconds)
                }
            }
            if ($delaySeconds -gt 120) { throw (New-Moe4AllSubmissionApiException -StatusCode $status -Code $code -Message '服务要求稍后重试；请保留报告。The service requested a longer wait; keep the report and retry later.') }
            $retryReason = "HTTP $status"
        } catch {
            if (Test-WizardOperationCancelled $_.Exception) { throw }
            $statusData = $_.Exception.Data['StatusCode']
            if ($null -ne $statusData) { throw }
            $cause = $_.Exception
            $transportFailure = $false
            $timedOut = $false
            while ($null -ne $cause) {
                if ($cause -is [System.TimeoutException] -or $cause -is [System.OperationCanceledException]) { $timedOut = $true }
                if ($cause -is [System.Net.Http.HttpRequestException] -or $cause -is [System.Net.WebException] -or $cause -is [System.IO.IOException] -or $cause -is [System.TimeoutException] -or $cause -is [System.OperationCanceledException]) { $transportFailure = $true }
                $cause = $cause.InnerException
            }
            if (-not $transportFailure) { throw }
            if ($attempt -eq $MaximumRetries) {
                $failureCode = if ($timedOut) { 'request_timeout' } else { 'transport_error' }
                throw (New-Moe4AllSubmissionApiException -StatusCode 0 -Code $failureCode -Message '上传服务请求失败，已停止自动重试；本地文件仍保留。Submission request failed; automatic retries stopped and the local file is preserved.')
            }
            $delaySeconds = [Math]::Pow(2, $attempt + 1)
            $retryReason = if ($timedOut) { '请求超时 / Request timed out' } else { '连接失败 / Connection failed' }
        } finally {
            if ($null -ne $response) { $response.Dispose() }
            if ($null -ne $request) { $request.Dispose() }
            $client.Dispose()
            $handler.Dispose()
        }
        # Dispose each attempt before waiting or starting the next connection.
        $delayMilliseconds = [int]($delaySeconds * 1000 + (Get-Random -Minimum 0 -Maximum 501))
        Write-Host ("{0}；{1:N1} 秒后重试（第 {2}/{3} 次）。 / Retrying in {1:N1}s (attempt {2}/{3})." -f $retryReason, ($delayMilliseconds / 1000.0), ($attempt + 2), ($MaximumRetries + 1)) -ForegroundColor Yellow
        Wait-WizardOperation -Milliseconds $delayMilliseconds
    }
    throw '提交服务请求失败。Submission service request failed.'
}

function Start-Moe4AllSubmissionVerification {
    param([Parameter(Mandatory = $true)][Uri]$VerificationUri)
    try {
        Start-Process -FilePath $VerificationUri.AbsoluteUri
        return $true
    } catch {
        return $false
    }
}

function Save-Moe4AllSubmissionReceipt {
    param(
        [Parameter(Mandatory = $true)]$Receipt,
        [Parameter(Mandatory = $true)][Uri]$ApiBase
    )
    $allowedStatuses = @('forwarding', 'pending_review', 'published', 'rejected', 'delivery_failed')
    if ([string]$Receipt.receipt_id -notmatch '^[A-Za-z0-9_-]+$' -or
        [string]::IsNullOrWhiteSpace([string]$Receipt.receipt_token) -or
        [string]$Receipt.status -notin $allowedStatuses) {
        throw '提交服务返回的回执无效。The submission service returned an invalid receipt.'
    }
    $directory = Join-Path $dataDir 'private\submission-receipts'
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $path = Join-Path $directory ("{0}.private-receipt.json" -f [string]$Receipt.receipt_id)
    $saved = [ordered]@{
        api_version = 1
        api_base = $ApiBase.AbsoluteUri.TrimEnd('/')
        receipt_id = [string]$Receipt.receipt_id
        receipt_token = [string]$Receipt.receipt_token
        kind = [string]$Receipt.kind
        report_id = [string]$Receipt.report_id
        file_sha256 = [string]$Receipt.file_sha256
        submitted_file_sha256 = [string]$Receipt.submitted_file_sha256
        payload_sha256 = [string]$Receipt.payload_sha256
        status = [string]$Receipt.status
        duplicate = [bool]$Receipt.duplicate
        received_at = [string]$Receipt.received_at
        updated_at = [string]$Receipt.updated_at
        public_url = [string]$Receipt.public_url
        saved_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    if($Receipt.PSObject.Properties['engine_version'] -and [string]$Receipt.engine_version -match '^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$'){$saved.engine_version=[string]$Receipt.engine_version}
    [System.IO.File]::WriteAllText($path, (($saved | ConvertTo-Json -Depth 5) + "`n"), [System.Text.UTF8Encoding]::new($false))
    return $path
}

function Invoke-WebsiteSubmissionUpload {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][ValidateSet('benchmark', 'error')][string]$Kind
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) { throw "找不到提交文件。Submission file not found: $fullPath" }
    $expectedExtension = if ($Kind -eq 'benchmark') { '.m4b' } else { '.m4e' }
    if ([System.IO.Path]::GetExtension($fullPath) -ine $expectedExtension) { throw "提交文件类型与 $Kind 不匹配。Submission extension does not match $Kind." }
    $verified = Invoke-NativeCommandResult -Executable $infrPath -Arguments @('submission', 'verify', $fullPath)
    if ($verified.ExitCode -ne 0) { throw '提交文件未通过本地校验，未上传。Local submission validation failed; nothing was uploaded.' }
    $payload = Read-MoE4AllSubmissionPayload -Path $fullPath
    $reportId = [string]$payload.id
    if ($reportId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$' -or [System.IO.Path]::GetFileNameWithoutExtension($fullPath) -ne $reportId) {
        throw '提交文件名或报告 ID 无效。Submission filename or report ID is invalid.'
    }
    if ($Kind -eq 'error' -and [string]::IsNullOrWhiteSpace([string]$payload.description)) {
        throw '错误报告必须先填写问题描述。Error reports require a non-empty description before upload.'
    }
    $bytes = [System.IO.File]::ReadAllBytes($fullPath)
    if ($bytes.Length -lt 1 -or $bytes.Length -gt 131072) { throw '提交文件大小必须在 1 到 131072 字节之间。Submission size must be between 1 and 131072 bytes.' }
    try {
        $armorText = [System.Text.UTF8Encoding]::new($false, $true).GetString($bytes)
    } catch {
        throw '提交文件不是有效 UTF-8。Submission file is not valid UTF-8.'
    }
    $kindMatches = [regex]::Matches($armorText, '(?m)^kind: (benchmark|error)\r?$')
    if ($kindMatches.Count -ne 1 -or $kindMatches[0].Groups[1].Value -ne $Kind) {
        throw '提交封装中的类型与所选上传类型不一致。Submission envelope kind does not match the selected upload kind.'
    }
    $fileHash = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $config = Get-Moe4AllSubmissionApiConfig
    $sessionUri = [Uri]::new($config.ApiBase.AbsoluteUri.TrimEnd('/') + '/upload-sessions')

    Write-Host ''
    Write-Host '正在创建安全上传会话... / Creating a secure upload session...' -ForegroundColor Cyan
    $opened = Invoke-Moe4AllSubmissionApi -Method POST -Uri $sessionUri -JsonBody ([ordered]@{
        kind = $Kind
        report_id = $reportId
        file_sha256 = $fileHash
        size_bytes = $bytes.Length
        summary = @(Get-SubmissionReadableSummary -Text $armorText)
    })
    $session = $opened.Value
    if ([int]$session.api_version -ne 1 -or [string]$session.state -ne 'awaiting_verification' -or [string]::IsNullOrWhiteSpace([string]$session.session_id) -or [string]::IsNullOrWhiteSpace([string]$session.session_token)) {
        throw '提交服务返回了无效会话。The submission service returned an invalid session.'
    }
    if ([int64]$session.max_upload_bytes -lt $bytes.Length) { throw '提交服务声明的上传上限小于当前文件。The service upload limit is smaller than this file.' }
    $verificationUri = $null
    if (-not [Uri]::TryCreate([string]$session.verification_url, [UriKind]::Absolute, [ref]$verificationUri)) {
        throw '提交服务返回了无效确认地址。The submission service returned an invalid verification URL.'
    }
    $verificationOrigin = $verificationUri.GetLeftPart([UriPartial]::Authority).TrimEnd('/')
    if ($verificationOrigin -ine $config.SiteOrigin -or $verificationUri.AbsolutePath -ne '/community/upload.html' -or [string]::IsNullOrWhiteSpace($verificationUri.Fragment)) {
        throw '提交确认地址不属于配置的网站来源，已拒绝打开。The verification URL is outside the configured site origin.'
    }
    if (-not (Start-Moe4AllSubmissionVerification -VerificationUri $verificationUri)) {
        Write-Host '无法自动打开浏览器。请只在受信任的本机环境中打开以下一次性确认地址：' -ForegroundColor Yellow
        Write-Host 'Could not open the browser. Open this one-time verification URL only on this trusted machine:' -ForegroundColor Yellow
        Write-Host $verificationUri.AbsoluteUri -ForegroundColor Cyan
    }
    Write-Host '请在浏览器确认公开提交；确认后后端会创建公开 GitHub PR，向导会继续上传。' -ForegroundColor Cyan
    Write-Host 'Confirm the public submission in the browser; the backend will create a public GitHub PR after upload.' -ForegroundColor DarkGray

    $expires = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParse([string]$session.expires_at, [ref]$expires)) { throw '提交会话缺少有效过期时间。Submission session has no valid expiration time.' }
    $pollSeconds = [Math]::Max(3, [Math]::Min(30, [int]$session.poll_after_seconds))
    $sessionId = [Uri]::EscapeDataString([string]$session.session_id)
    $stateUri = [Uri]::new($config.ApiBase.AbsoluteUri.TrimEnd('/') + '/upload-sessions/' + $sessionId)
    while ($true) {
        Assert-WizardOperationActive
        if ([DateTimeOffset]::UtcNow -ge $expires.ToUniversalTime()) { throw '上传会话已过期；本地文件仍完整保留。Upload session expired; the local file remains intact.' }
        $stateResult = Invoke-Moe4AllSubmissionApi -Method GET -Uri $stateUri -Token ([string]$session.session_token)
        if ([int]$stateResult.Value.api_version -ne 1 -or [string]$stateResult.Value.kind -ne $Kind -or [string]$stateResult.Value.report_id -ne $reportId -or [string]$stateResult.Value.file_sha256 -ne $fileHash -or [int64]$stateResult.Value.size_bytes -ne $bytes.Length) {
            throw '上传会话状态与本地文件不匹配。Upload session state does not match the local file.'
        }
        $state = [string]$stateResult.Value.state
        if ($state -in @('ready', 'uploaded')) { break }
        if ($state -eq 'cancelled') {
            Write-Host '已在浏览器取消上传；本地文件仍完整保留。Upload cancelled in the browser; the local file remains intact.' -ForegroundColor Yellow
            return $false
        }
        if ($state -ne 'awaiting_verification') { throw "未知上传会话状态：$state / Unknown upload session state." }
        $pollSeconds = [Math]::Max(3, [Math]::Min(30, [int]$stateResult.Value.poll_after_seconds))
        for ($pollTick=0; $pollTick -lt $pollSeconds*5; $pollTick++) { Assert-WizardOperationActive; Start-Sleep -Milliseconds 200 }
    }

    Write-Host '确认完成，正在上传原始提交文件... / Confirmed; uploading the original submission bytes...' -ForegroundColor Cyan
    $reportUri = [Uri]::new($stateUri.AbsoluteUri + '/report')
    $uploaded = Invoke-Moe4AllSubmissionApi -Method POST -Uri $reportUri -Token ([string]$session.session_token) -FileBytes $bytes
    $receipt = $uploaded.Value
    $allowedStatuses = @('forwarding', 'pending_review', 'published', 'rejected', 'delivery_failed')
    $submittedHash = if (-not [string]::IsNullOrWhiteSpace([string]$receipt.submitted_file_sha256)) {
        [string]$receipt.submitted_file_sha256
    } else {
        [string]$receipt.file_sha256
    }
    if ([int]$receipt.api_version -ne 1 -or [string]$receipt.kind -ne $Kind -or [string]$receipt.report_id -ne $reportId -or $submittedHash -ne $fileHash -or [string]$receipt.status -notin $allowedStatuses) {
        throw '提交回执与本地文件不匹配。Submission receipt does not match the local file.'
    }
    $reportVersion=[string]$payload.engine_version
    if ($Kind -eq 'benchmark' -and $reportVersion -match '^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$') { $receipt | Add-Member -NotePropertyName engine_version -NotePropertyValue $reportVersion -Force }
    $receiptPath = Save-Moe4AllSubmissionReceipt -Receipt $receipt -ApiBase $config.ApiBase
    if (Test-BenchmarkBodyReceipt $receipt) { Set-BenchmarkSubmissionSucceeded -Kind $Kind -ReportVersion $reportVersion }
    if ([string]$receipt.status -eq 'delivery_failed') {
        Write-Warning '该报告此前转发失败，正文已清理；请重新导出新报告。This report previously failed delivery; export a new report to try again.'
    } elseif ([string]$receipt.status -eq 'rejected') {
        Write-Warning '该报告的 PR 已退回。The report PR was closed without merging.'
    } else { Write-Host ("已收到，状态：{0} / Received, status: {0}" -f [string]$receipt.status) -ForegroundColor Green }
    if (-not [string]::IsNullOrWhiteSpace([string]$receipt.public_url)) {
        Write-Host ("公开地址 / Public URL: {0}" -f [string]$receipt.public_url) -ForegroundColor Cyan
    } else {
        Write-Host '需要查看进度时，在网站“我的回执”导入下方保存的私密回执文件。' -ForegroundColor DarkGray
        Write-Host 'To check progress, import the saved private receipt under My receipts on the website.' -ForegroundColor DarkGray
    }
    Write-Host ("Receipt ID: {0}" -f [string]$receipt.receipt_id) -ForegroundColor Cyan
    Write-Host "Private receipt: $receiptPath" -ForegroundColor DarkGray
    return $true
}

function Get-SubmissionReadableSummary {
    param([Parameter(Mandatory = $true)][string]$Text)
    foreach ($line in @([regex]::Matches($Text, '(?m)^summary-[0-9]{3}: (.*)\r?$') | Select-Object -First 8)) {
        $value = $line.Groups[1].Value.TrimEnd("`r")
        $end = 0; $count = 0
        while ($end -lt $value.Length -and $count -lt 500) {
            if ([char]::IsHighSurrogate($value[$end]) -and $end + 1 -lt $value.Length -and [char]::IsLowSurrogate($value[$end + 1])) { $end++ }
            $end++; $count++
        }
        $value.Substring(0, $end)
    }
}


function Get-ErrorSubmissionSnapshot {
    $root = Join-Path (Join-Path $repoRoot 'diagnostics') 'reports'
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { return @() }
    return @(Get-ChildItem -LiteralPath $root -Filter '*.m4e' -File -Recurse -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
}

function Read-ProblemDescription {
    param([AllowEmptyString()][string]$Initial = '')

    $description = $Initial
    $mode = if ($description) { 'append' } else { 'rewrite' }
    while ($true) {
        if ($mode -eq 'rewrite') { $description = '' }
        Write-Host ''
        Write-Host '请输入问题描述；可输入多行，单独输入一个句点 . 才结束编辑。' -ForegroundColor Cyan
        Write-Host 'Describe the problem. Multiple lines are allowed; enter a single period . to finish editing.' -ForegroundColor Cyan
        $lines = [System.Collections.Generic.List[string]]::new()
        while ($true) {
            $line = Read-Host
            if ($line -eq '.') { break }
            [void]$lines.Add($line)
        }
        $added = ($lines -join "`n").Trim()
        if ($mode -eq 'append' -and $description -and $added) {
            $description = ($description.TrimEnd() + "`n" + $added)
        } elseif ($added -or $mode -eq 'rewrite') {
            $description = $added
        }

        Write-Host "`n问题描述预览 / Description preview" -ForegroundColor Green
        if ($description) { Write-Host $description } else { Write-Host '(空 / empty)' -ForegroundColor DarkGray }
        $decision = Read-Choice -Label '确认这段描述？/ Confirm this description?' -DefaultValue 'confirm' -Options @(
            [pscustomobject]@{ Key = '1'; Value = 'confirm'; Label = '确认并生成报告 / Confirm and create report' }
            [pscustomobject]@{ Key = '2'; Value = 'rewrite'; Label = '全部重写 / Rewrite all' }
            [pscustomobject]@{ Key = '3'; Value = 'append'; Label = '继续补充 / Append' }
            [pscustomobject]@{ Key = '4'; Value = 'cancel'; Label = '取消 / Cancel' }
        )
        switch ($decision) {
            'confirm' {
                if ([string]::IsNullOrWhiteSpace($description)) {
                    Write-Host '问题描述不能为空。The problem description cannot be empty.' -ForegroundColor Yellow
                    $mode = 'rewrite'
                    continue
                }
                if ((Get-UnicodeCodePointCount -Value $description) -gt 4096) {
                    Write-Host '问题描述不能超过 4096 个 Unicode 字符。The problem description cannot exceed 4096 Unicode code points.' -ForegroundColor Yellow
                    $mode = 'rewrite'
                    continue
                }
                return $description
            }
            'rewrite' { $mode = 'rewrite' }
            'append' { $mode = 'append' }
            'cancel' { return $null }
        }
    }
}

function Save-ErrorUploadPreference {
    if ($DryRun) { return }
    New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
    $value = if (Test-Path -LiteralPath $statePath -PathType Leaf) {
        try { [System.IO.File]::ReadAllText($statePath) | ConvertFrom-Json } catch { [pscustomobject]@{} }
    } else {
        [pscustomobject]@{}
    }
    if ($value.PSObject.Properties.Name -contains 'error_upload') {
        $value.error_upload = $script:ErrorUploadDefault
    } else {
        $value | Add-Member -NotePropertyName 'error_upload' -NotePropertyValue $script:ErrorUploadDefault
    }
    [System.IO.File]::WriteAllText($statePath, (($value | ConvertTo-Json -Depth 20) + "`n"), [System.Text.UTF8Encoding]::new($true))
}

function Complete-ErrorReport {
    param([AllowEmptyString()][string]$Source = '')

    $description = Read-ProblemDescription
    if ($null -eq $description) {
        Write-Host '已取消；已有诊断文件不会删除。Cancelled; existing diagnostics were not deleted.' -ForegroundColor Yellow
        return $null
    }
    New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
    $descriptionPath = Join-Path $dataDir ("error-description-{0}.txt" -f [Guid]::NewGuid().ToString('N'))
    try {
        [System.IO.File]::WriteAllText($descriptionPath, $description, [System.Text.UTF8Encoding]::new($false))
        $reportArgs = [System.Collections.Generic.List[string]]::new()
        [void]$reportArgs.Add('report')
        if ($Source) {
            [void]$reportArgs.Add('--source')
            [void]$reportArgs.Add($Source)
        } else {
            [void]$reportArgs.Add('--diagnostics-dir')
            [void]$reportArgs.Add((Join-Path $repoRoot 'diagnostics'))
        }
        [void]$reportArgs.Add('--description-file')
        [void]$reportArgs.Add($descriptionPath)
        $created = Invoke-NativeCommandResult -Executable $infrPath -Arguments $reportArgs.ToArray()
        if ($created.ExitCode -ne 0) { throw "error report creation failed: $($created.Output -join ' ')" }
        $marker = @($created.Output | Where-Object { $_ -like 'MOE4ALL_SUBMISSION_PATH=*' } | Select-Object -Last 1)
        if ($marker.Count -ne 1) { throw 'error report creation did not return a result path' }
        $path = [string]$marker[0].Substring('MOE4ALL_SUBMISSION_PATH='.Length)
        Write-Host "`n错误报告已生成 / Error report created: $path" -ForegroundColor Green
        $upload = Read-YesNo -Label '愿意公开提交这份错误报告吗？/ Submit this error report publicly?' -Default $script:ErrorUploadDefault
        $script:ErrorUploadDefault = $upload
        Save-ErrorUploadPreference
        if ($upload) { [void](Offer-SubmissionUpload -Path $path -Kind 'error' -ConsentAlreadyGranted) }
        return $path
    } finally {
        Remove-Item -LiteralPath $descriptionPath -Force -ErrorAction SilentlyContinue
    }
}

function Read-SubmissionNextAction {
    param([switch]$Browser)
    if ($Browser) {
        # A remembered "sent" must never confirm a different report by pressing Enter.
        $default = $script:SubmissionBrowserActionDefault
        if ($default -notin @('reopen', 'cloudflare', 'later')) { $default = 'later' }
        $value = Read-Choice -Label 'GitHub 网页提交 / GitHub web submission' -DefaultValue $default -Options @(
            [pscustomobject]@{ Key = '1'; Value = 'sent'; Label = '已成功提交 / Successfully submitted' }
            [pscustomobject]@{ Key = '2'; Value = 'reopen'; Label = '重新打开 GitHub / Reopen GitHub' }
            [pscustomobject]@{ Key = '3'; Value = 'cloudflare'; Label = '改用 Cloudflare / Use Cloudflare instead' }
            [pscustomobject]@{ Key = '4'; Value = 'later'; Label = '下次再发 / Send later' }
        )
        $script:SubmissionBrowserActionDefault = $value
    } else {
        $default = $script:SubmissionFailureActionDefault
        if ($default -notin @('retry', 'channel', 'later')) { $default = 'later' }
        $value = Read-Choice -Label 'Cloudflare 上传未完成 / Cloudflare upload incomplete' -DefaultValue $default -Options @(
            [pscustomobject]@{ Key = '1'; Value = 'retry'; Label = '重试 Cloudflare / Retry Cloudflare' }
            [pscustomobject]@{ Key = '2'; Value = 'channel'; Label = '返回渠道选择 / Return to channel selection' }
            [pscustomobject]@{ Key = '3'; Value = 'later'; Label = '下次再发 / Send later' }
        )
        $script:SubmissionFailureActionDefault = $value
    }
    Save-BenchmarkUploadPreferences
    return $value
}

function Offer-SubmissionUpload {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [ValidateSet('benchmark', 'error')][string]$Kind = 'benchmark',
        [switch]$ConsentAlreadyGranted
    )

    try { $submission = Test-SubmissionForUpload -Path $Path -Kind $Kind } catch { Write-Warning $_.Exception.Message; return $false }
    if (-not $ConsentAlreadyGranted) {
        if ($Kind -eq 'error') {
            $consent = Read-YesNo -Label '愿意公开提交这份错误报告吗？/ Submit this error report publicly?' -Default $script:ErrorUploadDefault
            $script:ErrorUploadDefault = $consent
            Save-ErrorUploadPreference
        } else { $consent = Read-BenchmarkUploadConsent }
        if (-not $consent) { return $false }
    }
    try { $pendingPath = Save-PendingSubmission -Path $Path -Kind $Kind -Quiet } catch { Write-Warning $_.Exception.Message; return $false }
    $channel = 'choose'
    while ($true) {
        if ($channel -eq 'choose') {
            $hasGithub = Read-YesNo -Label '你有 GitHub 账号吗？/ Do you have a GitHub account?' -Default $script:BenchmarkUploadHasGithubDefault
            $script:BenchmarkUploadHasGithubDefault = $hasGithub
            Save-BenchmarkUploadPreferences
            $channel = if ($hasGithub) { 'github' } else { 'cloudflare' }
        }
        if ($channel -eq 'github') {
            $script:SubmissionUploadCancelled = $false
            $script:SubmissionValidationFailed = $false
            $script:SubmissionGhConfirmedHash = ''
            $gh = Get-Command gh -ErrorAction SilentlyContinue
            $ghAuthenticated = $false
            if ($null -ne $gh) {
                try { $ghAuthenticated = Invoke-WithSubmissionGithubRoute -Action { Test-GhAuthenticated -GhCommand $gh } } catch { }
            }
            if ($ghAuthenticated) {
                for ($attempt = 1; $attempt -le 2; $attempt++) {
                    try {
                        $gitUploaded = Invoke-WithSubmissionGithubRoute -Action {
                            Write-Host "GitHub CLI 上传 / GitHub CLI upload: $attempt/2"
                            return (Invoke-GhSubmissionUpload -Path $Path -Kind $Kind -Retry:($attempt -gt 1))
                        }
                        if ($gitUploaded) { Remove-PendingSubmission -Path $pendingPath; return $true }
                    } catch { if (Test-WizardOperationCancelled $_.Exception) { throw }; Write-Warning "GitHub CLI 上传失败：$($_.Exception.Message) / GitHub CLI upload failed." }
                    if ($script:SubmissionUploadCancelled -or $script:SubmissionValidationFailed) { return $false }
                }
            }
            Write-Host '没有可用的已登录 CLI 或上传未完成，改用 GitHub 网页。No signed-in CLI is available or CLI upload did not complete; opening GitHub.' -ForegroundColor Cyan
            $channel = 'browser'
        }
        if ($channel -eq 'browser') {
            try { Open-SubmissionBrowserUpload -Path $Path -Kind $Kind } catch {
                if (Test-WizardOperationCancelled $_.Exception) { throw }
                Write-Warning "未能自动打开 GitHub，请保留本地报告。Could not open GitHub: $($_.Exception.Message)"
            }
            Write-Host '完成网页操作后再选择；向导不会查询网页提交结果。Choose after completing the web steps; the wizard does not query their result.'
            $action = Read-SubmissionNextAction -Browser
            if ($action -eq 'sent') { Set-BenchmarkSubmissionSucceeded -Kind $Kind -ReportVersion ([string]$submission.engine_version); Remove-PendingSubmission -Path $pendingPath; return $true }
            if ($action -eq 'reopen') { continue }
            if ($action -eq 'cloudflare') { $channel = 'cloudflare' } else { break }
        }
        if ($channel -eq 'cloudflare') {
            try {
                if (Invoke-WebsiteSubmissionUpload -Path $Path -Kind $Kind) { Remove-PendingSubmission -Path $pendingPath; return $true }
            } catch { if (Test-WizardOperationCancelled $_.Exception) { throw }; Write-Warning "Cloudflare 中转失败：$($_.Exception.Message) / Cloudflare relay failed." }
            $action = Read-SubmissionNextAction
            if ($action -eq 'retry') { continue }
            if ($action -eq 'channel') { $channel = 'choose'; continue }
            break
        }
    }
    Write-Host "提交暂存，待下次启动重试 / Submission saved for a later retry: $pendingPath" -ForegroundColor Yellow
    return $false
}

function Offer-BenchmarkUpload {
    param([Parameter(Mandatory = $true)][string]$Path, [switch]$ConsentAlreadyGranted)
    Offer-SubmissionUpload -Path $Path -Kind 'benchmark' -ConsentAlreadyGranted:$ConsentAlreadyGranted
}

function Invoke-BenchmarkUploadSafely {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$ConsentAlreadyGranted
    )

    try {
        [void](Offer-BenchmarkUpload -Path $Path -ConsentAlreadyGranted:$ConsentAlreadyGranted)
    } catch {
        Write-Warning "Benchmark 已完成，但上传辅助流程失败：$($_.Exception.Message) / Benchmark completed, but the upload helper failed."
    }
}

function Open-BenchmarkLeaderboard {
    $browser = [Diagnostics.ProcessStartInfo]::new()
    $browser.FileName = 'https://moe4all.com/community/'
    $browser.UseShellExecute = $true
    [void][Diagnostics.Process]::Start($browser)
}

function Offer-BenchmarkLeaderboard {
    try {
        if ($DesktopExecutionPlanPath -and $env:MOE4ALL_DESKTOP_IO -eq '1') {
            $open = (Read-WizardRuntimeResponse -Prompt '要打开性能排行榜看看吗？/ Would you like to see the benchmark leaderboard?' -InputType 'confirm' -DefaultValue 'n' -Title 'Benchmark · 排行榜 / Benchmark · Leaderboard' -AffirmativeText '打开排行榜 / Open leaderboard' -NegativeText '不了，结束 / No, finish') -eq 'y'
        } else {
            $open = Read-YesNo -Label '要打开性能排行榜看看吗？/ Would you like to see the benchmark leaderboard?' -Default $false
        }
        if ($open) { Open-BenchmarkLeaderboard }
    } catch {
        Write-Warning ('排行榜未打开；测试结果已保留。The leaderboard did not open; benchmark results are retained. ' + $_.Exception.Message)
    }
}

function Offer-PendingSubmissions {
    param([switch]$ExplicitAction)
    if ($script:PendingSubmissionPrompted -and -not $ExplicitAction) { return }
    $pending = @(Get-PendingSubmissions)
    if ($pending.Count -eq 0) { $script:PendingSubmissionPrompted = $true; return }
    $script:PendingSubmissionPrompted = $true
    $default = if ($null -eq $script:SubmissionPendingRetryDefault) { $true } else { [bool]$script:SubmissionPendingRetryDefault }
    $retry = Read-YesNo -Label "发现 $($pending.Count) 份待发报告，当前重试？/ Found $($pending.Count) pending report(s). Retry now?" -Default $default
    $script:SubmissionPendingRetryDefault = $retry
    Save-BenchmarkUploadPreferences
    if (-not $retry) { return }
    foreach ($item in $pending) {
        $kind = if ($item.Extension -ieq '.m4e') { 'error' } else { 'benchmark' }
        if (Offer-SubmissionUpload -Path $item.FullName -Kind $kind -ConsentAlreadyGranted) {
            Remove-PendingSubmission -Path $item.FullName
        }
    }
}


function Select-EmbeddingModelPath {
    param([AllowEmptyString()][string]$Default = '')

    if (-not $Default) { Show-RecommendedModelDownloads -Kind 'embedding' }
    while ($true) {
        $inputPath = Read-TextValue -Label 'Embedding GGUF 文件或目录，D 自动下载，R 查看推荐 / GGUF file or directory, D to download, R for recommendations' -Default $Default -Required
        if ($inputPath -match '^(d|download|下载)$') {
            try { return Invoke-RecommendedModelDownload -Id 'qwen3-embedding-06b-q8' } catch { Write-Host $_.Exception.Message -ForegroundColor Yellow; continue }
        }
        if ($inputPath -match '^(r|recommended|推荐)$') { Show-RecommendedModelDownloads -Kind 'embedding'; continue }
        try {
            $path = ConvertTo-FullPath $inputPath
            if (Test-Path -LiteralPath $path -PathType Leaf) {
                if ([System.IO.Path]::GetExtension($path) -ine '.gguf') {
                    throw '文件扩展名不是 .gguf。The file extension is not .gguf.'
                }
                return $path
            }
            if (Test-Path -LiteralPath $path -PathType Container) {
                $models = @(Get-ChildItem -LiteralPath $path -File | Where-Object {
                    $_.Extension -ieq '.gguf'
                } | Sort-Object Name)
                if ($models.Count -eq 1) {
                    return $models[0].FullName
                }
                if ($models.Count -eq 0) {
                    throw '目录中没有 GGUF 文件。The directory contains no GGUF file.'
                }
                throw "目录中有 $($models.Count) 个 GGUF 文件，请输入具体文件路径。The directory contains $($models.Count) GGUF files; enter the exact file path."
            }
            throw '路径不存在。The path does not exist.'
        } catch {
            Write-Host $_.Exception.Message -ForegroundColor Yellow
            $Default = ''
        }
    }
}

function Read-ServerModules {
    param([bool]$LlmEnabled, [bool]$EmbeddingEnabled, [bool]$TranscriptionEnabled, [bool]$SpeechEnabled = $false)
    while ($true) {
        $LlmEnabled = Read-YesNo -Label '启用大语言模型 API？/ Enable the language model API?' -Default $LlmEnabled
        $EmbeddingEnabled = Read-YesNo -Label '启用 Embedding API？/ Enable the Embedding API?' -Default $EmbeddingEnabled
        $TranscriptionEnabled = Read-YesNo -Label '启用本地语音转录 API？/ Enable the local transcription API?' -Default $TranscriptionEnabled
        $SpeechEnabled = Read-YesNo -Label '启用本地语音合成 API？/ Enable the local speech synthesis API?' -Default $SpeechEnabled
        if ($LlmEnabled -or $EmbeddingEnabled -or $TranscriptionEnabled -or $SpeechEnabled) {
            return [pscustomobject]@{ Llm = $LlmEnabled; Embedding = $EmbeddingEnabled; Transcription = $TranscriptionEnabled; Speech = $SpeechEnabled }
        }
        Write-Host '请至少启用一个 API 模块。/ Enable at least one API module.' -ForegroundColor Yellow
    }
}

function Test-SavedWizardModels {
    $isServer = (Get-SavedValue 'launch_mode' 'chat') -eq 'server'
    $paths = @()
    if (-not $isServer -or ([string](Get-SavedValue 'server_llm_enabled' $true) -ieq 'true')) {
        $paths += [string](Get-SavedValue 'model' '')
    }
    if ($isServer) {
        if ([string](Get-SavedValue 'server_embedding' $false) -ieq 'true') { $paths += [string](Get-SavedValue 'embedding_model' '') }
        if ([string](Get-SavedValue 'server_transcription' $false) -ieq 'true') { $paths += [string](Get-SavedValue 'transcription_model' '') }
        if ([string](Get-SavedValue 'server_speech' $false) -ieq 'true') {
            try { Add-SpeechArguments -Arguments ([Collections.Generic.List[string]]::new()) -Settings (Get-SpeechSettings) } catch { return $false }
            $paths += [string](Get-SavedValue 'speech_model' '')
        }
    }
    if ($paths.Count -eq 0) { return $false }
    foreach ($path in $paths) {
        if (-not $path -or -not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
    }
    return $true
}

function Restore-DisabledLlmState {
    param([System.Collections.IDictionary]$State)
    # Preserve exact saved values, including legacy settings normalized during initialization.
    $updatedKeys = @(
        'launch_mode', 'device', 'server_llm_enabled', 'server_addr', 'server_auth',
        'server_embedding', 'embedding_model', 'embedding_idle_timeout', 'server_transcription',
        'transcription_model', 'transcription_threads', 'transcription_idle_timeout', 'transcription_upload_dir',
        'server_speech', 'speech_model', 'speech_lexicon', 'speech_threads', 'speech_idle_timeout', 'speech_max_output_seconds',
        'benchmark_upload', 'benchmark_upload_has_github', 'benchmark_upload_gh_auth', 'benchmark_upload_fallback',
        'submission_browser_action', 'submission_failure_action', 'submission_pending_retry', 'error_upload', 'last_command',
        'api_key', 'server_api_key'
    )
    if ($null -eq $script:Saved) { return }
    foreach ($property in $script:Saved.PSObject.Properties) {
        # Retain legacy/unknown LLM fields too, for a later LLM-enabled launch.
        if ($property.Name -notin $updatedKeys) { $State[$property.Name] = $property.Value }
    }
}

function Select-TranscriptionModelPath {
    param([AllowEmptyString()][string]$Default = '')

    if (-not $Default) { Show-RecommendedModelDownloads -Kind 'audio' }
    while ($true) {
        $inputPath = Read-TextValue -Label '语音 GGUF 文件或目录，D 自动下载，R 查看推荐 / Audio GGUF file or directory, D to download, R for recommendations' -Default $Default -Required
        if ($inputPath -match '^(d|download|下载)$') {
            try { return Invoke-RecommendedModelDownload -Id 'sensevoice-small-q8' } catch { Write-Host $_.Exception.Message -ForegroundColor Yellow; continue }
        }
        if ($inputPath -match '^(r|recommended|推荐)$') { Show-RecommendedModelDownloads -Kind 'audio'; continue }
        try {
            $path = ConvertTo-FullPath $inputPath
            if (Test-Path -LiteralPath $path -PathType Container) {
                $models = @(Get-ChildItem -LiteralPath $path -File | Where-Object { $_.Extension -ieq '.gguf' })
                if ($models.Count -ne 1) { throw '请输入准确的 GGUF 文件路径。Enter the exact GGUF file path.' }
                $path = $models[0].FullName
            }
            if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or [IO.Path]::GetExtension($path) -ine '.gguf') {
                throw '未找到 GGUF 文件。GGUF file not found.'
            }
            return $path
        } catch { Write-Host $_.Exception.Message -ForegroundColor Yellow; $Default = '' }
    }
}

function Get-TranscriptionSettings {
    # Do not cast a saved string "false" to [bool]: PowerShell treats it as true.
    return [pscustomobject]@{
        Enabled = ([string](Get-SavedValue 'server_transcription' $false) -ieq 'true')
        Model = [string](Get-SavedValue 'transcription_model' '')
        Threads = [string](Get-SavedValue 'transcription_threads' '4')
        IdleTimeout = [string](Get-SavedValue 'transcription_idle_timeout' '300')
        UploadDir = [string](Get-SavedValue 'transcription_upload_dir' (Join-Path $dataDir 'audio-uploads'))
    }
}

function Read-TranscriptionSettings {
    param($Settings)
    if (-not $Settings.Enabled) { return }
    Write-Host '语音转录固定使用 CPU，默认 4 线程；模型载入后约占 300 MiB RAM。/ Transcription uses the CPU with 4 threads by default; the loaded model uses about 300 MiB RAM.' -ForegroundColor DarkGray
    Write-Host '语音模型在首次请求时载入；不额外占用 GPU 权重，无需外部 runtime。/ The transcription model loads on the first request; no extra GPU weights or external runtime are needed.' -ForegroundColor DarkGray
    Write-Host '首版支持 WAV/PCM；不支持全格式或真正逐 token 流式。/ First release: WAV/PCM; no all-format support or true token-by-token streaming.' -ForegroundColor DarkGray
    $Settings.Model = Select-TranscriptionModelPath -Default $Settings.Model
    $Settings.Threads = Read-IntegerValue -Label '语音 CPU 线程数 / Transcription CPU threads' -Default $Settings.Threads -Minimum 1 -Maximum 256
    $Settings.IdleTimeout = Read-IntegerValue -Label '语音空闲释放秒数，0 为服务期间常驻 / Transcription idle eviction seconds, 0 keeps resident' -Default $Settings.IdleTimeout -Minimum 0 -Maximum 999999999
    $Settings.UploadDir = ConvertTo-FullPath (Read-TextValue -Label '语音上传目录 / Transcription upload directory' -Default $Settings.UploadDir -Required)
}

function Add-TranscriptionArguments {
    param([System.Collections.Generic.List[string]]$Arguments, $Settings)
    if (-not $Settings.Enabled) { return }
    $path = ConvertTo-FullPath $Settings.Model
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or [IO.Path]::GetExtension($path) -ine '.gguf') {
        throw '未找到语音 GGUF 文件。Transcription GGUF file not found.'
    }
    $threadCount = 0; $idleSeconds = 0
    if (-not [int]::TryParse($Settings.Threads, [ref]$threadCount) -or $threadCount -lt 1 -or $threadCount -gt 256 -or
        -not [int]::TryParse($Settings.IdleTimeout, [ref]$idleSeconds) -or $idleSeconds -lt 0 -or $idleSeconds -gt 999999999 -or
        [string]::IsNullOrWhiteSpace($Settings.UploadDir)) { throw '语音参数无效。Invalid transcription settings.' }
    $Settings.Model = $path
    $Settings.UploadDir = ConvertTo-FullPath $Settings.UploadDir
    foreach ($value in @('--transcription-model', $path, '--transcription-dev', 'cpu', '--transcription-threads', [string]$threadCount,
        '--transcription-idle-timeout', [string]$idleSeconds, '--transcription-upload-dir', $Settings.UploadDir)) { [void]$Arguments.Add($value) }
}

function Get-SpeechSettings {
    return [pscustomobject]@{
        Enabled = ([string](Get-SavedValue 'server_speech' $false) -ieq 'true')
        Model = [string](Get-SavedValue 'speech_model' '')
        Lexicon = [string](Get-SavedValue 'speech_lexicon' '')
        Threads = [string](Get-SavedValue 'speech_threads' '4')
        IdleTimeout = [string](Get-SavedValue 'speech_idle_timeout' '300')
        MaxOutputSeconds = [string](Get-SavedValue 'speech_max_output_seconds' '30')
    }
}

function Read-SpeechSettings {
    param($Settings)
    if (-not $Settings.Enabled) { return }
    Write-Host '语音合成使用独立 CPU 服务，首次请求加载，空闲后释放；需要支持 TTS 的引擎。/ Speech synthesis uses an independent lazy CPU service and requires a TTS-enabled build.' -ForegroundColor DarkGray
    Write-Host '选择配套的原生 Melo GGUF 与资源目录；输入 D 查看本版本安装与转换说明。/ Select matching native Melo assets; enter D for this version''s setup and conversion guide.' -ForegroundColor DarkGray
    Write-Host '输出 WAV/PCM 完整片段；音色、语速和格式由 API 请求指定。/ Returns complete WAV/PCM clips; voice, speed and format are set per API request.' -ForegroundColor DarkGray
    while ($true) {
        $modelInput = Read-TextValue -Label '合成 GGUF 文件 / Speech GGUF file' -Default $Settings.Model -Required
        if ($modelInput -match '^(d|download|下载)$') {
            [void](Select-RecommendedDownloads -Kind speech)
            continue
        }
        $Settings.Model = $modelInput
        $Settings.Lexicon = Read-TextValue -Label '合成资源目录，含 lexicon.txt、tokens.txt、config.json / Speech asset directory' -Default $Settings.Lexicon -Required
        $Settings.Threads = Read-IntegerValue -Label '合成 CPU 线程数 / Speech CPU threads' -Default $Settings.Threads -Minimum 1 -Maximum 256
        $Settings.IdleTimeout = Read-TextValue -Label '合成空闲释放秒数，0 为常驻 / Speech idle seconds, 0 keeps resident' -Default $Settings.IdleTimeout -Required
        $Settings.MaxOutputSeconds = Read-TextValue -Label '每段合成时长上限，1-120 秒 / Speech clip duration cap, 1-120 seconds' -Default $Settings.MaxOutputSeconds -Required
        try { Add-SpeechArguments -Arguments ([Collections.Generic.List[string]]::new()) -Settings $Settings; return }
        catch { Write-Host $_.Exception.Message -ForegroundColor Yellow }
    }
}

function Add-SpeechArguments {
    param([System.Collections.Generic.List[string]]$Arguments, $Settings)
    if (-not $Settings.Enabled) { return }
    foreach ($value in @($Settings.Model, $Settings.Lexicon)) {
        if ([string]::IsNullOrWhiteSpace([string]$value) -or [string]$value -match '[\x00-\x1f\x7f]') { throw 'Speech requires a model file and asset directory without control characters.' }
    }
    $model = ConvertTo-FullPath $Settings.Model
    $lexicon = ConvertTo-FullPath $Settings.Lexicon
    if (-not (Test-Path -LiteralPath $model -PathType Leaf) -or [IO.Path]::GetExtension($model) -ine '.gguf') { throw 'Speech requires a native Melo GGUF file.' }
    if (-not (Test-Path -LiteralPath $lexicon -PathType Container)) { throw 'Speech assets must be a directory containing lexicon.txt, tokens.txt and config.json.' }
    foreach ($asset in @('lexicon.txt', 'tokens.txt', 'config.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $lexicon $asset) -PathType Leaf)) { throw "Speech asset directory is missing $asset" }
    }
    $threadCount = 0; $idleSeconds = [UInt64]0; $seconds = [single]0
    if ([string]$Settings.Threads -notmatch '^\d+$' -or -not [int]::TryParse([string]$Settings.Threads, [ref]$threadCount) -or $threadCount -lt 1 -or $threadCount -gt 256) { throw 'Speech CPU threads must be in 1..256.' }
    if ([string]$Settings.IdleTimeout -notmatch '^\d+$' -or -not [UInt64]::TryParse([string]$Settings.IdleTimeout, [ref]$idleSeconds)) { throw 'Speech idle timeout must be unsigned whole seconds; 0 keeps resident.' }
    if (-not [single]::TryParse([string]$Settings.MaxOutputSeconds, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$seconds) -or [single]::IsNaN($seconds) -or [single]::IsInfinity($seconds) -or $seconds -lt 1 -or $seconds -gt 120) { throw 'Speech output duration must be in 1..120 seconds.' }
    $Settings.Model = $model; $Settings.Lexicon = $lexicon
    foreach ($value in @('--speech-model', $model, '--speech-lexicon', $lexicon, '--speech-threads', [string]$threadCount,
        '--speech-idle-timeout', $idleSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
        '--speech-max-output-seconds', $seconds.ToString('R', [Globalization.CultureInfo]::InvariantCulture))) { [void]$Arguments.Add($value) }
}

function Find-VisionProjectorPath {
    param([Parameter(Mandatory = $true)][string]$ModelPath)

    try {
        $directory = Split-Path -Parent (ConvertTo-FullPath $ModelPath)
        $projectors = @(Get-ChildItem -LiteralPath $directory -File | Where-Object {
            $_.Extension -ieq '.gguf' -and $_.Name -like 'mmproj*'
        } | Sort-Object Name)
        if ($projectors.Count -eq 1) {
            return $projectors[0].FullName
        }
    } catch {
        # Discovery is only a default; the explicit prompt below remains authoritative.
    }
    return ''
}

function Select-VisionProjectorPath {
    param([AllowEmptyString()][string]$Default = '')

    if (-not $Default) { Show-RecommendedModelDownloads -Kind 'vision' }
    while ($true) {
        $inputPath = Read-TextValue -Label '视觉 mmproj GGUF 文件或目录，D 自动下载，R 查看推荐 / Vision mmproj GGUF file or directory, D to download, R for recommendations' -Default $Default -Required
        if ($inputPath -match '^(d|download|下载)$') {
            try { return Invoke-RecommendedModelDownload -Id 'flashnext-vision' } catch { Write-Host $_.Exception.Message -ForegroundColor Yellow; continue }
        }
        if ($inputPath -match '^(r|recommended|推荐)$') { Show-RecommendedModelDownloads -Kind 'vision'; continue }
        try {
            $path = ConvertTo-FullPath $inputPath
            if (Test-Path -LiteralPath $path -PathType Leaf) {
                if ([System.IO.Path]::GetExtension($path) -ine '.gguf') {
                    throw '文件扩展名不是 .gguf。The file extension is not .gguf.'
                }
                return $path
            }
            if (Test-Path -LiteralPath $path -PathType Container) {
                $projectors = @(Get-ChildItem -LiteralPath $path -File | Where-Object {
                    $_.Extension -ieq '.gguf' -and $_.Name -like 'mmproj*'
                } | Sort-Object Name)
                if ($projectors.Count -eq 1) {
                    return $projectors[0].FullName
                }
                if ($projectors.Count -eq 0) {
                    throw '目录中没有 mmproj*.gguf。The directory contains no mmproj*.gguf file.'
                }
                throw "目录中有 $($projectors.Count) 个 mmproj GGUF，请输入具体文件路径。The directory contains $($projectors.Count) mmproj GGUF files; enter the exact file path."
            }
            throw '路径不存在。The path does not exist.'
        } catch {
            Write-Host $_.Exception.Message -ForegroundColor Yellow
            $Default = ''
        }
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

# The refusal-direction control vector `[uncensor] vector` projects. Pinned by repository, file name
# AND sha256 on purpose: the hub also carries a same-named look-alike with DIFFERENT numbers, and a
# wrong direction is exactly the kind of mistake nobody notices until the model behaves oddly. A
# function rather than script variables because the wizard's own tests load function definitions
# one at a time, where a script-scope assignment would never have run.
function Get-UncensorVectorSpec {
    return @{
        Repo = 'cjxzdzh/Qwen3.8-Flash-Next-refusal-direction-qwen4exp-cvec'
        File = 'cvec-qwen4exp.gguf'
        Sha256 = 'EF0724C5B79297E481017BE85769832BC26EB220C984ECBA2B0E1B0C8426312B'
    }
}

function Get-UncensorVectorPath {
    $spec = Get-UncensorVectorSpec
    return Join-Path (Join-Path $repoRoot 'models\control-vectors') $spec.File
}

function Test-UncensorVectorCurrent {
    $spec = Get-UncensorVectorSpec
    $path = Get-UncensorVectorPath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -eq $spec.Sha256
}

# Whether a host answers at all, decided in about a second instead of in the downloader's 30 s
# request timeout times three attempts. A TCP connect is deliberately the only test: it separates
# "this machine has no route to the hub" (mainland China boxes that can only reach the mirror) from
# "the hub is up but this file is slow", which the download itself has to discover.
function Test-EndpointReachable {
    param(
        [Parameter(Mandatory = $true)][string]$BaseUrl,
        [int]$TimeoutMs = 1500
    )
    try {
        $probeHost = ([System.Uri]$BaseUrl).Host
    } catch { return $false }
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $wait = $client.ConnectAsync($probeHost, 443).Wait($TimeoutMs)
        return ($wait -and $client.Connected)
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

# huggingface.co first, then the mirror: some machines cannot reach the origin at all and others can
# only reach the origin. A partial download, or one whose hash does not match, is DELETED - never
# kept as if it were the direction. An endpoint that does not answer the probe is skipped outright,
# and one that answers gets a short first attempt, so "no route" costs ~1 s and "reachable but
# stalling" costs 8 s rather than minute-and-a-half of retries before the mirror is even tried.
function Get-UncensorDirectionVector {
    $spec = Get-UncensorVectorSpec
    $path = Get-UncensorVectorPath
    if (Test-UncensorVectorCurrent) { return $path }
    New-Item -ItemType Directory -Path (Split-Path -Parent $path) -Force | Out-Null
    $attempts = 0
    foreach ($origin in @('https://huggingface.co', 'https://hf-mirror.com')) {
        if (-not (Test-EndpointReachable -BaseUrl $origin)) {
            Write-Host "连不上，跳过 / no route to $origin, going straight to the next endpoint" -ForegroundColor DarkGray
            continue
        }
        $url = "$origin/$($spec.Repo)/resolve/main/$($spec.File)"
        Write-Host "获取方向文件 / Fetching the direction vector: $url" -ForegroundColor DarkGray
        # The origin gets one short attempt because it is the one that may be a black hole; the
        # mirror, once reached, is worth the downloader's full retry ladder.
        $attempts++
        $isOrigin = $origin -eq 'https://huggingface.co'
        try {
            Invoke-ModelFileDownload -Url $url -Destination $path -Label $spec.File `
                -MaximumAttempts $(if ($isOrigin) { 1 } else { 3 }) `
                -AttemptTimeoutMs $(if ($isOrigin) { 8000 } else { 30000 })
        } catch {
            Write-Host "该来源失败，尝试下一个 / this endpoint failed, trying the next one: $($_.Exception.Message)" -ForegroundColor Yellow
            continue
        }
        if (Test-UncensorVectorCurrent) { return $path }
        Write-Host "校验和不匹配，已删除 / sha256 mismatch, discarded" -ForegroundColor Yellow
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    }
    if ($attempts -eq 0) {
        Write-Host '两个来源都连不上 / neither endpoint answered, so no vector could be fetched' -ForegroundColor Yellow
    }
    return ''
}

function Show-UncensorNotice {
    Write-Host ''
    Write-Host '去审查投影 / Uncensor (refusal-direction projection)' -ForegroundColor Yellow
    Write-Host '  只支持 Qwen3.8-Flash-Next（qwen4exp）；给别的模型指定方向文件会直接启动失败，不会被静默忽略。' 
    Write-Host '  Supported on Qwen3.8-Flash-Next (qwen4exp) ONLY; naming a file for another model fails startup.'
    Write-Host '  它移除的是模型训练出来的拒绝行为：本服务器之后回答什么由你负责，安全、合规与法律风险自行承担，'
    Write-Host '  模型许可证的使用条款仍然适用。开启即视为同意。默认：关闭。'
    Write-Host '  It REMOVES A TRAINED REFUSAL: safety, compliance and legal risk are yours to carry, the model'
    Write-Host '  licence still applies, and enabling it counts as consent. Default: OFF.'
    Write-Host ''
}

function Get-BenchmarkEstimateArguments {
    if (-not (Test-Path -LiteralPath $infrPath -PathType Leaf) -or [string]::IsNullOrWhiteSpace($modelPath)) {
        return @()
    }
    $estimateArgs = [System.Collections.Generic.List[string]]::new()
    foreach ($value in @('bench', '--estimate-only')) {
        [void]$estimateArgs.Add($value)
    }
    if (-not [string]::IsNullOrWhiteSpace($device)) {
        [void]$estimateArgs.Add('--dev'); [void]$estimateArgs.Add($device)
    }
    if ($setupMode -eq 'manual') {
        if (-not [string]::IsNullOrWhiteSpace($configPath)) { [void]$estimateArgs.Add('--config'); [void]$estimateArgs.Add($configPath) }
        if (-not [string]::IsNullOrWhiteSpace($ubatch)) { [void]$estimateArgs.Add('--ubatch'); [void]$estimateArgs.Add($ubatch) }
        if (-not [string]::IsNullOrWhiteSpace($threads)) { [void]$estimateArgs.Add('--threads'); [void]$estimateArgs.Add($threads) }
        if ($configureMemory) {
            if ($vramBudget) { Add-SetArgument $estimateArgs 'device.vram_budget' $vramBudget }
            if ($vramReserve) { Add-SetArgument $estimateArgs 'device.vram_reserve' $vramReserve }
            if ($ramBudget) { Add-SetArgument $estimateArgs 'device.ram_budget' $ramBudget }
        }
    } else {
        Add-SetArgument $estimateArgs 'device.auto_profile' $setupMode
    }
    [void]$estimateArgs.Add($modelPath)
    return $estimateArgs.ToArray()
}

function Start-BenchmarkEstimateProbe {
    if ($launchMode -ne 'benchmark' -or $null -ne $script:BenchmarkEstimateProbe -or $script:BenchmarkEstimateMinutes.Count) {
        return
    }
    $estimateArgs = @(Get-BenchmarkEstimateArguments)
    if (-not $estimateArgs.Count) {
        $script:BenchmarkEstimateError = 'engine or model is unavailable'
        return
    }
    try {
        $runner = [powershell]::Create()
        [void]$runner.AddScript({
            param([string]$Engine, [string[]]$NativeArguments)
            $output = @(& $Engine @NativeArguments 2>$null)
            [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = [string[]]$output }
        }).AddArgument($infrPath).AddArgument([object]([string[]]$estimateArgs))
        $async = $runner.BeginInvoke()
        $script:BenchmarkEstimateProbe = [pscustomobject]@{ Runner = $runner; Async = $async }
    } catch {
        if ($null -ne $runner) { $runner.Dispose() }
        $script:BenchmarkEstimateError = $_.Exception.Message
    }
}

function Complete-BenchmarkEstimateProbe {
    if ($script:BenchmarkEstimateMinutes.Count -or $script:BenchmarkEstimateError) { return }
    if ($null -eq $script:BenchmarkEstimateProbe) { Start-BenchmarkEstimateProbe }
    if ($null -eq $script:BenchmarkEstimateProbe) { return }
    $probe = $script:BenchmarkEstimateProbe
    $script:BenchmarkEstimateProbe = $null
    try {
        $result = @($probe.Runner.EndInvoke($probe.Async)) | Select-Object -Last 1
        if ($null -eq $result -or [int]$result.ExitCode -ne 0) {
            throw "estimate process exited with code $($result.ExitCode)"
        }
        $line = @($result.Output) | Where-Object { $_ -like 'BENCH_ESTIMATE_MENU_V1|*' } | Select-Object -Last 1
        $parts = @($line -split '\|')
        if ($parts.Count -ne 13 -or $parts[0] -ne 'BENCH_ESTIMATE_MENU_V1') {
            throw 'estimate matrix protocol response is missing'
        }
        $index = 1
        foreach ($suiteName in @('quick', 'standard', 'full')) {
            foreach ($parallelCount in 1..4) {
                $displayMinutes = 0
                if (-not [int]::TryParse($parts[$index], [ref]$displayMinutes) -or $displayMinutes -lt 1) {
                    throw 'estimate matrix contains invalid minutes'
                }
                $script:BenchmarkEstimateMinutes["${suiteName}:$parallelCount"] = $displayMinutes
                $index++
            }
        }
    } catch {
        $script:BenchmarkEstimateError = $_.Exception.Message
    } finally {
        $probe.Runner.Dispose()
    }
}

function Get-BenchmarkEstimateDetail {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('quick', 'standard', 'full')][string]$Suite,
        [Parameter(Mandatory = $true)][ValidateRange(1, 4)][int]$Parallel
    )
    Complete-BenchmarkEstimateProbe
    $key = "${Suite}:$Parallel"
    if ($script:BenchmarkEstimateMinutes.ContainsKey($key)) {
        $displayMinutes = [int]$script:BenchmarkEstimateMinutes[$key]
        return "预计约 $displayMinutes 分钟 / Estimated about $displayMinutes min."
    }
    return '暂时无法估算时间。Estimate currently unavailable.'
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

function Get-LatestEngineRelease {
    param([Parameter(Mandatory = $true)][string]$CurrentVersion)

    $headers = @{
        Accept = 'application/vnd.github+json'
        'User-Agent' = "MoE4All-Wizard/$CurrentVersion"
        'X-GitHub-Api-Version' = '2022-11-28'
    }
    $response = Invoke-RestMethod -Uri 'https://api.github.com/repos/Headmaster218/MoE4All/releases?per_page=30' -Headers $headers -TimeoutSec 5 -ErrorAction Stop
    $release = Select-Moe4AllEngineRelease -Releases @($response)
    if ($null -eq $release) {
        throw 'No stable Engine release with a Windows archive and SHA-256 file was found.'
    }
    return $release
}

function Show-EngineReleaseNotes {
    param([Parameter(Mandatory = $true)]$Release)

    Write-Host "`n$($Release.Name)" -ForegroundColor Cyan
    if (-not [string]::IsNullOrWhiteSpace([string]$Release.PublishedAt)) {
        Write-Host "Published: $($Release.PublishedAt)" -ForegroundColor DarkGray
    }
    Write-Host (Format-Moe4AllReleaseNotes -Notes ([string]$Release.Notes))
    Write-Host "`n$($Release.Url)" -ForegroundColor Cyan
}

function Read-EngineUpdateAction {
    param([Parameter(Mandatory = $true)]$Release, [switch]$Window, [string]$CurrentVersion = '')

    if ($Window) {
        $language=[string](Get-SavedValue 'wizard_language' '')
        $english=$language -eq 'en' -or (-not $language -and [Globalization.CultureInfo]::CurrentUICulture.TwoLetterISOLanguageName -ne 'zh')
        $url='https://github.com/Headmaster218/MoE4All/releases/tag/'+[string]$Release.Tag
        return [MoE4All.Launcher.NativeStartupPrompts]::Update($CurrentVersion,[string]$Release.Version,(Format-Moe4AllReleaseNotes -Notes ([string]$Release.Notes)),$url,[bool]$english)
    }

    while ($true) {
        Write-Host "`n发现新版本 / Update available: $($Release.Tag)" -ForegroundColor Yellow
        if ($Release.PSObject.Properties['Notes']) { Write-Host (Format-Moe4AllReleaseNotes -Notes ([string]$Release.Notes)) }
        Write-Host ('Release Note: https://github.com/Headmaster218/MoE4All/releases/tag/'+[string]$Release.Tag) -ForegroundColor Cyan
        Write-Host '   [1] 更新 / Update'
        Write-Host ' * [2] 下次更新（默认）/ Next time (default)'
        Write-Host '   [3] 三天后再提醒 / Remind me in 3 days'
        Write-Host '   [R] 打开完整发行说明 / Open full release notes'
        $value = (Read-Host '选择 [2] / Select [2]').Trim().ToLowerInvariant()
        if (-not $value) { return 'skip' }
        switch ($value) {
            { $_ -in @('r', 'release', 'notes') } {
                Open-EngineReleaseNotes -Release $Release
                continue
            }
            { $_ -in @('1', 'y', 'yes', 'update') } { return 'update' }
            { $_ -in @('2', 'n', 'no', 'later', 'skip') } { return 'skip' }
            { $_ -in @('3', 'snooze') } { return 'snooze' }
            default {
                Write-Host '请选择 1、2、3 或 R；回车下次更新。Select 1, 2, 3 or R; Enter skips this time.' -ForegroundColor Yellow
            }
        }
    }
}

function Open-EngineReleaseNotes {
    param([Parameter(Mandatory=$true)]$Release)
    Show-EngineReleaseNotes -Release $Release
    if([string]$Release.Tag -notmatch '^release-\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$'){return}
    try{Start-Process -FilePath ('https://github.com/Headmaster218/MoE4All/releases/tag/'+[string]$Release.Tag) -ErrorAction Stop|Out-Null}catch{}
}

function Test-EngineProcessRunning {
    foreach ($process in @(Get-Process -Name 'infr' -ErrorAction SilentlyContinue)) {
        try {
            if ($process.Path -and ([System.IO.Path]::GetFullPath($process.Path)).Equals($releaseInfrPath, [System.StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
        } catch {
            # Process paths can be inaccessible across integrity levels.
        }
    }
    return $false
}

function Invoke-EngineSelfUpdate {
    param([Parameter(Mandatory = $true)]$Release)

    if (-not $script:IsPackagedInstall) {
        Write-Host '源码工作区不会自动覆盖；请更新源码后重新构建。Source checkouts are not overwritten automatically.' -ForegroundColor Yellow
        Write-Host $Release.Url -ForegroundColor Cyan
        return $false
    }
    if (Test-EngineProcessRunning) {
        Write-Host '当前安装目录中的 infr.exe 正在运行；请停止服务后重试。The installed infr.exe is running; stop it and try again.' -ForegroundColor Yellow
        return $false
    }

    $updateRoot = Join-Path $repoRoot '.update'
    $runRoot = Join-Path $updateRoot ("staging-" + [guid]::NewGuid().ToString('N'))
    $archivePath = Join-Path $runRoot $Release.ArchiveName
    $checksumPath = Join-Path $runRoot $Release.ChecksumName
    $extractRoot = Join-Path $runRoot 'extracted'
    New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

    try {
        $headers = @{ 'User-Agent' = "MoE4All-Wizard/$productVersion" }
        Write-Host "`n正在下载 $($Release.ArchiveName) / Downloading $($Release.ArchiveName)..." -ForegroundColor Cyan
        Invoke-FileDownloadWithProgress `
            -Url $Release.ArchiveUrl `
            -Destination $archivePath `
            -Label $Release.ArchiveName `
            -ExpectedBytes $Release.ArchiveSize `
            -ProgressId 81 `
            -UserAgent $headers['User-Agent'] `
            -ReplaceExisting
        Invoke-FileDownloadWithProgress `
            -Url $Release.ChecksumUrl `
            -Destination $checksumPath `
            -Label $Release.ChecksumName `
            -ExpectedBytes $Release.ChecksumSize `
            -ProgressId 82 `
            -UserAgent $headers['User-Agent'] `
            -ReplaceExisting

        if ($Release.ArchiveSize -gt 0 -and (Get-Item -LiteralPath $archivePath).Length -ne $Release.ArchiveSize) {
            throw 'Downloaded archive size does not match the GitHub release asset.'
        }
        $checksumText = Get-Content -LiteralPath $checksumPath -Raw -Encoding UTF8
        $checksumPattern = '(?im)^\s*([a-f0-9]{64})\s+\*?' + [regex]::Escape($Release.ArchiveName) + '\s*$'
        if ($checksumText -notmatch $checksumPattern) {
            throw "The SHA-256 file does not contain an entry for $($Release.ArchiveName)."
        }
        $expectedHash = $Matches[1]
        $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
        if (-not $actualHash.Equals($expectedHash, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'Downloaded archive failed SHA-256 verification.'
        }

        Write-Host '正在校验更新包 / Validating update package...' -ForegroundColor Cyan
        Expand-Archive -LiteralPath $archivePath -DestinationPath $extractRoot -Force
        $manifests = @(Get-ChildItem -LiteralPath $extractRoot -Filter 'install-manifest.json' -File -Recurse)
        if ($manifests.Count -ne 1) { throw 'The update archive must contain exactly one install-manifest.json.' }
        $packageRoot = $manifests[0].Directory.FullName
        $manifest = Read-Moe4AllInstallManifest -ManifestPath $manifests[0].FullName -PackageRoot $packageRoot -VerifyFiles
        if ([string]$manifest.version -ne [string]$Release.Version) {
            throw "Release version $($Release.Version) does not match package version $($manifest.version)."
        }
        $stagedBinary = Join-Path $packageRoot 'infr.exe'
        $versionOutput = (& $stagedBinary --version 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $versionOutput -notmatch [regex]::Escape("infr $($Release.Version)")) {
            throw "The staged engine failed version validation: $versionOutput"
        }

        $stagedHelper = Join-Path $packageRoot 'scripts\apply-engine-update.ps1'
        $temporaryHelper = Join-Path ([System.IO.Path]::GetTempPath()) ("MoE4All-Engine-Updater-" + [guid]::NewGuid().ToString('N') + '.ps1')
        Copy-Item -LiteralPath $stagedHelper -Destination $temporaryHelper -Force
        $launcherProcessId = 0
        try {
            $currentProcess = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $PID" -ErrorAction Stop
            $parent = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $([int]$currentProcess.ParentProcessId)" -ErrorAction Stop
            $parentCommandLine = [string]$parent.CommandLine
            if ([string]$parent.Name -ieq 'cmd.exe' -and $parentCommandLine -match '(?i)(?:^|\s)/c(?:\s|$)' -and $parentCommandLine -match '(?i)Start-INFR-Wizard\.cmd') {
                $launcherProcessId = [int]$parent.ProcessId
            }
        } catch {
            $launcherProcessId = 0
        }
        $argumentText = @(
            '-NoLogo'
            '-NoProfile'
            '-File'
            ('"' + $temporaryHelper + '"')
            '-InstallRoot'
            ('"' + $repoRoot + '"')
            '-StagedPackageRoot'
            ('"' + $packageRoot + '"')
            '-ExpectedVersion'
            ('"' + $Release.Version + '"')
            '-ParentProcessId'
            ([string]$PID)
            '-LauncherProcessId'
            ([string]$launcherProcessId)
            '-RelaunchEntry'
            $(if ($WindowWizard) { 'window' } else { 'console' })
            '-CleanupRoot'
            ('"' + $runRoot + '"')
            '-DeleteSelf'
        ) -join ' '
        $updateWindowStyle = if ($WindowWizard) { 'Hidden' } else { 'Normal' }
        Start-Process -FilePath 'powershell.exe' -ArgumentList $argumentText -WorkingDirectory $repoRoot -WindowStyle $updateWindowStyle | Out-Null
        Write-Host '更新包已准备完成，向导将退出并由更新助手完成替换。Update is staged; the helper will finish after this wizard exits.' -ForegroundColor Green
        exit 42
    } catch {
        if (Test-Path -LiteralPath $runRoot) {
            Remove-Item -LiteralPath $runRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Host "更新失败，现有安装未改变。Update failed; the current installation was not changed.`n$($_.Exception.Message)" -ForegroundColor Red
        Write-Host $Release.Url -ForegroundColor Cyan
        return $false
    }
}

function Show-PendingUpdateResult {
    $resultPath = Join-Path $repoRoot '.update\last-update-result.json'
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) { return }
    try {
        $result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $color = if ([string]$result.status -eq 'success') { 'Green' } else { 'Red' }
        Write-Host ([string]$result.message) -ForegroundColor $color
    } catch {
        Write-Host '上次更新留下了无法读取的结果。The previous update left an unreadable result.' -ForegroundColor Yellow
    } finally {
        Remove-Item -LiteralPath $resultPath -Force -ErrorAction SilentlyContinue
    }
}

function Get-EngineUpdateReminder {
    $result=[pscustomobject]@{release_tag='';next_remind_at=[long]0}
    try {
        $path=Join-Path $dataDir 'private\update-reminder.json'
        if(Test-Path -LiteralPath $path -PathType Leaf){
            $saved=[IO.File]::ReadAllText($path)|ConvertFrom-Json;$time=[long]0
            if([string]$saved.release_tag -match '^release-\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$' -and [long]::TryParse([string]$saved.next_remind_at,[ref]$time) -and $time-ge 0){$result.release_tag=[string]$saved.release_tag;$result.next_remind_at=$time}
        }
    }catch{}
    return $result
}

function Save-EngineUpdateReminder {
    param([string]$Tag,[ValidateSet('update','skip','snooze')][string]$Action)
    if($DryRun){return};$temporary=$null
    try {
        $directory=Join-Path $dataDir 'private';[void][IO.Directory]::CreateDirectory($directory)
        $path=Join-Path $directory 'update-reminder.json';$temporary=$path+'.'+[guid]::NewGuid().ToString('N')+'.tmp'
        $next=if($Action-eq'snooze'){(Get-BenchmarkGuideNow)+259200}else{0}
        $value=[ordered]@{schema_version=1;release_tag=$Tag;next_remind_at=$next}
        [IO.File]::WriteAllText($temporary,($value|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
        if([IO.File]::Exists($path)){[IO.File]::Replace($temporary,$path,[Management.Automation.Language.NullString]::Value)}else{[IO.File]::Move($temporary,$path)}
    }catch{Write-Warning '无法保存更新提醒，仍可正常启动。Could not save the update reminder; startup continues.'}
    finally{if($temporary -and [IO.File]::Exists($temporary)){[IO.File]::Delete($temporary)}}
}

function Start-EngineUpdateProbe {
    param([string]$CurrentVersion)
    $disabled=[Environment]::GetEnvironmentVariable('MOE4ALL_NO_UPDATE_CHECK','Process')
    if($SkipUpdateCheck -or $DryRun -or $disabled-match '^(1|true|yes|on)$' -or $script:EngineUpdateProbe){return}
    $runner=$null
    try{
        $runner=[PowerShell]::Create()
        $text='param($common,$version) . $common; function Get-LatestEngineRelease {'+${function:Get-LatestEngineRelease}.ToString()+'}; Get-LatestEngineRelease -CurrentVersion $version'
        [void]$runner.AddScript($text).AddArgument($updateCommonPath).AddArgument($CurrentVersion)
        $script:EngineUpdateProbe=[pscustomobject]@{Runner=$runner;Async=$runner.BeginInvoke()}
    }catch{if($runner){$runner.Dispose()};$script:EngineUpdateProbe=$null}
}

function Show-UpdateStatus {
    param([Parameter(Mandatory = $true)][string]$CurrentVersion,[switch]$Window,[switch]$Force)

    $disabled = [Environment]::GetEnvironmentVariable('MOE4ALL_NO_UPDATE_CHECK', 'Process')
    if ($SkipUpdateCheck -or $DryRun -or $disabled -match '^(1|true|yes|on)$') { return }
    $current = ConvertTo-Moe4AllSemVer $CurrentVersion
    if ($null -eq $current) { return }

    try {
        if($script:EngineUpdateProbe){
            $probe=$script:EngineUpdateProbe;$script:EngineUpdateProbe=$null
            try{if(-not $probe.Async.AsyncWaitHandle.WaitOne(6000)){throw 'Update check timed out'};$release=@($probe.Runner.EndInvoke($probe.Async))|Select-Object -First 1;if(-not $release){throw 'Update check returned no release'}}finally{$probe.Runner.Dispose()}
        }else{$release = Get-LatestEngineRelease -CurrentVersion $CurrentVersion}
        if ((Compare-Moe4AllSemVer $release.SemVer $current) -gt 0) {
            $reminder=Get-EngineUpdateReminder
            if(-not $Force -and $reminder.release_tag-eq$release.Tag -and (Get-BenchmarkGuideNow)-lt$reminder.next_remind_at){return}
            $action = Read-EngineUpdateAction -Release $release -Window:$Window -CurrentVersion $CurrentVersion
            Save-EngineUpdateReminder -Tag $release.Tag -Action $action
            if ($action -eq 'update') { [void](Invoke-EngineSelfUpdate -Release $release) }
        } else {
            Write-Host "更新检查 / Update check: v$CurrentVersion is current" -ForegroundColor DarkGray
        }
    } catch {
        Write-Host "更新检查不可用，继续离线启动。Update check unavailable; continuing offline. $($_.Exception.Message)" -ForegroundColor DarkGray
    }
}

function Write-PostUpdateProbe {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Version
    )

    $updateRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot '.update')).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
    $probePath = [System.IO.Path]::GetFullPath($Path)
    $updatePrefix = $updateRoot + [System.IO.Path]::DirectorySeparatorChar
    if (-not $probePath.StartsWith($updatePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'The post-update probe path must remain inside the installation update directory.'
    }
    if ($ExpectedPostUpdateVersion -and $Version -ne $ExpectedPostUpdateVersion) {
        throw "The relaunched wizard reported v$Version; expected v$ExpectedPostUpdateVersion."
    }

    New-Item -ItemType Directory -Path $updateRoot -Force | Out-Null
    $probe = [ordered]@{
        version = $Version
        entry = $(if ($WindowWizard) { 'window' } else { 'console' })
        process_id = $PID
        at = [DateTimeOffset]::Now.ToString('o')
    }
    $temporary = "$probePath.moe4all-new-$([guid]::NewGuid().ToString('N'))"
    [System.IO.File]::WriteAllText($temporary, ($probe | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $probePath -Force
}

function Get-WizardExecutionVariableNames {
    # Never restore caller-selected variable names, paths, or secrets from JSON.
    @('nativeArgs', 'launchMode', 'modelPath', 'serverAddr', 'serverAuth',
      'serverEmbedding', 'serverVision', 'llmEnabled', 'transcription', 'speech',
      'benchKind', 'manualBenchKind', 'promptTokens', 'genTokens', 'depthMode',
      'depthTokens', 'reps', 'productVersion', 'device', 'setupMode', 'kvTypeK', 'kvTypeV')
}

function Get-WizardExecutionPreferenceNames {
    @('BenchmarkUploadDefault', 'BenchmarkUploadHasGithubDefault', 'BenchmarkUploadGhAuthDefault',
      'BenchmarkUploadFallbackDefault', 'SubmissionBrowserActionDefault', 'SubmissionFailureActionDefault',
      'SubmissionPendingRetryDefault', 'ErrorUploadDefault', 'BenchmarkGuideActive', 'PendingSubmissionPrompted')
}

function New-WizardExecutionPlan {
    $values = [ordered]@{}
    foreach ($name in Get-WizardExecutionVariableNames) {
        $values[$name] = (Get-Variable -Name $name -Scope 1 -ValueOnly -ErrorAction SilentlyContinue)
    }
    $values['nativeArgs'] = [string[]]@($values['nativeArgs'])
    # Only Enabled is needed for the server startup message; model/paths already live in argv.
    $values['transcription'] = [pscustomobject]@{ Enabled = [bool]$values['transcription'].Enabled }
    $values['speech'] = [pscustomobject]@{ Enabled = ([string]$values['speech'].Enabled -ieq 'true') }
    $preferences = [ordered]@{}
    foreach ($name in Get-WizardExecutionPreferenceNames) {
        $preferences[$name] = Get-Variable -Name $name -Scope Script -ValueOnly -ErrorAction SilentlyContinue
    }
    [pscustomobject]@{ SchemaVersion = 1; RepoRoot = $repoRoot; EngineIdentity = $null
        Variables = [pscustomobject]$values; Preferences = [pscustomobject]$preferences }
}

function Get-WizardExecutionPlanDirectory {
    # A source checkout can also contain infr.exe; .git takes priority over package detection.
    if ($script:IsPackagedInstall -and -not (Test-Path -LiteralPath (Join-Path $repoRoot '.git'))) {
        return [IO.Path]::GetFullPath((Join-Path $repoRoot 'gui-data\private\launch'))
    }
    $hash = [Security.Cryptography.SHA256]::Create()
    try {
        $identity = [BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($repoRoot.ToLowerInvariant()))).Replace('-', '').Substring(0, 20).ToLowerInvariant()
    } finally { $hash.Dispose() }
    $generatedRoot = if ($env:MOE4ALL_WORK_ROOT) { [IO.Path]::GetFullPath($env:MOE4ALL_WORK_ROOT) }
        elseif ($repoRoot.StartsWith('D:\AIinfr\', [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath 'D:\AIinfr\disk_usage_rule.md')) { 'D:\AIinfr\infr\local-work' }
        else { Join-Path $repoRoot 'local-work' }
    return (Join-Path $generatedRoot "runtime\moe4all\wizard-$identity\launch")
}

function Assert-WizardExecutionPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $current = [IO.Path]::GetFullPath($Path)
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            if (((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw 'Execution plans cannot use reparse points.'
            }
        }
        $current = [IO.Path]::GetDirectoryName($current)
    }
}

function Get-WizardExecutionEngineIdentity {
    $stream = [IO.File]::Open($infrPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    $hash = [Security.Cryptography.SHA256]::Create()
    try {
        [pscustomobject]@{ Path = [IO.Path]::GetFullPath($infrPath); Length = $stream.Length
            LastWriteTimeUtcTicks = [IO.File]::GetLastWriteTimeUtc($infrPath).Ticks
            Sha256 = [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    } finally { $hash.Dispose(); $stream.Dispose() }
}

function Read-WizardExecutionPlan {
    param([Parameter(Mandatory = $true)][string]$Path)
    $fullPath = [IO.Path]::GetFullPath($Path)
    $directory = Get-WizardExecutionPlanDirectory
    if (-not [string]::Equals([IO.Path]::GetDirectoryName($fullPath), $directory, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($fullPath) -notmatch '^launch-[a-f0-9]{32}\.json$') {
        throw 'Execution plan is outside the controlled launch directory.'
    }
    Assert-WizardExecutionPath -Path $fullPath
    $stream = [IO.File]::Open($fullPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try {
        if ($stream.Length -eq 0 -or $stream.Length -gt 131072) { throw 'Execution plan must be at most 128 KiB.' }
        $reader = [IO.StreamReader]::new($stream, [Text.UTF8Encoding]::new($false, $true), $true)
        try { $plan = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
    } finally { $stream.Dispose() }
    if ($null -eq $plan -or $plan.SchemaVersion -ne 1 -or
        -not [string]::Equals([string]$plan.RepoRoot, $repoRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Execution plan does not match this installation or schema.'
    }
    $actual = Get-WizardExecutionEngineIdentity
    foreach ($name in @('Path', 'Length', 'LastWriteTimeUtcTicks', 'Sha256')) {
        if ($null -eq $plan.EngineIdentity -or [string]$plan.EngineIdentity.$name -ine [string]$actual.$name) {
            throw 'Execution plan does not match the current engine file.'
        }
    }
    foreach ($section in @('Variables', 'Preferences')) {
        $allowed = if ($section -eq 'Variables') { @(Get-WizardExecutionVariableNames) } else { @(Get-WizardExecutionPreferenceNames) }
        $names = @($plan.$section.PSObject.Properties.Name)
        if ($names.Count -ne $allowed.Count -or @($names | Where-Object { $_ -cnotin $allowed }).Count -or
            @($allowed | Where-Object { $_ -cnotin $names }).Count) { throw 'Invalid execution plan fields.' }
    }
    if ($plan.Variables.launchMode -notin @('chat', 'server', 'benchmark') -or
        $plan.Variables.nativeArgs -isnot [array] -or $plan.Variables.nativeArgs.Count -eq 0 -or
        @($plan.Variables.nativeArgs | Where-Object { $_ -isnot [string] }).Count) { throw 'Invalid execution plan command.' }
    return $plan
}

function Invoke-WizardDesktopExecution {
    param([Parameter(Mandatory = $true)][object]$Plan, [bool]$English = $false)
    $planPath = $null
    $ownsPlan = $false
    $desktopCompleted = $false
    $previousApiKey = [Environment]::GetEnvironmentVariable('INFR_API_KEY', 'Process')
    $previousDesktopIo = [Environment]::GetEnvironmentVariable('MOE4ALL_DESKTOP_IO', 'Process')
    try {
        # The desktop conversation uses the existing streaming API: multiline input,
        # roles and reasoning stay structured instead of parsing terminal output.
        $chatEndpoint=''
        if($Plan.Variables.launchMode -eq 'chat' -and $Plan.Variables.nativeArgs[0] -eq 'run'){
            $Plan=$Plan|ConvertTo-Json -Depth 8 -Compress|ConvertFrom-Json
            $portProbe=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0)
            $portProbe.Start();try{$chatPort=([Net.IPEndPoint]$portProbe.LocalEndpoint).Port}finally{$portProbe.Stop()}
            $chatArguments=[Collections.Generic.List[string]]::new([string[]]$Plan.Variables.nativeArgs)
            $chatArguments[0]='serve';$chatArguments.Add('--addr');$chatArguments.Add("127.0.0.1:$chatPort");$chatArguments.Add('--parallel');$chatArguments.Add('1')
            $Plan.Variables.nativeArgs=$chatArguments.ToArray()
            $chatEndpoint="http://127.0.0.1:$chatPort/v1"
            [Environment]::SetEnvironmentVariable('INFR_API_KEY',[guid]::NewGuid().ToString('N'),'Process')
        }
        $Plan.EngineIdentity = Get-WizardExecutionEngineIdentity
        $payload = [Text.UTF8Encoding]::new($false).GetBytes(($Plan | ConvertTo-Json -Depth 8 -Compress))
        if ($payload.Length -gt 131072) { throw 'Execution plan exceeds 128 KiB.' }
        $directory = Get-WizardExecutionPlanDirectory
        Assert-WizardExecutionPath -Path $directory
        [void][IO.Directory]::CreateDirectory($directory)
        Assert-WizardExecutionPath -Path $directory
        $planPath = Join-Path $directory ('launch-' + [guid]::NewGuid().ToString('N') + '.json')
        $stream = [IO.File]::Open($planPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $ownsPlan = $true
        try { $stream.Write($payload, 0, $payload.Length) } finally { $stream.Dispose() }
        if ($Plan.Variables.launchMode -eq 'server' -and $Plan.Variables.serverAuth) {
            [Environment]::SetEnvironmentVariable('INFR_API_KEY', $serverApiKey, 'Process')
        }
        [Environment]::SetEnvironmentVariable('MOE4ALL_DESKTOP_IO', '1', 'Process')
        $childArguments = [string[]]@('-NoLogo', '-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass', '-File',
            (Join-Path $PSScriptRoot 'infr-wizard.ps1'), '-DesktopExecutionPlanPath', $planPath)
        $desktopEndpoint = if($chatEndpoint){$chatEndpoint}elseif ($Plan.Variables.launchMode -eq 'server') { 'http://' + (Get-ClientAddress ([string]$Plan.Variables.serverAddr)) + '/v1' } else { '' }
        $desktopResult = [MoE4All.Launcher.NativeRuntime]::Run('powershell.exe', $childArguments, $repoRoot,
            [string]$Plan.Variables.launchMode, [string]$Plan.Variables.modelPath, $desktopEndpoint, $English)
        $desktopCompleted=$true
        return $desktopResult
    } finally {
        [Environment]::SetEnvironmentVariable('INFR_API_KEY', $previousApiKey, 'Process')
        [Environment]::SetEnvironmentVariable('MOE4ALL_DESKTOP_IO', $previousDesktopIo, 'Process')
        if ($ownsPlan) { Remove-Item -LiteralPath $planPath -Force -ErrorAction SilentlyContinue }
        if ($desktopCompleted -and ('MoE4All.Launcher.NativeWizard' -as [type])) { [MoE4All.Launcher.NativeWizard]::RestoreConsole() }
    }
}

function Invoke-WizardExecutionPlan {
    param([Parameter(Mandatory = $true)][object]$Plan)
    foreach ($name in Get-WizardExecutionVariableNames) {
        Set-Variable -Name $name -Scope Local -Value $Plan.Variables.$name
    }
    $nativeArgs = [string[]]@($nativeArgs)
    foreach ($name in Get-WizardExecutionPreferenceNames) {
        Set-Variable -Name $name -Scope Script -Value $Plan.Preferences.$name
    }
    if ($DesktopExecutionPlanPath) {
        $serverApiKey = [Environment]::GetEnvironmentVariable('INFR_API_KEY', 'Process')
    }
    if ($launchMode -eq 'server') {
        $clientAddress = Get-ClientAddress $serverAddr
        Write-Host "`nAPI 地址 / API base URL: http://$clientAddress/v1" -ForegroundColor Cyan
        Write-Host "健康检查 / Health check: http://$clientAddress/health"
        Write-Host '兼容 OpenAI 客户端时，将 Base URL 设为上面的 /v1 地址。' -ForegroundColor DarkGray
        Write-Host 'For OpenAI clients, use the /v1 address above as the Base URL.' -ForegroundColor DarkGray
        if ($serverEmbedding) {
            Write-Host "Embedding API: http://$clientAddress/v1/embeddings" -ForegroundColor Cyan
        }
        if ($transcription.Enabled) {
            Write-Host "本地语音转录 API / Local transcription API: http://$clientAddress/v1/audio/transcriptions (WAV/PCM, CPU)" -ForegroundColor Cyan
        }
        if ($speech.Enabled) {
            Write-Host "本地语音合成 API / Local speech synthesis API: http://$clientAddress/v1/audio/speech (WAV/PCM, CPU)" -ForegroundColor Cyan
        }
        if ($llmEnabled -and $serverVision) {
            Write-Host '视觉聊天已启用：在 /v1/chat/completions 的 content parts 中发送 image_url（data URI、裸 base64 或 http(s) URL）。' -ForegroundColor Cyan
            Write-Host 'Vision chat enabled: send an image_url in /v1/chat/completions content parts — data URI, bare base64, or http(s) URL.' -ForegroundColor Cyan
            if ($imageUrlAllowPrivate) {
                Write-Host 'image_url 允许抓取本机与内网地址：这台服务器可以被请求指向私网。' -ForegroundColor Yellow
                Write-Host 'image_url may address loopback and private hosts: a request can point this server into your network.' -ForegroundColor Yellow
            }
        }
        Write-Host '按 Ctrl+C 停止服务器。Press Ctrl+C to stop the server.' -ForegroundColor Yellow
    } elseif ($launchMode -eq 'chat') {
        if($DesktopExecutionPlanPath){
            Write-Host "`n正在准备聊天；模型加载完成后会自动进入对话。" -ForegroundColor Cyan
            Write-Host 'Preparing chat; the conversation opens automatically when the model is ready.' -ForegroundColor DarkGray
        }else{
            Write-Host "`n模型加载完成后，在 > 提示符输入消息；输入 exit、quit 或 :q 退出。" -ForegroundColor Cyan
            Write-Host 'After the model loads, type at the > prompt; use exit, quit or :q to leave.' -ForegroundColor DarkGray
        }
    }

    Write-Host "`n启动中 / Starting..." -ForegroundColor Green
    $benchmarkFilesBefore = @()
    if ($launchMode -eq 'benchmark') { $benchmarkFilesBefore = @(Get-BenchmarkSubmissionSnapshot) }
    $errorReportsBefore = @(Get-ErrorSubmissionSnapshot)
    $benchmarkResultPath = $null
    $previousApiKey = [Environment]::GetEnvironmentVariable('INFR_API_KEY', 'Process')
    $hadApiKey = $null -ne $previousApiKey
    $previousDiagnosticDir = [Environment]::GetEnvironmentVariable('INFR_DIAGNOSTIC_DIR', 'Process')
    $hadDiagnosticDir = $null -ne $previousDiagnosticDir
    [Environment]::SetEnvironmentVariable('INFR_DIAGNOSTIC_DIR', (Join-Path $repoRoot 'diagnostics'), 'Process')
    if ($launchMode -eq 'server' -and $serverAuth) {
        [Environment]::SetEnvironmentVariable('INFR_API_KEY', $serverApiKey, 'Process')
    }
    Push-Location $repoRoot
    try {
        if ($launchMode -eq 'benchmark' -and $benchKind -eq 'manual') {
            $manualJson = ((& $infrPath @nativeArgs) -join "`n").Trim()
            $exitCode = $LASTEXITCODE
            if ($exitCode -eq 0) {
                try {
                    $benchmarkResultPath = Write-ManualBenchmarkResult -RawJson $manualJson -ModelPath $modelPath -Kind $manualBenchKind -PromptTokens $promptTokens -GenTokens $genTokens -DepthMode $depthMode -DepthTokens $depthTokens -Repetitions $reps
                } catch {
                    Write-Warning "手动 benchmark 结果保存失败：$($_.Exception.Message) / Could not save the manual benchmark result."
                }
            }
        } else {
            & $infrPath @nativeArgs
            $exitCode = $LASTEXITCODE
        }
    } finally {
        Pop-Location
        if ($launchMode -eq 'server' -and $serverAuth) {
            $restoreApiKey = if ($hadApiKey) { $previousApiKey } else { $null }
            [Environment]::SetEnvironmentVariable('INFR_API_KEY', $restoreApiKey, 'Process')
        }
        $restoreDiagnosticDir = if ($hadDiagnosticDir) { $previousDiagnosticDir } else { $null }
        [Environment]::SetEnvironmentVariable('INFR_DIAGNOSTIC_DIR', $restoreDiagnosticDir, 'Process')
    }
    if ($null -eq $exitCode) { $exitCode = 0 }
    Write-Host "`nMoE4All / infr 退出码 / exit code: $exitCode" -ForegroundColor $(if ($exitCode -eq 0) { 'Green' } else { 'Red' })
    if ($launchMode -eq 'benchmark' -and $benchKind -ne 'manual') {
        $before = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        foreach ($path in $benchmarkFilesBefore) { [void]$before.Add($path) }
        $benchmarkResultPath = @(Get-BenchmarkSubmissionSnapshot | Where-Object { -not $before.Contains($_) } | Sort-Object { (Get-Item -LiteralPath $_).LastWriteTimeUtc } -Descending | Select-Object -First 1)
        if ($benchmarkResultPath.Count -gt 0) { $benchmarkResultPath = [string]$benchmarkResultPath[0] } else { $benchmarkResultPath = $null }
    }
    $benchmarkUploadRequested = $false
    if ($benchmarkResultPath) {
        if($DesktopExecutionPlanPath){try{Show-WizardRuntimeBenchmarkResult -Path $benchmarkResultPath}catch{Write-Warning ('结果界面加载失败，原始报告保留。Could not display results; the report is retained. '+$_.Exception.Message)}}
        $benchmarkUploadRequested = Read-BenchmarkUploadConsent
    }
    if ($benchmarkResultPath -and $benchmarkUploadRequested) {
        Invoke-BenchmarkUploadSafely -Path $benchmarkResultPath -ConsentAlreadyGranted
    }
    $knownReports = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($path in $errorReportsBefore) { [void]$knownReports.Add($path) }
    if ($exitCode -ne 0) {
        $fatalReport = @(Get-ErrorSubmissionSnapshot | Where-Object { -not $knownReports.Contains($_) } | Sort-Object { (Get-Item -LiteralPath $_).LastWriteTimeUtc } -Descending | Select-Object -First 1)
        if ($fatalReport.Count -gt 0) {
            try { [void](Complete-ErrorReport -Source ([string]$fatalReport[0])) } catch { Write-Warning "错误报告补充失败：$($_.Exception.Message) / Could not complete the error report." }
        }
    }
    if ($launchMode -eq 'benchmark' -and $benchmarkResultPath) { Offer-BenchmarkLeaderboard }
    exit $exitCode
}

if ($DesktopExecutionPlanPath) {
    if ($DryRun -or $PreviewUi) { throw 'Desktop execution cannot be combined with DryRun or PreviewUi.' }
    # Native child prompts remain in the owning runtime window; no second GUI host.
    . (Join-Path $PSScriptRoot 'wizard-runtime-bridge.ps1')
    $desktopPlan = Read-WizardExecutionPlan -Path $DesktopExecutionPlanPath
    Invoke-WizardExecutionPlan -Plan $desktopPlan
}

if ($PreviewUi) {
    . (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
    if ($WindowWizard -or (Test-WizardConsoleInput)) { $null = Invoke-WizardDashboard -Saved $null -Demo }
    else {
        . (Join-Path $PSScriptRoot 'infr-dashboard-state.ps1')
        Initialize-DashboardRuntime
        $demoState = Get-DashboardInitialState -Saved $null
        $demoSchema = Get-DashboardSchema -State $demoState -DeviceOptions @()
        Write-Output ([MoE4All.Launcher.Dashboard]::Snapshot(($demoState|ConvertTo-Json -Depth 20 -Compress),($demoSchema|ConvertTo-Json -Depth 20 -Compress),'home',132,42,$false))
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $infrPath -PathType Leaf)) {
    throw "找不到 infr.exe。发布包请把 infr.exe 与 Start-INFR-Wizard.cmd 放在同一目录；源码构建请先运行 cargo build --release --locked -p infr-cli。`nExecutable not found beside the launcher or under target\release."
}

$productVersion = Get-EngineVersion
if ($PostUpdateProbePath) {
    Write-PostUpdateProbe -Path $PostUpdateProbePath -Version $productVersion
    if ($ExitAfterPostUpdateProbe) { exit 43 }
}
if (-not $WindowWizard) {
Clear-Host
Write-Host '============================================================' -ForegroundColor Green
$banner = @'
 __  __       _____ _  _     _    _     _
|  \/  | ___ | ____| || |   / \  | |   | |           ___ ___  _ __ ___
| |\/| |/ _ \|  _| | || |_ / _ \ | |   | |          / __/ _ \| '_ ` _ \
| |  | | (_) | |___|__   _/ ___ \| |___| |___   _  | (_| (_) | | | | | |
|_|  |_|\___/|_____|  |_|/_/   \_\_____|_____| (_)  \___\___/|_| |_| |_|
'@

Write-Host $banner -ForegroundColor Green
Write-Host "  MoE4All v$productVersion" -ForegroundColor Green
Write-Host '  https://moe4all.com' -ForegroundColor Cyan
Write-Host '  Run models far larger than VRAM on gaming GPUs.'
Write-Host '  让游戏显卡跑起远超显存容量的大模型。'
Write-Host '  John / Headmaster218  https://github.com/Headmaster218/MoE4All' -ForegroundColor DarkGray
Write-Host '============================================================' -ForegroundColor Green
Write-Host 'MoE4All 启动向导 / MoE4All Launch Wizard' -ForegroundColor Green
Write-Host '上次设置会作为默认值；直接回车即可复用。Press Enter to reuse the previous value.'
Write-Host "引擎 / Engine: $infrPath" -ForegroundColor DarkGray
Show-PendingUpdateResult
}
$script:DashboardAccepted = $false
$script:DashboardApiKey = ''
$script:DashboardPendingPrompted = $false
$script:DashboardBenchmarkGuideActive = $false
$useDashboard = [bool]$WindowWizard -and -not $DryRun
if ($useDashboard) {
    try {
        . (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
        $dashboardResult = Invoke-WizardDashboard -Saved $script:Saved
        if ($null -eq $dashboardResult) { exit 44 }
        $script:Saved = $dashboardResult.State
        $script:DashboardApiKey = $dashboardResult.ApiKey
        $script:DashboardPendingPrompted = [bool]$script:PendingSubmissionPrompted
        $script:DashboardAccepted = $true
        $script:NativePromptsEnabled = $true
    } catch {
        $dashboardException = $_.Exception
        while ($null -ne $dashboardException) {
            if ($dashboardException.GetType().FullName -eq 'MoE4All.Launcher.NativePromptCanceledException') { throw }
            $dashboardException = $dashboardException.InnerException
        }
        if ('MoE4All.Launcher.NativeWizard' -as [type]) { [MoE4All.Launcher.NativeWizard]::RestoreConsole() }
        # The EXE entry has no terminal for the compatibility wizard to read from.
        # Let the bootstrap present the error instead of waiting on invisible input.
        if ($WindowWizard) { throw }
        Write-Warning ('工作台不可用，将进入兼容向导。Dashboard unavailable; using the compatible wizard. ' + $_.Exception.Message)
    }
}
if (-not $useDashboard -or -not $script:DashboardAccepted) { Show-UpdateStatus -CurrentVersion $productVersion }
$reuseSavedSettings = $false
$script:BenchmarkGuideActive = if ($script:DashboardAccepted) { [bool]$script:DashboardBenchmarkGuideActive } else { Invoke-BenchmarkStartupGuide -Interactive (Test-BenchmarkGuideInteractive) }
$savedModel = [string](Get-SavedValue 'model' '')
if ($script:DashboardAccepted) { $reuseSavedSettings = $true }
elseif (-not $script:BenchmarkGuideActive -and $null -ne $script:Saved -and -not $InitialModelPath) {
    if (Test-SavedWizardModels) {
        $reuseSavedSettings = Read-YesNo -Label '是否按上次设置直接启动？/ Start with the previous settings?' -Default $true
    } else {
        Write-Warning '上次启用模块的模型不可用，请重新选择。A previously enabled module has no available model; choose again.'
    }
}

$launchMode = [string](Get-SavedValue 'launch_mode' 'chat')
$modelPath = $savedModel
$setupModeDefault = 'conservative'
if ($null -ne $script:Saved) {
    $savedSetupMode = $script:Saved.PSObject.Properties['setup_mode']
    # States written by the older all-advanced wizard keep their previous behavior after upgrade.
    $setupModeDefault = if ($null -eq $savedSetupMode) {
        'manual'
    } else {
        switch ([string]$savedSetupMode.Value) {
            'quick' { 'conservative' }
            'advanced' { 'manual' }
            'conservative' { 'conservative' }
            'aggressive' { 'aggressive' }
            'manual' { 'manual' }
            default { 'conservative' }
        }
    }
}
$setupMode = $setupModeDefault
$device = [string](Get-SavedValue 'device' '')
$context = [string](Get-SavedValue 'context' '')
$ubatch = [string](Get-SavedValue 'ubatch' '')
$threads = [string](Get-SavedValue 'threads' '')
$configPath = [string](Get-SavedValue 'config_path' '')
$kvPreset = [string](Get-SavedValue 'kv_preset' 'auto')
$kvTypeK = [string](Get-SavedValue 'kv_type_k' '')
$kvTypeV = [string](Get-SavedValue 'kv_type_v' '')
$configureMemory = [bool](Get-SavedValue 'configure_memory' $false)

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
$submitMode = [string](Get-SavedValue 'submit_mode' 'auto')
$submitCap = [string](Get-SavedValue 'submit_cap' '64')
$configureDiagnostics = [bool](Get-SavedValue 'configure_diagnostics' $false)
$pagerStats = [bool](Get-SavedValue 'pager_stats' $false)
$pagerProfile = [bool](Get-SavedValue 'pager_profile' $false)
$stageProfile = [bool](Get-SavedValue 'stage_profile' $false)
$vramProfile = [bool](Get-SavedValue 'vram_profile' $false)
$benchKind = [string](Get-SavedValue 'bench_kind' 'quick')
$benchParallel = [string](Get-SavedValue 'bench_parallel' '1')
$manualBenchKind = [string](Get-SavedValue 'manual_bench_kind' 'decode')
$promptTokens = [string](Get-SavedValue 'prompt_tokens' '6144')
$genTokens = [string](Get-SavedValue 'gen_tokens' '512')
$depthMode = [string](Get-SavedValue 'depth_mode' 'none')
$depthTokens = [string](Get-SavedValue 'depth_tokens' '0')
$reps = [string](Get-SavedValue 'reps' '1')
$jsonOutput = [bool](Get-SavedValue 'json_output' $false)
$thinkMode = [string](Get-SavedValue 'think_mode' 'default')
$reasoningEffort = [string](Get-SavedValue 'reasoning_effort' 'default')
$script:ReasoningProbeModel = [string](Get-SavedValue 'reasoning_probe_model' '')
$script:ReasoningProbeLength = [int64](Get-SavedValue 'reasoning_probe_length' -1)
$script:ReasoningProbeMtimeTicks = [int64](Get-SavedValue 'reasoning_probe_mtime_ticks' -1)
$script:ReasoningProbeEfforts = @((Get-SavedValue 'reasoning_probe_efforts' @()))
$script:ModelReasoningProbe = $null
$script:VulkanDeviceProbe = $null
$script:VulkanDeviceProbeError = ''
$script:BenchmarkEstimateProbe = $null
$script:BenchmarkEstimateMinutes = @{}
$script:BenchmarkEstimateError = ''
$script:BenchmarkUploadDefault = [bool](Get-SavedValue 'benchmark_upload' $false)
$script:BenchmarkUploadHasGithubDefault = [bool](Get-SavedValue 'benchmark_upload_has_github' $true)
$script:BenchmarkUploadGhAuthDefault = [bool](Get-SavedValue 'benchmark_upload_gh_auth' $true)
$script:BenchmarkUploadFallbackDefault = [string](Get-SavedValue 'benchmark_upload_fallback' 'folder')
$script:SubmissionBrowserActionDefault = [string](Get-SavedValue 'submission_browser_action' 'later')
$script:SubmissionFailureActionDefault = [string](Get-SavedValue 'submission_failure_action' 'later')
$script:SubmissionPendingRetryDefault = [bool](Get-SavedValue 'submission_pending_retry' $true)
$script:ErrorUploadDefault = [bool](Get-SavedValue 'error_upload' $false)
if ($script:BenchmarkUploadFallbackDefault -notin @('folder', 'clipboard', 'skip')) { $script:BenchmarkUploadFallbackDefault = 'folder' }
$script:PendingSubmissionPrompted = [bool]$script:DashboardPendingPrompted
if (-not $DryRun) { Offer-PendingSubmissions }
$maxNew = [string](Get-SavedValue 'max_new' '65536')
$configureSampling = [bool](Get-SavedValue 'configure_sampling' $false)
$temperature = [string](Get-SavedValue 'temperature' '')
$topK = [string](Get-SavedValue 'top_k' '')
$topP = [string](Get-SavedValue 'top_p' '')
$seed = [string](Get-SavedValue 'seed' '')
$serverAddr = [string](Get-SavedValue 'server_addr' '127.0.0.1:8080')
$serverParallel = [string](Get-SavedValue 'server_parallel' '1')
$defaultSessionCacheDir = Join-Path $repoRoot 'kv-sessions'
$legacySessionCacheDir = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'MoE4All\kv-sessions'
$serverSessionCache = [bool](Get-SavedValue 'server_session_cache' $true)
$sessionCacheDir = [string](Get-SavedValue 'session_cache_dir' $defaultSessionCacheDir)
if ([string]::Equals($sessionCacheDir, $legacySessionCacheDir, [System.StringComparison]::OrdinalIgnoreCase)) {
    $sessionCacheDir = $defaultSessionCacheDir
}
$sessionIdleSecs = [string](Get-SavedValue 'session_idle_secs' '120')
$sessionCacheMax = [string](Get-SavedValue 'session_cache_max' '5GiB')
$sessionCacheTtlHours = [string](Get-SavedValue 'session_cache_ttl_hours' '24')
$kvRamCacheMax = [string](Get-SavedValue 'kv_ram_cache_max' '')
$kvPagedCacheMax = [string](Get-SavedValue 'kv_paged_cache_max' '')
# The tree and its RAM frame pool are two choices now. A state file from before `kv_paged_tree`
# asked for a pool whenever it named one, so a saved size still implies the tree.
$kvPagedTree = [bool](Get-SavedValue 'kv_paged_tree' ([bool]$kvPagedCacheMax))
$kvPagedPageTokens = [string](Get-SavedValue 'kv_paged_page_tokens' '')
$serverAuth = [bool](Get-SavedValue 'server_auth' $false)
$serverApiKey = ''
$serverVision = [bool](Get-SavedValue 'server_vision' $false)
$visionProjectorPath = [string](Get-SavedValue 'vision_projector' '')
# A state file from before this key asked for nothing of the sort, and the engine default is the
# closed one — so an absent key means private image URLs stay refused, never an accidental open.
$imageUrlAllowPrivate = [bool](Get-SavedValue 'image_url_allow_private' $false)
$serverLlmEnabled = ([string](Get-SavedValue 'server_llm_enabled' $true) -ieq 'true')
$serverEmbedding = ([string](Get-SavedValue 'server_embedding' $false) -ieq 'true')
$embeddingModelPath = [string](Get-SavedValue 'embedding_model' '')
$embeddingIdleTimeout = [string](Get-SavedValue 'embedding_idle_timeout' '300')
$transcription = Get-TranscriptionSettings
$speech = Get-SpeechSettings

$configureThinking = [bool](Get-SavedValue 'configure_thinking' ($thinkMode -ne 'default' -or $reasoningEffort -ne 'default'))
$cpuMissEnabled = [bool](Get-SavedValue 'cpu_miss_enabled' $false)
$cpuMissMax = [string](Get-SavedValue 'cpu_miss_max' '1')
$cpuMissCores = [string](Get-SavedValue 'cpu_miss_cores' '')
$customSets = [string](Get-SavedValue 'custom_sets' '')
$llmEnabled = ($launchMode -ne 'server' -or $serverLlmEnabled)

if (-not $reuseSavedSettings) {
    if ($script:BenchmarkGuideActive) {
        $launchMode = 'benchmark'; $benchKind = 'quick'; $benchParallel = '1'; $jsonOutput = $false
    } else {
    $launchMode = Read-Choice -Label '你想做什么？/ What would you like to do?' -DefaultValue ([string](Get-SavedValue 'launch_mode' 'chat')) -Options @(
        [pscustomobject]@{ Key = '1'; Value = 'chat'; Label = '实时终端对话（推荐）/ Interactive terminal chat (recommended)' }
        [pscustomobject]@{ Key = '2'; Value = 'server'; Label = '启动 OpenAI 兼容 API / Start OpenAI-compatible API server' }
        [pscustomobject]@{ Key = '3'; Value = 'benchmark'; Label = '性能测试 / Benchmark' }
        [pscustomobject]@{ Key = '4'; Value = 'report'; Label = '报告当前运行问题 / Report a current runtime problem' }
    )
    }
    if ($launchMode -eq 'report') {
        try { [void](Complete-ErrorReport) } catch { Write-Warning "错误报告生成失败：$($_.Exception.Message) / Could not create the error report." }
        exit 0
    }
    if ($launchMode -eq 'server') {
        $modules = Read-ServerModules -LlmEnabled $serverLlmEnabled -EmbeddingEnabled $serverEmbedding -TranscriptionEnabled $transcription.Enabled -SpeechEnabled $speech.Enabled
        $serverLlmEnabled = $modules.Llm
        $serverEmbedding = $modules.Embedding
        $transcription.Enabled = $modules.Transcription
        $speech.Enabled = $modules.Speech
    }
    $llmEnabled = ($launchMode -ne 'server' -or $serverLlmEnabled)
    if ($llmEnabled -or $serverEmbedding) { Start-VulkanDeviceProbe }
    if ($llmEnabled) {
        $modelPath = Select-ModelPath -InitialPath $InitialModelPath
        if ($launchMode -ne 'benchmark') { Start-ModelReasoningProbe -ModelPath $modelPath }
    }
    if ($launchMode -eq 'server' -and $llmEnabled) {
        if (-not $visionProjectorPath -or -not (Test-Path -LiteralPath $visionProjectorPath -PathType Leaf)) {
            $visionProjectorPath = Find-VisionProjectorPath -ModelPath $modelPath
        }
        $serverVision = Read-YesNo -Label '启用视觉图片理解？/ Enable image understanding?' -Default $serverVision
        if ($serverVision) {
            Write-Host '视觉权重按图片请求从 SSD 临时载入统一显存，处理完全部图片后立即释放。API 图片可用 data URI、裸 base64，或 http(s) URL。' -ForegroundColor DarkGray
            Write-Host 'Vision weights use request-scoped unified VRAM and are released after the image batch. API images may be data URIs, bare base64, or http(s) URLs.' -ForegroundColor DarkGray
            $visionProjectorPath = Select-VisionProjectorPath -Default $visionProjectorPath
            Write-Host '抓取 URL 是服务端替客户端出网，所以上了三层界：单图 32 MiB、每跳 10 秒、且只允许公网可路由地址——回环、RFC1918、链路本地（云元数据端点就在 169.254.169.254）、CGNAT 与保留段全部拒绝。打开它就是把这台机器所在内网交给请求方探测，默认关。' -ForegroundColor DarkGray
            Write-Host 'A URL fetch is the server going out on the caller''s behalf, so it is capped at 32 MiB and 10 s per hop, and only publicly routable addresses are allowed. Opening this hands your local network to whoever can send a request.' -ForegroundColor DarkGray
            $imageUrlAllowPrivate = Read-YesNo -Label '允许 image_url 抓取本机或局域网地址（NAS、网络摄像头）？/ Allow image_url to fetch loopback or LAN hosts?' -Default $imageUrlAllowPrivate
        }
    }
    if ($launchMode -eq 'server') {
        if ($serverEmbedding) {
            Write-Host 'Embedding 模型与 Backend 在首次请求时加载，空闲超时后卸载整个实例。/ The Embedding model and backend load on the first request; the entire instance unloads after the idle timeout.' -ForegroundColor DarkGray
            $embeddingModelPath = Select-EmbeddingModelPath -Default $embeddingModelPath
            $embeddingIdleTimeout = Read-IntegerValue -Label 'Embedding 空闲释放秒数，0 为服务期间常驻 / Idle eviction seconds, 0 keeps resident' -Default $embeddingIdleTimeout -Minimum 0
        }
        Read-TranscriptionSettings -Settings $transcription
        Read-SpeechSettings -Settings $speech
    }
    if ($llmEnabled -or $serverEmbedding) {
        if ($launchMode -eq 'server') {
            Write-Host 'LLM 和 Embedding 共用所选设备；语音转录固定使用 CPU。/ LLM and Embedding share the selected device; transcription always uses the CPU.' -ForegroundColor DarkGray
        }
        $device = Read-ComputeDevice -Default $device
    }
    if ($llmEnabled) {
        $setupMode = Read-Choice -Label '配置方式 / Configuration' -DefaultValue $setupModeDefault -Options @(
            [pscustomobject]@{ Key = '1'; Value = 'conservative'; Label = '自动配置：保守（推荐）/ Automatic: conservative (recommended)' }
            [pscustomobject]@{ Key = '2'; Value = 'aggressive'; Label = '自动配置：激进性能 / Automatic: aggressive performance' }
            [pscustomobject]@{ Key = '3'; Value = 'manual'; Label = '全手动配置 / Fully manual configuration' }
        )
        if ($setupMode -eq 'manual') {
            Write-Host "`n高级通用设置 / Advanced common settings" -ForegroundColor Cyan
            Write-Host '各项留空即可继续使用引擎的硬件探测与自动预算。Leave values blank to keep engine auto-detection.' -ForegroundColor DarkGray
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
        } elseif ($setupMode -eq 'aggressive') {
            Write-Host "`n将自动探测硬件，并减少 RAM/VRAM 保留、提高 Ubatch、探索更大的 Submit cap。" -ForegroundColor Yellow
            Write-Host 'Hardware stays auto-detected, with tighter RAM/VRAM headroom, a larger Ubatch and a wider Submit cap search.' -ForegroundColor DarkGray
            Write-Host '适合追求吞吐且能接受更高资源压力；启动失败或系统换页时请改用保守档。' -ForegroundColor Yellow
            Write-Host 'Use conservative mode if allocation fails or Windows starts paging.' -ForegroundColor DarkGray
            $kvPreset = 'auto'
            $kvTypeK = ''
            $kvTypeV = ''
        } else {
            Write-Host "`n将自动预算所选设备的显存和 RAM；上下文留空时由引擎自动确定。" -ForegroundColor DarkGray
            Write-Host 'VRAM and RAM budgets use the selected device; leave context blank for engine sizing.' -ForegroundColor DarkGray
            $kvPreset = 'auto'
            $kvTypeK = ''
            $kvTypeV = ''
        }
        if ($setupMode -eq 'manual' -and $configureMemory) {
            Write-Host '大小可写 21GiB、512MiB、80%；留空表示自动。Sizes accept 21GiB, 512MiB or 80%; blank means auto.' -ForegroundColor DarkGray
            $vramBudget = Read-TextValue -Label '总显存预算 / Total VRAM budget' -Default $vramBudget
            $vramReserve = Read-TextValue -Label '额外显存保留 / Additional VRAM reserve' -Default $vramReserve
            $expertCache = Read-TextValue -Label 'GPU 专家缓存 / GPU expert cache' -Default $expertCache
            Write-Host '留空＝自动：按物理内存标定——保守档＝当前空闲减 3 GiB，激进档＝总量减 13 GiB。Windows 的 commit 余量只记录、不绑定：系统托管的页面文件会在两次启动之间移动这条线，同一份 71.73 GiB 的 store 先见过 104.6 GB 的上限、又见过 116.8 GB，拿它标定等于拿将要变长的数丢吞吐。你把熔断的 commit 轴（device.mem_guard_headroom）显式打开时，自动档才改为留 2 GiB commit 余量。填了＝按你说的兑现，引擎不再替你缩减。/ Blank = automatic, sized from PHYSICAL RAM: conservative gives free-minus-3 GiB, aggressive total-minus-13 GiB. Commit headroom is reported, never obeyed, because a system-managed page file moves that line between launches. Arm the commit axis of the breaker (device.mem_guard_headroom) and the automatic store leaves 2 GiB of commit slack instead. A value is honored verbatim; the engine does not trim it for you.' -ForegroundColor DarkGray
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
        if ($setupMode -eq 'manual') {
            $submitMode = Read-Choice -Label 'Submit splitter' -DefaultValue $submitMode -Options @(
                [pscustomobject]@{ Key = '1'; Value = 'auto'; Label = '自动反馈 / Automatic feedback' }
                [pscustomobject]@{ Key = '2'; Value = 'disabled'; Label = '禁用，no-split / Disabled, no-split' }
                [pscustomobject]@{ Key = '3'; Value = 'fixed'; Label = '固定 cap / Fixed cap' }
            )
            if ($submitMode -eq 'fixed') {
                $submitCap = Read-IntegerValue -Label '固定 dispatch cap / Fixed dispatch cap' -Default $submitCap -Minimum 1
            }
        }
        if ($setupMode -eq 'manual') {
            $configureDiagnostics = Read-YesNo -Label '设置统计或 profiler？/ Configure statistics or profilers?' -Default $configureDiagnostics
        }
        if ($setupMode -eq 'manual' -and $configureDiagnostics) {
            $pagerStats = Read-YesNo -Label '输出 pager 命中统计？/ Print pager hit statistics?' -Default $pagerStats
            $pagerProfile = Read-YesNo -Label '启用聚合 pager profiler？/ Enable aggregate pager profiler?' -Default $pagerProfile
            $stageProfile = Read-YesNo -Label '启用阶段计时？/ Enable stage timings?' -Default $stageProfile
            $vramProfile = Read-YesNo -Label '输出实时显存信息？/ Print live VRAM information?' -Default $vramProfile
        }
        if ($setupMode -eq 'manual') {
            $customSets = Read-TextValue -Label '额外 --set，以分号分隔，留空为无 / Extra --set entries separated by semicolons, blank for none' -Default $customSets
        }
        if ($launchMode -ne 'benchmark') {
            $configureThinking = Read-YesNo -Label '是否配置默认思考？/ Configure default reasoning?' -Default $configureThinking
            if ($configureThinking) {
                $thinkingEnabled = Read-YesNo -Label '默认开启思考？/ Enable reasoning by default?' -Default ($thinkMode -ne 'no-think')
                $thinkMode = if ($thinkingEnabled) { 'think' } else { 'no-think' }
                $reasoningEffort = 'default'
                if ($thinkingEnabled) {
                    $efforts = @(Get-ModelReasoningEfforts -ModelPath $modelPath)
                    if ($efforts.Count) {
                        $effortDefault = [string](Get-SavedValue 'reasoning_effort' 'default')
                        if ($effortDefault -ne 'default' -and $effortDefault -notin $efforts) { $effortDefault = 'default' }
                        $effortOptions = @([pscustomobject]@{ Key = '1'; Value = 'default'; Label = '模板默认 / Template default' })
                        foreach ($effort in $efforts) {
                            $effortOptions += [pscustomobject]@{ Key = ($effortOptions.Count + 1).ToString(); Value = $effort; Label = $effort }
                        }
                        $reasoningEffort = Read-Choice -Label '默认思考强度 / Default reasoning effort' -DefaultValue $effortDefault -Options $effortOptions
                    }
                }
            } else {
                $thinkMode = 'default'
                $reasoningEffort = 'default'
            }
            if (-not $maxNew) { $maxNew = '65536' }
            $maxNew = Read-IntegerValue -Label '每轮最大生成 token（思考与正文合计）/ Max generated tokens per reply (reasoning + answer)' -Default $maxNew -Minimum 1
            $configureSampling = Read-YesNo -Label '设置默认采样参数？/ Configure default sampling?' -Default $configureSampling
            if ($configureSampling) {
                $temperature = Read-TextValue -Label 'Temperature，留空为模型默认 / blank for model default' -Default $temperature
                $topK = Read-IntegerValue -Label 'Top-K，留空为模型默认 / blank for model default' -Default $topK -Minimum 0 -AllowBlank
                $topP = Read-TextValue -Label 'Top-P，留空为模型默认 / blank for model default' -Default $topP
                $seed = Read-IntegerValue -Label '随机种子，留空为随机 / Seed, blank for random' -Default $seed -Minimum 0 -AllowBlank
            }
        }

        $cpuMissEnabled = Read-YesNo -Label '启用实验性 CPU 专家 miss 计算？/ Enable experimental CPU expert-miss computation?' -Default ([bool](Get-SavedValue 'cpu_miss_enabled' $false))
        if ($cpuMissEnabled) {
            $cpuTopology = Get-CpuMissTopology
            $defaultCpuMissCores = [Math]::Max(1, $cpuTopology.Physical - 2)
            Write-Host ('物理核心 {0}，性能核心 {1}；默认计算核数为 max(1, 总物理核数 - 2) = {2}。' -f $cpuTopology.Physical, $cpuTopology.Preferred, $defaultCpuMissCores) -ForegroundColor DarkGray
            Write-Host ('Physical cores: {0}; performance cores: {1}; default compute cores: max(1, physical - 2) = {2}.' -f $cpuTopology.Physical, $cpuTopology.Preferred, $defaultCpuMissCores) -ForegroundColor DarkGray
            if ($cpuTopology.Preferred -lt $cpuTopology.Physical) {
                Write-Host '大小核 CPU 优先预留两个大核：其余大核 → 小核 → 预留大核；选满不预留。' -ForegroundColor DarkGray
                Write-Host 'Hybrid CPU: other performance cores, then efficiency cores, then the two reserved performance cores; selecting all cores leaves no reserve.' -ForegroundColor DarkGray
            }
            if (-not $cpuTopology.Detected) {
                Write-Warning '未能识别大小核，默认按总物理核心计算；请确认核心数。Hybrid-core detection unavailable; confirm the compute-core count.'
            }
            $parsedCores = 0
            if (-not [int]::TryParse($cpuMissCores, [ref]$parsedCores) -or $parsedCores -lt 1 -or $parsedCores -gt $cpuTopology.Physical) {
                $cpuMissCores = $defaultCpuMissCores.ToString()
            }
            $parsedMissMax = 0
            if (-not [int]::TryParse($cpuMissMax, [ref]$parsedMissMax) -or $parsedMissMax -lt 1 -or $parsedMissMax -gt 3) { $cpuMissMax = '1' }
            $cpuMissMax = Read-IntegerValue -Label '最多交给 CPU 的 miss 数（1-3）/ Maximum CPU miss count (1-3)' -Default $cpuMissMax -Minimum 1 -Maximum 3
            $cpuMissCores = Read-IntegerValue -Label 'CPU 计算核心数 / CPU compute cores' -Default $cpuMissCores -Minimum 1 -Maximum $cpuTopology.Physical
            Write-Host '要求 AVX2 + FMA3、支持的量化和完整 RAM 专家缓存；超出 miss 上限或其他不支持工况自动使用 GPU。' -ForegroundColor DarkGray
            Write-Host 'Requires AVX2 + FMA3, supported quantization and full-RAM experts; unsupported cases or larger miss sets use GPU.' -ForegroundColor DarkGray
            Write-Host '性能收益尚未保证；2/3 miss 未做整机性能验收。Performance gains are not guaranteed; 2/3-miss model throughput is unvalidated.' -ForegroundColor Yellow
        }
        if ($launchMode -eq 'server') {
            $serverParallel = Read-IntegerValue -Label '并发会话数（每个会话有独立 KV）/ Concurrent slots (one KV cache each)' -Default $serverParallel -Minimum 1
        }

        if ($launchMode -ne 'benchmark') {
            $context = Read-TextValue -Label '上下文窗口，留空为自动 / Context window, blank for auto' -Default $context
        }
        if ($launchMode -eq 'server') {
            $serverSessionCache = Read-YesNo -Label '将闲置会话 KV 缓存到 SSD？/ Cache idle-session KV on SSD?' -Default $serverSessionCache
            if ($serverSessionCache -and (Read-YesNo -Label '修改 SSD KV 缓存参数？/ Customize SSD KV cache settings?' -Default $false)) {
                $sessionCacheDir = Read-TextValue -Label 'KV 缓存目录 / KV cache directory' -Default $sessionCacheDir -Required
                $sessionCacheDir = ConvertTo-FullPath $sessionCacheDir
                $sessionIdleSecs = Read-IntegerValue -Label '会话闲置多少秒后写入 SSD / Spill after idle seconds' -Default $sessionIdleSecs -Minimum 0
                $sessionCacheMax = Read-TextValue -Label 'SSD KV 缓存总上限（绝对大小）/ Total SSD KV cache limit (absolute size)' -Default $sessionCacheMax -Required
                $sessionCacheTtlHours = Read-IntegerValue -Label '缓存保留小时数，0 为不按时间清理 / Cache TTL hours, 0 disables age expiry' -Default $sessionCacheTtlHours -Minimum 0
            }

        }
    }
    if ($launchMode -eq 'server') {
        # The paged/RAM KV tiers are LLM concepts; the author's module interview can now produce an
        # audio-only server with no LLM, so gate these prompts exactly like the KV prompts above.
        if ($llmEnabled) {
        $kvPagedTree = Read-YesNo -Label '改用地基树分页 KV 前缀缓存（试验性，零拷贝帧共享，命中只改写页表）？/ Use the paged radix-frame KV prefix cache instead (experimental, zero-copy frame sharing; a hit rewrites page tables)?' -Default $kvPagedTree
        # The tree IS the segmented KV layout, and that layout only exists on a Q8_0/Q8_0 cache with
        # the RAM overflow off. A manual KV choice outside that pair therefore meant a command line
        # the engine refused to boot, so resolve the conflict here the same way the mutually
        # exclusive RAM/SSD tiers below already do.
        if ($kvPagedTree -and -not (Test-PagedKvEligible -Preset $kvPreset -TypeK $kvTypeK -TypeV $kvTypeV -Overflow $kvOverflow)) {
            $treeBlocker = if ($kvOverflow) { 'KV overflow is on' } else { "KV is $kvTypeK/$kvTypeV" }
            Write-Host "分页树只建立在 Q8_0/Q8_0 动态分段 KV 上：$treeBlocker，已关闭分页树而不是让启动失败。" -ForegroundColor Yellow
            Write-Host "The paged tree only exists on the dynamic segmented Q8_0/Q8_0 layout; $treeBlocker, so it was turned off instead of refusing at startup. Pick KV auto/Q8 and keep overflow off to use it." -ForegroundColor DarkGray
            $kvPagedTree = $false
            $kvPagedCacheMax = ''
            $kvPagedPageTokens = ''
        }
        if ($kvPagedTree) {
            # One prerequisite the wizard cannot test: the corridor is claimed from the unified VRAM
            # arena, and a session whose experts all loaded resident never builds one. Say so before
            # the operator commits, since the startup refusal now names the same condition.
            Write-Host '前置条件：Qwen3.5/3.6/3.8 架构、Q8_0/Q8_0 KV、kv.dynamic 开启、且存在统一显存 arena（专家被分页才会建立；整机全部常驻的显存充裕设备没有 arena）。不满足时启动会拒绝并逐条列出未满足的条件。' -ForegroundColor Yellow
            Write-Host 'Prerequisites: a Qwen3.5/3.6/3.8 model, Q8_0/Q8_0 KV, kv.dynamic on, and a unified VRAM arena (it is only built when MoE experts are paged - a fully resident model on a roomy device has none). A startup that misses a prerequisite now lists exactly which.' -ForegroundColor DarkGray
            # The tree and its host-RAM frame pool are separate choices: with no pool every frame
            # stays inside the VRAM corridor and a cold prefix evicts instead of sinking. The
            # corridor is sized to the LIVE sessions' own context, so without a pool several
            # conversations evict exactly the chain the next request would hit and every revisit
            # reports cached_tokens=0. Recommend the pool on a fresh install; a reconfiguration
            # keeps whatever the operator saved (an explicit decline saves an empty cap).
            Write-Host '推荐开启 RAM 帧池：显存走廊只装得下活跃会话自己的上下文，没有帧池时几个会话的 KV 加起来超过走廊就会互相驱逐，交替请求永远命中不了（cached_tokens=0）。实测三个会话各 1812 token、ctx 4096：开 2GiB 帧池后回访命中 1792/1812（98.9%），单次耗时从约 1200ms 降到约 190ms。' -ForegroundColor Yellow
            Write-Host 'Recommended: the VRAM corridor only holds the live sessions own context, so without a frame pool several conversations evict exactly the chain the next request would hit and every revisit reports cached_tokens=0. Measured with three 1812-token conversations at ctx 4096: a 2GiB pool gives 1792/1812 (98.9%) reuse on revisit and cuts a request from ~1200ms to ~190ms.' -ForegroundColor DarkGray
            $ramPoolDefault = if ($null -eq $script:Saved) { $true } else { [bool]$kvPagedCacheMax }
            $kvPagedRamPool = Read-YesNo -Label '为分页树分配 RAM 帧池（推荐；显存走廊放不下的冷前缀整链 DMA 下沉到内存、命中再换回）？/ Give the paged tree a host-RAM frame pool (recommended; cold chains the VRAM corridor cannot hold DMA down and promote back on a hit)?' -Default $ramPoolDefault
            if ($kvPagedRamPool) {
                if (-not $kvPagedCacheMax) { $kvPagedCacheMax = '16GiB' }
                Write-Host '帧按实际下沉量惰性增长、开机不预占，也不从专家 store 的额度里预扣——分页树建立时按当时的空闲物理内存夹紧帧数（留 1 GiB 不花），夹不住的冷前缀直接逐出、不再下沉。/ Frames grow lazily with actual demotions, reserve nothing at boot and are no longer pre-carved from the expert store budget: on Windows the paged tree clamps the frame count against the free physical RAM it sees, leaving 1 GiB unspent, and a cold prefix that cannot fit evicts instead of sinking.' -ForegroundColor DarkGray
                $kvPagedCacheMax = Read-TextValue -Label '分页 KV 的 RAM 帧池上限 / Paged KV RAM frame-pool cap' -Default $kvPagedCacheMax -Required
            } else {
                $kvPagedCacheMax = ''
                Write-Host '不分配 RAM 帧池：树只使用显存帧走廊，冷前缀直接逐出、不再下沉。多个会话的 KV 加起来超过走廊时会互相驱逐，交替请求的复用率会是 0；只有单个会话在走廊内重放才命中。混合递归模型续接所需的状态快照仍保留 512 MiB 主机内存上限，按快照逐个分配、不预占。' -ForegroundColor Yellow
                Write-Host 'No RAM frame pool: every frame stays in the VRAM corridor and cold prefixes evict instead of sinking. Once several conversations hold more KV than the corridor they evict each other, so alternating requests reuse nothing; only a single conversation replayed inside the corridor hits. Recurrent/PLE state snapshots keep a 512 MiB host cap, claimed one snapshot at a time.' -ForegroundColor DarkGray
                Write-Host '没有 RAM 帧池时磁盘帧层不生效：冷链不经过内存就无法落到磁盘。No disk frame tier without a pool: a cold chain cannot reach the file without a host frame to sink through.' -ForegroundColor DarkGray
            }
            if (-not $kvPagedPageTokens) { $kvPagedPageTokens = '64' }
            $kvPagedPageTokens = Read-TextValue -Label '每帧 token 数（2 的幂，如 64；越小共享越细、重算尾巴越短）/ Tokens per frame (power of two, e.g. 64; smaller shares finer with shorter recompute tails)' -Default $kvPagedPageTokens -Required
            Write-Host '试验性特性：分页走廊占用全部 KV 显存（会话与缓存树共用，不再为每个会话单独保留 KV）。' -ForegroundColor Yellow
            Write-Host 'Experimental: the frame corridor claims the FULL KV VRAM space (resident sessions and the cache tree share it; no per-slot KV reserve).' -ForegroundColor DarkGray
            Write-Host '命中只到帧边界，尾帧重算；多模态会话不入树；与 MTP 联用未验收（draft 状态在树恢复时丢弃，且 draft KV 在走廊之外另占显存），建议关闭 MTP。' -ForegroundColor Yellow
            Write-Host 'Hits are frame-aligned (tail frame recomputed); multimodal sessions never enter the tree; the MTP combination is unvalidated (draft state is dropped on tree restore and the draft KV costs VRAM outside the corridor) - keep MTP off.' -ForegroundColor DarkGray
            if ($kvRamCacheMax) {
                Write-Host '分页缓存开启，RAM 基数树缓存已关闭（两者互斥）。The RAM blob cache was turned off: it is mutually exclusive with the paged cache.' -ForegroundColor Yellow
                $kvRamCacheMax = ''
            }
            if ($serverSessionCache) {
                Write-Host '分页缓存开启，磁盘会话缓存已关闭。The disk session cache was turned off because the paged cache is on.' -ForegroundColor Yellow
                $serverSessionCache = $false
            }
        } else {
            $kvPagedCacheMax = ''
            $kvPagedPageTokens = ''
        }
        $kvRamCacheEnabled = $false
        if (-not $kvPagedTree) {
            $kvRamCacheEnabled = Read-YesNo -Label '改用内存基数树前缀缓存闲置 KV（不落盘，优先于 SSD）？/ Keep idle-session KV in a RAM radix-tree prefix cache instead (no disk, takes precedence over SSD)?' -Default ([bool]$kvRamCacheMax)
        }
        if ($kvRamCacheEnabled) {
            if (-not $kvRamCacheMax) { $kvRamCacheMax = '16GiB' }
            $kvRamCacheMax = Read-TextValue -Label 'RAM KV 缓存上限（绝对大小或内存百分比，如 16GiB / 25%）/ RAM KV cache limit (absolute size or percent, e.g. 16GiB / 25%)' -Default $kvRamCacheMax -Required
            Write-Host '试验性特性：此缓存独立分配，不并入进程总 RAM 预算，会从专家 Host Store 的可用内存中挤占；自动策略或核显设备可能触发换页或内存分配失败，建议使用物理内存充裕的设备。' -ForegroundColor Yellow
            Write-Host 'Experimental: this cache allocates OUTSIDE the process-wide RAM budget and competes with the expert host store; the auto strategy or iGPU devices may page or fail to allocate. Recommended on RAM-rich machines only.' -ForegroundColor DarkGray
            if ($serverSessionCache) {
                Write-Host 'RAM 缓存开启，磁盘会话缓存已关闭。The disk session cache was turned off because the RAM cache is on.' -ForegroundColor Yellow
                $serverSessionCache = $false
            }
        } else {
            $kvRamCacheMax = ''
        }
        }
        Write-Host "`nAPI 服务器 / API server" -ForegroundColor Cyan
        $serverAddr = Read-ListenAddress -Label '监听地址（IP:端口）/ Listen address (IP:port)' -Default $serverAddr
        $serverAuth = Read-YesNo -Label '启用 Bearer API key 鉴权？/ Enable Bearer API-key authentication?' -Default $serverAuth
    }

    if ($launchMode -eq 'benchmark') {
        if ($benchKind -notin @('quick', 'standard', 'full', 'manual')) {
            $manualBenchKind = $benchKind
            $benchKind = 'manual'
        }
        if (-not $script:BenchmarkGuideActive) {
        $benchKind = Read-Choice -Label '测试方案 / Benchmark plan' -DefaultValue $benchKind -Options @(
            [pscustomobject]@{ Key = '1'; Value = 'quick'; Label = '自动快速 / Automatic quick'; Detail = @('6K Prefill + 1024 Decode。单发，一次，可追加并发。', '6K Prefill + 1024 Decode. x1, one run; optional concurrency.') }
            [pscustomobject]@{ Key = '2'; Value = 'standard'; Label = '自动标准 / Automatic standard'; Detail = @('6K、30K Prefill + 1024 Decode；1-2 并发，各一次，共 4 次。', '6K, 30K Prefill + 1024 Decode; x1-x2, one run each, 4 runs. Shared KV, different questions.') }
            [pscustomobject]@{ Key = '3'; Value = 'full'; Label = '自动完整 / Automatic full'; Detail = @('6K、30K、150K Prefill + 1024 Decode；1-4 并发，各一次，共 12 次。', '6K, 30K, 150K Prefill + 1024 Decode; x1-x4, one run each, 12 runs. Shared KV, different questions.') }
            [pscustomobject]@{ Key = '4'; Value = 'manual'; Label = '手动测试 / Manual benchmark'; Detail = @('按顺序添加 Prefill、Decode 和上下文深度。', 'Build one test step by step: Prefill, Decode and context depth.') }
        )
        }
        if ($benchKind -ne 'manual') {
            $profileParallel = switch ($benchKind) { 'quick' { '1' } 'standard' { '2' } default { '4' } }
            if (-not $script:BenchmarkGuideActive) { $benchParallel = Read-IntegerValue -Label '最大并发路数（会依次测试 x1 到 xN）/ Maximum concurrency (tests x1 through xN)' -Default $profileParallel -Minimum 1 -Maximum 4 }
        } else {
            if ($manualBenchKind -notin @('decode', 'prefill', 'mixed')) { $manualBenchKind = 'decode' }
            $manualBenchKind = Read-Choice -Label '第一步：选择要添加的工作 / Step 1: choose work' -DefaultValue $manualBenchKind -Options @(
                [pscustomobject]@{ Key = '1'; Value = 'prefill'; Label = '只测 Prefill / Prefill only'; Detail = @('处理一段提示词，不生成 token。', 'Process a prompt without generating tokens.') }
                [pscustomobject]@{ Key = '2'; Value = 'decode'; Label = '只测 Decode / Decode only'; Detail = @('生成 token；下一步可添加已有上下文。', 'Generate tokens; existing context can be added next.') }
                [pscustomobject]@{ Key = '3'; Value = 'mixed'; Label = 'Prefill + Decode'; Detail = @('同一轮先处理提示词，再生成 token。', 'Process a prompt and then generate in the same turn.') }
            )
            if ($promptTokens -notmatch '^[1-9][0-9]*$') { $promptTokens = '6144' }
            if ($genTokens -notmatch '^[1-9][0-9]*$') { $genTokens = '512' }
            switch ($manualBenchKind) {
            'decode' {
                $promptTokens = '0'
                $genTokens = Read-IntegerValue -Label 'Decode token 数 / Decode tokens' -Default $genTokens -Minimum 1
                Show-ManualBenchmarkPlan -Kind $manualBenchKind -GenTokens $genTokens
            }
            'prefill' {
                $promptTokens = Read-IntegerValue -Label 'Prefill token 数 / Prefill tokens' -Default $promptTokens -Minimum 1
                $genTokens = '0'
                $depthMode = 'none'
                $depthTokens = '0'
                Show-ManualBenchmarkPlan -Kind $manualBenchKind -PromptTokens $promptTokens
            }
            'mixed' {
                $promptTokens = Read-IntegerValue -Label 'Prefill token 数 / Prefill tokens' -Default $promptTokens -Minimum 1
                Show-ManualBenchmarkPlan -Kind $manualBenchKind -PromptTokens $promptTokens -GenTokens '尚未添加 / not added'
                $genTokens = Read-IntegerValue -Label 'Decode token 数 / Decode tokens' -Default $genTokens -Minimum 1
                Show-ManualBenchmarkPlan -Kind $manualBenchKind -PromptTokens $promptTokens -GenTokens $genTokens
            }
            }
            if ($manualBenchKind -ne 'prefill') {
                $depthMode = Read-Choice -Label '下一步：添加测量前上下文 / Next: add context before measurement' -DefaultValue $depthMode -Options @(
                    [pscustomobject]@{ Key = '1'; Value = 'none'; Label = '不添加 / None'; Detail = @('从空 KV 开始测量。', 'Measure from an empty KV cache.') }
                    [pscustomobject]@{ Key = '2'; Value = 'real'; Label = '真实上下文 / Real context'; Detail = @('先真实 Prefill 指定长度，再开始计时。', 'Really prefill the selected depth before timing.') }
                    [pscustomobject]@{ Key = '3'; Value = 'synthetic'; Label = 'Synthetic KV 深度 / Synthetic KV depth'; Detail = @('只构造 KV 深度与注意力形状，不计作真实 Prefill。', 'Create KV depth and attention shape without a real Prefill.') }
                )
                if ($depthMode -ne 'none') {
                    if ($depthTokens -notmatch '^[1-9][0-9]*$') { $depthTokens = '6144' }
                    $depthTokens = Read-IntegerValue -Label '已有上下文 token 数 / Existing context tokens' -Default $depthTokens -Minimum 1
                } else {
                    $depthTokens = '0'
                }
                Show-ManualBenchmarkPlan -Kind $manualBenchKind -PromptTokens $promptTokens -GenTokens $genTokens -DepthMode $depthMode -DepthTokens $depthTokens
            }
            $reps = Read-IntegerValue -Label '重复次数 / Repetitions' -Default $reps -Minimum 1
            $jsonOutput = $true
            Show-ManualBenchmarkPlan -Kind $manualBenchKind -PromptTokens $promptTokens -GenTokens $genTokens -DepthMode $depthMode -DepthTokens $depthTokens -Repetitions $reps
            Write-Host '结果会自动保存为 JSON。Results will be saved as JSON automatically.' -ForegroundColor DarkGray
        }
    }
}

if ($launchMode -eq 'server') {
    if ($serverAuth) {
        if ($script:DashboardAccepted -and $script:DashboardApiKey) { $serverApiKey = $script:DashboardApiKey }
        else { $serverApiKey = Read-SecretValue -Label 'API key（隐藏输入且不会保存）/ API key (hidden and not saved)' }
    } elseif (-not (Test-IsLoopbackAddress $serverAddr)) {
        Write-Host '警告：该监听地址可能被其他设备访问，且当前未启用鉴权。' -ForegroundColor Yellow
        Write-Host 'Warning: this address may be reachable by other devices and authentication is disabled.' -ForegroundColor Yellow
        if (-not (Read-YesNo -Label '仍然继续？/ Continue anyway?' -Default $false)) { exit 0 }
    }
}

if ($launchMode -eq 'server' -and -not ($serverLlmEnabled -or $serverEmbedding -or $transcription.Enabled -or $speech.Enabled)) {
    throw '请至少启用一个 API 模块。/ Enable at least one API module.'
}
$nativeArgs = [System.Collections.Generic.List[string]]::new()
[void]$nativeArgs.Add($(switch ($launchMode) {
    'benchmark' { 'bench' }
    'server' { 'serve' }
    default { 'run' }
}))
$commandDevice = if ($llmEnabled -or $serverEmbedding) { $device } else { 'cpu' }
if (-not [string]::IsNullOrWhiteSpace($commandDevice)) { [void]$nativeArgs.Add('--dev'); [void]$nativeArgs.Add($commandDevice) }
if ($llmEnabled) {
    if ($setupMode -eq 'manual') {
        if (-not [string]::IsNullOrWhiteSpace($configPath)) { [void]$nativeArgs.Add('--config'); [void]$nativeArgs.Add($configPath) }
        if (-not [string]::IsNullOrWhiteSpace($ubatch)) { [void]$nativeArgs.Add('--ubatch'); [void]$nativeArgs.Add($ubatch) }
        if (-not [string]::IsNullOrWhiteSpace($threads)) { [void]$nativeArgs.Add('--threads'); [void]$nativeArgs.Add($threads) }
    }
    if ($launchMode -ne 'benchmark' -and -not [string]::IsNullOrWhiteSpace($context)) {
        [void]$nativeArgs.Add('--ctx'); [void]$nativeArgs.Add($context)
    }

    if ($setupMode -eq 'conservative' -or $setupMode -eq 'aggressive') {
        Add-SetArgument $nativeArgs 'device.auto_profile' $setupMode
        # Also migrate direct reuse of older automatic settings; the engine owns format support.
        $kvPreset = 'auto'
        $kvTypeK = ''
        $kvTypeV = ''
    }

    if ($kvPreset -ne 'auto') {
        Add-SetArgument $nativeArgs 'kv.type_k' $kvTypeK
        Add-SetArgument $nativeArgs 'kv.type_v' $kvTypeV
    }
    if ($setupMode -eq 'manual' -and $configureMemory) {
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
    if ($setupMode -eq 'manual') {
        switch ($submitMode) {
            'disabled' { Add-SetArgument $nativeArgs 'device.submit_dispatches' '0' }
            'fixed' { Add-SetArgument $nativeArgs 'device.submit_dispatches' $submitCap }
        }
    }
    if ($setupMode -eq 'manual' -and $configureDiagnostics) {
        Add-SetArgument $nativeArgs 'paging.stats' $pagerStats.ToString().ToLowerInvariant()
        Add-SetArgument $nativeArgs 'prof.pager_profile' $pagerProfile.ToString().ToLowerInvariant()
        Add-SetArgument $nativeArgs 'prof.stages' $stageProfile.ToString().ToLowerInvariant()
        Add-SetArgument $nativeArgs 'prof.vram' $vramProfile.ToString().ToLowerInvariant()
    }
    Add-SetArgument $nativeArgs 'kernels.vulkan.cpu_miss_threads' $(if ($cpuMissEnabled) { $cpuMissCores } else { '0' })
    if ($cpuMissEnabled) {
        Add-SetArgument $nativeArgs 'kernels.vulkan.cpu_miss_max' $cpuMissMax
        Add-SetArgument $nativeArgs 'kernels.vulkan.cpu_miss_push' 'true'
        Add-SetArgument $nativeArgs 'kernels.vulkan.cpu_miss_host_result' 'true'
        Add-SetArgument $nativeArgs 'kernels.vulkan.cpu_miss_token_park' 'true'
        Add-SetArgument $nativeArgs 'kernels.vulkan.cpu_miss_spin' '262144'
    }
    if ($setupMode -eq 'manual' -and -not [string]::IsNullOrWhiteSpace($customSets)) {
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
            if ($settingPath.StartsWith('kernels.vulkan.cpu_miss_', [System.StringComparison]::Ordinal) -and
                @($nativeArgs | Where-Object { $_.StartsWith("$settingPath=", [System.StringComparison]::Ordinal) }).Count -gt 0) {
                Write-Warning "CPU-miss 参数 $settingPath 已由向导设置，忽略同名额外配置。The wizard controls this CPU-miss setting; the duplicate extra setting is ignored."
                continue
            }
            Add-SetArgument $nativeArgs $settingPath $entry.Substring($equals + 1).Trim()
        }
    }
    if ($launchMode -ne 'benchmark') {
        # The engine still supports MTP through manual CLI/config use, but the release wizard keeps a
        # deterministic non-MTP path and overrides stale saved or environment configuration.
        Add-SetArgument $nativeArgs 'spec.mtp' 'false'
    }
}

if ($launchMode -eq 'benchmark') {
    if ($benchKind -in @('quick', 'standard', 'full')) {
        [void]$nativeArgs.Add('--suite'); [void]$nativeArgs.Add($benchKind)
        [void]$nativeArgs.Add('--parallel'); [void]$nativeArgs.Add($benchParallel)
        [void]$nativeArgs.Add('--result-dir'); [void]$nativeArgs.Add((Join-Path $repoRoot 'benchmark-database'))
    } elseif ($manualBenchKind -eq 'mixed') {
        [void]$nativeArgs.Add('--pg'); [void]$nativeArgs.Add("$promptTokens,$genTokens")
    } else {
        [void]$nativeArgs.Add('-p'); [void]$nativeArgs.Add($promptTokens)
        [void]$nativeArgs.Add('-n'); [void]$nativeArgs.Add($genTokens)
    }
    if ($benchKind -eq 'manual') {
        switch ($depthMode) {
            'real' { [void]$nativeArgs.Add('-d'); [void]$nativeArgs.Add($depthTokens) }
            'synthetic' { [void]$nativeArgs.Add('--synthetic-depth'); [void]$nativeArgs.Add($depthTokens) }
        }
        [void]$nativeArgs.Add('-r'); [void]$nativeArgs.Add($reps)
        [void]$nativeArgs.Add('--json')
    }
} elseif ($llmEnabled) {
    switch ($thinkMode) {
        'think' { [void]$nativeArgs.Add('--think') }
        'no-think' { [void]$nativeArgs.Add('--no-think') }
    }
    if ($thinkMode -ne 'no-think' -and $reasoningEffort -ne 'default') {
        [void]$nativeArgs.Add('--reasoning-effort'); [void]$nativeArgs.Add($reasoningEffort)
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
    if ($llmEnabled) {
        [void]$nativeArgs.Add('--parallel'); [void]$nativeArgs.Add($serverParallel)
        # Reuse mode never re-runs the question campaign, so the eligibility the fresh path enforced
        # while asking has to be enforced here too: a state file saved with F16 KV (or overflow) plus a
        # tree would otherwise emit a command line the engine refuses to boot. Correcting $kvPagedTree
        # before the state write also heals the saved profile.
        if ($kvPagedTree -and -not (Test-PagedKvEligible -Preset $kvPreset -TypeK $kvTypeK -TypeV $kvTypeV -Overflow $kvOverflow)) {
            $treeBlocker = if ($kvOverflow) { 'KV overflow is on' } else { "KV is $kvTypeK/$kvTypeV" }
            Write-Host "分页树只建立在 Q8_0/Q8_0 动态分段 KV 上：$treeBlocker，已关闭分页树而不是让启动失败。" -ForegroundColor Yellow
            Write-Host "The paged tree only exists on the dynamic segmented Q8_0/Q8_0 layout; $treeBlocker, so it was turned off instead of refusing at startup. Pick KV auto/Q8 and keep overflow off to use it." -ForegroundColor DarkGray
            $kvPagedTree = $false
            $kvPagedCacheMax = ''
        }
        if ($kvPagedTree) {
            # `kv.paged` opens the tree; the host-RAM frame pool is a separate budget that only exists
            # when the wizard named one. Pages must be smaller than the legacy 32K segment or the engine
            # rejects the config, so a state file that predates the split still gets a usable geometry.
            Add-SetArgument $nativeArgs 'kv.paged' 'true'
            if (-not $kvPagedPageTokens) { $kvPagedPageTokens = '64' }
            Add-SetArgument $nativeArgs 'kv.paged_page_tokens' $kvPagedPageTokens
            if ($kvPagedCacheMax) { Add-SetArgument $nativeArgs 'kv.paged_cache_max' $kvPagedCacheMax }
        }
        if ($kvRamCacheMax) {
            Add-SetArgument $nativeArgs 'kv.ram_cache_max' $kvRamCacheMax
        }
        if ($serverSessionCache) {
            Add-SetArgument $nativeArgs 'kv.session_cache_dir' $sessionCacheDir
            Add-SetArgument $nativeArgs 'kv.session_idle_secs' $sessionIdleSecs
            Add-SetArgument $nativeArgs 'kv.session_cache_max' $sessionCacheMax
            Add-SetArgument $nativeArgs 'kv.session_cache_ttl_hours' $sessionCacheTtlHours
        } else {
            # An explicit empty CLI layer disables a cache inherited from infr.toml or the environment.
            Add-SetArgument $nativeArgs 'kv.session_cache_dir' ''
        }
        if ($serverVision) {
            [void]$nativeArgs.Add('--mmproj'); [void]$nativeArgs.Add($visionProjectorPath)
            # Always emitted, both ways: an answer of No must also be able to turn off a value the
            # operator set in infr.toml or the environment, exactly like the other boolean switches.
            Add-SetArgument $nativeArgs 'vision.image_url_allow_private' $imageUrlAllowPrivate.ToString().ToLowerInvariant()
        }
    }
    if ($serverEmbedding) {
        [void]$nativeArgs.Add('--embedding-model'); [void]$nativeArgs.Add($embeddingModelPath)
        [void]$nativeArgs.Add('--embedding-idle-timeout'); [void]$nativeArgs.Add($embeddingIdleTimeout)
    }
    Add-TranscriptionArguments -Arguments $nativeArgs -Settings $transcription
    Add-SpeechArguments -Arguments $nativeArgs -Settings $speech
}
# Uncensor: the architecture gate is the reasoning probe (only qwen4exp exposes low/medium/xhigh), so
# a model the projection cannot apply to is never even asked about. A saved ON is reused WITHOUT
# re-asking — the yes that saved it already carried the consent — but the notice is printed again,
# because "what this server answers is on you" has to be visible on the run that does it. This sits
# before the model path: a named file is the request for the projection, so handing it over IS the
# opt-in, and it happens outside the manual-memory block so automatic profiles carry it too.
if ($llmEnabled -and $null -ne $script:ModelReasoningProbe -and $script:ModelReasoningProbe.Async.IsCompleted) {
    [void](Get-ModelReasoningEfforts -ModelPath $modelPath)
}
$uncensorVector = [string](Get-SavedValue 'uncensor_vector' '')
if ($launchMode -ne 'benchmark' -and (@($script:ReasoningProbeEfforts) -contains 'xhigh')) {
    if ($uncensorVector) {
        Show-UncensorNotice
        Write-Host "沿用上次开启 / staying on from the last run: $uncensorVector" -ForegroundColor DarkGray
    } elseif (-not $reuseSavedSettings) {
        Show-UncensorNotice
        if (Read-YesNo -Label '开启去审查投影？/ Enable the refusal-direction projection?' -Default $false) {
            $fetched = Get-UncensorDirectionVector
            if ($fetched) {
                $uncensorVector = $fetched
            } else {
                Write-Host '未能取得方向文件，本次不开启 / no direction vector could be fetched; staying off' -ForegroundColor Yellow
            }
        }
    }
}
if ($uncensorVector) {
    Add-SetArgument $nativeArgs 'uncensor.vector' $uncensorVector
    Write-Host '去审查投影：开 / uncensor: ON' -ForegroundColor Yellow
}
# An audio-only server has no LLM model path; only append it when the LLM is enabled.
if ($llmEnabled) { [void]$nativeArgs.Add($modelPath) }

$commandText = '& ' + (Format-PowerShellArgument $infrPath) + ' ' + (($nativeArgs | ForEach-Object { Format-PowerShellArgument $_ }) -join ' ')
$state = [ordered]@{
    wizard_language = [string](Get-SavedValue 'wizard_language' '')
    wizard_theme = [string](Get-SavedValue 'wizard_theme' 'system')
    download_dir = Get-ModelDownloadDirectory
    launch_mode = $launchMode; setup_mode = $setupMode; model = $modelPath
    model_history = @(if ($llmEnabled) { Get-UpdatedModelHistory -SelectedModel $modelPath } else { Get-SavedValue 'model_history' @() })
    device = $device; context = $context
    ubatch = $ubatch; threads = $threads; config_path = $configPath
    kv_preset = $kvPreset; kv_type_k = $kvTypeK; kv_type_v = $kvTypeV
    configure_memory = $configureMemory; vram_budget = $vramBudget; vram_reserve = $vramReserve
    expert_cache = $expertCache; ram_budget = $ramBudget; host_dma = $hostDma
    pager_ring = $pagerRing; pager_ring_slots = $pagerRingSlots
    kv_overflow = $kvOverflow; kv_overflow_vram_mb = $kvOverflowVram; kv_overflow_reserve_mb = $kvOverflowReserve
    submit_mode = $submitMode; submit_cap = $submitCap
    configure_diagnostics = $configureDiagnostics; pager_stats = $pagerStats
    pager_profile = $pagerProfile; stage_profile = $stageProfile; vram_profile = $vramProfile
    bench_kind = $benchKind; bench_parallel = $benchParallel; manual_bench_kind = $manualBenchKind
    prompt_tokens = $promptTokens; gen_tokens = $genTokens
    depth_mode = $depthMode; depth_tokens = $depthTokens; reps = $reps; json_output = $jsonOutput
    uncensor_vector = $uncensorVector
    benchmark_upload = $script:BenchmarkUploadDefault
    benchmark_upload_has_github = $script:BenchmarkUploadHasGithubDefault
    benchmark_upload_gh_auth = $script:BenchmarkUploadGhAuthDefault
    benchmark_upload_fallback = $script:BenchmarkUploadFallbackDefault
    submission_browser_action = $script:SubmissionBrowserActionDefault
    submission_failure_action = $script:SubmissionFailureActionDefault
    submission_pending_retry = $script:SubmissionPendingRetryDefault
    error_upload = $script:ErrorUploadDefault
    think_mode = $thinkMode; configure_thinking = $configureThinking
    max_new = $maxNew; configure_sampling = $configureSampling
    reasoning_effort = $reasoningEffort
    reasoning_probe_model = $script:ReasoningProbeModel
    reasoning_probe_length = $script:ReasoningProbeLength
    reasoning_probe_mtime_ticks = $script:ReasoningProbeMtimeTicks
    reasoning_probe_efforts = @($script:ReasoningProbeEfforts)
    temperature = $temperature; top_k = $topK; top_p = $topP; seed = $seed
    server_addr = $serverAddr; server_parallel = $serverParallel; server_auth = $serverAuth
    server_session_cache = $serverSessionCache; session_cache_dir = $sessionCacheDir
    session_idle_secs = $sessionIdleSecs; session_cache_max = $sessionCacheMax
    session_cache_ttl_hours = $sessionCacheTtlHours; kv_ram_cache_max = $kvRamCacheMax
    kv_paged_cache_max = $kvPagedCacheMax; kv_paged_page_tokens = $kvPagedPageTokens
    kv_paged_tree = $kvPagedTree
    server_vision = $serverVision; vision_projector = $visionProjectorPath
    image_url_allow_private = $imageUrlAllowPrivate
    server_llm_enabled = $serverLlmEnabled
    server_embedding = $serverEmbedding; embedding_model = $embeddingModelPath
    embedding_idle_timeout = $embeddingIdleTimeout
    server_transcription = $transcription.Enabled; transcription_model = $transcription.Model
    transcription_threads = $transcription.Threads; transcription_idle_timeout = $transcription.IdleTimeout
    transcription_upload_dir = $transcription.UploadDir
    server_speech = $speech.Enabled; speech_model = $speech.Model; speech_lexicon = $speech.Lexicon
    speech_threads = $speech.Threads; speech_idle_timeout = $speech.IdleTimeout; speech_max_output_seconds = $speech.MaxOutputSeconds
    cpu_miss_enabled = $cpuMissEnabled; cpu_miss_max = $cpuMissMax; cpu_miss_cores = $cpuMissCores
    custom_sets = $customSets; last_command = $commandText
}
if (-not $llmEnabled) { Restore-DisabledLlmState -State $state }
if (-not $DryRun -and -not $script:PendingSubmissionPrompted) { Offer-PendingSubmissions }
Write-Host "`n最终命令 / Final command" -ForegroundColor Cyan
Write-Host $commandText -ForegroundColor DarkGray
if ($launchMode -eq 'server' -and $serverAuth) {
    Write-Host 'API key 将通过当前子进程环境传入，未显示在命令中，也不会保存。' -ForegroundColor DarkGray
    Write-Host 'The API key is passed through the child-process environment; it is hidden above and not saved.' -ForegroundColor DarkGray
}
$startLabel = if ($launchMode -eq 'server') { '现在启动服务器？/ Start the server now?' } else { '现在启动？/ Start now?' }
if (-not $DryRun -and -not $reuseSavedSettings -and -not (Read-YesNo -Label $startLabel -Default $true)) {
    exit 0
}
if (-not $DryRun -and -not $script:BenchmarkGuideActive) {
    New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
    $state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statePath -Encoding UTF8
    Write-Host "`n设置已保存 / Settings saved: $statePath" -ForegroundColor DarkGray
}
if ($DryRun) {
    Write-Host 'DryRun：未启动。DryRun: command was not started.' -ForegroundColor Yellow
    exit 0
}
$executionPlan = New-WizardExecutionPlan
if ($script:DashboardAccepted -and -not $ClassicWizard -and -not $DesktopExecutionPlanPath) {
    $desktopLanguage = [string]$script:Saved.wizard_language
    $desktopEnglish = $desktopLanguage -eq 'en' -or (-not $desktopLanguage -and [Globalization.CultureInfo]::CurrentUICulture.TwoLetterISOLanguageName -ne 'zh')
    $desktopExitCode = Invoke-WizardDesktopExecution -Plan $executionPlan -English $desktopEnglish
    # Internal CMD sentinels avoid pause/console output after the native completion screen.
    if ($desktopExitCode -eq 0) { exit 45 }
    exit 46
}
Invoke-WizardExecutionPlan -Plan $executionPlan
