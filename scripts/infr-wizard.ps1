[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$InitialModelPath = '',
    [switch]$DryRun,
    [switch]$SkipUpdateCheck,
    [string]$PostUpdateProbePath = '',
    [string]$ExpectedPostUpdateVersion = '',
    [switch]$ExitAfterPostUpdateProbe
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
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
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
    $flashDirectory = 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64'
    $flashFiles = @()
    for ($i = 1; $i -le 33; $i++) {
        $fileName = 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-{0:D5}-of-00033.gguf' -f $i
        $flashFiles += [pscustomobject]@{
            Url = "https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/$flashDirectory/$fileName`?download=true"
            RelativePath = Join-Path $flashDirectory $fileName
        }
    }

    return @(
        [pscustomobject]@{
            Id = 'qwen35-balanced'
            Kind = 'llm'
            Name = 'Qwen3.6 35B APEX-I-Balanced'
            InfoUrl = 'https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/blob/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf'
            Primary = 'Qwen3.6-35B-A3B-APEX-I-Balanced\Qwen3.6-35B-A3B-APEX-I-Balanced.gguf'
            Files = @([pscustomobject]@{
                Url = 'https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true'
                RelativePath = 'Qwen3.6-35B-A3B-APEX-I-Balanced\Qwen3.6-35B-A3B-APEX-I-Balanced.gguf'
            })
        }
        [pscustomobject]@{
            Id = 'flashnext-427'
            Kind = 'llm'
            Name = 'Qwen3.8 Flash-Next AD-4.27bpw-Q4_K_M-M64（33 个分片 / 33 shards）'
            InfoUrl = "https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/main/$flashDirectory"
            Primary = "$flashDirectory\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
            Files = $flashFiles
        }
        [pscustomobject]@{
            Id = 'flashnext-vision'
            Kind = 'vision'
            Name = 'Qwen3.8 Flash-Next F16 视觉文件 / vision projector'
            InfoUrl = 'https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/blob/main/mmproj-Qwen3.8-Flash-Next-F16.gguf'
            Primary = "$flashDirectory\mmproj-Qwen3.8-Flash-Next-F16.gguf"
            Files = @([pscustomobject]@{
                Url = 'https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true'
                RelativePath = "$flashDirectory\mmproj-Qwen3.8-Flash-Next-F16.gguf"
            })
        }
        [pscustomobject]@{
            Id = 'qwen3-embedding-06b-q8'
            Kind = 'embedding'
            Name = 'Qwen3-Embedding 0.6B Q8_0'
            InfoUrl = 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/blob/main/Qwen3-Embedding-0.6B-Q8_0.gguf'
            Primary = 'Qwen3-Embedding-0.6B-Q8_0\Qwen3-Embedding-0.6B-Q8_0.gguf'
            Files = @([pscustomobject]@{
                Url = 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true'
                RelativePath = 'Qwen3-Embedding-0.6B-Q8_0\Qwen3-Embedding-0.6B-Q8_0.gguf'
            })
        }
    )
}

function Show-RecommendedModelDownloads {
    param([ValidateSet('all', 'llm', 'vision', 'embedding')][string]$Kind = 'all')

    Write-Host "`n官方推荐的模型 / Official recommended models" -ForegroundColor Cyan
    foreach ($download in @(Get-RecommendedModelCatalog)) {
        if ($Kind -ne 'all' -and $download.Kind -ne $Kind) { continue }
        Write-Host " $($download.Name)"
        Write-Host "   $($download.InfoUrl)" -ForegroundColor DarkGray
    }
    Write-Host '在模型列表选择 D，可一次选择一个或多个项目下载到安装目录下的 models 文件夹。' -ForegroundColor DarkGray
    Write-Host 'Choose D in the model list to download one or more items into the models folder beside the launcher.' -ForegroundColor DarkGray
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
        [switch]$ReplaceExisting
    )

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $existingFile = Get-Item -LiteralPath $Destination
        if (-not $ReplaceExisting -and $existingFile.Length -gt 0) {
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
            $request.Timeout = 30000
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
            Move-Item -LiteralPath $partialPath -Destination $Destination -Force
            $completeArguments = @{ Id = $ProgressId; Activity = $Label; Completed = $true }
            if ($ParentProgressId -ge 0) { $completeArguments.ParentId = $ParentProgressId }
            Write-Progress @completeArguments
            return
        } catch {
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
        [int]$MaximumAttempts = 3
    )

    Invoke-FileDownloadWithProgress `
        -Url $Url `
        -Destination $Destination `
        -Label $Label `
        -MaximumAttempts $MaximumAttempts `
        -ProgressId 71 `
        -ParentProgressId 70
}

function Invoke-RecommendedModelDownload {
    param([Parameter(Mandatory = $true)][string]$Id)

    $bundle = @(Get-RecommendedModelCatalog | Where-Object { $_.Id -eq $Id }) | Select-Object -First 1
    if ($null -eq $bundle) { throw "Unknown recommended download: $Id" }
    $modelsRoot = Join-Path $repoRoot 'models'
    $files = @($bundle.Files)
    Write-Host "`n下载 / Download: $($bundle.Name)" -ForegroundColor Cyan
    Write-Host "目录 / Folder: $modelsRoot" -ForegroundColor DarkGray
    try {
        for ($i = 0; $i -lt $files.Count; $i++) {
            $file = $files[$i]
            $destination = Join-Path $modelsRoot $file.RelativePath
            $activity = "[$($i + 1)/$($files.Count)] $([System.IO.Path]::GetFileName($destination))"
            Write-Progress -Id 70 -Activity $bundle.Name -Status $activity -PercentComplete ([int](100 * $i / $files.Count))
            Invoke-ModelFileDownload -Url $file.Url -Destination $destination -Label $activity
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

function Select-RecommendedDownloads {
    $catalog = @(Get-RecommendedModelCatalog)
    while ($true) {
        Write-Host "`n下载模型与组件 / Download models and components" -ForegroundColor Cyan
        for ($i = 0; $i -lt $catalog.Count; $i++) {
            $bundle = $catalog[$i]
            $primary = Join-Path (Join-Path $repoRoot 'models') $bundle.Primary
            $status = if (Test-Path -LiteralPath $primary -PathType Leaf) { ' [已下载 / installed]' } else { '' }
            Write-Host (" [{0}] {1}{2}" -f ($i + 1), $bundle.Name, $status)
            Write-Host "     $($bundle.InfoUrl)" -ForegroundColor DarkGray
        }
        Write-Host ' [A] 全部 / All'
        Write-Host ' [B] 返回 / Back'
        $raw = (Read-Host '选择一个或多个编号，用逗号分隔 / Select one or more numbers, comma-separated').Trim()
        if ($raw -match '^(b|back|返回)$') { return @() }

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
        $candidate = Join-Path (Join-Path $repoRoot 'models') $bundle.Primary
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
                $candidate = Join-Path (Join-Path $repoRoot 'models') $bundle.Primary
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
    try { $cpuName = [string](Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name) } catch { }
    try { $ramBytes = [uint64](Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory) } catch { }
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
    Write-Host 'AI/自动化：请向用户展示结果并等待其明确决定是否上传；不要自行上传或静默忽略。' -ForegroundColor Yellow
    Write-Host 'AI/automation: report the result and wait for the user to explicitly decide whether to upload; do not upload or silently skip.' -ForegroundColor Yellow
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
        $nativeOutput = @(& $Executable @Arguments 2>&1)
        $nativeExitCode = $LASTEXITCODE
        $textOutput = @($nativeOutput | ForEach-Object { [string]$_ })
        return [pscustomobject]@{ ExitCode = $nativeExitCode; Output = [string[]]$textOutput }
    } catch {
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
    if ([string]::IsNullOrWhiteSpace($statePath)) { return }
    try {
        $preferences = $script:state
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
    $answer = Read-YesNo -Label '愿意公开提交这份 benchmark 结果吗？/ Submit this benchmark result publicly?' -Default $script:BenchmarkUploadDefault
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

function Invoke-Moe4AllSubmissionApi {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('GET', 'POST')][string]$Method,
        [Parameter(Mandatory = $true)][Uri]$Uri,
        [AllowEmptyString()][string]$Token = '',
        $JsonBody = $null,
        [byte[]]$FileBytes = $null
    )

    Add-Type -AssemblyName System.Net.Http
    $maximumRetries = 3
    for ($attempt = 0; $attempt -le $maximumRetries; $attempt++) {
        $handler = [System.Net.Http.HttpClientHandler]::new()
        $handler.AllowAutoRedirect = $false
        $proxy = Get-SubmissionProxy -ProbeUri $Uri
        if ($null -ne $proxy) { $handler.Proxy = $proxy }
        $client = [System.Net.Http.HttpClient]::new($handler)
        $client.Timeout = [TimeSpan]::FromSeconds(30)
        $request = $null
        $response = $null
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
            $response = $client.SendAsync($request).GetAwaiter().GetResult()
            $status = [int]$response.StatusCode
            $bodyText = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            if ($bodyText.Length -gt 1048576) { throw [System.Exception]::new('提交服务响应超过 1 MiB，已拒绝处理。Submission service response exceeded 1 MiB.') }
            $value = $null
            if (-not [string]::IsNullOrWhiteSpace($bodyText)) {
                try { $value = $bodyText | ConvertFrom-Json } catch {
                    if ($status -ge 200 -and $status -lt 300) { throw [System.Exception]::new('提交服务返回了无效 JSON。The submission service returned invalid JSON.') }
                }
            }
            if ($status -ge 200 -and $status -lt 300) {
                if ($null -eq $value) { throw [System.Exception]::new('提交服务返回了空响应。The submission service returned an empty response.') }
                return [pscustomobject]@{ StatusCode = $status; Value = $value }
            }

            $code = ''
            $message = ''
            $retryable = $status -eq 429 -or $status -ge 500
            if ($null -ne $value -and $null -ne $value.error) {
                $code = [string]$value.error.code
                $message = [string]$value.error.message
                if ($value.error.PSObject.Properties.Name -contains 'retryable') { $retryable = $retryable -or [bool]$value.error.retryable }
            }
            if (-not $retryable -or $attempt -eq $maximumRetries) {
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
            Start-Sleep -Milliseconds ([int]($delaySeconds * 1000 + (Get-Random -Minimum 0 -Maximum 501)))
        } catch {
            $statusData = $_.Exception.Data['StatusCode']
            if ($null -ne $statusData -or $attempt -eq $maximumRetries) { throw }
            Start-Sleep -Milliseconds ([int]([Math]::Pow(2, $attempt + 1) * 1000 + (Get-Random -Minimum 0 -Maximum 501)))
        } finally {
            if ($null -ne $response) { $response.Dispose() }
            if ($null -ne $request) { $request.Dispose() }
            $client.Dispose()
            $handler.Dispose()
        }
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
        Start-Sleep -Seconds $pollSeconds
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
    $receiptPath = Save-Moe4AllSubmissionReceipt -Receipt $receipt -ApiBase $config.ApiBase
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
        $default = $script:SubmissionBrowserActionDefault
        if ($default -notin @('sent', 'reopen', 'cloudflare', 'later')) { $default = 'later' }
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

    try { [void](Test-SubmissionForUpload -Path $Path -Kind $Kind) } catch { Write-Warning $_.Exception.Message; return $false }
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
                    } catch { Write-Warning "GitHub CLI 上传失败：$($_.Exception.Message) / GitHub CLI upload failed." }
                    if ($script:SubmissionUploadCancelled -or $script:SubmissionValidationFailed) { return $false }
                }
            }
            Write-Host '没有可用的已登录 CLI 或上传未完成，改用 GitHub 网页。No signed-in CLI is available or CLI upload did not complete; opening GitHub.' -ForegroundColor Cyan
            $channel = 'browser'
        }
        if ($channel -eq 'browser') {
            try { Open-SubmissionBrowserUpload -Path $Path -Kind $Kind } catch {
                Write-Warning "未能自动打开 GitHub，请保留本地报告。Could not open GitHub: $($_.Exception.Message)"
            }
            Write-Host '完成网页操作后再选择；向导不会查询网页提交结果。Choose after completing the web steps; the wizard does not query their result.'
            $action = Read-SubmissionNextAction -Browser
            if ($action -eq 'sent') { Remove-PendingSubmission -Path $pendingPath; return $true }
            if ($action -eq 'reopen') { continue }
            if ($action -eq 'cloudflare') { $channel = 'cloudflare' } else { break }
        }
        if ($channel -eq 'cloudflare') {
            try {
                if (Invoke-WebsiteSubmissionUpload -Path $Path -Kind $Kind) { Remove-PendingSubmission -Path $pendingPath; return $true }
            } catch { Write-Warning "Cloudflare 中转失败：$($_.Exception.Message) / Cloudflare relay failed." }
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

function Offer-PendingSubmissions {
    if ($script:PendingSubmissionPrompted) { return }
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

function ConvertFrom-SmbiosMemoryType {
    param([int]$Type)
    switch ($Type) {
        20 { 'DDR' }
        21 { 'DDR2' }
        22 { 'DDR2 FB-DIMM' }
        24 { 'DDR3' }
        26 { 'DDR4' }
        27 { 'LPDDR' }
        28 { 'LPDDR2' }
        29 { 'LPDDR3' }
        30 { 'LPDDR4' }
        34 { 'DDR5' }
        35 { 'LPDDR5' }
        default { $null }
    }
}

function Get-PcieTheoreticalBandwidthGbps {
    param([int]$Generation, [int]$Width)
    if ($Width -le 0) { return $null }
    $perLane = switch ($Generation) {
        1 { 0.250 }
        2 { 0.500 }
        3 { 0.985 }
        4 { 1.969 }
        5 { 3.938 }
        6 { 7.563 }
        default { return $null }
    }
    return [Math]::Round($perLane * $Width, 3)
}

function ConvertFrom-StorageBusType {
    param([int]$Type)
    switch ($Type) {
        3 { 'ATA' }
        7 { 'USB' }
        8 { 'RAID' }
        10 { 'SAS' }
        11 { 'SATA' }
        12 { 'SD' }
        13 { 'MMC' }
        17 { 'NVMe' }
        18 { 'SCM' }
        19 { 'UFS' }
        default { $null }
    }
}

function ConvertFrom-StorageMediaType {
    param([int]$Type)
    switch ($Type) {
        3 { 'HDD' }
        4 { 'SSD' }
        5 { 'SCM' }
        default { $null }
    }
}

function Get-PhysicalDiskReadCounter {
    param([Parameter(Mandatory = $true)][int]$DiskIndex)
    try {
        $prefix = "^$([regex]::Escape([string]$DiskIndex))(?:\s|$)"
        return Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk -ErrorAction Stop |
            Where-Object { [string]$_.Name -match $prefix } |
            Select-Object -First 1
    } catch {
        return $null
    }
}

function New-PhysicalDiskReadPerformanceCounter {
    param([Parameter(Mandatory = $true)][int]$DiskIndex)
    try {
        $prefix = "^$([regex]::Escape([string]$DiskIndex))(?:\s|$)"
        $category = [System.Diagnostics.PerformanceCounterCategory]::new('PhysicalDisk')
        $instance = @($category.GetInstanceNames() | Where-Object { $_ -match $prefix } | Select-Object -First 1)
        if ($instance.Count -ne 1) { return $null }
        return [System.Diagnostics.PerformanceCounter]::new('PhysicalDisk', 'Disk Read Bytes/sec', [string]$instance[0], $true)
    } catch {
        return $null
    }
}

function Measure-ModelStorageRead {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int]$DiskIndex
    )

    $file = Get-Item -LiteralPath $Path -ErrorAction Stop
    if ($file.Length -lt 65536) { return $null }
    $performanceCounter = New-PhysicalDiskReadPerformanceCounter -DiskIndex $DiskIndex
    $before = $null
    $beforeBytes = $null
    if ($null -ne $performanceCounter) {
        try { $beforeBytes = [uint64]$performanceCounter.RawValue } catch { }
    }
    if ($null -eq $beforeBytes) { $before = Get-PhysicalDiskReadCounter -DiskIndex $DiskIndex }
    $seqBuffer = New-Object byte[] (4MB)
    $seqStream = [System.IO.FileStream]::new($file.FullName, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite, $seqBuffer.Length, [System.IO.FileOptions]::SequentialScan)
    try {
        $seqClock = [System.Diagnostics.Stopwatch]::StartNew()
        [int64]$seqBytes = 0
        while ($seqClock.Elapsed.TotalSeconds -lt 0.5) {
            $read = $seqStream.Read($seqBuffer, 0, $seqBuffer.Length)
            if ($read -le 0) { $seqStream.Position = 0; continue }
            $seqBytes += $read
        }
        $seqSeconds = $seqClock.Elapsed.TotalSeconds
    } finally {
        $seqStream.Dispose()
    }

    $randomBlockBytes = 64KB
    $randomBuffer = New-Object byte[] $randomBlockBytes
    $randomStream = [System.IO.FileStream]::new($file.FullName, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite, $randomBlockBytes, [System.IO.FileOptions]::RandomAccess)
    try {
        $random = [System.Random]::new(218)
        $blockCount = [Math]::Floor($file.Length / $randomBlockBytes)
        $randomClock = [System.Diagnostics.Stopwatch]::StartNew()
        [int64]$randomBytes = 0
        [int64]$randomOps = 0
        while ($randomClock.Elapsed.TotalSeconds -lt 0.5) {
            $block = [int64]($random.NextDouble() * $blockCount)
            $randomStream.Position = $block * $randomBlockBytes
            $read = $randomStream.Read($randomBuffer, 0, $randomBuffer.Length)
            if ($read -gt 0) { $randomBytes += $read; $randomOps++ }
        }
        $randomSeconds = $randomClock.Elapsed.TotalSeconds
    } finally {
        $randomStream.Dispose()
    }
    $after = $null
    $afterBytes = $null
    if ($null -ne $performanceCounter) {
        try { $afterBytes = [uint64]$performanceCounter.RawValue } catch { }
        $performanceCounter.Dispose()
    }
    if ($null -eq $afterBytes) { $after = Get-PhysicalDiskReadCounter -DiskIndex $DiskIndex }

    $applicationBytes = $seqBytes + $randomBytes
    $physicalBytes = $null
    if ($null -eq $beforeBytes -and $null -ne $before) { $beforeBytes = [uint64]$before.DiskReadBytesPersec }
    if ($null -eq $afterBytes -and $null -ne $after) { $afterBytes = [uint64]$after.DiskReadBytesPersec }
    if ($null -ne $beforeBytes -and $null -ne $afterBytes -and $afterBytes -ge $beforeBytes) {
        $physicalBytes = [uint64]($afterBytes - $beforeBytes)
    }
    $cacheDisposition = $null
    $physicalFraction = $null
    if ($null -ne $physicalBytes -and $applicationBytes -gt 0) {
        $physicalFraction = [Math]::Round([Math]::Min(1.0, $physicalBytes / [double]$applicationBytes), 4)
        $cacheDisposition = if ($physicalFraction -ge 0.75) { 'mostly_physical' } elseif ($physicalFraction -le 0.10) { 'mostly_cached' } else { 'mixed' }
    }
    return [pscustomobject][ordered]@{
        mode = 'buffered_read_only_before_benchmark'
        sampled_file_bytes = [uint64]$file.Length
        sequential = [pscustomobject][ordered]@{
            seconds = [Math]::Round($seqSeconds, 6)
            bytes = [uint64]$seqBytes
            mib_per_second = [Math]::Round($seqBytes / 1MB / $seqSeconds, 3)
        }
        random_64k = [pscustomobject][ordered]@{
            seconds = [Math]::Round($randomSeconds, 6)
            operations = [uint64]$randomOps
            bytes = [uint64]$randomBytes
            mib_per_second = [Math]::Round($randomBytes / 1MB / $randomSeconds, 3)
            iops = [Math]::Round($randomOps / $randomSeconds, 3)
        }
        physical_disk_read_bytes = $physicalBytes
        physical_read_fraction = $physicalFraction
        cache_disposition = $cacheDisposition
    }
}

function Get-WindowsBenchmarkModelStorageDetails {
    param([Parameter(Mandatory = $true)][string]$ModelPath)

    try {
        $root = [System.IO.Path]::GetPathRoot($ModelPath)
        if ($root -notmatch '^([A-Za-z]):\\$') { return $null }
        $logical = Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='{0}:'" -f $Matches[1]) -ErrorAction Stop
        $partition = @(Get-CimAssociatedInstance -InputObject $logical -Association Win32_LogicalDiskToPartition -ErrorAction Stop | Select-Object -First 1)
        $drive = @($partition | ForEach-Object { Get-CimAssociatedInstance -InputObject $_ -Association Win32_DiskDriveToDiskPartition -ErrorAction Stop } | Select-Object -First 1)
        if ($drive.Count -eq 0) { return $null }

        $diskIndex = [int]$drive[0].Index
        $physical = $null
        try {
            $physical = Get-CimInstance -Namespace 'root/Microsoft/Windows/Storage' -ClassName MSFT_PhysicalDisk -Filter ("DeviceId='{0}'" -f $diskIndex) -ErrorAction Stop | Select-Object -First 1
        } catch { }
        $busType = if ($null -ne $physical) { ConvertFrom-StorageBusType -Type ([int]$physical.BusType) } elseif ([string]$drive[0].PNPDeviceID -match 'NVME') { 'NVMe' } else { [string]$drive[0].InterfaceType }
        $mediaType = if ($null -ne $physical) { ConvertFrom-StorageMediaType -Type ([int]$physical.MediaType) } else { [string]$drive[0].MediaType }
        return [pscustomobject][ordered]@{
            model = [string]$drive[0].Model
            media_type = $mediaType
            bus_type = $busType
            size_bytes = [uint64]$drive[0].Size
            filesystem = [string]$logical.FileSystem
            read_probe = Measure-ModelStorageRead -Path $ModelPath -DiskIndex $diskIndex
        }
    } catch {
        return $null
    }
}

function Get-WindowsBenchmarkHardwareDetails {
    param(
        [Parameter(Mandatory = $true)][string]$ModelPath,
        [AllowEmptyString()][string]$SelectedGpuName = '',
        $ModelStorage = $null
    )

    $modules = @()
    try {
        $modules = @(Get-CimInstance Win32_PhysicalMemory -ErrorAction Stop | ForEach-Object {
            $generation = ConvertFrom-SmbiosMemoryType -Type ([int]$_.SMBIOSMemoryType)
            [pscustomobject][ordered]@{
                capacity_bytes = [uint64]$_.Capacity
                generation = $generation
                speed_mt_s = if ($_.Speed) { [uint64]$_.Speed } else { $null }
                configured_speed_mt_s = if ($_.ConfiguredClockSpeed) { [uint64]$_.ConfiguredClockSpeed } else { $null }
                manufacturer = if ($_.Manufacturer) { ([string]$_.Manufacturer).Trim() } else { $null }
                part_number = if ($_.PartNumber) { ([string]$_.PartNumber).Trim() } else { $null }
            }
        })
    } catch { }
    $generations = @($modules | ForEach-Object { $_.generation } | Where-Object { $_ } | Sort-Object -Unique)
    $memoryGeneration = if ($generations.Count -eq 1) { $generations[0] } elseif ($generations.Count -gt 1) { 'Mixed: ' + ($generations -join ', ') } else { $null }

    $gpuLinks = @()
    if (-not [string]::IsNullOrWhiteSpace($SelectedGpuName)) {
        try {
            $linkKeys = @(
                'DEVPKEY_PciDevice_CurrentLinkSpeed', 'DEVPKEY_PciDevice_CurrentLinkWidth',
                'DEVPKEY_PciDevice_MaxLinkSpeed', 'DEVPKEY_PciDevice_MaxLinkWidth'
            )
            $gpuLinks = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object {
                [string]$_.PNPDeviceID -like 'PCI\*' -and [string]$_.Name -eq $SelectedGpuName
            } | Select-Object -First 1 | ForEach-Object {
                $properties = @(Get-PnpDeviceProperty -InstanceId $_.PNPDeviceID -KeyName $linkKeys -ErrorAction SilentlyContinue)
                $get = { param($key) $entry = $properties | Where-Object KeyName -eq $key | Select-Object -First 1; if ($null -ne $entry -and $null -ne $entry.Data -and [int]$entry.Data -gt 0) { [int]$entry.Data } else { $null } }
                $currentGeneration = & $get 'DEVPKEY_PciDevice_CurrentLinkSpeed'
                $currentWidth = & $get 'DEVPKEY_PciDevice_CurrentLinkWidth'
                $maxGeneration = & $get 'DEVPKEY_PciDevice_MaxLinkSpeed'
                $maxWidth = & $get 'DEVPKEY_PciDevice_MaxLinkWidth'
                [pscustomobject][ordered]@{
                    name = [string]$_.Name
                    current_generation = $currentGeneration
                    current_width = $currentWidth
                    max_generation = $maxGeneration
                    max_width = $maxWidth
                    current_theoretical_one_way_gb_per_second = Get-PcieTheoreticalBandwidthGbps -Generation $currentGeneration -Width $currentWidth
                    max_theoretical_one_way_gb_per_second = Get-PcieTheoreticalBandwidthGbps -Generation $maxGeneration -Width $maxWidth
                    source = 'windows_pnp'
                }
            })
        } catch { }
    }

    if ($null -eq $ModelStorage) { $ModelStorage = Get-WindowsBenchmarkModelStorageDetails -ModelPath $ModelPath }
    return [pscustomobject]@{
        MemoryGeneration = $memoryGeneration
        MemoryModules = $modules
        GpuLinks = $gpuLinks
        ModelStorage = $ModelStorage
    }
}

function Set-JsonProperty {
    param([Parameter(Mandatory = $true)]$Object, [Parameter(Mandatory = $true)][string]$Name, $Value)
    if ($Object.PSObject.Properties.Name -contains $Name) { $Object.$Name = $Value } else { $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value }
}

function Add-BenchmarkHardwareDetails {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Details
    )
    $value = Read-MoE4AllSubmissionPayload -Path $Path
    if ($null -eq $value.hardware) { return }
    if ($null -ne $value.hardware.ram) {
        Set-JsonProperty -Object $value.hardware.ram -Name 'memory_generation' -Value $Details.MemoryGeneration
        Set-JsonProperty -Object $value.hardware.ram -Name 'module_count' -Value @($Details.MemoryModules).Count
        Set-JsonProperty -Object $value.hardware.ram -Name 'modules' -Value @($Details.MemoryModules)
    }
    if ($null -ne $value.hardware.gpu -and $value.hardware.gpu.name) {
        $matches = @($Details.GpuLinks | Where-Object { [string]$_.name -eq [string]$value.hardware.gpu.name })
        if ($matches.Count -eq 1) {
            $link = $matches[0]
            Set-JsonProperty -Object $value.hardware.gpu -Name 'pcie_generation' -Value $link.current_generation
            Set-JsonProperty -Object $value.hardware.gpu -Name 'pcie_width' -Value $link.current_width
            Set-JsonProperty -Object $value.hardware.gpu -Name 'pcie_max_generation' -Value $link.max_generation
            Set-JsonProperty -Object $value.hardware.gpu -Name 'pcie_max_width' -Value $link.max_width
            Set-JsonProperty -Object $value.hardware.gpu -Name 'pcie_theoretical_one_way_gb_per_second' -Value $link.current_theoretical_one_way_gb_per_second
            Set-JsonProperty -Object $value.hardware.gpu -Name 'pcie_max_theoretical_one_way_gb_per_second' -Value $link.max_theoretical_one_way_gb_per_second
            Set-JsonProperty -Object $value.hardware.gpu -Name 'pcie_source' -Value $link.source
        }
    }
    Set-JsonProperty -Object $value.hardware -Name 'model_storage' -Value $Details.ModelStorage
    New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
    $temporary = Join-Path $dataDir ("benchmark-enrichment-{0}.json" -f [Guid]::NewGuid().ToString('N'))
    try {
        [System.IO.File]::WriteAllText($temporary, (($value | ConvertTo-Json -Depth 20 -Compress) + "`n"), [System.Text.UTF8Encoding]::new($false))
        $repacked = Invoke-NativeCommandResult -Executable $infrPath -Arguments @('submission', 'repack-benchmark', '--source', $Path, '--input', $temporary)
        if ($repacked.ExitCode -ne 0) {
            throw "submission repack failed: $($repacked.Output -join ' ')"
        }
    } finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
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
    param([Parameter(Mandatory = $true)]$Release)

    while ($true) {
        Write-Host "`n发现新版本 / Update available: $($Release.Tag)" -ForegroundColor Yellow
        Write-Host '   [1] 查看更新说明 / View release notes'
        Write-Host '   [2] 下载并更新 / Download and update'
        Write-Host ' * [3] 暂不更新（默认）/ Not now (default)'
        $value = (Read-Host '选择 [3] / Select [3]').Trim().ToLowerInvariant()
        if (-not $value) { return 'skip' }
        switch ($value) {
            { $_ -in @('1', 'r', 'release', 'notes') } {
                Show-EngineReleaseNotes -Release $Release
                continue
            }
            { $_ -in @('2', 'y', 'yes', 'update') } { return 'update' }
            { $_ -in @('3', 'n', 'no', 'later') } { return 'skip' }
            default {
                Write-Host '请选择 1、2 或 3；回车暂不更新。Select 1, 2 or 3; Enter skips the update.' -ForegroundColor Yellow
            }
        }
    }
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
            '-CleanupRoot'
            ('"' + $runRoot + '"')
            '-DeleteSelf'
        ) -join ' '
        Start-Process -FilePath 'powershell.exe' -ArgumentList $argumentText -WorkingDirectory $repoRoot -WindowStyle Normal | Out-Null
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

function Show-UpdateStatus {
    param([Parameter(Mandatory = $true)][string]$CurrentVersion)

    $disabled = [Environment]::GetEnvironmentVariable('MOE4ALL_NO_UPDATE_CHECK', 'Process')
    if ($SkipUpdateCheck -or $disabled -match '^(1|true|yes|on)$') { return }
    $current = ConvertTo-Moe4AllSemVer $CurrentVersion
    if ($null -eq $current) { return }

    try {
        $release = Get-LatestEngineRelease -CurrentVersion $CurrentVersion
        if ((Compare-Moe4AllSemVer $release.SemVer $current) -gt 0) {
            $action = Read-EngineUpdateAction -Release $release
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
        process_id = $PID
        at = [DateTimeOffset]::Now.ToString('o')
    }
    $temporary = "$probePath.moe4all-new-$([guid]::NewGuid().ToString('N'))"
    [System.IO.File]::WriteAllText($temporary, ($probe | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $probePath -Force
}

if (-not (Test-Path -LiteralPath $infrPath -PathType Leaf)) {
    throw "找不到 infr.exe。发布包请把 infr.exe 与 Start-INFR-Wizard.cmd 放在同一目录；源码构建请先运行 cargo build --release --locked -p infr-cli。`nExecutable not found beside the launcher or under target\release."
}

$productVersion = Get-EngineVersion
if ($PostUpdateProbePath) {
    Write-PostUpdateProbe -Path $PostUpdateProbePath -Version $productVersion
    if ($ExitAfterPostUpdateProbe) { exit 43 }
}
Clear-Host
Write-Host '============================================================' -ForegroundColor Green
$banner = @'
 __  __       _____ _  _     _   _ _
|  \/  | ___ | ____| || |   / \ | | |
| |\/| |/ _ \|  _| | || |_ / _ \| | |
| |  | | (_) | |___|__   _/ ___ \ | |___
|_|  |_|\___/|_____|  |_|/_/   \_\_|_____|
'@

Write-Host $banner -ForegroundColor Green
Write-Host "  MoE4All v$productVersion" -ForegroundColor Green
Write-Host '  Run models far larger than VRAM on gaming GPUs.'
Write-Host '  让游戏显卡跑起远超显存容量的大模型。'
Write-Host '  John / Headmaster218  https://github.com/Headmaster218/MoE4All' -ForegroundColor DarkGray
Write-Host '============================================================' -ForegroundColor Green
Write-Host 'MoE4All 启动向导 / MoE4All Launch Wizard' -ForegroundColor Green
Write-Host '上次设置会作为默认值；直接回车即可复用。Press Enter to reuse the previous value.'
Write-Host "引擎 / Engine: $infrPath" -ForegroundColor DarkGray
Show-PendingUpdateResult
Show-UpdateStatus -CurrentVersion $productVersion

$reuseSavedSettings = $false
$savedModel = [string](Get-SavedValue 'model' '')
if ($null -ne $script:Saved -and -not $InitialModelPath -and $savedModel) {
    if (Test-Path -LiteralPath $savedModel -PathType Leaf) {
        $reuseSavedSettings = Read-YesNo -Label '是否按上次设置直接启动？/ Start with the previous settings?' -Default $true
    } else {
        Write-Warning '上次模型路径已不存在，请重新选择。The previous model is missing; choose a model again.'
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
$script:PendingSubmissionPrompted = $false
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
$kvPagedPageTokens = [string](Get-SavedValue 'kv_paged_page_tokens' '')
$serverAuth = [bool](Get-SavedValue 'server_auth' $false)
$serverApiKey = ''
$serverVision = [bool](Get-SavedValue 'server_vision' $false)
$visionProjectorPath = [string](Get-SavedValue 'vision_projector' '')
$serverEmbedding = [bool](Get-SavedValue 'server_embedding' $false)
$embeddingModelPath = [string](Get-SavedValue 'embedding_model' '')
$embeddingIdleTimeout = [string](Get-SavedValue 'embedding_idle_timeout' '300')

$configureThinking = [bool](Get-SavedValue 'configure_thinking' ($thinkMode -ne 'default' -or $reasoningEffort -ne 'default'))
$cpuMissEnabled = [bool](Get-SavedValue 'cpu_miss_enabled' $false)
$cpuMissMax = [string](Get-SavedValue 'cpu_miss_max' '1')
$cpuMissCores = [string](Get-SavedValue 'cpu_miss_cores' '')
$customSets = [string](Get-SavedValue 'custom_sets' '')

if (-not $reuseSavedSettings) {
    $launchMode = Read-Choice -Label '你想做什么？/ What would you like to do?' -DefaultValue ([string](Get-SavedValue 'launch_mode' 'chat')) -Options @(
        [pscustomobject]@{ Key = '1'; Value = 'chat'; Label = '实时终端对话（推荐）/ Interactive terminal chat (recommended)' }
        [pscustomobject]@{ Key = '2'; Value = 'server'; Label = '启动 OpenAI 兼容 API / Start OpenAI-compatible API server' }
        [pscustomobject]@{ Key = '3'; Value = 'benchmark'; Label = '性能测试 / Benchmark' }
        [pscustomobject]@{ Key = '4'; Value = 'report'; Label = '报告当前运行问题 / Report a current runtime problem' }
    )
    if ($launchMode -eq 'report') {
        try { [void](Complete-ErrorReport) } catch { Write-Warning "错误报告生成失败：$($_.Exception.Message) / Could not create the error report." }
        exit 0
    }
    Start-VulkanDeviceProbe
    $modelPath = Select-ModelPath -InitialPath $InitialModelPath
    if ($launchMode -ne 'benchmark') {
        Start-ModelReasoningProbe -ModelPath $modelPath
    }
    if ($launchMode -eq 'server') {
        if (-not $visionProjectorPath -or -not (Test-Path -LiteralPath $visionProjectorPath -PathType Leaf)) {
            $visionProjectorPath = Find-VisionProjectorPath -ModelPath $modelPath
        }
        $serverVision = Read-YesNo -Label '启用视觉图片理解？/ Enable image understanding?' -Default $serverVision
        if ($serverVision) {
            Write-Host '视觉权重按图片请求从 SSD 临时载入统一显存，处理完全部图片后立即释放。API 图片请使用 data URI 或 base64。' -ForegroundColor DarkGray
            Write-Host 'Vision weights use request-scoped unified VRAM and are released after the image batch. API images must be data URIs or base64.' -ForegroundColor DarkGray
            $visionProjectorPath = Select-VisionProjectorPath -Default $visionProjectorPath
        }
        $serverEmbedding = Read-YesNo -Label '同时提供 Embedding API？/ Also serve the Embedding API?' -Default $serverEmbedding
        if ($serverEmbedding) {
            Write-Host 'Embedding 首次请求时从 GGUF/SSD 载入统一显存；空闲超时后释放，不建立额外 RAM 权重缓存。' -ForegroundColor DarkGray
            Write-Host 'Weights load from GGUF/SSD into unified VRAM on demand and are released after the idle timeout; no extra RAM weight cache is kept.' -ForegroundColor DarkGray
            $embeddingModelPath = Select-EmbeddingModelPath -Default $embeddingModelPath
            $embeddingIdleTimeout = Read-IntegerValue -Label 'Embedding 空闲释放秒数，0 为服务期间常驻 / Idle eviction seconds, 0 keeps resident' -Default $embeddingIdleTimeout -Minimum 0
        }
    }

    $device = Read-ComputeDevice -Default $device
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
        $kvPagedCacheEnabled = Read-YesNo -Label '改用地基树分页 KV 前缀缓存（试验性，零拷贝帧共享，冷前缀下沉 RAM）？/ Use the paged radix-frame KV prefix cache instead (experimental, zero-copy frame sharing, cold prefixes sink to RAM)?' -Default ([bool]$kvPagedCacheMax)
        if ($kvPagedCacheEnabled) {
            if (-not $kvPagedCacheMax) { $kvPagedCacheMax = '16GiB' }
            $kvPagedCacheMax = Read-TextValue -Label '分页 KV 的 RAM 帧池上限（显存走廊放不下的冷前缀整链 DMA 下沉到此，命中再换回；host 内存按实际下沉量惰性增长，不预占）/ Paged KV RAM frame-pool cap (cold chains evicted from the VRAM corridor DMA down here and promote back on a hit; the host backing grows lazily with actual demotions, no up-front reservation)' -Default $kvPagedCacheMax -Required
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
        if (-not $kvPagedCacheEnabled) {
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

        Write-Host "`nAPI 服务器 / API server" -ForegroundColor Cyan
        $serverAddr = Read-ListenAddress -Label '监听地址（IP:端口）/ Listen address (IP:port)' -Default $serverAddr
        $serverAuth = Read-YesNo -Label '启用 Bearer API key 鉴权？/ Enable Bearer API-key authentication?' -Default $serverAuth
    }

    if ($launchMode -eq 'benchmark') {
        if ($benchKind -notin @('quick', 'standard', 'full', 'manual')) {
            $manualBenchKind = $benchKind
            $benchKind = 'manual'
        }
        $benchKind = Read-Choice -Label '测试方案 / Benchmark plan' -DefaultValue $benchKind -Options @(
            [pscustomobject]@{ Key = '1'; Value = 'quick'; Label = '自动快速 / Automatic quick'; Detail = @('6K Prefill + 1024 Decode。单发，一次，可追加并发。', '6K Prefill + 1024 Decode. x1, one run; optional concurrency.') }
            [pscustomobject]@{ Key = '2'; Value = 'standard'; Label = '自动标准 / Automatic standard'; Detail = @('6K、30K Prefill + 1024 Decode；1-2 并发，各一次，共 4 次。', '6K, 30K Prefill + 1024 Decode; x1-x2, one run each, 4 runs. Shared KV, different questions.') }
            [pscustomobject]@{ Key = '3'; Value = 'full'; Label = '自动完整 / Automatic full'; Detail = @('6K、30K、150K Prefill + 1024 Decode；1-4 并发，各三次，共 36 次。', '6K, 30K, 150K Prefill + 1024 Decode; x1-x4, three runs each, 36 runs. Shared KV, different questions.') }
            [pscustomobject]@{ Key = '4'; Value = 'manual'; Label = '手动测试 / Manual benchmark'; Detail = @('按顺序添加 Prefill、Decode 和上下文深度。', 'Build one test step by step: Prefill, Decode and context depth.') }
        )
        if ($benchKind -ne 'manual') {
            $profileParallel = switch ($benchKind) { 'quick' { '1' } 'standard' { '2' } default { '4' } }
            $benchParallel = Read-IntegerValue -Label '最大并发路数（会依次测试 x1 到 xN）/ Maximum concurrency (tests x1 through xN)' -Default $profileParallel -Minimum 1 -Maximum 4
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
        $serverApiKey = Read-SecretValue -Label 'API key（隐藏输入且不会保存）/ API key (hidden and not saved)'
    } elseif (-not (Test-IsLoopbackAddress $serverAddr)) {
        Write-Host '警告：该监听地址可能被其他设备访问，且当前未启用鉴权。' -ForegroundColor Yellow
        Write-Host 'Warning: this address may be reachable by other devices and authentication is disabled.' -ForegroundColor Yellow
        if (-not (Read-YesNo -Label '仍然继续？/ Continue anyway?' -Default $false)) { exit 0 }
    }
}

$nativeArgs = [System.Collections.Generic.List[string]]::new()
[void]$nativeArgs.Add($(switch ($launchMode) {
    'benchmark' { 'bench' }
    'server' { 'serve' }
    default { 'run' }
}))
if (-not [string]::IsNullOrWhiteSpace($device)) { [void]$nativeArgs.Add('--dev'); [void]$nativeArgs.Add($device) }
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
} else {
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
    [void]$nativeArgs.Add('--parallel'); [void]$nativeArgs.Add($serverParallel)
    if ($kvPagedCacheMax) {
        Add-SetArgument $nativeArgs 'kv.paged_cache_max' $kvPagedCacheMax
        Add-SetArgument $nativeArgs 'kv.paged_page_tokens' $kvPagedPageTokens
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
    }
    if ($serverEmbedding) {
        [void]$nativeArgs.Add('--embedding-model'); [void]$nativeArgs.Add($embeddingModelPath)
        [void]$nativeArgs.Add('--embedding-idle-timeout'); [void]$nativeArgs.Add($embeddingIdleTimeout)
    }
}
[void]$nativeArgs.Add($modelPath)

$commandText = '& ' + (Format-PowerShellArgument $infrPath) + ' ' + (($nativeArgs | ForEach-Object { Format-PowerShellArgument $_ }) -join ' ')
if ($null -ne $script:ModelReasoningProbe -and $script:ModelReasoningProbe.Async.IsCompleted) {
    [void](Get-ModelReasoningEfforts -ModelPath $modelPath)
}
$state = [ordered]@{
    launch_mode = $launchMode; setup_mode = $setupMode; model = $modelPath; model_history = @(Get-UpdatedModelHistory -SelectedModel $modelPath); device = $device; context = $context
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
    server_vision = $serverVision; vision_projector = $visionProjectorPath
    server_embedding = $serverEmbedding; embedding_model = $embeddingModelPath
    embedding_idle_timeout = $embeddingIdleTimeout
    cpu_miss_enabled = $cpuMissEnabled; cpu_miss_max = $cpuMissMax; cpu_miss_cores = $cpuMissCores
    custom_sets = $customSets; last_command = $commandText
}
if (-not $DryRun -and -not $script:PendingSubmissionPrompted) { Offer-PendingSubmissions }
Write-Host "`n最终命令 / Final command" -ForegroundColor Green
Write-Host $commandText -ForegroundColor White
if ($launchMode -eq 'server' -and $serverAuth) {
    Write-Host 'API key 将通过当前子进程环境传入，未显示在命令中，也不会保存。' -ForegroundColor DarkGray
    Write-Host 'The API key is passed through the child-process environment; it is hidden above and not saved.' -ForegroundColor DarkGray
}
$startLabel = if ($launchMode -eq 'server') { '现在启动服务器？/ Start the server now?' } else { '现在启动？/ Start now?' }
if (-not $DryRun -and -not $reuseSavedSettings -and -not (Read-YesNo -Label $startLabel -Default $true)) {
    exit 0
}
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statePath -Encoding UTF8
Write-Host "`n设置已保存 / Settings saved: $statePath" -ForegroundColor DarkGray
if ($DryRun) {
    Write-Host 'DryRun：未启动。DryRun: command was not started.' -ForegroundColor Yellow
    exit 0
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
    if ($serverVision) {
        Write-Host '视觉聊天已启用：在 /v1/chat/completions 的 content parts 中发送 image_url data URI。' -ForegroundColor Cyan
        Write-Host 'Vision chat enabled: send an image_url data URI in /v1/chat/completions content parts.' -ForegroundColor Cyan
    }
    Write-Host '按 Ctrl+C 停止服务器。Press Ctrl+C to stop the server.' -ForegroundColor Yellow
} elseif ($launchMode -eq 'chat') {
    Write-Host "`n模型加载完成后，在 > 提示符输入消息；输入 exit、quit 或 :q 退出。" -ForegroundColor Cyan
    Write-Host 'After the model loads, type at the > prompt; use exit, quit or :q to leave.' -ForegroundColor DarkGray
}

Write-Host "`n启动中 / Starting..." -ForegroundColor Green
$benchmarkFilesBefore = @()
if ($launchMode -eq 'benchmark') { $benchmarkFilesBefore = @(Get-BenchmarkSubmissionSnapshot) }
$errorReportsBefore = @(Get-ErrorSubmissionSnapshot)
$benchmarkResultPath = $null
$benchmarkModelStorage = $null
if ($launchMode -eq 'benchmark') {
    Write-Host '正在进行模型盘短读测（只读，约 2 秒）... / Running a short read-only model-storage probe (about 2 seconds)...' -ForegroundColor DarkGray
    try {
        $benchmarkModelStorage = Get-WindowsBenchmarkModelStorageDetails -ModelPath $modelPath
    } catch {
        Write-Warning "模型盘信息采集失败，将继续 benchmark：$($_.Exception.Message) / Model-storage collection failed; benchmark will continue."
    }
}
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
    $benchmarkUploadRequested = Read-BenchmarkUploadConsent
}
if ($benchmarkResultPath) {
    try {
        $resultGpuName = ''
        try {
            $resultValue = Read-MoE4AllSubmissionPayload -Path $benchmarkResultPath
            $resultGpuName = [string]$resultValue.hardware.gpu.name
        } catch { }
        if ([string]::IsNullOrWhiteSpace($resultGpuName)) {
            $deviceProbe = Invoke-NativeCommandResult -Executable $infrPath -Arguments @('devices')
            if ($deviceProbe.ExitCode -eq 0) {
                $devicePattern = '^\s*' + [regex]::Escape($device) + ':\s+(.+?)\s+\[[^\]]+\]'
                $deviceLine = @($deviceProbe.Output | Where-Object { [string]$_ -match $devicePattern } | Select-Object -First 1)
                if ($deviceLine.Count -eq 1 -and [string]$deviceLine[0] -match $devicePattern) { $resultGpuName = $Matches[1] }
            }
        }
        $benchmarkHardwareDetails = Get-WindowsBenchmarkHardwareDetails -ModelPath $modelPath -SelectedGpuName $resultGpuName -ModelStorage $benchmarkModelStorage
        Add-BenchmarkHardwareDetails -Path $benchmarkResultPath -Details $benchmarkHardwareDetails
    } catch {
        Write-Warning "Benchmark 提交的硬件信息补全失败：$($_.Exception.Message) / Could not enrich benchmark submission hardware details."
    }
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
exit $exitCode
