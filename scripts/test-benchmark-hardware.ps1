param([Parameter(Mandatory = $true)][string]$FixtureRoot)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = Split-Path $PSScriptRoot -Parent
$fixture = [IO.Path]::GetFullPath($FixtureRoot)
if ($env:MOE4ALL_WORK_ROOT) {
    $allowed = [IO.Path]::GetFullPath((Join-Path $env:MOE4ALL_WORK_ROOT 'tmp\moe4all')).TrimEnd('\')
    if ($fixture -ne $allowed -and -not $fixture.StartsWith($allowed + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'FixtureRoot must be inside MOE4ALL_WORK_ROOT/tmp/moe4all when MOE4ALL_WORK_ROOT is set'
    }
}
[void][IO.Directory]::CreateDirectory($fixture)
$env:TEMP = $fixture; $env:TMP = $fixture
$model = Join-Path $fixture 'sensitive-model [fixture].bin'
[IO.File]::WriteAllBytes($model, (New-Object byte[] (1MB)))
$source = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'benchmark-hardware.ps1'))
$tokens = $null; $parseErrors = $null
[void][Management.Automation.Language.Parser]::ParseInput($source, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
. (Join-Path $PSScriptRoot 'benchmark-hardware.ps1')
$script:mode = 'normal'
$script:gpuNames = @('Fixture GPU', 'Fixture GPU')
function Assert-True { param($Condition, [string]$Message) if (-not $Condition) { throw $Message } }
function Get-CimInstance {
    param($ClassName, $Filter, $Namespace, $OperationTimeoutSec, $ErrorAction)
    switch ($ClassName) {
        Win32_Processor { [pscustomobject]@{ Name = 'Fixture CPU'; NumberOfLogicalProcessors = 16; SystemName = 'machine-private' } }
        Win32_ComputerSystem { [pscustomobject]@{ TotalPhysicalMemory = 34359738368; UserName = 'private-user' } }
        Win32_PhysicalMemory {
            if ($script:mode -eq 'ram-denied') { throw [UnauthorizedAccessException]::new('private-user') }
            [pscustomobject]@{ Capacity = 17179869184; SMBIOSMemoryType = 34; Speed = 6000; ConfiguredClockSpeed = 5600; Manufacturer = 'Fixture'; PartNumber = 'Part'; SerialNumber = 'SECRET_SERIAL' }
        }
        Win32_VideoController {
            for ($i=0; $i -lt $script:gpuNames.Count; $i++) {
                $device = if ($i -eq 0) { '744C' } else { '6611' }
                [pscustomobject]@{ Name = $script:gpuNames[$i]; PNPDeviceID = "PCI\VEN_1002&DEV_$device&FIXTURE$i" }
            }
        }
        Win32_SystemSlot { [pscustomobject]@{ SlotDesignation = 'PCIEX1_1'; CurrentUsage = 4 } }
        Win32_LogicalDisk {
            if ($script:mode -eq 'disk-denied') { throw [UnauthorizedAccessException]::new($model) }
            [pscustomobject]@{ FileSystem = 'NTFS' }
        }
        MSFT_PhysicalDisk { [pscustomobject]@{ BusType = 17; MediaType = 4 } }
        Win32_PerfRawData_PerfDisk_PhysicalDisk { [pscustomobject]@{ Name = '0 Fixture'; DiskReadBytesPersec = 1024 } }
        default { throw "Unexpected fixture class: $ClassName" }
    }
}
function Get-CimAssociatedInstance {
    param($InputObject, $Association, $OperationTimeoutSec, $ErrorAction)
    if ($Association -eq 'Win32_LogicalDiskToPartition') { [pscustomobject]@{ DiskIndex = 0 } }
    else { [pscustomobject]@{ Index = 0; Model = 'Fixture SSD'; Size = 1000000000; PNPDeviceID = 'NVME'; InterfaceType = 'SCSI'; MediaType = 'Fixed'; SerialNumber = 'SECRET_SERIAL' } }
}
function Get-PnpDeviceProperty {
    param($InstanceId, $KeyName, $ErrorAction)
    $isBridge = $InstanceId -like '*BRIDGE*'
    foreach ($key in $KeyName) {
        $data = switch ($key) {
            DEVPKEY_Device_BusNumber { if ($isBridge) { 0 } elseif ($InstanceId -like '*FIXTURE0') { 1 } else { 2 } }
            DEVPKEY_Device_Address { if ($isBridge) { 65536 } else { 0 } }
            DEVPKEY_Device_Parent {
                if ($InstanceId -like '*FIXTURE0') { 'PCI\VEN_1234&DEV_0001&BRIDGE0' }
                elseif ($InstanceId -like '*FIXTURE1') { 'PCI\VEN_1234&DEV_0002&BRIDGE1' }
                else { 'ACPI\PNP0A08\0' }
            }
            DEVPKEY_Device_LocationPaths { @('ACPI(_SB_)#ACPI(PCI0)#ACPI(X1_1)#PCI(0000)') }
            DEVPKEY_PciDevice_CurrentLinkSpeed { if ($isBridge) { 3 } else { 4 } }
            DEVPKEY_PciDevice_MaxLinkSpeed { if ($isBridge) { 4 } else { 5 } }
            DEVPKEY_PciDevice_CurrentLinkWidth { if ($isBridge) { 4 } else { 8 } }
            DEVPKEY_PciDevice_MaxLinkWidth { if ($isBridge) { 4 } else { 16 } }
        }
        [pscustomobject]@{ KeyName = $key; Data = $data }
    }
}

$checks = 0
Assert-True ((ConvertFrom-SmbiosMemoryType 26) -eq 'DDR4' -and (ConvertFrom-SmbiosMemoryType 34) -eq 'DDR5') 'Legacy SMBIOS generation mapping changed'
Assert-True ((ConvertFrom-StorageBusType 17) -eq 'NVMe' -and (ConvertFrom-StorageMediaType 4) -eq 'SSD') 'Legacy storage enum mapping changed'
Assert-True ((Get-PcieTheoreticalBandwidthGbps 4 16) -eq 31.504) 'Legacy PCIe bandwidth mapping changed'
$checks++
# Exercise exactly the parent's ScriptBlock.Create contract with native probes mocked.
$raw = @(& ([scriptblock]::Create($source)) -ModelPath $model -Backend Vulkan -SelectedGpuName 'wrong name ignored for BDF' -PciAddress '02:00.0' -VendorId 1002 -DeviceId 6611 -VulkanIndex 1)
Assert-True ($raw.Count -eq 1) 'Collector must return exactly one stdout record'
Assert-True ([Text.Encoding]::UTF8.GetByteCount($raw[0]) -lt 65536) 'JSON exceeds 64 KiB'
$json = $raw[0] | ConvertFrom-Json
Assert-True ($json.hardware.gpu.pci_address -eq '0000:02:00.0') 'BDF selection failed'
Assert-True ($json.hardware.gpu.pcie_generation -eq 4 -and $json.hardware.gpu.pcie_width -eq 8) 'PCIe values lost'
Assert-True ($json.hardware.gpu.pcie_endpoint_current_generation -eq 4 -and $json.hardware.gpu.pcie_endpoint_max_generation -eq 5) 'PCIe endpoint current and capability were conflated'
Assert-True ($json.hardware.gpu.pcie_end_to_end_current_generation -eq 3 -and $json.hardware.gpu.pcie_end_to_end_current_width -eq 4) 'PCIe physical-path bottleneck was not selected'
Assert-True ($json.hardware.gpu.pcie_end_to_end_current_theoretical_one_way_gb_per_second -eq 3.94 -and -not $json.hardware.gpu.pcie_end_to_end_unknown_reason) 'PCIe end-to-end bandwidth scope is wrong'
Assert-True ($json.hardware.gpu.pcie_max_scope -eq 'selected_gpu_endpoint_capability_not_slot' -and $json.hardware.gpu.pcie_path.Count -eq 2) 'PCIe endpoint/path scope is not explicit'
Assert-True ($json.hardware.gpu.pcie_physical_slot.designation -eq 'PCIEX1_1' -and $json.hardware.gpu.pcie_physical_slot.connector_width_hint -eq 1) 'Physical slot attribution was lost'
Assert-True ($json.hardware.gpu.vendor_id -eq 0x1002 -and $json.hardware.gpu.device_id -eq 0x6611 -and $json.hardware.gpu.vulkan_index -eq 1) 'Vulkan identity lost'
Assert-True ($json.hardware.ram.memory_generation -eq 'DDR5') 'SMBIOS generation lost'
Assert-True ($json.hardware.model_storage.bus_type -eq 'NVMe') 'Storage metadata lost'
foreach ($mode in @('sequential', 'random_64k')) {
    $read = $json.hardware.model_storage.read_probe.$mode
    Assert-True ($read.bytes -gt 0 -and $read.bytes -le 268435456 -and $read.seconds -lt 3) 'Read probe is not bounded'
}
Assert-True ($raw[0] -notmatch 'SECRET_SERIAL|private-user|machine-private|sensitive-model|FIXTURE[01]') 'Private data leaked'
$checks++
$script:diagnostics = [ordered]@{}
$selected = Get-WindowsSelectedGpu 'Fixture GPU' ''
Assert-True ($null -eq $selected -and $script:diagnostics['gpu.selection'].reason -eq 'ambiguous_selected_gpu') 'Duplicate GPUs must not pick the first'
$selected = Get-WindowsSelectedGpu 'Fixture GPU' '' '1002' '744c' '0'
Assert-True ($selected.device_id -eq 0x744c -and $selected.vulkan_index -eq 0) 'PCI identity did not disambiguate duplicate names'
$checks++
$script:gpuNames = @('Fixture GPU')
$selected = Get-WindowsSelectedGpu 'fixture gpu' ''
Assert-True ($null -eq $selected) 'GPU name selection must be exact and case sensitive'
$selected = Get-WindowsSelectedGpu 'Fixture GPU' 'not-a-bdf'
Assert-True ($null -eq $selected -and $script:diagnostics['gpu.selection'].reason -eq 'invalid_pci_address') 'Invalid BDF must not fall back to name'
$selected = Get-WindowsSelectedGpu 'Fixture GPU' '' 'not-hex' '744c' '0'
Assert-True ($null -eq $selected -and $script:diagnostics['gpu.selection'].reason -eq 'invalid_pci_identity') 'Invalid PCI identity must not fall back to name'
$checks++
$script:mode = 'disk-denied'
$script:diagnostics = [ordered]@{}
$storage = Get-WindowsBenchmarkModelStorageDetails $model
Assert-True ($script:diagnostics['model_storage.metadata'].status -eq 'permission_denied') 'Access denial was not diagnosed'
Assert-True ($storage.read_probe.sequential.bytes -gt 0 -and $storage.read_probe.random_64k.bytes -gt 0) 'Metadata failure must not skip reads'
$checks++
$script:mode = 'normal'
$script:diagnostics = [ordered]@{}
$storage = Get-WindowsBenchmarkModelStorageDetails (Join-Path $fixture 'missing.bin')
Assert-True ($storage.model -eq 'Fixture SSD') 'Read failure must retain metadata'
Assert-True ($script:diagnostics['model_storage.sequential'].status -eq 'failed') 'Missing file must be diagnosed'
Assert-True ($script:diagnostics['model_storage.random_64k'].status -eq 'failed') 'Random probe must fail independently'
$checks++
$script:diagnostics = [ordered]@{}
$extended = '\\?\' + $model
$storage = Get-WindowsBenchmarkModelStorageDetails $extended
Assert-True ($storage.read_probe.sequential.bytes -gt 0 -and $storage.model -eq 'Fixture SSD') 'Extended Windows paths are not supported'
$checks++
$script:mode = 'ram-denied'
$result = Invoke-BenchmarkHardwareCollector '' '' ''
Assert-True ($result.collection.probes['ram.modules'].status -eq 'permission_denied') 'RAM denial was not isolated'
Assert-True ($result.hardware.cpu.name -eq 'Fixture CPU') 'Other probes were lost'
Assert-True ($result.collection.probes['model_storage.sequential'].status -eq 'not_applicable') 'No-model case is not explicit'
$checks++
$script:mode = 'normal'
$result = Invoke-BenchmarkHardwareCollector (Join-Path $fixture 'missing.bin') '' ''
Assert-True ($result.hardware.cpu.name -eq 'Fixture CPU' -and $result.hardware.ram.total_bytes -gt 0) 'Missing model must preserve CPU/RAM'
Assert-True ($result.collection.probes['model_storage.sequential'].status -eq 'failed') 'Missing model read status lost'
$checks++
# Package a fixture payload, never execute a binary or compile the engine.
$packageOutput = Join-Path $fixture 'package'
$null = & (Join-Path $PSScriptRoot 'package-windows.ps1') -BinaryPath $model -OutputDirectory $packageOutput -Version '1.0.0-collector-test' -WithoutImageCodecs 6>&1
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath = Join-Path $packageOutput 'MoE4All-Windows-x86_64-v1.0.0-collector-test.zip'
$zip = [IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $entry = @($zip.Entries | Where-Object { $_.FullName.Replace('\', '/') -like '*/scripts/benchmark-hardware.ps1' })
    Assert-True ($entry.Count -eq 1) 'Windows package did not include the collector'
    $reader = [IO.StreamReader]::new($entry[0].Open())
    try { Assert-True ($reader.ReadToEnd() -ceq $source) 'Packaged collector differs from the source' } finally { $reader.Dispose() }
    $manifestEntry = @($zip.Entries | Where-Object { $_.FullName.Replace('\', '/') -like '*/install-manifest.json' })
    $reader = [IO.StreamReader]::new($manifestEntry[0].Open())
    try { $manifest = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
    Assert-True (@($manifest.files | Where-Object path -eq 'scripts/benchmark-hardware.ps1').Count -eq 1) 'Collector missing from manifest'
} finally { $zip.Dispose() }
$checks++
# Capture actual PS 5.1 stdout bytes through the same stdin/ScriptBlock path.
$utf8Mock = @'
function Get-CimInstance {
    param($ClassName, $OperationTimeoutSec, $ErrorAction)
    switch ($ClassName) {
        Win32_Processor { [pscustomobject]@{ Name = ('Fixture ' + [char]0x5904 + [char]0x7406); NumberOfLogicalProcessors = 2 } }
        Win32_ComputerSystem { [pscustomobject]@{ TotalPhysicalMemory = 1048576 } }
        Win32_PhysicalMemory { }
        default { throw 'Unexpected native probe in UTF8 fixture' }
    }
}
& {
'@
$utf8Input = $utf8Mock + "`n" + $source + "`n}"
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$start.Arguments = '-NoLogo -NoProfile -NonInteractive -Command "$s = [Console]::In.ReadToEnd(); & ([scriptblock]::Create($s))"'
$start.UseShellExecute = $false; $start.CreateNoWindow = $true
$start.RedirectStandardInput = $true; $start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true
$process = [Diagnostics.Process]::new(); $process.StartInfo = $start
[void]$process.Start()
$process.StandardInput.Write($utf8Input); $process.StandardInput.Close()
$bytes = [IO.MemoryStream]::new()
$process.StandardOutput.BaseStream.CopyTo($bytes)
$childError = $process.StandardError.ReadToEnd()
Assert-True ($process.WaitForExit(12000) -and $process.ExitCode -eq 0) 'UTF8 child failed'
Assert-True ($childError -eq '') 'Collector contaminated stderr'
$strictUtf8 = [Text.UTF8Encoding]::new($false, $true)
$text = $strictUtf8.GetString($bytes.ToArray())
$bytes.Dispose(); $process.Dispose()
$utf8Json = $text | ConvertFrom-Json
Assert-True ($utf8Json.hardware.cpu.name -ceq ('Fixture ' + [char]0x5904 + [char]0x7406)) 'PS5 stdout was not UTF8'
$checks++
[ordered]@{ status = 'passed'; checks = $checks; powershell = $PSVersionTable.PSVersion.ToString(); probes = 'mocked native metadata; real reads of generated 1 MiB fixture only' } | ConvertTo-Json -Compress
