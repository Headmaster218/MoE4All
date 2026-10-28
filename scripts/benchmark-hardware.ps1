# Standalone PS 5.1 collector, also suitable for embedding as script text.
# Call before benchmark warmup. No wizard, engine, GPU API or output file is used.
[CmdletBinding()]
param([string]$ModelPath = '', [string]$SelectedGpuName = '', [string]$PciAddress = '')

$ErrorActionPreference = 'Stop'
$WarningPreference = 'SilentlyContinue'
$ProgressPreference = 'SilentlyContinue'
# PS 5.1 otherwise writes redirected native stdout using the active OEM codepage.
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
$OutputEncoding = [Text.UTF8Encoding]::new($false)

function Set-ProbeStatus {
    param([string]$Probe, [string]$Status, [string]$Reason, [string]$Source)
    $script:diagnostics[$Probe] = [ordered]@{ status = $Status; reason = $Reason; source = $Source }
}

function Set-ProbeError {
    param([string]$Probe, $Failure, [string]$Source)
    $status = 'failed'
    $exception = $Failure.Exception
    while ($null -ne $exception) {
        if ($exception -is [UnauthorizedAccessException] -or $exception -is [System.Security.SecurityException] -or
            $exception.HResult -eq -2147024891 -or $exception.HResult -eq -2147217405) { $status = 'permission_denied' }
        $exception = $exception.InnerException
    }
    if ($Failure.CategoryInfo.Category -eq 'PermissionDenied') { $status = 'permission_denied' }
    if ($Failure.Exception -is [System.Management.Automation.CommandNotFoundException]) { $status = 'unsupported' }
    # Exception messages can contain model paths, account names and device serials.
    Set-ProbeStatus $Probe $status 'native_probe_error' $Source
}

function Get-SafeHardwareText {
    param($Value)
    if ($null -eq $Value) { return $null }
    $text = ([string]$Value -replace '[\x00-\x1f\x7f]', '').Trim()
    if (-not $text) { return $null }
    return $text.Substring(0, [Math]::Min(160, $text.Length))
}

# Memory, PCIe rates and storage enum mappings originate in infr-wizard.ps1.
function ConvertFrom-SmbiosMemoryType {
    param([int]$Type)
    switch ($Type) {
        20 { 'DDR' } 21 { 'DDR2' } 22 { 'DDR2 FB-DIMM' } 24 { 'DDR3' }
        26 { 'DDR4' } 27 { 'LPDDR' } 28 { 'LPDDR2' } 29 { 'LPDDR3' }
        30 { 'LPDDR4' } 34 { 'DDR5' } 35 { 'LPDDR5' } default { $null }
    }
}

function Get-PcieTheoreticalBandwidthGbps {
    param([int]$Generation, [int]$Width)
    if ($Width -le 0) { return $null }
    $perLane = switch ($Generation) {
        1 { 0.250 } 2 { 0.500 } 3 { 0.985 } 4 { 1.969 } 5 { 3.938 } 6 { 7.563 }
        default { return $null }
    }
    return [Math]::Round($perLane * $Width, 3)
}

function ConvertFrom-StorageBusType {
    param([int]$Type)
    switch ($Type) {
        3 { 'ATA' } 7 { 'USB' } 8 { 'RAID' } 10 { 'SAS' } 11 { 'SATA' }
        12 { 'SD' } 13 { 'MMC' } 17 { 'NVMe' } 18 { 'SCM' } 19 { 'UFS' } default { $null }
    }
}

function ConvertFrom-StorageMediaType {
    param([int]$Type)
    switch ($Type) { 3 { 'HDD' } 4 { 'SSD' } 5 { 'SCM' } default { $null } }
}

function ConvertTo-PciAddress {
    param([string]$Address)
    if ($Address -match '^(?:([0-9a-fA-F]{4}):)?([0-9a-fA-F]{2}):([0-1][0-9a-fA-F])\.([0-7])$') {
        $domain = if ($Matches[1]) { $Matches[1] } else { '0000' }
        return ('{0}:{1}:{2}.{3}' -f $domain, $Matches[2], $Matches[3], $Matches[4]).ToLowerInvariant()
    }
    return $null
}

function Get-NativePciAddress {
    param([string]$InstanceId)
    $properties = @(Get-PnpDeviceProperty -InstanceId $InstanceId -KeyName @(
        'DEVPKEY_Device_BusNumber', 'DEVPKEY_Device_Address'
    ) -ErrorAction Stop)
    $bus = @($properties | Where-Object KeyName -eq 'DEVPKEY_Device_BusNumber')
    $address = @($properties | Where-Object KeyName -eq 'DEVPKEY_Device_Address')
    if ($bus.Count -ne 1 -or $address.Count -ne 1 -or $null -eq $bus[0].Data -or $null -eq $address[0].Data) { return $null }
    $number = [uint32]$address[0].Data
    $device = $number -shr 16
    $function = $number -band 0xffff
    if ([uint32]$bus[0].Data -gt 255 -or $device -gt 31 -or $function -gt 7) { return $null }
    # Windows exposes bus/device/function, but no PCI segment here. Only segment 0 is matchable.
    return '0000:{0:x2}:{1:x2}.{2:x1}' -f [uint32]$bus[0].Data, $device, $function
}

function Get-WindowsSelectedGpu {
    param([string]$Name, [string]$Address)
    if (-not $Name -and -not $Address) {
        Set-ProbeStatus 'gpu.selection' 'not_applicable' 'no_selected_gpu' 'windows_pnp'; return $null
    }
    $bdf = ConvertTo-PciAddress $Address
    if ($Address -and -not $bdf) {
        Set-ProbeStatus 'gpu.selection' 'failed' 'invalid_pci_address' 'windows_pnp'; return $null
    }
    $candidates = @(Get-CimInstance Win32_VideoController -OperationTimeoutSec 3 -ErrorAction Stop | Select-Object -First 64)
    $found = @()
    foreach ($candidate in $candidates) {
        if ($bdf) {
            if ([string]$candidate.PNPDeviceID -notlike 'PCI\*') { continue }
            $native = Get-NativePciAddress ([string]$candidate.PNPDeviceID)
            if ($native -ceq $bdf) { $found += [pscustomobject]@{ device = $candidate; bdf = $native } }
        } elseif ([string]$candidate.Name -ceq $Name) {
            $found += [pscustomobject]@{ device = $candidate; bdf = $null }
        }
    }
    if ($found.Count -ne 1) {
        $reason = if ($found.Count -gt 1) { 'ambiguous_selected_gpu' } else { 'selected_gpu_not_found' }
        Set-ProbeStatus 'gpu.selection' 'failed' $reason 'windows_pnp'; return $null
    }
    Set-ProbeStatus 'gpu.selection' 'ok' 'unique_native_match' 'windows_pnp'
    $selected = $found[0]
    $gpu = [ordered]@{ name = Get-SafeHardwareText $selected.device.Name; pci_address = $selected.bdf }
    if ([string]$selected.device.PNPDeviceID -notlike 'PCI\*') {
        Set-ProbeStatus 'gpu.pcie' 'not_applicable' 'non_pci_device' 'windows_pnp'; return $gpu
    }
    try {
        $properties = @(Get-PnpDeviceProperty -InstanceId $selected.device.PNPDeviceID -KeyName @(
            'DEVPKEY_PciDevice_CurrentLinkSpeed', 'DEVPKEY_PciDevice_CurrentLinkWidth',
            'DEVPKEY_PciDevice_MaxLinkSpeed', 'DEVPKEY_PciDevice_MaxLinkWidth'
        ) -ErrorAction Stop)
        $fields = [ordered]@{ CurrentLinkSpeed = 'pcie_generation'; CurrentLinkWidth = 'pcie_width'; MaxLinkSpeed = 'pcie_max_generation'; MaxLinkWidth = 'pcie_max_width' }
        $complete = $true
        foreach ($key in $fields.Keys) {
            $entry = @($properties | Where-Object KeyName -eq "DEVPKEY_PciDevice_$key")
            $value = if ($entry.Count -eq 1 -and $null -ne $entry[0].Data -and [int]$entry[0].Data -gt 0) { [int]$entry[0].Data } else { $null }
            $gpu[$fields[$key]] = $value
            if ($null -eq $value) { $complete = $false }
        }
        $gpu.pcie_theoretical_one_way_gb_per_second = Get-PcieTheoreticalBandwidthGbps $gpu.pcie_generation $gpu.pcie_width
        $gpu.pcie_max_theoretical_one_way_gb_per_second = Get-PcieTheoreticalBandwidthGbps $gpu.pcie_max_generation $gpu.pcie_max_width
        $gpu.pcie_source = 'windows_pnp'
        if ($complete) { Set-ProbeStatus 'gpu.pcie' 'ok' 'native_link_properties' 'windows_pnp' }
        else { Set-ProbeStatus 'gpu.pcie' 'unsupported' 'link_properties_unavailable' 'windows_pnp' }
    } catch { Set-ProbeError 'gpu.pcie' $_ 'windows_pnp' }
    return $gpu
}

function Get-PhysicalDiskReadCounter {
    param([int]$DiskIndex)
    $prefix = "^$DiskIndex(?:\s|$)"
    return Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk -OperationTimeoutSec 3 -ErrorAction Stop |
        Where-Object { [string]$_.Name -match $prefix } | Select-Object -First 1
}

function Measure-ModelStorageRead {
    param([string]$Path)
    $result = [ordered]@{ mode = 'buffered_read_only_before_benchmark'; target_seconds_per_probe = 0.5; max_bytes_per_probe = 268435456; cache_disposition = 'unknown'; physical_disk_read_bytes = $null; physical_read_fraction = $null }
    foreach ($mode in @('sequential', 'random_64k')) {
        $stream = $null
        try {
            $blockSize = if ($mode -eq 'sequential') { 4MB } else { 64KB }
            $options = if ($mode -eq 'sequential') { [IO.FileOptions]::SequentialScan } else { [IO.FileOptions]::RandomAccess }
            $stream = [IO.FileStream]::new($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite, $blockSize, $options)
            $result.sampled_file_bytes = $stream.Length
            if ($stream.Length -lt 64KB) {
                Set-ProbeStatus "model_storage.$mode" 'not_applicable' 'file_smaller_than_64k' 'dotnet_filestream'; continue
            }
            $buffer = New-Object byte[] $blockSize
            $random = [Random]::new(218)
            $blocks = [Math]::Floor($stream.Length / 64KB)
            $clock = [Diagnostics.Stopwatch]::StartNew()
            [long]$bytes = 0; [long]$operations = 0
            while ($clock.Elapsed.TotalSeconds -lt 0.5 -and $bytes -lt 256MB) {
                if ($mode -eq 'random_64k') { $stream.Position = [long]($random.NextDouble() * ($blocks - 1)) * 64KB }
                elseif ($stream.Position -eq $stream.Length) { $stream.Position = 0 }
                $read = $stream.Read($buffer, 0, [int][Math]::Min($buffer.Length, 256MB - $bytes))
                if ($read -le 0) { break }
                $bytes += $read; $operations++
            }
            $seconds = [Math]::Max($clock.Elapsed.TotalSeconds, 0.000001)
            $result[$mode] = [ordered]@{ seconds = [Math]::Round($seconds, 6); bytes = $bytes; operations = $operations; mib_per_second = [Math]::Round($bytes / 1MB / $seconds, 3) }
            if ($mode -eq 'random_64k') { $result[$mode].iops = [Math]::Round($operations / $seconds, 3) }
            if ($bytes -gt 0) { Set-ProbeStatus "model_storage.$mode" 'ok' 'buffered_read_includes_cache' 'dotnet_filestream' }
            else { Set-ProbeStatus "model_storage.$mode" 'failed' 'empty_read' 'dotnet_filestream' }
        } catch { Set-ProbeError "model_storage.$mode" $_ 'dotnet_filestream' }
        finally { if ($null -ne $stream) { $stream.Dispose() } }
    }
    return $result
}

function Get-WindowsBenchmarkModelStorageDetails {
    param([string]$Path)
    $storage = [ordered]@{}
    $diskIndex = $null
    try {
        $lookup = $Path
        if ($lookup.StartsWith('\\?\UNC\', [StringComparison]::OrdinalIgnoreCase)) { $lookup = '\\' + $lookup.Substring(8) }
        elseif ($lookup.StartsWith('\\?\')) { $lookup = $lookup.Substring(4) }
        $root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($lookup))
        if ($root -notmatch '^([A-Za-z]):\\$') {
            Set-ProbeStatus 'model_storage.metadata' 'unsupported' 'non_drive_path' 'windows_cim'
        } else {
            $logical = Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='{0}:'" -f $Matches[1]) -OperationTimeoutSec 3 -ErrorAction Stop
            $storage.filesystem = Get-SafeHardwareText $logical.FileSystem
            $partitions = @(Get-CimAssociatedInstance -InputObject $logical -Association Win32_LogicalDiskToPartition -OperationTimeoutSec 3 -ErrorAction Stop)
            $drives = @($partitions | ForEach-Object { Get-CimAssociatedInstance -InputObject $_ -Association Win32_DiskDriveToDiskPartition -OperationTimeoutSec 3 -ErrorAction Stop })
            if ($drives.Count -ne 1) {
                Set-ProbeStatus 'model_storage.metadata' 'unsupported' 'non_unique_physical_disk' 'windows_cim'
            } else {
                $drive = $drives[0]; $diskIndex = [int]$drive.Index
                $storage.model = Get-SafeHardwareText $drive.Model
                $storage.size_bytes = [uint64]$drive.Size
                $storage.bus_type = if ([string]$drive.PNPDeviceID -match 'NVME') { 'NVMe' } else { Get-SafeHardwareText $drive.InterfaceType }
                $storage.media_type = Get-SafeHardwareText $drive.MediaType
                Set-ProbeStatus 'model_storage.metadata' 'ok' 'native_disk_association' 'windows_cim'
                try {
                    $physical = Get-CimInstance -Namespace 'root/Microsoft/Windows/Storage' -ClassName MSFT_PhysicalDisk -Filter ("DeviceId='{0}'" -f $diskIndex) -OperationTimeoutSec 3 -ErrorAction Stop | Select-Object -First 1
                    if ($null -ne $physical) {
                        $storage.bus_type = ConvertFrom-StorageBusType ([int]$physical.BusType)
                        $storage.media_type = ConvertFrom-StorageMediaType ([int]$physical.MediaType)
                        Set-ProbeStatus 'model_storage.media' 'ok' 'native_physical_disk' 'windows_cim'
                    } else { Set-ProbeStatus 'model_storage.media' 'unsupported' 'physical_disk_unavailable' 'windows_cim' }
                } catch { Set-ProbeError 'model_storage.media' $_ 'windows_cim' }
            }
        }
    } catch { Set-ProbeError 'model_storage.metadata' $_ 'windows_cim' }
    $before = $null
    if ($null -ne $diskIndex) {
        try { $before = Get-PhysicalDiskReadCounter $diskIndex } catch { Set-ProbeError 'model_storage.physical_reads' $_ 'windows_perf_raw' }
    }
    # Reads are independent of WMI/storage mapping and of each other.
    $storage.read_probe = Measure-ModelStorageRead $Path
    if ($null -ne $before) {
        try {
            $after = Get-PhysicalDiskReadCounter $diskIndex
            if ($null -ne $after -and [uint64]$after.DiskReadBytesPersec -ge [uint64]$before.DiskReadBytesPersec) {
                $delta = [uint64]$after.DiskReadBytesPersec - [uint64]$before.DiskReadBytesPersec
                $storage.read_probe.physical_disk_read_bytes = $delta
                # Disk-wide counters include other processes; never label this an exact cache ratio.
                Set-ProbeStatus 'model_storage.physical_reads' 'ok' 'disk_wide_includes_other_processes' 'windows_perf_raw'
            } else { Set-ProbeStatus 'model_storage.physical_reads' 'unsupported' 'counter_unavailable_or_reset' 'windows_perf_raw' }
        } catch { Set-ProbeError 'model_storage.physical_reads' $_ 'windows_perf_raw' }
    }
    return $storage
}

function Invoke-BenchmarkHardwareCollector {
    param([string]$Path, [string]$GpuName, [string]$Address)
    $script:diagnostics = [ordered]@{}
    foreach ($probe in @('cpu', 'ram.total', 'ram.modules', 'gpu.selection', 'gpu.pcie', 'model_storage.metadata', 'model_storage.media', 'model_storage.sequential', 'model_storage.random_64k', 'model_storage.physical_reads')) {
        Set-ProbeStatus $probe 'unsupported' 'native_data_unavailable' 'windows_native'
    }
    $hardware = [ordered]@{ cpu = $null; ram = [ordered]@{}; gpu = $null; model_storage = $null }
    try {
        $cpus = @(Get-CimInstance Win32_Processor -OperationTimeoutSec 3 -ErrorAction Stop | Select-Object -First 64)
        if ($cpus.Count -gt 0) {
            $hardware.cpu = [ordered]@{ name = Get-SafeHardwareText $cpus[0].Name; logical_threads = [int](($cpus | Measure-Object NumberOfLogicalProcessors -Sum).Sum) }
            Set-ProbeStatus 'cpu' 'ok' 'native_processor' 'windows_cim'
        }
    } catch { Set-ProbeError 'cpu' $_ 'windows_cim' }
    try {
        $system = Get-CimInstance Win32_ComputerSystem -OperationTimeoutSec 3 -ErrorAction Stop | Select-Object -First 1
        if ($null -ne $system) {
            $hardware.ram.total_bytes = [uint64]$system.TotalPhysicalMemory
            Set-ProbeStatus 'ram.total' 'ok' 'native_memory_total' 'windows_cim'
        }
    } catch { Set-ProbeError 'ram.total' $_ 'windows_cim' }
    try {
        $modules = @(Get-CimInstance Win32_PhysicalMemory -OperationTimeoutSec 3 -ErrorAction Stop | Select-Object -First 32 | ForEach-Object {
            [ordered]@{
                capacity_bytes = [uint64]$_.Capacity; generation = ConvertFrom-SmbiosMemoryType ([int]$_.SMBIOSMemoryType)
                speed_mt_s = if ($_.Speed) { [uint64]$_.Speed } else { $null }
                configured_speed_mt_s = if ($_.ConfiguredClockSpeed) { [uint64]$_.ConfiguredClockSpeed } else { $null }
                manufacturer = Get-SafeHardwareText $_.Manufacturer; part_number = Get-SafeHardwareText $_.PartNumber
            }
        })
        $hardware.ram.modules = $modules; $hardware.ram.module_count = $modules.Count
        $generations = @($modules | ForEach-Object { $_.generation } | Where-Object { $_ } | Sort-Object -Unique)
        $hardware.ram.memory_generation = if ($generations.Count -eq 1) { $generations[0] } elseif ($generations.Count -gt 1) { 'Mixed: ' + ($generations -join ', ') } else { $null }
        if ($modules.Count) { Set-ProbeStatus 'ram.modules' 'ok' 'smbios_modules' 'windows_cim' }
    } catch { Set-ProbeError 'ram.modules' $_ 'windows_cim' }
    try { $hardware.gpu = Get-WindowsSelectedGpu $GpuName $Address } catch { Set-ProbeError 'gpu.selection' $_ 'windows_pnp' }
    if ($Path) { $hardware.model_storage = Get-WindowsBenchmarkModelStorageDetails $Path }
    else {
        foreach ($probe in @('metadata', 'media', 'sequential', 'random_64k', 'physical_reads')) { Set-ProbeStatus "model_storage.$probe" 'not_applicable' 'no_model_path' 'windows_native' }
    }
    return [ordered]@{ schema_version = 1; hardware = $hardware; collection = [ordered]@{ phase = 'before_benchmark_warmup'; probes = $script:diagnostics } }
}

if ($MyInvocation.InvocationName -ne '.') {
    try { $result = Invoke-BenchmarkHardwareCollector $ModelPath $SelectedGpuName $PciAddress }
    catch { $result = [ordered]@{ schema_version = 1; hardware = @{ cpu = $null; ram = $null; gpu = $null; model_storage = $null }; collection = @{ probes = @{ collector = @{ status = 'failed'; reason = 'collector_error' } } } } }
    $result | ConvertTo-Json -Depth 12 -Compress
}
