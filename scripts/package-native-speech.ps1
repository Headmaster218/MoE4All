[CmdletBinding(DefaultParameterSetName = 'Package')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Package')][string]$SourceDirectory,
    [Parameter(Mandatory = $true, ParameterSetName = 'Package')][string]$ValidationManifest,
    [Parameter(ParameterSetName = 'Package')][string]$TaskId = '20261105-031000-runtime-experience-c81e',
    [Parameter(ParameterSetName = 'Package')][string]$WorkRoot = 'D:\AIinfr\infr\local-work',
    [Parameter(ParameterSetName = 'Package')][string]$OutputDirectory = '',
    [Parameter(Mandatory = $true, ParameterSetName = 'Verify')][string]$VerifyPackage,
    [Parameter(ParameterSetName = 'Verify')][string]$ExpectedArchiveSha256 = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression

function Get-NativeSpeechFileSpec {
    # This allowlist pins the exact voice used by the historical CPU CLI/HTTP
    # acceptance. No checkpoint, conversion report, cache, log or user data ships.
    @(
        [pscustomobject]@{ Name='melotts-chinese-f32.gguf'; Source='melotts-chinese-f32.gguf'; Bytes=164882176; Sha256='6d0d6fffe43cf6dea169f584c07bbb1a43756f37958ba9f9b78e27e8f9b32cc2' }
        [pscustomobject]@{ Name='config.json'; Source='config.json'; Bytes=2296; Sha256='d58b5acdab89ad2bbd65325affab309ae3cb964834b02f9a60587474e81c8bb9' }
        [pscustomobject]@{ Name='lexicon.txt'; Source='lexicon.txt'; Bytes=6837671; Sha256='7236884b02435ac5d10cf69b4be40a61b45aa676b5300f0e412f185748fee528' }
        [pscustomobject]@{ Name='tokens.txt'; Source='tokens.txt'; Bytes=655; Sha256='d18664a7e12bd7ea1022ddaf951e534e136815016c5a809d6b64156bffb4369d' }
        [pscustomobject]@{ Name='frontend.json'; Source='frontend.json'; Bytes=166; Sha256='8849152a1c02c74a0b28ac52d0a4aec66920e74ddecc31c8e82a51c8caef4bfe' }
        [pscustomobject]@{ Name='custom-lexicon.txt'; Source='custom-lexicon.txt'; Bytes=730; Sha256='f9a6a029dd35a500350c6e518de0b3dee8a3c8a9c7f851887f693adfe34419b0' }
        [pscustomobject]@{ Name='LICENSE'; Source='LICENSE'; Bytes=1053; Sha256='88a50e5a02bbc2a5c2f084dc19da751aa97b1690f5fda76cd8005c8634d1ca70' }
        [pscustomobject]@{ Name='UPSTREAM-README.md'; Source='README.md'; Bytes=4445; Sha256='40b5b860bacbb3947cbef838719e660d2fe4961eba5b484290b4bedc0253e208' }
    )
}

function Get-NativeSpeechPackageNames {
    @((Get-NativeSpeechFileSpec).Name) + @('SOURCES.json', 'README.txt', 'NOTICE.txt', 'Verify-NativeSpeech.ps1', 'VERIFIER-LICENSE.txt', 'manifest.json', 'manifest.sha256')
}

function Get-NativeSpeechStreamHash {
    param([IO.Stream]$Stream, [long]$ExpectedBytes = -1)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        if ($ExpectedBytes -lt 0) { return [BitConverter]::ToString($algorithm.ComputeHash($Stream)).Replace('-', '').ToLowerInvariant() }
        $buffer=New-Object byte[] 81920; $total=[long]0
        while (($count=$Stream.Read($buffer,0,$buffer.Length)) -gt 0) {
            $total+=$count
            if ($total -gt $ExpectedBytes) { throw 'Native speech stream exceeds its declared size.' }
            [void]$algorithm.TransformBlock($buffer,0,$count,$buffer,0)
        }
        if ($total -ne $ExpectedBytes) { throw 'Native speech stream is shorter than its declared size.' }
        [void]$algorithm.TransformFinalBlock([byte[]]@(),0,0)
        return [BitConverter]::ToString($algorithm.Hash).Replace('-', '').ToLowerInvariant()
    }
    finally { $algorithm.Dispose() }
}

function Get-NativeSpeechFileHash {
    param([string]$Path)
    $stream = [IO.File]::OpenRead($Path)
    try { Get-NativeSpeechStreamHash $stream } finally { $stream.Dispose() }
}

function Read-NativeSpeechEntry {
    param([string]$Name, [long]$Length, [IO.Stream]$Stream)
    $pinned = @(Get-NativeSpeechFileSpec | Where-Object { $_.Name -ceq $Name })
    if ($pinned.Count) {
        if ($Length -ne $pinned[0].Bytes) { throw "Native speech size mismatch: $Name" }
    } elseif ($Length -le 0 -or $Length -gt 131072) { throw "Native speech metadata exceeds bounds: $Name" }
    $record = [ordered]@{ Name=$Name; Bytes=$Length; Sha256=''; Text='' }
    if ($Name -in @('manifest.json', 'manifest.sha256', 'SOURCES.json')) {
        $memory = [IO.MemoryStream]::new()
        try {
            $buffer=New-Object byte[] 8192
            while (($count=$Stream.Read($buffer,0,$buffer.Length)) -gt 0) {
                if (($memory.Length+$count) -gt $Length) { throw 'Native speech metadata exceeds its declared size.' }
                $memory.Write($buffer,0,$count)
            }
            if ($memory.Length -ne $Length) { throw 'Native speech metadata is shorter than its declared size.' }
            $memory.Position=0
            $record.Sha256=Get-NativeSpeechStreamHash $memory
            $record.Text=[Text.Encoding]::UTF8.GetString($memory.ToArray())
        } finally { $memory.Dispose() }
    } else { $record.Sha256=Get-NativeSpeechStreamHash -Stream $Stream -ExpectedBytes $Length }
    return [pscustomobject]$record
}

function Test-NativeSpeechPackage {
    param([Parameter(Mandatory = $true)][string]$Path, [string]$ExpectedArchiveSha256 = '')
    $pathFull = [IO.Path]::GetFullPath($Path)
    $bundleId = 'MeloTTS-Chinese-Native-F32-v1'
    $allowed = @(Get-NativeSpeechPackageNames)
    $records = @{}; $archiveHash = ''
    if (Test-Path -LiteralPath $pathFull -PathType Container) {
        if ($ExpectedArchiveSha256) { throw 'Archive SHA256 requires a ZIP, not a directory.' }
        foreach ($file in @(Get-ChildItem -LiteralPath $pathFull -Force)) {
            if ($file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $file.Name -cnotin $allowed) { throw ('Unexpected native speech payload: ' + $file.Name) }
            $stream=[IO.File]::OpenRead($file.FullName)
            try { $records[$file.Name]=Read-NativeSpeechEntry $file.Name $file.Length $stream } finally { $stream.Dispose() }
        }
    } else {
        $archiveHash=Get-NativeSpeechFileHash $pathFull
        if ($ExpectedArchiveSha256 -and ($ExpectedArchiveSha256 -notmatch '^[a-fA-F0-9]{64}$' -or $archiveHash -ine $ExpectedArchiveSha256)) { throw 'Native speech archive SHA256 mismatch.' }
        $archive=[IO.Compression.ZipFile]::OpenRead($pathFull)
        try {
            foreach ($entry in $archive.Entries) {
                if ($entry.FullName -ceq ($bundleId+'/')) { continue }
                if (-not $entry.FullName.StartsWith(($bundleId+'/'), [StringComparison]::Ordinal)) { throw 'Unexpected native speech archive root.' }
                $name=$entry.FullName.Substring($bundleId.Length+1)
                if ($name -cnotin $allowed -or $records.ContainsKey($name)) { throw "Unexpected or duplicate native speech payload: $name" }
                $stream=$entry.Open()
                try { $records[$name]=Read-NativeSpeechEntry $name $entry.Length $stream } finally { $stream.Dispose() }
            }
        } finally { $archive.Dispose() }
    }
    foreach ($name in $allowed) { if (-not $records.ContainsKey($name)) { throw "Missing native speech payload: $name" } }
    if ($records.Count -ne $allowed.Count) { throw 'Unexpected native speech file count.' }
    foreach ($spec in Get-NativeSpeechFileSpec) {
        if ($records[$spec.Name].Sha256 -cne $spec.Sha256) { throw ('Pinned native speech SHA256 mismatch: ' + $spec.Name) }
    }
    if ($records['manifest.sha256'].Text.Trim() -cne ($records['manifest.json'].Sha256+'  manifest.json')) { throw 'Native speech manifest SHA256 mismatch.' }
    $manifest=$records['manifest.json'].Text | ConvertFrom-Json
    if ($manifest.schema_version -ne 1 -or $manifest.package_id -cne $bundleId -or $manifest.license -cne 'MIT' -or $manifest.verifier_license -cne 'Apache-2.0') { throw 'Invalid native speech manifest identity.' }
    $seen=@{}
    foreach ($file in @($manifest.files)) {
        if ($file.name -cnotin $allowed -or $file.name -in @('manifest.json','manifest.sha256') -or $seen.ContainsKey([string]$file.name)) { throw 'Invalid native speech manifest file list.' }
        $seen[[string]$file.name]=$true
        $actual=$records[[string]$file.name]
        if ($actual.Bytes -ne $file.bytes -or $actual.Sha256 -cne $file.sha256) { throw ('Native speech manifest payload mismatch: ' + $file.name) }
    }
    if ($seen.Count -ne ($allowed.Count-2)) { throw 'Native speech manifest omits files.' }
    $sources=$records['SOURCES.json'].Text | ConvertFrom-Json
    if ($sources.model_revision -cne 'af5d207a364ea4208c6f589c89f57f88414bdd16' -or
        $sources.frontend_revision -cne 'a0d5c6a264c0ef92d70d8661d8cc502d79627cd6' -or
        $sources.conversion -cne 'native-f32-zero-bert-dp-v1') { throw 'Invalid native speech provenance.' }
    return [pscustomobject]@{ PackagePath=$pathFull; PackageId=$bundleId; Files=$records.Count
        ModelSha256=$records['melotts-chinese-f32.gguf'].Sha256; ModelBytes=$records['melotts-chinese-f32.gguf'].Bytes
        ArchiveSha256=$archiveHash; License='MIT'; VerifierLicense='Apache-2.0'; OfflineVerified=$true }
}

function Assert-NativeSpeechOutputPath {
    param([string]$Path, [string[]]$AllowedRoots)
    $full=[IO.Path]::GetFullPath($Path)
    $inside=$false
    foreach ($root in $AllowedRoots) { if ($full.StartsWith(([IO.Path]::GetFullPath($root).TrimEnd('\')+'\'), [StringComparison]::OrdinalIgnoreCase)) { $inside=$true } }
    if (-not $inside) { throw "Native speech output is outside the task's packages/tmp roots: $full" }
    $ancestor=$full
    while ($ancestor) {
        if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Native speech outputs cannot use reparse points.' }
        $ancestor=[IO.Path]::GetDirectoryName($ancestor)
    }
    return $full
}

function Write-NativeSpeechNewFile {
    param([string]$Path, [byte[]]$Bytes)
    $stream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try { $stream.Write($Bytes,0,$Bytes.Length) } finally { $stream.Dispose() }
}

if ($PSCmdlet.ParameterSetName -eq 'Verify') {
    Test-NativeSpeechPackage -Path $VerifyPackage -ExpectedArchiveSha256 $ExpectedArchiveSha256
    return
}

if ($TaskId -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]+$') { throw 'TaskId must be one safe directory component.' }
$WorkRoot=[IO.Path]::GetFullPath($WorkRoot)
if ((Test-Path -LiteralPath 'D:\AIinfr\disk_usage_rule.md') -and $WorkRoot -ine 'D:\AIinfr\infr\local-work') { throw 'This host requires D:\AIinfr\infr\local-work.' }
$packageRoot=Join-Path $WorkRoot ('packages\moe4all\'+$TaskId)
$taskTmp=Join-Path $WorkRoot ('tmp\moe4all\'+$TaskId+'\native-speech')
if (-not $OutputDirectory) { $OutputDirectory=Join-Path $packageRoot 'native-speech' }
$OutputDirectory=Assert-NativeSpeechOutputPath $OutputDirectory @($packageRoot,(Join-Path $WorkRoot ('tmp\moe4all\'+$TaskId)))
$taskTmp=Assert-NativeSpeechOutputPath $taskTmp @((Join-Path $WorkRoot ('tmp\moe4all\'+$TaskId)))
$sourceFull=[IO.Path]::GetFullPath($SourceDirectory)
$specs=@(Get-NativeSpeechFileSpec)
foreach ($spec in $specs) {
    $file=Get-Item -LiteralPath (Join-Path $sourceFull $spec.Source)
    if ($file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $file.Length -ne $spec.Bytes -or (Get-NativeSpeechFileHash $file.FullName) -cne $spec.Sha256) { throw ('Source voice does not match the validated native package: ' + $spec.Source) }
}
$sourceManifest=Get-Content -LiteralPath (Join-Path $sourceFull 'sources.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$conversion=Get-Content -LiteralPath (Join-Path $sourceFull 'conversion.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ((Get-NativeSpeechFileHash (Join-Path $sourceFull 'sources.json')) -cne 'e49baec101e6b1e09e6f95a4ece98d9bef12de64474eed696ef0a74d80f4c9f6') { throw 'Original input source manifest does not match the preserved voice provenance.' }
if ($sourceManifest.model_repo -cne 'myshell-ai/MeloTTS-Chinese' -or $sourceManifest.model_revision -cne 'af5d207a364ea4208c6f589c89f57f88414bdd16' -or
    $sourceManifest.frontend_repo -cne 'csukuangfj/vits-melo-tts-zh_en' -or $sourceManifest.frontend_revision -cne 'a0d5c6a264c0ef92d70d8661d8cc502d79627cd6' -or
    $conversion.conversion -cne 'native-f32-zero-bert-dp-v1' -or $conversion.sha256 -cne $specs[0].Sha256 -or $conversion.bytes -ne $specs[0].Bytes -or $conversion.tensor_count -ne 565) { throw 'Source conversion/provenance is not the validated native voice.' }
$validation=Get-Content -LiteralPath $ValidationManifest -Raw -Encoding UTF8 | ConvertFrom-Json
if ($validation.task_id -cne '20261104-200651-melo-native-j3a67' -or $validation.status -cne 'passed_final_service_running' -or
    $validation.waveform.sha256 -ine '44F77F73149C60A9EC26A2C29532665F28619A475B652339C96DD12583A94BC5' -or
    $validation.waveform.sample_rate -ne 44100 -or $validation.waveform.samples -ne 193024) { throw 'Expected the preserved successful native CLI/HTTP validation record.' }

$bundleId='MeloTTS-Chinese-Native-F32-v1'
$archivePath=Join-Path $OutputDirectory ($bundleId+'.zip')
$partialPath=$archivePath+'.partial'
foreach ($path in @($archivePath,$partialPath,($archivePath+'.sha256'),(Join-Path $OutputDirectory 'catalog-template.json'),(Join-Path $OutputDirectory 'Verify-NativeSpeech.ps1'))) {
    if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite a native speech artifact: $path" }
}
$sourceBytes=[long](($specs | Measure-Object Bytes -Sum).Sum)
$drive=[IO.DriveInfo]::new([IO.Path]::GetPathRoot($OutputDirectory))
if ($drive.AvailableFreeSpace -lt (($sourceBytes*3)+64MB)) { throw 'Insufficient free disk space for the native speech package and verification.' }
[void][IO.Directory]::CreateDirectory($OutputDirectory)
[void][IO.Directory]::CreateDirectory($taskTmp)
Write-Host ('Packages: '+$OutputDirectory)
Write-Host ('Temp: '+$taskTmp+'; no build, model acquisition, GPU, service or publication step.')
$previousTemp=$env:TEMP; $previousTmp=$env:TMP; $previousTmpdir=$env:TMPDIR
try {
    $env:TEMP=$taskTmp; $env:TMP=$taskTmp; $env:TMPDIR=$taskTmp
    $utf8=[Text.UTF8Encoding]::new($false)
    $verifier = @'
# SPDX-License-Identifier: Apache-2.0; see VERIFIER-LICENSE.txt.
[CmdletBinding()]
param([string]$PackagePath = $PSScriptRoot, [string]$ExpectedArchiveSha256 = '')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression
'@
    foreach ($name in @('Get-NativeSpeechFileSpec','Get-NativeSpeechPackageNames','Get-NativeSpeechStreamHash','Get-NativeSpeechFileHash','Read-NativeSpeechEntry','Test-NativeSpeechPackage')) {
        $verifier += "`nfunction $name {`n" + (Get-Command $name -CommandType Function).ScriptBlock.ToString() + "`n}`n"
    }
    $verifier += "`nTest-NativeSpeechPackage -Path `$PackagePath -ExpectedArchiveSha256 `$ExpectedArchiveSha256`n"
    $sources=[ordered]@{
        model_repo=$sourceManifest.model_repo; model_revision=$sourceManifest.model_revision
        frontend_repo=$sourceManifest.frontend_repo; frontend_revision=$sourceManifest.frontend_revision
        license='MIT'; license_file='LICENSE'; upstream_model_card='UPSTREAM-README.md'
        verifier_license='Apache-2.0'; verifier_license_file='VERIFIER-LICENSE.txt'
        verifier_license_source='MoE4All repository LICENSE (Apache License, Version 2.0)'
        conversion='native-f32-zero-bert-dp-v1'; source_commit='66a4434a518a3b6d124756952977f0b10f0e9a76'
        conversion_script='scripts/convert_melotts_chinese.py'
        conversion_changes=@('F32 inference tensors; folded weight normalization','zero BERT with projection biases retained; deterministic duration predictor','external token/lexicon frontend and explicit technical-term spellings')
        original_inputs=@($sourceManifest.assets)
        historical_validation=[ordered]@{ task_id=$validation.task_id; verified_at=$validation.verified_at
            kind='Successful native infr CLI and HTTP synthesis; historical evidence, not a new inference run'
            engine_sha256=$validation.final_exe_sha256; waveform_sha256=$validation.waveform.sha256
            sample_rate=$validation.waveform.sample_rate; samples=$validation.waveform.samples }
    }
    $readme=@'
Native Melo Chinese voice for INFR (CPU), ready to use; no conversion required.
Requires an infr executable built with the optional tts feature. No Python,
PyTorch, ONNX runtime or checkpoint is needed by the end user.

1. Verify the ZIP offline using the adjacent Verify-NativeSpeech.ps1:
   powershell.exe -NoProfile -File .\Verify-NativeSpeech.ps1 -PackagePath .\MeloTTS-Chinese-Native-F32-v1.zip -ExpectedArchiveSha256 <published SHA256>
2. Extract this folder under models beside your launcher.
3. Enable Speech synthesis. Choose melotts-chinese-f32.gguf as speech_model,
   and this same folder as speech_lexicon. Keep the entire folder together.
4. Default API settings: CPU threads 4, idle timeout 300 seconds, clip cap 30
   seconds (supported cap 1..120). POST /v1/audio/speech returns WAV or PCM.

Do not substitute an ONNX model, a different GGUF export, or unrelated assets.
The frontend supports the package's Chinese voice (speaker 1); unsupported
words can require explicit pronunciations. This is not a general English G2P.

To verify an extracted folder, run its Verify-NativeSpeech.ps1 with no arguments.
Verification hashes files only and never loads a model or starts a service.
See SOURCES.json, UPSTREAM-README.md and LICENSE for provenance and attribution.
Voice/assets are MIT. The standalone verifier is Apache-2.0; its full license
text is preserved in VERIFIER-LICENSE.txt.
'@
    $notice=@'
MeloTTS model and frontend assets: Copyright (c) 2024 MyShell.ai, MIT.
The original LICENSE text and original model card are included without changes.
Model source: myshell-ai/MeloTTS-Chinese @ af5d207a364ea4208c6f589c89f57f88414bdd16
Lexicon/token/license source: csukuangfj/vits-melo-tts-zh_en @ a0d5c6a264c0ef92d70d8661d8cc502d79627cd6
INFR native conversion: native-f32-zero-bert-dp-v1, commit 66a4434a518a3b6d124756952977f0b10f0e9a76.
This is a derived runtime package, not a new upstream model. Transformations
and original input hashes are recorded in SOURCES.json. Runtime and generated
technical-term frontend data are distributed as one matching voice package.
Standalone offline verifier: MoE4All project, Apache-2.0. Its original license
text is included as VERIFIER-LICENSE.txt; this is separate from the voice MIT license.
'@
    $payload=[Collections.Generic.List[object]]::new()
    foreach ($spec in $specs) { [void]$payload.Add([pscustomobject]@{Name=$spec.Name;Path=(Join-Path $sourceFull $spec.Source);Bytes=$null;Length=$spec.Bytes;Sha256=$spec.Sha256}) }
    $verifierLicensePath=Join-Path ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))) 'LICENSE'
    $verifierLicense=[IO.File]::ReadAllBytes($verifierLicensePath)
    if ([Text.Encoding]::UTF8.GetString($verifierLicense) -notmatch 'Apache License[\s\S]*Version 2\.0') { throw 'Expected the MoE4All Apache-2.0 LICENSE for the standalone verifier.' }
    $memory=[IO.MemoryStream]::new($verifierLicense,$false)
    try { $verifierLicenseHash=Get-NativeSpeechStreamHash $memory } finally { $memory.Dispose() }
    [void]$payload.Add([pscustomobject]@{Name='VERIFIER-LICENSE.txt';Path='';Bytes=$verifierLicense;Length=$verifierLicense.Length;Sha256=$verifierLicenseHash})
    foreach ($item in @(
        @{Name='SOURCES.json';Text=($sources|ConvertTo-Json -Depth 10)}, @{Name='README.txt';Text=$readme},
        @{Name='NOTICE.txt';Text=$notice}, @{Name='Verify-NativeSpeech.ps1';Text=$verifier}
    )) {
        $bytes=$utf8.GetBytes(($item.Text+"`n")); $memory=[IO.MemoryStream]::new($bytes,$false)
        try { $hash=Get-NativeSpeechStreamHash $memory } finally { $memory.Dispose() }
        [void]$payload.Add([pscustomobject]@{Name=$item.Name;Path='';Bytes=$bytes;Length=$bytes.Length;Sha256=$hash})
    }
    $manifest=[ordered]@{schema_version=1;package_id=$bundleId;license='MIT';verifier_license='Apache-2.0';model='melotts-chinese-f32.gguf';lexicon='.'
        files=@($payload | ForEach-Object {[ordered]@{name=$_.Name;bytes=$_.Length;sha256=$_.Sha256}})}
    $manifestBytes=$utf8.GetBytes((($manifest|ConvertTo-Json -Depth 8)+"`n"))
    $memory=[IO.MemoryStream]::new($manifestBytes,$false)
    try { $manifestHash=Get-NativeSpeechStreamHash $memory } finally { $memory.Dispose() }
    [void]$payload.Add([pscustomobject]@{Name='manifest.json';Path='';Bytes=$manifestBytes;Length=$manifestBytes.Length;Sha256=$manifestHash})
    [void]$payload.Add([pscustomobject]@{Name='manifest.sha256';Path='';Bytes=$utf8.GetBytes(($manifestHash+'  manifest.json'+"`n"));Length=0;Sha256=''})
    $zipStream=[IO.File]::Open($partialPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try {
        $zip=[IO.Compression.ZipArchive]::new($zipStream,[IO.Compression.ZipArchiveMode]::Create,$true)
        try {
            foreach ($file in $payload) {
                $entry=$zip.CreateEntry(($bundleId+'/'+$file.Name),[IO.Compression.CompressionLevel]::Optimal)
                $entry.LastWriteTime=[DateTimeOffset]::Parse('2026-11-04T00:00:00Z',[Globalization.CultureInfo]::InvariantCulture)
                $output=$entry.Open()
                try {
                    if ($file.Path) { $inputStream=[IO.File]::OpenRead($file.Path);try {$inputStream.CopyTo($output)}finally{$inputStream.Dispose()} }
                    else { $output.Write($file.Bytes,0,$file.Bytes.Length) }
                } finally { $output.Dispose() }
            }
        } finally { $zip.Dispose() }
    } finally { $zipStream.Dispose() }
    $verified=Test-NativeSpeechPackage $partialPath
    [IO.File]::Move($partialPath,$archivePath)
    Write-NativeSpeechNewFile ($archivePath+'.sha256') ($utf8.GetBytes(($verified.ArchiveSha256+'  '+[IO.Path]::GetFileName($archivePath)+"`n")))
    Write-NativeSpeechNewFile (Join-Path $OutputDirectory 'Verify-NativeSpeech.ps1') ($utf8.GetBytes($verifier))
    # Fill URLs only after authorized hosting and hash verification. The existing
    # wizard downloads individual files; no archive extraction dependency is needed.
    $template=[ordered]@{Id='melotts-chinese-native-f32';Kind='speech';Downloadable=$false
        UnavailableReason='Native package is ready; authorized public hosting is pending.'
        Primary=($bundleId+'/melotts-chinese-f32.gguf');SpeechLexicon=$bundleId
        PackageFileName=($bundleId+'.zip');PackageSha256=$verified.ArchiveSha256
        Files=@($payload|ForEach-Object{[ordered]@{Url=$null;RelativePath=($bundleId+'/'+$_.Name);Sha256=$_.Sha256;Bytes=$_.Length}})}
    # manifest.sha256 is not self-listed in the package manifest; hash it here.
    $last=$template.Files[$template.Files.Count-1];$memory=[IO.MemoryStream]::new($payload[$payload.Count-1].Bytes,$false)
    try {$last.Sha256=Get-NativeSpeechStreamHash $memory;$last.Bytes=$memory.Length}finally{$memory.Dispose()}
    Write-NativeSpeechNewFile (Join-Path $OutputDirectory 'catalog-template.json') ($utf8.GetBytes((($template|ConvertTo-Json -Depth 8)+"`n")))
    $verified.PackagePath=$archivePath
    return $verified
} finally { $env:TEMP=$previousTemp; $env:TMP=$previousTmp; $env:TMPDIR=$previousTmpdir }
