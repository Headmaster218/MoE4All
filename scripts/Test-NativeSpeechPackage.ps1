[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PackagePath,
    [Parameter(Mandatory = $true)][string]$WorkDir
)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression
$WorkDir=[IO.Path]::GetFullPath($WorkDir)
if (-not $WorkDir.StartsWith('D:\AIinfr\infr\local-work\tmp\moe4all\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Use local-work/tmp/moe4all/<task>/native-speech.' }
$ancestor=$WorkDir
while($ancestor){
    if((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'Test output cannot use reparse points.'}
    $ancestor=[IO.Path]::GetDirectoryName($ancestor)
}
[void][IO.Directory]::CreateDirectory($WorkDir)
$testRoot=Join-Path $WorkDir ('package-tests-'+[guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($testRoot)
$packScript=Join-Path $PSScriptRoot 'package-native-speech.ps1'
$tokens=$null;$parseErrors=$null
$tree=[Management.Automation.Language.Parser]::ParseFile($packScript,[ref]$tokens,[ref]$parseErrors)
if($parseErrors.Count){throw ($parseErrors|Out-String)}
# Import pure hashing/verification/path-policy functions; never execute package
# assembly, inference, downloads, services or the launcher entry point here.
foreach($definition in $tree.FindAll({param($node)$node -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){
    . ([scriptblock]::Create($definition.Extent.Text))
}
$script:checks=0
function Check([bool]$Condition,[string]$Message){if(-not $Condition){throw $Message};$script:checks++}
function Reject([scriptblock]$Action,[string]$Pattern,[string]$Message){
    $caught='';try{& $Action|Out-Null}catch{$caught=$_.Exception.Message}
    Check ($caught -match $Pattern) ($Message+': '+$caught)
}
$PackagePath=[IO.Path]::GetFullPath($PackagePath)
$publishedHash=([IO.File]::ReadAllText($PackagePath+'.sha256') -split '\s+')[0]
$oversized=[IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes('12345'),$false)
try{Reject {Read-NativeSpeechEntry 'SOURCES.json' 2 $oversized} 'exceeds its declared size' 'Metadata reads are bounded independently of ZIP headers'}finally{$oversized.Dispose()}
$shortStream=[IO.MemoryStream]::new([byte[]]@(1,2),$false)
try{Reject {Get-NativeSpeechStreamHash -Stream $shortStream -ExpectedBytes 3} 'shorter than' 'Bounded streaming hash detects truncation'}finally{$shortStream.Dispose()}
Check ($publishedHash -match '^[a-f0-9]{64}$') 'Outer archive checksum is present'
$originalArchiveHash=Get-NativeSpeechFileHash $PackagePath
Check ($originalArchiveHash -ceq $publishedHash) 'Archive matches external checksum'
$verified=Test-NativeSpeechPackage -Path $PackagePath -ExpectedArchiveSha256 $publishedHash
Check ($verified.OfflineVerified -and $verified.Files -eq 15 -and $verified.License -eq 'MIT' -and $verified.VerifierLicense -eq 'Apache-2.0') 'Real archive verifies offline with distinct voice/verifier licenses'
Check ($verified.ModelBytes -eq 164882176 -and $verified.ModelSha256 -ceq '6d0d6fffe43cf6dea169f584c07bbb1a43756f37958ba9f9b78e27e8f9b32cc2') 'Exact previously successful native weights'
Reject {Test-NativeSpeechPackage -Path $PackagePath -ExpectedArchiveSha256 ('0'*64)} 'archive SHA256 mismatch' 'Reject wrong externally supplied archive hash'
$bundleId='MeloTTS-Chinese-Native-F32-v1'
$archive=[IO.Compression.ZipFile]::OpenRead($PackagePath)
try{
    $entryNames=@($archive.Entries.FullName)
    Check ($entryNames.Count -eq 15) 'Exactly fifteen allowlisted archive entries'
    foreach($name in Get-NativeSpeechPackageNames){Check (($bundleId+'/'+$name) -cin $entryNames) ('Contains '+$name)}
    Check (@($entryNames|Where-Object{$_ -match 'checkpoint|conversion\.json|\.exe$|\.onnx$|gui-data|kv-sessions|\.log$'}).Count -eq 0) 'Exclude original checkpoint, binaries, ONNX, logs and user/runtime state'
}finally{$archive.Dispose()}

# Extract only the already verified archive into this task's isolated fixture.
$relocated=Join-Path $testRoot ("relocated voice's [test] "+[char]0x4e2d)
[void][IO.Directory]::CreateDirectory($relocated)
[IO.Compression.ZipFile]::ExtractToDirectory($PackagePath,$relocated)
$voice=Join-Path $relocated $bundleId
$folderResult=Test-NativeSpeechPackage $voice
Check ($folderResult.ModelSha256 -ceq $verified.ModelSha256) 'Relocated Unicode/spaces/apostrophes/brackets directory verifies'
$readme=[IO.File]::ReadAllText((Join-Path $voice 'README.txt'))
Check ($readme -match 'no conversion required' -and $readme -match 'No Python') 'End users receive ready-to-use weights'
Check ($readme -match 'this same folder as speech_lexicon') 'Instructions preserve model/resource pairing'
$sourceText=[IO.File]::ReadAllText((Join-Path $voice 'SOURCES.json'))
Check ($sourceText -notmatch '[A-Za-z]:\\|local-work|127\.0\.0\.1|\bpid\b') 'Public provenance contains no host paths or runtime identifiers'
$sources=$sourceText|ConvertFrom-Json
Check ($sources.model_repo -eq 'myshell-ai/MeloTTS-Chinese' -and $sources.frontend_repo -eq 'csukuangfj/vits-melo-tts-zh_en') 'Both original upstream sources are preserved'
Check ($sources.original_inputs.Count -eq 6) 'All original input identities are recorded without checkpoint payload'
Check ($sources.historical_validation.waveform_sha256 -ieq '44F77F73149C60A9EC26A2C29532665F28619A475B652339C96DD12583A94BC5' -and $sources.historical_validation.sample_rate -eq 44100) 'Historical successful HTTP waveform anchors provenance'
Check ($sources.source_commit -eq '66a4434a518a3b6d124756952977f0b10f0e9a76') 'Record the native TTS implementation commit'
$license=[IO.File]::ReadAllText((Join-Path $voice 'LICENSE'))
Check ($license -match 'Copyright \(c\) 2024 MyShell.ai' -and $license -match 'permission notice shall be included') 'Original upstream MIT attribution and notice included'
Check ([IO.File]::ReadAllText((Join-Path $voice 'VERIFIER-LICENSE.txt')) -match 'Apache License[\s\S]*Version 2\.0') 'Verifier retains the MoE4All Apache-2.0 license text'

# Exercise the distributed verifier as a separate Windows PowerShell 5.1 process.
$standalone=Join-Path $voice 'Verify-NativeSpeech.ps1'
$verifierOutput=& powershell.exe -NoLogo -NoProfile -NonInteractive -File $standalone -PackagePath $voice 2>&1
$verifyExit=$LASTEXITCODE
Check ($verifyExit -eq 0 -and ($verifierOutput -join "`n") -match 'OfflineVerified\s+: True') 'Distributed verifier works independently after relocation under WinPS 5.1'
$scriptResult=& $packScript -VerifyPackage $PackagePath -ExpectedArchiveSha256 $publishedHash
Check ($scriptResult.OfflineVerified) 'Packaging script also supports verify-only mode'

$configPath=Join-Path $voice 'config.json'
$configBefore=[IO.File]::ReadAllBytes($configPath)
try{
    $tampered=[byte[]]$configBefore.Clone();$tampered[0]=$tampered[0] -bxor 1
    [IO.File]::WriteAllBytes($configPath,$tampered)
    Reject {Test-NativeSpeechPackage $voice} 'Pinned native speech SHA256 mismatch: config.json' 'Reject same-size asset corruption even before trusting manifest'
}finally{[IO.File]::WriteAllBytes($configPath,$configBefore)}
$manifestPath=Join-Path $voice 'manifest.json';$manifestHashPath=Join-Path $voice 'manifest.sha256'
$manifestBefore=[IO.File]::ReadAllBytes($manifestPath);$manifestHashBefore=[IO.File]::ReadAllBytes($manifestHashPath)
try{
    $manifest=[IO.File]::ReadAllText($manifestPath)|ConvertFrom-Json
    $manifest.files[0].sha256='0'*64
    [IO.File]::WriteAllText($manifestPath,($manifest|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($manifestHashPath,((Get-NativeSpeechFileHash $manifestPath)+'  manifest.json'),[Text.UTF8Encoding]::new($false))
    Reject {Test-NativeSpeechPackage $voice} 'manifest payload mismatch' 'Reject incorrect per-file manifest even with an updated manifest checksum'
}finally{[IO.File]::WriteAllBytes($manifestPath,$manifestBefore);[IO.File]::WriteAllBytes($manifestHashPath,$manifestHashBefore)}
try{
    [IO.File]::WriteAllText($manifestHashPath,('0'*64+'  manifest.json'))
    Reject {Test-NativeSpeechPackage $voice} 'manifest SHA256 mismatch' 'Reject a corrupted manifest sidecar'
}finally{[IO.File]::WriteAllBytes($manifestHashPath,$manifestHashBefore)}
Reject {Test-NativeSpeechPackage -Path $voice -ExpectedArchiveSha256 $publishedHash} 'requires a ZIP' 'Do not apply an archive hash to an extracted folder'
Check ((Test-NativeSpeechPackage $voice).OfflineVerified) 'Restored fixture still verifies'

function New-TinyZip([string]$Name,[object[]]$Entries){
    $path=Join-Path $testRoot $Name
    $stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew)
    try{
        $zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$true)
        try{foreach($item in $Entries){$entry=$zip.CreateEntry($item.name);$output=$entry.Open();try{$output.Write($item.bytes,0,$item.bytes.Length)}finally{$output.Dispose()}}}finally{$zip.Dispose()}
    }finally{$stream.Dispose()}
    return $path
}
$utf8=[Text.UTF8Encoding]::new($false)
$traversal=New-TinyZip 'traversal.zip' @(@{name=$bundleId+'/../outside.txt';bytes=$utf8.GetBytes('fixture')})
Reject {Test-NativeSpeechPackage $traversal} 'Unexpected or duplicate' 'Reject ZIP traversal without extracting'
$wrongRoot=New-TinyZip 'wrong-root.zip' @(@{name='unrelated/config.json';bytes=$configBefore})
Reject {Test-NativeSpeechPackage $wrongRoot} 'archive root' 'Reject another bundle root'
$duplicate=New-TinyZip 'duplicate.zip' @(@{name=$bundleId+'/config.json';bytes=$configBefore},@{name=$bundleId+'/config.json';bytes=$configBefore})
Reject {Test-NativeSpeechPackage $duplicate} 'Unexpected or duplicate' 'Reject duplicate ZIP paths'
$missing=New-TinyZip 'missing.zip' @(@{name=$bundleId+'/config.json';bytes=$configBefore})
Reject {Test-NativeSpeechPackage $missing} 'Missing native speech payload' 'Reject incomplete ZIP'
Reject {Assert-NativeSpeechOutputPath 'D:\AIinfr\infr\scripts\not-a-package.zip' @($testRoot)} 'outside' 'Output policy rejects source tree'
Reject {Assert-NativeSpeechOutputPath ([IO.Path]::GetFullPath((Join-Path $testRoot '..\escaped.zip'))) @($testRoot)} 'outside' 'Output policy rejects traversal outside task root'
Reject {Assert-NativeSpeechOutputPath $testRoot @($testRoot)} 'outside' 'Output policy rejects root itself'
$extra=Join-Path $voice 'private-user-state.json';[IO.File]::WriteAllText($extra,'fixture')
Reject {Test-NativeSpeechPackage $voice} 'Unexpected native speech payload' 'Extracted package rejects extra files'
Check ((Get-NativeSpeechFileHash $PackagePath) -ceq $originalArchiveHash) 'Original distributable archive remains unchanged'
Write-Output ("PASS: {0} native speech package checks; hashes/archives only, no inference, GPU, downloads or service changes." -f $script:checks)
