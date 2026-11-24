[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [string]$Version = '',
    [string]$BuildDirectory = ''
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $Version) {
    $match = [regex]::Match([IO.File]::ReadAllText((Join-Path $repoRoot 'Cargo.toml')), '(?m)^version\s*=\s*"([^"\r\n]+)"')
    if (-not $match.Success) { throw 'Cannot read the launcher version.' }
    $Version = $match.Groups[1].Value
}
$versionMatch = [regex]::Match($Version, '^(\d+)\.(\d+)\.(\d+)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$')
if (-not $versionMatch.Success) { throw 'Invalid launcher version.' }
$assemblyVersion = "$($versionMatch.Groups[1].Value).$($versionMatch.Groups[2].Value).$($versionMatch.Groups[3].Value).0"
$source = Join-Path $PSScriptRoot 'wizard-entry.cs'
$icon = Join-Path $PSScriptRoot 'moe4all.ico'
if (-not (Test-Path -LiteralPath $icon -PathType Leaf)) { throw 'Required launcher icon is missing: moe4all.ico' }
if (-not $BuildDirectory) {
    $workRoot = if ($env:MOE4ALL_WORK_ROOT) { $env:MOE4ALL_WORK_ROOT } else { Join-Path $repoRoot 'local-work' }
    $BuildDirectory = Join-Path $workRoot 'build-cache\agent\wizard-entry-netfx'
}
$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
[void][IO.Directory]::CreateDirectory($BuildDirectory)
[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($OutputPath))
$hash = [Security.Cryptography.SHA256]::Create()
try { $identity = [BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes([IO.File]::ReadAllText($source) + $Version + 'MoE4All Engine' + [Convert]::ToBase64String([IO.File]::ReadAllBytes($icon))))).Replace('-','').Substring(0,20) } finally { $hash.Dispose() }
$cached = Join-Path $BuildDirectory ("wizard-entry-$identity.exe")
if (-not (Test-Path -LiteralPath $cached -PathType Leaf)) {
    $compilerCandidates = @(
        (Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe')
        (Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe')
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    if ($compilerCandidates.Count -eq 0) { throw 'Windows .NET Framework compiler is unavailable.' }
    $compilerErrors = @()
    $compiled = $false
    foreach ($compiler in $compilerCandidates) {
        $tag = [guid]::NewGuid().ToString('N')
        $attributes = Join-Path $BuildDirectory ("entry-$tag.cs")
        $temporary = Join-Path $BuildDirectory ("entry-$tag.exe")
        [IO.File]::WriteAllText($attributes, ('[assembly:System.Reflection.AssemblyTitle("MoE4All Engine")]' + "`n" + '[assembly:System.Reflection.AssemblyVersion("'+$assemblyVersion+'")]' + "`n" + '[assembly:System.Reflection.AssemblyFileVersion("'+$assemblyVersion+'")]' + "`n" + '[assembly:System.Reflection.AssemblyInformationalVersion("'+$Version+'")]'), [Text.UTF8Encoding]::new($true))
        $info = [Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $compiler
        $info.Arguments = '/nologo /noconfig /optimize+ /target:winexe /platform:anycpu /utf8output /r:System.dll /r:System.Core.dll /r:System.Windows.Forms.dll /win32icon:"'+$icon+'" /out:"'+$temporary+'" "'+$source+'" "'+$attributes+'"'
        $info.UseShellExecute=$false; $info.CreateNoWindow=$true; $info.RedirectStandardOutput=$true; $info.RedirectStandardError=$true
        if ($null -ne $info.EnvironmentVariables) {
            $info.EnvironmentVariables['TEMP']=$BuildDirectory
            $info.EnvironmentVariables['TMP']=$BuildDirectory
        }
        $process = [Diagnostics.Process]::new(); $process.StartInfo=$info
        try {
            [void]$process.Start(); $stdout=$process.StandardOutput.ReadToEndAsync(); $stderr=$process.StandardError.ReadToEndAsync(); $process.WaitForExit()
            if ($process.ExitCode -ne 0) {
                $compilerErrors += "Compiler: $compiler`n$($stdout.Result)$($stderr.Result)"
                continue
            }
            try { [IO.File]::Move($temporary,$cached) } catch { if (-not (Test-Path -LiteralPath $cached -PathType Leaf)) { throw } }
            $compiled = $true
            break
        } finally {
            $process.Dispose()
            foreach ($file in @($attributes,$temporary)) { if (Test-Path -LiteralPath $file -PathType Leaf) { Remove-Item -LiteralPath $file } }
        }
    }
    if (-not $compiled) { throw "Launcher compilation failed:`n$($compilerErrors -join "`n")" }
}
if (-not [string]::Equals($cached,$OutputPath,[StringComparison]::OrdinalIgnoreCase)) { Copy-Item -LiteralPath $cached -Destination $OutputPath -Force }
Write-Output $OutputPath
