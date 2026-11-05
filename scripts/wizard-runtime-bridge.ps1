$script:RuntimePromptLines=[Collections.Generic.List[string]]::new()
# Private child-process UI protocol. No engine changes, uploads or implicit consent.
function Write-WizardRuntimeEvent {
    param([Parameter(Mandatory=$true)][hashtable]$Event)
    if ($env:MOE4ALL_DESKTOP_IO -ne '1') { return }
    $json=ConvertTo-Json -InputObject $Event -Depth 8 -Compress
    $encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json))
    if($encoded.Length -gt 60000){throw 'Runtime UI message exceeds its bounded frame.'}
    if($env:MOE4ALL_DESKTOP_CONTROL_PIPE){
        if($null -eq $script:RuntimeControlWriter){
            $script:RuntimeControlPipe=[IO.Pipes.NamedPipeClientStream]::new('.', $env:MOE4ALL_DESKTOP_CONTROL_PIPE, [IO.Pipes.PipeDirection]::Out)
            $script:RuntimeControlPipe.Connect(3000)
            $script:RuntimeControlWriter=[IO.StreamWriter]::new($script:RuntimeControlPipe,[Text.UTF8Encoding]::new($false))
            $script:RuntimeControlWriter.AutoFlush=$true
        }
        $script:RuntimeControlWriter.WriteLine($encoded)
        return
    }
    [Console]::Out.WriteLine("`n__MOE4ALL_UI__:${encoded}:__END_UI__")
    [Console]::Out.Flush()
}

function Write-Host {
    [CmdletBinding()]
    param([Parameter(Position=0,ValueFromRemainingArguments=$true)][object[]]$Object,[string]$Separator=' ',[switch]$NoNewLine,[ConsoleColor]$ForegroundColor='Gray',[ConsoleColor]$BackgroundColor='Black')
    $text=($Object|ForEach-Object {[string]$_}) -join $Separator
    if($env:MOE4ALL_DESKTOP_IO -eq '1'){
        if($null -eq $script:RuntimePromptLines){$script:RuntimePromptLines=[Collections.Generic.List[string]]::new()}
        foreach($line in ($text -split "`r?`n")){if($line.Trim().Length){$script:RuntimePromptLines.Add($line)}}
        while($script:RuntimePromptLines.Count -gt 12){$script:RuntimePromptLines.RemoveAt(0)}
    }
    Microsoft.PowerShell.Utility\Write-Host @PSBoundParameters
    if($env:MOE4ALL_DESKTOP_IO -eq '1' -and $text -match '(?i)浏览器|browser|上传|upload|cloudflare|github|https://github.com/.+/pull/'){Write-WizardRuntimeEvent @{kind='notice';message=$text}}
}

function Read-WizardRuntimeResponse {
    param([string]$Prompt='', [ValidateSet('confirm','choice','text','ack')][string]$InputType='text',
        [string]$DefaultValue='', [object[]]$Options=@(), [switch]$Secret,
        [string]$AffirmativeText='', [string]$NegativeText='', [string]$Title='')
    $context=if($null -ne $script:RuntimePromptLines){($script:RuntimePromptLines -join "`n")}else{''}
    if($context.Length -gt 3500){$context=$context.Substring($context.Length-3500)}
    $id=[guid]::NewGuid().ToString('N')
    $choices=@($Options | ForEach-Object { @{key=[string]$_.Key;value=[string]$_.Value;label=[string]$_.Label;detail=$(if($_.PSObject.Properties['Detail']){@($_.Detail)-join "`n"}else{''})} })
    Write-WizardRuntimeEvent @{kind='prompt';id=$id;input_type=$InputType;prompt=$Prompt;secret=[bool]$Secret;context=$context;default=$DefaultValue;options=$choices;yes_label=$AffirmativeText;no_label=$NegativeText;title=$Title}
    if($null -ne $script:RuntimePromptLines){$script:RuntimePromptLines.Clear()}
    $secure=$null
    if($Secret){
        $secure=Microsoft.PowerShell.Utility\Read-Host -AsSecureString
        $pointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        try{$value=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)}finally{[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)}
    }else{$value=[Console]::ReadLine()}
    if($null -eq $value -or $value -eq "__MOE4ALL_UI_CANCEL_${id}__"){
        if($null -ne $secure){$secure.Dispose()}
        throw [OperationCanceledException]::new('Submission cancelled; the local report is retained.')
    }
    if($Secret){return $secure}
    if($value.Length -eq 0){return $DefaultValue}
    return $value
}

function Read-Host {
    [CmdletBinding()]
    param([Parameter(Position=0)][object]$Prompt='', [switch]$AsSecureString)
    if($env:MOE4ALL_DESKTOP_IO -ne '1'){return (Microsoft.PowerShell.Utility\Read-Host @PSBoundParameters)}
    $inputType=if([string]$Prompt -match '\[[Yy]/[Nn]\]'){ 'confirm' }elseif([string]::IsNullOrWhiteSpace([string]$Prompt) -or [string]$Prompt -match '(?i)按回车|按 Enter|Press Enter|Enter to'){ 'ack' }else{ 'text' }
    if($inputType -eq 'confirm'){return (Read-WizardRuntimeResponse -Prompt ([string]$Prompt) -InputType 'confirm' -DefaultValue $(if([string]$Prompt -cmatch '\[Y/n\]'){'y'}else{'n'}))}
    return (Read-WizardRuntimeResponse -Prompt ([string]$Prompt) -InputType $inputType -Secret:$AsSecureString)
}

function Show-WizardRuntimeBenchmarkResult {
    param([Parameter(Mandatory=$true)][string]$Path)
    if($env:MOE4ALL_DESKTOP_IO -ne '1'){return}
    $payload=Read-MoE4AllSubmissionPayload -Path $Path
    $rows=[Collections.Generic.List[object]]::new()
    function Format-UiSpeed($Value){if($null -eq $Value){return '-'};return ([double]$Value).ToString('F1',[Globalization.CultureInfo]::InvariantCulture)}
    foreach($context in @($payload.contexts)){
        $label=if($context.requested_context_tokens -ge 1024){([double]$context.requested_context_tokens/1024).ToString('0.#',[Globalization.CultureInfo]::InvariantCulture)+'K'}else{[string]$context.requested_context_tokens}
        if($null -ne $context.prefill){$rows.Add(@{context=$label;workload='Prefill';first='-';middle='-';last='-';overall=(Format-UiSpeed $context.prefill.tokens_per_second)})}
        foreach($decode in @($context.decode)){$sections=@($decode.section_mean_tokens_per_second);$rows.Add(@{context=$label;workload=('Decode x'+$decode.parallel);first=$(if($sections.Count -gt 0){Format-UiSpeed $sections[0]}else{'-'});middle=$(if($sections.Count -gt 1){Format-UiSpeed $sections[1]}else{'-'});last=$(if($sections.Count -gt 2){Format-UiSpeed $sections[2]}else{'-'});overall=(Format-UiSpeed $decode.overall_tokens_per_second)})}
    }
    Write-WizardRuntimeEvent @{kind='result';path=$Path;rows=@($rows.ToArray())}
}
