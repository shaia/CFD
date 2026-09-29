# Worker for Start-LongRun.ps1: runs one executable and maintains its status file. Not for direct use.
param([Parameter(Mandatory)][string]$SpecFile)
$ErrorActionPreference = 'Continue'
$spec = Get-Content $SpecFile -Raw | ConvertFrom-Json

function Write-Status($extra) {
    $s = [ordered]@{
        name = $spec.name; exe = $spec.exe; args = @($spec.args); workDir = $spec.workDir; log = $spec.log
    }
    foreach ($k in $extra.Keys) { $s[$k] = $extra[$k] }
    $s | ConvertTo-Json -Depth 4 | Set-Content -Path $spec.status -Encoding UTF8
}

if ($spec.env) {
    foreach ($prop in $spec.env.PSObject.Properties) { Set-Item -Path "Env:$($prop.Name)" -Value $prop.Value }
}
Set-Location $spec.workDir
$start = Get-Date
"[$($start.ToString('s'))] start: $($spec.exe) $($spec.args -join ' ')" | Set-Content -Path $spec.log -Encoding UTF8

$outFile = "$($spec.log).out"
$errFile = "$($spec.log).err"
try {
    $startArgs = @{
        FilePath = $spec.exe; NoNewWindow = $true; PassThru = $true
        RedirectStandardOutput = $outFile; RedirectStandardError = $errFile
    }
    if ($spec.args.Count -gt 0) { $startArgs.ArgumentList = @($spec.args) }
    $proc = Start-Process @startArgs
    Write-Status @{ state = 'running'; pid = $proc.Id; start = $start.ToString('o') }
    $proc.WaitForExit()
    $exit = $proc.ExitCode
} catch {
    "[error] $($_.Exception.Message)" | Add-Content -Path $spec.log
    $exit = -1
}
$end = Get-Date
foreach ($f in $outFile, $errFile) {
    if (Test-Path $f) { Get-Content $f | Add-Content -Path $spec.log; Remove-Item $f -ErrorAction SilentlyContinue }
}
$elapsed = [int]($end - $start).TotalSeconds
"[$($end.ToString('s'))] exit: $exit  elapsed: ${elapsed}s" | Add-Content -Path $spec.log
$state = if ($exit -eq 0) { 'done' } else { 'failed' }
Write-Status @{ state = $state; pid = $null; start = $start.ToString('o'); end = $end.ToString('o'); exit = $exit; elapsedSeconds = $elapsed }
