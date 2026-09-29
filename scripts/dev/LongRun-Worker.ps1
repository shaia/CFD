# Worker for Start-LongRun.ps1: runs one executable and maintains its status file. Not for direct use.
param([Parameter(Mandatory)][string]$SpecFile)
$ErrorActionPreference = 'Continue'
$spec = Get-Content -LiteralPath $SpecFile -Raw | ConvertFrom-Json

function Write-Status($extra) {
    $s = [ordered]@{
        name = $spec.name; exe = $spec.exe; args = @($spec.args); workDir = $spec.workDir; log = $spec.log
    }
    foreach ($k in $extra.Keys) { $s[$k] = $extra[$k] }
    $s | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $spec.status -Encoding UTF8
}

if ($spec.env) {
    foreach ($prop in $spec.env.PSObject.Properties) { Set-Item -Path "Env:$($prop.Name)" -Value $prop.Value }
}
Set-Location -LiteralPath $spec.workDir
$start = Get-Date

# One writer for the whole run, flushed line by line and shared for reading, so Get-LongRun.ps1 can
# tail the log while the job runs and a killed worker leaves everything written so far.
$stream = [IO.FileStream]::new($spec.log, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite)
$writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
$writer.AutoFlush = $true
$log = [IO.TextWriter]::Synchronized($writer)
$log.WriteLine("[$($start.ToString('s'))] start: $($spec.exe) $(@($spec.args) -join ' ')")

$exit = -1
try {
    $psi = [Diagnostics.ProcessStartInfo]::new($spec.exe)
    # ArgumentList quotes each element, so an argument that contains spaces stays one argument.
    foreach ($a in @($spec.args)) { if ($null -ne $a) { $psi.ArgumentList.Add([string]$a) } }
    $psi.WorkingDirectory = $spec.workDir
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $proc = [Diagnostics.Process]::Start($psi)
    Write-Status @{ state = 'running'; pid = $proc.Id; start = $start.ToString('o') }

    # stderr is drained on its own thread while stdout is read here; reading only one of the two
    # pipes would block the child as soon as the other fills.
    $errJob = $null; $errTask = $null
    if (Get-Command Start-ThreadJob -ErrorAction SilentlyContinue) {
        $errJob = Start-ThreadJob -ScriptBlock {
            param($p, $w)
            while ($null -ne ($l = $p.StandardError.ReadLine())) { $w.WriteLine($l) }
        } -ArgumentList $proc, $log
    } else {
        $errTask = $proc.StandardError.ReadToEndAsync()
    }
    while ($null -ne ($line = $proc.StandardOutput.ReadLine())) { $log.WriteLine($line) }
    $proc.WaitForExit()
    if ($errJob) { $errJob | Wait-Job | Remove-Job }
    if ($errTask) { $rest = $errTask.GetAwaiter().GetResult(); if ($rest) { $log.Write($rest) } }
    $exit = $proc.ExitCode
} catch {
    $log.WriteLine("[error] $($_.Exception.Message)")
}
$end = Get-Date
$elapsed = [int]($end - $start).TotalSeconds
$log.WriteLine("[$($end.ToString('s'))] exit: $exit  elapsed: ${elapsed}s")
$log.Dispose()
$state = if ($exit -eq 0) { 'done' } else { 'failed' }
Write-Status @{ state = $state; pid = $null; start = $start.ToString('o'); end = $end.ToString('o'); exit = $exit; elapsedSeconds = $elapsed }
