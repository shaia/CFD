# Shared by Start-LongRun.ps1 and Get-LongRun.ps1 (dot-sourced). Not called directly.

# Runs are tracked where they are launched from: <top of the current directory's repository>\output\runs,
# or <current directory>\output\runs outside a repository. One definition, so the launcher and the
# status script always look in the same place.
function Get-LongRunDir([string]$RunDir) {
    if ($RunDir) { return $RunDir }
    $top = (git rev-parse --show-toplevel 2>$null)
    $base = if ($top) { $top.Trim() } else { (Get-Location).Path }
    return (Join-Path $base 'output\runs')
}

# A recorded process id alone is not proof of life: ids are reused after a kill or a reboot. The run
# is alive only if a process with that id exists and started when the status says the job started.
function Test-LongRunAlive($Status) {
    if (-not $Status.pid) { return $false }
    $p = Get-Process -Id $Status.pid -ErrorAction SilentlyContinue
    if (-not $p) { return $false }
    if (-not $Status.pidStarted) { return $true }
    try {
        return ([math]::Abs(($p.StartTime - [datetime]$Status.pidStarted).TotalSeconds) -lt 2)
    } catch {
        return $true
    }
}
