<#
.SYNOPSIS
Show the state of detached runs started with Start-LongRun.ps1.

.DESCRIPTION
States: starting (launched, worker not up yet), running, done (exit 0), failed (non-zero exit), and
lost (the status says starting or running but no such process exists, so no exit was recorded).

.EXAMPLE
.\scripts\dev\Get-LongRun.ps1            # one line per job
.\scripts\dev\Get-LongRun.ps1 rich257    # plus the last 30 log lines
.\scripts\dev\Get-LongRun.ps1 rich257 -Tail 100
#>
param(
    [string]$Name,
    [string]$RunDir,
    [int]$Tail = 30
)
$repo = (git rev-parse --show-toplevel 2>$null)
if (-not $RunDir) { $RunDir = Join-Path ($(if ($repo) { $repo.Trim() } else { (Get-Location).Path })) 'output\runs' }
if (-not (Test-Path -LiteralPath $RunDir)) { Write-Host "no runs ($RunDir does not exist)"; return }

$files = @(Get-ChildItem -LiteralPath $RunDir -Filter '*.status.json' | Sort-Object LastWriteTime -Descending)
if ($Name) { $files = @($files | Where-Object { $_.Name -eq "$Name.status.json" }) }
if ($files.Count -eq 0) { Write-Host "no runs$(if ($Name) { " named '$Name'" })"; return }

foreach ($f in $files) {
    $s = Get-Content -LiteralPath $f.FullName -Raw | ConvertFrom-Json
    $state = $s.state
    if ($state -eq 'running') {
        $alive = $s.pid -and (Get-Process -Id $s.pid -ErrorAction SilentlyContinue)
        if (-not $alive) { $state = 'lost' }
        $elapsed = [int]((Get-Date) - [datetime]$s.start).TotalSeconds
    } elseif ($state -eq 'starting') {
        $elapsed = [int]((Get-Date) - [datetime]$s.start).TotalSeconds
        if ($elapsed -gt 60) { $state = 'lost' }
    } else {
        $elapsed = $s.elapsedSeconds
    }
    $exitTxt = if ($null -ne $s.exit) { " exit=$($s.exit)" } else { '' }
    '{0,-20} {1,-8} pid={2,-7} elapsed={3,6}s{4}  log={5}' -f $s.name, $state, $s.pid, $elapsed, $exitTxt, $s.log
}
if ($Name) {
    $s = Get-Content -LiteralPath $files[0].FullName -Raw | ConvertFrom-Json
    if (Test-Path -LiteralPath $s.log) {
        Write-Host "--- last $Tail lines of $($s.log) ---"
        Get-Content -LiteralPath $s.log -Tail $Tail
    }
}
