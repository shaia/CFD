<#
.SYNOPSIS
Launch a long-running executable detached from the current session and track it with a status file.

.DESCRIPTION
Anything expected to run longer than about ten minutes (full cavity validation, Richardson grids
above 129x129, channel DNS) goes through this script. It starts LongRun-Worker.ps1 as an independent
pwsh process (Start-Process, so it survives the shell that launched it), which runs the executable,
writes its output to <RunDir>\<Name>.log line by line as it arrives, and keeps
<RunDir>\<Name>.status.json up to date (state, pid, start, exit, end). Check progress with
Get-LongRun.ps1.

Each element of -ExeArgs reaches the executable as one argument, including elements with spaces.

.EXAMPLE
.\scripts\dev\Start-LongRun.ps1 -Name rich257 -Exe build-rel\Release\test_cavity_richardson.exe -ExeArgs 257
.\scripts\dev\Start-LongRun.ps1 -Name ctest-full -Exe ctest -ExeArgs '--test-dir','build','-C','Debug','-L','validation' -Env @{ OMP_NUM_THREADS = '4' }
#>
param(
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][string]$Exe,
    [string[]]$ExeArgs = @(),
    [string]$WorkDir = (Get-Location).Path,
    [string]$RunDir,
    [hashtable]$Env = @{},
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
if ($Name -notmatch '^[A-Za-z0-9._-]+$') { throw "Name must be [A-Za-z0-9._-]+ (got '$Name')" }
$repo = (git -C $WorkDir rev-parse --show-toplevel 2>$null)
if (-not $RunDir) { $RunDir = Join-Path ($(if ($repo) { $repo.Trim() } else { $WorkDir })) 'output\runs' }
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$status = Join-Path $RunDir "$Name.status.json"
if ((Test-Path -LiteralPath $status) -and -not $Force) {
    $s = Get-Content -LiteralPath $status -Raw | ConvertFrom-Json
    if ($s.state -eq 'running' -and $s.pid -and (Get-Process -Id $s.pid -ErrorAction SilentlyContinue)) {
        throw "'$Name' is still running (pid $($s.pid)); pick another -Name or pass -Force"
    }
    $age = ((Get-Date) - (Get-Item -LiteralPath $status).LastWriteTime).TotalSeconds
    if ($s.state -eq 'starting' -and $age -lt 60) {
        throw "'$Name' was started $([int]$age)s ago and is still starting; pick another -Name or pass -Force"
    }
}

$resolvedExe = if (Test-Path -LiteralPath $Exe) { (Resolve-Path -LiteralPath $Exe).Path } else { $Exe }
$spec = [ordered]@{
    name    = $Name
    exe     = $resolvedExe
    args    = @($ExeArgs)
    workDir = (Resolve-Path -LiteralPath $WorkDir).Path
    log     = Join-Path $RunDir "$Name.log"
    status  = $status
    env     = $Env
}
$specFile = Join-Path $RunDir "$Name.spec.json"
$spec | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $specFile -Encoding UTF8

# Replace any earlier run's status before the worker starts, so a reused name never reports the
# previous run's state or exit code in the moment before the new worker writes its own.
[ordered]@{
    name = $Name; exe = $resolvedExe; args = @($ExeArgs); workDir = $spec.workDir; log = $spec.log
    state = 'starting'; pid = $null; start = (Get-Date).ToString('o')
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $status -Encoding UTF8

# Start-Process joins -ArgumentList with spaces and does not quote, so the two paths are quoted here.
$worker = Join-Path $PSScriptRoot 'LongRun-Worker.ps1'
$workerArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$worker`"", '-SpecFile', "`"$specFile`"")
$p = Start-Process -FilePath 'pwsh' -ArgumentList $workerArgs -WindowStyle Hidden -PassThru
Write-Host "started '$Name' (worker pid $($p.Id))"
Write-Host "status : $status"
Write-Host "log    : $($spec.log)"
Write-Host "check  : scripts/dev/Get-LongRun.ps1 $Name"
