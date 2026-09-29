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

The default run directory is output\runs under the repository of the current directory, the same
place Get-LongRun.ps1 looks, whatever -WorkDir is.

.PARAMETER Force
Launch although the status file says an earlier launch under this name is still starting. It never
replaces a run that is alive: stop that run first, or pick another name.

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
. (Join-Path $PSScriptRoot 'LongRun-Common.ps1')
if ($Name -notmatch '^[A-Za-z0-9._-]+$') { throw "Name must be [A-Za-z0-9._-]+ (got '$Name')" }

$explicitRunDir = [bool]$RunDir
$RunDir = Get-LongRunDir $RunDir
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
# Absolute from here on: the worker changes to -WorkDir before it writes, so a relative run
# directory would give the launcher and the worker two different status files.
$RunDir = (Resolve-Path -LiteralPath $RunDir).Path

$status = Join-Path $RunDir "$Name.status.json"
if (Test-Path -LiteralPath $status) {
    $s = Get-Content -LiteralPath $status -Raw | ConvertFrom-Json
    if ($s.state -eq 'running' -and (Test-LongRunAlive $s)) {
        throw "'$Name' is still running (pid $($s.pid)). Stop it first (Stop-Process -Id $($s.pid)) or pick another -Name; -Force does not replace a live run"
    }
    $age = ((Get-Date) - (Get-Item -LiteralPath $status).LastWriteTime).TotalSeconds
    if ($s.state -eq 'starting' -and $age -lt 60 -and -not $Force) {
        throw "'$Name' was launched $([int]$age)s ago and is still starting; wait, pick another -Name, or pass -Force if that launch failed"
    }
}

$resolvedExe = if (Test-Path -LiteralPath $Exe) { (Resolve-Path -LiteralPath $Exe).Path } else { $Exe }
$spec = [ordered]@{
    name    = $Name
    runId   = [guid]::NewGuid().ToString()
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
    name = $Name; runId = $spec.runId; exe = $resolvedExe; args = @($ExeArgs); workDir = $spec.workDir
    log = $spec.log; state = 'starting'; pid = $null; start = (Get-Date).ToString('o')
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $status -Encoding UTF8

# Start-Process joins -ArgumentList with spaces and does not quote, so the two paths are quoted here.
$worker = Join-Path $PSScriptRoot 'LongRun-Worker.ps1'
$workerArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$worker`"", '-SpecFile', "`"$specFile`"")
$p = Start-Process -FilePath 'pwsh' -ArgumentList $workerArgs -WindowStyle Hidden -PassThru
$check = "scripts/dev/Get-LongRun.ps1 $Name" + $(if ($explicitRunDir) { " -RunDir `"$RunDir`"" } else { '' })
Write-Host "started '$Name' (worker pid $($p.Id))"
Write-Host "status : $status"
Write-Host "log    : $($spec.log)"
Write-Host "check  : $check"
