<#
.SYNOPSIS
Launch a long-running executable detached from the current session and track it with a status file.

.DESCRIPTION
Anything expected to exceed the Claude Code 10-minute tool timeout (full cavity validation, Richardson
grids above 129x129, channel DNS) goes through this script. It starts LongRun-Worker.ps1 as an
independent pwsh process (Start-Process, so it survives the session that launched it), which runs the
executable, appends its output to <RunDir>\<Name>.log and keeps <RunDir>\<Name>.status.json up to
date (state, pid, start, exit, end). Check progress with Get-LongRun.ps1.

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
if ((Test-Path $status) -and -not $Force) {
    $s = Get-Content $status -Raw | ConvertFrom-Json
    if ($s.state -eq 'running' -and $s.pid -and (Get-Process -Id $s.pid -ErrorAction SilentlyContinue)) {
        throw "'$Name' is still running (pid $($s.pid)); pick another -Name or pass -Force"
    }
}

$resolvedExe = if (Test-Path $Exe) { (Resolve-Path $Exe).Path } else { $Exe }
$spec = [ordered]@{
    name    = $Name
    exe     = $resolvedExe
    args    = @($ExeArgs)
    workDir = (Resolve-Path $WorkDir).Path
    log     = Join-Path $RunDir "$Name.log"
    status  = $status
    env     = $Env
}
$specFile = Join-Path $RunDir "$Name.spec.json"
$spec | ConvertTo-Json -Depth 4 | Set-Content -Path $specFile -Encoding UTF8

$worker = Join-Path $PSScriptRoot 'LongRun-Worker.ps1'
$p = Start-Process -FilePath 'pwsh' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $worker, '-SpecFile', $specFile) -WindowStyle Hidden -PassThru
Write-Host "started '$Name' (worker pid $($p.Id))"
Write-Host "status : $status"
Write-Host "log    : $($spec.log)"
Write-Host "check  : scripts/dev/Get-LongRun.ps1 $Name"
