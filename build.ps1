#requires -Version 5.1
<#
.SYNOPSIS
    Windows build wrapper for the CFD Framework that runs CMake/ctest under a pruned PATH.

.DESCRIPTION
    On Windows, an overly long inherited PATH (this machine's grew to ~7140 chars / 164 entries)
    overflows cmd.exe's ~8191-char command-line limit once MSVC env vars are appended. When CMake
    builds the CUDA backend, nvcc runs with --use-local-env and spawns a `cmd /c` subprocess to
    materialize the host-compiler environment; that subprocess hits the limit and dies with a
    swallowed "exit 1" and no diagnostic.

    This wrapper deduplicates the PATH (case-insensitive) and drops non-existent directories for the
    lifetime of the process only -- it never writes the persistent environment -- which keeps every
    real tool (CUDA, MSVC, cmake, ninja, vcpkg) on PATH while bringing the length well under the
    limit. To also fix non-wrapper IDE/terminal builds, prune the persistent user PATH once (see
    README "Windows CUDA builds").

.EXAMPLE
    .\build.ps1 all                 # configure + build + fast test subset (CUDA preset)
.EXAMPLE
    .\build.ps1 configure -Preset windows-ninja-cuda
.EXAMPLE
    .\build.ps1 build -Config Release
.EXAMPLE
    .\build.ps1 test -All           # include long-running validation/cross-arch tests
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('configure', 'build', 'test', 'all', 'rebuild', 'clean', 'help')]
    [string]$Command = 'help',

    # CMake configure preset from CMakePresets.json (default: Windows MSVC + CUDA).
    [string]$Preset = 'windows-msvc-cuda',

    # Build configuration for multi-config generators (Visual Studio).
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Config = 'Debug',

    # Binary dir; defaults from the preset name (build-ninja for *ninja* presets, else build).
    [string]$BuildDir = '',

    # Run the full test suite including long-running validation/cross-arch tests.
    [switch]$All,

    # Remaining args are forwarded verbatim to the underlying cmake/ctest invocation.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Passthrough
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Info    { param([string]$m) Write-Host "[INFO] $m"    -ForegroundColor Blue }
function Write-Good    { param([string]$m) Write-Host "[OK] $m"      -ForegroundColor Green }
function Write-Warn    { param([string]$m) Write-Host "[WARN] $m"    -ForegroundColor Yellow }
function Write-Err     { param([string]$m) Write-Host "[ERROR] $m"   -ForegroundColor Red }

# Deduplicate (case-insensitive, keep first) and drop non-existent dirs from a PATH string.
# Inaccessible directories (ACL denied) are kept -- they exist, we just can't stat them.
function Get-PrunedPath {
    param([string]$Raw)
    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $out  = [System.Collections.Generic.List[string]]::new()
    foreach ($e in ($Raw -split ';')) {
        $entry = $e.Trim()
        if ($entry -eq '') { continue }
        $key = $entry.TrimEnd('\')
        if (-not $seen.Add($key)) { continue }              # duplicate
        $exists = $true
        try { $exists = Test-Path -LiteralPath $entry -PathType Container -ErrorAction Stop }
        catch { $exists = $true }                           # inaccessible -> keep, never risk dropping a real dir
        if (-not $exists) { continue }                      # dead entry
        $out.Add($entry)
    }
    return ($out -join ';')
}

# Prune the process PATH for the lifetime of this script only. Never touches the persistent env.
function Set-PrunedProcessPath {
    $before = $env:PATH
    $after  = Get-PrunedPath $before
    $bc = ($before -split ';').Count
    $ac = ($after  -split ';').Count
    Write-Info "PATH pruned: $($before.Length) -> $($after.Length) chars, $bc -> $ac entries"
    $env:PATH = $after
    if ($after.Length -gt 7000) {
        Write-Warn "Pruned PATH is still $($after.Length) chars; nvcc --use-local-env may still overflow cmd's ~8191 limit."
        Write-Warn "Consider pruning your persistent user PATH (see README: 'Windows CUDA builds')."
    }
}

function Resolve-BuildDir {
    if ($BuildDir) { return $BuildDir }
    if ($Preset -like '*ninja*') { return Join-Path $ScriptDir 'build-ninja' }
    return Join-Path $ScriptDir 'build'
}

function Invoke-Native {
    param([string]$Exe, [string[]]$CmdArgs)
    Write-Info "$Exe $($CmdArgs -join ' ')"
    & $Exe @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Err "$Exe exited with code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

function Get-Jobs {
    if ($env:NUMBER_OF_PROCESSORS) { return $env:NUMBER_OF_PROCESSORS } else { return 4 }
}

function Do-Configure {
    Invoke-Native 'cmake' (@('--preset', $Preset) + $Passthrough)
}

function Do-Build {
    $dir = Resolve-BuildDir
    Invoke-Native 'cmake' (@('--build', $dir, '--config', $Config, '-j', (Get-Jobs)) + $Passthrough)
}

function Do-Test {
    $dir = Resolve-BuildDir
    # Bound per-test OpenMP threads to avoid oversubscription under parallel ctest (CLAUDE.md).
    if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = '2' }
    $ctestArgs = @('--test-dir', $dir, '-C', $Config, '--output-on-failure', '-j', (Get-Jobs))
    if (-not $All) {
        # Fast pre-commit subset: exclude long-running validation/cross-arch tests.
        $ctestArgs += @('-LE', 'cross-arch|validation')
    } else {
        Write-Info 'Including long-running validation and cross-arch tests'
    }
    Invoke-Native 'ctest' ($ctestArgs + $Passthrough)
}

function Do-Clean {
    $dir = Resolve-BuildDir
    if (Test-Path -LiteralPath $dir) {
        Write-Info "Removing $dir"
        Remove-Item -LiteralPath $dir -Recurse -Force
        Write-Good 'Build directory cleaned'
    } else {
        Write-Warn "Build directory does not exist: $dir"
    }
}

function Show-Help {
    @"
CFD Framework - Windows build wrapper (runs CMake/ctest under a pruned PATH)

Usage: .\build.ps1 <command> [-Preset <name>] [-Config <cfg>] [-BuildDir <dir>] [-All] [-- <extra args>]

Commands:
  configure   cmake --preset <Preset>
  build       cmake --build <BuildDir> --config <Config> -j N
  test        ctest in <BuildDir> (fast subset; -All includes validation/cross-arch)
  all         configure + build + test
  rebuild     clean + configure + build
  clean       remove the build directory
  help        show this message

Options:
  -Preset    Configure preset (default: windows-msvc-cuda). Others: windows-msvc,
             windows-ninja, windows-ninja-cuda  (see CMakePresets.json)
  -Config    Debug | Release | RelWithDebInfo | MinSizeRel  (default: Debug)
  -BuildDir  Override binary dir (default: build, or build-ninja for *ninja* presets)
  -All       Include long-running validation/cross-arch tests in 'test'

Examples:
  .\build.ps1 all
  .\build.ps1 configure -Preset windows-ninja-cuda
  .\build.ps1 build -Config Release
  .\build.ps1 test -All
"@ | Write-Host
}

if ($Command -eq 'help') { Show-Help; return }

Set-PrunedProcessPath

switch ($Command) {
    'configure' { Do-Configure }
    'build'     { Do-Build }
    'test'      { Do-Test }
    'all'       { Do-Configure; Do-Build; Do-Test }
    'rebuild'   { Do-Clean; Do-Configure; Do-Build }
    'clean'     { Do-Clean }
}

Write-Good "Done: $Command"
