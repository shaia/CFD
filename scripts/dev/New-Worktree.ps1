<#
.SYNOPSIS
Create a git worktree for this repo with the local tooling directory linked in, or retrofit an existing one.

.DESCRIPTION
.claude/ is gitignored, so a plain `git worktree add` leaves a new worktree without the local
tooling the main worktree has (instructions, skills, local settings). This script adds the worktree,
then creates a directory junction named .claude inside it that points at the same directory the main
worktree uses: the target of the main worktree's .claude junction, or that directory itself when it
is a real directory. Pass -HarnessDir to link something else.

Remove worktrees with `git worktree remove <path>`; never delete the directory by hand, because the
.claude junction points at a shared directory.

.EXAMPLE
.\scripts\dev\New-Worktree.ps1 -Branch feat/x
.\scripts\dev\New-Worktree.ps1 -Branch feat/x -Path ..\cfd-x -Base origin/master
.\scripts\dev\New-Worktree.ps1 -Existing ..\cfd-bcomp
.\scripts\dev\New-Worktree.ps1 -Existing ..\cfd-bcomp -WhatIf
#>
[CmdletBinding(DefaultParameterSetName = 'Create', SupportsShouldProcess)]
param(
    [Parameter(Mandatory, ParameterSetName = 'Create')][string]$Branch,
    [Parameter(ParameterSetName = 'Create')][string]$Path,
    [Parameter(ParameterSetName = 'Create')][string]$Base = 'origin/master',
    [Parameter(Mandatory, ParameterSetName = 'Retrofit')][string]$Existing,
    [string]$HarnessDir
)
$ErrorActionPreference = 'Stop'

function Resolve-HarnessDir([string]$RepoDir) {
    if ($HarnessDir) { return $HarnessDir }
    $first = git -C $RepoDir worktree list --porcelain | Select-Object -First 1
    $main = Join-Path $first.Substring(9) '.claude'
    if (-not (Test-Path -LiteralPath $main)) {
        throw "the main worktree has no .claude directory to link ($main); pass -HarnessDir"
    }
    $item = Get-Item -LiteralPath $main -Force
    if ($item.LinkType -and $item.Target) { return @($item.Target)[0] }
    return $item.FullName
}

function Add-HarnessJunction([string]$Worktree) {
    $link = Join-Path $Worktree '.claude'
    if (Test-Path -LiteralPath $link) {
        Write-Host "tooling : $link already exists"
        return
    }
    $target = Resolve-HarnessDir $Worktree
    if ($PSCmdlet.ShouldProcess($link, "create junction to $target")) {
        New-Item -ItemType Junction -Path $link -Target $target | Out-Null
        Write-Host "tooling : $link -> $target"
    }
}

if ($PSCmdlet.ParameterSetName -eq 'Retrofit') {
    $wt = (Resolve-Path -LiteralPath $Existing).Path
    if (-not (Test-Path -LiteralPath (Join-Path $wt '.git'))) { throw "$wt is not a git worktree" }
    Add-HarnessJunction $wt
    return
}

$repo = (git rev-parse --show-toplevel).Trim()
if (-not $Path) {
    $Path = Join-Path (Split-Path $repo -Parent) ('cfd-' + ($Branch -replace '[/\\]', '-'))
}
if (-not $PSCmdlet.ShouldProcess($Path, "git worktree add for branch $Branch")) { return }
git fetch origin --quiet
# An existing local branch is used as is; a branch that exists only on origin is tracked, not
# recreated from $Base; only a new name branches from $Base, and without an upstream: git would
# otherwise make $Base the upstream, so a pull would merge it and a plain push would be refused.
git show-ref --verify --quiet "refs/heads/$Branch"
if ($LASTEXITCODE -eq 0) {
    git worktree add $Path $Branch
} else {
    git show-ref --verify --quiet "refs/remotes/origin/$Branch"
    if ($LASTEXITCODE -eq 0) {
        git worktree add --track -b $Branch $Path "origin/$Branch"
    } else {
        git worktree add --no-track -b $Branch $Path $Base
    }
}
if ($LASTEXITCODE -ne 0) { throw 'git worktree add failed' }
$wt = (Resolve-Path -LiteralPath $Path).Path
Add-HarnessJunction $wt
Write-Host "worktree: $wt"
Write-Host "remove with: git worktree remove `"$wt`"  (never delete it by hand: .claude is a junction)"
