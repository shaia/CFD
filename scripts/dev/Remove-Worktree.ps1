<#
.SYNOPSIS
Remove a linked git worktree without deleting the shared tooling directory its .claude junction points at.

.DESCRIPTION
Worktrees made by New-Worktree.ps1 contain a .claude directory junction pointing at a directory every
worktree shares. Git for Windows follows that junction during `git worktree remove` and deletes the
shared directory's contents (this emptied the harness on 2026-10-03), so a bare `git worktree remove`
is as destructive as deleting the folder by hand.

This script removes the junction itself first (rmdir without /s deletes only the link), checks that the
shared directory still has its contents, and only then runs `git worktree remove`. It refuses the main
worktree, and any worktree with another link at its top level that it would not know how to treat.
-Force is passed through to `git worktree remove` (uncommitted or untracked changes are discarded).

.EXAMPLE
.\scripts\dev\Remove-Worktree.ps1 ..\cfd-x
.\scripts\dev\Remove-Worktree.ps1 ..\cfd-x -WhatIf
.\scripts\dev\Remove-Worktree.ps1 ..\cfd-x -Force
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory, Position = 0)][string]$Path,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'

$wt = (Resolve-Path -LiteralPath $Path).Path
$gitDir = (git -C $wt rev-parse --absolute-git-dir 2>$null)
$common = (git -C $wt rev-parse --path-format=absolute --git-common-dir 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $gitDir) { throw "$wt is not a git worktree" }
if ((Resolve-Path -LiteralPath $gitDir).Path -eq (Resolve-Path -LiteralPath $common).Path) {
    throw "$wt is the main worktree; refusing to remove it"
}

# Any top-level link other than .claude is unexpected: refuse rather than guess what it points at.
$links = @(Get-ChildItem -LiteralPath $wt -Force | Where-Object { $_.LinkType })
$others = @($links | Where-Object { $_.Name -ne '.claude' })
if ($others) {
    throw ("refusing: $wt has other links at its top level ({0}); remove them by hand first with " +
           "cmd /c rmdir, which deletes only the link" -f ($others.Name -join ', '))
}

$link = Join-Path $wt '.claude'
$item = Get-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
if ($item -and $item.LinkType) {
    $target = @($item.Target)[0]
    $before = @(Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue).Count
    if ($PSCmdlet.ShouldProcess($link, "remove the $($item.LinkType) (link only; $target is kept)")) {
        cmd /c rmdir "$link"
        if (Test-Path -LiteralPath $link) { throw "could not remove the link $link; nothing else was touched" }
        $after = @(Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue).Count
        if ($after -lt $before) {
            throw "the shared directory $target lost entries ($before -> $after) while unlinking; stopping"
        }
        Write-Host "unlinked: $link (target $target untouched, $after entries)"
    }
}

$gitArgs = @('worktree', 'remove')
if ($Force) { $gitArgs += '--force' }
$gitArgs += $wt
if ($PSCmdlet.ShouldProcess($wt, "git $($gitArgs -join ' ')")) {
    git @gitArgs
    if ($LASTEXITCODE -ne 0) { throw "git worktree remove failed (the .claude link is already gone; re-run with New-Worktree.ps1 -Existing to restore it)" }
    Write-Host "removed: $wt"
}
