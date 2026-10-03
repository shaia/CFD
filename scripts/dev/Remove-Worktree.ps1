<#
.SYNOPSIS
Remove a linked git worktree without deleting the shared tooling directory its .claude junction points at.

.DESCRIPTION
Worktrees made by New-Worktree.ps1 contain a .claude directory junction pointing at a directory every
worktree shares. Git for Windows follows that junction during `git worktree remove` and deletes the
shared directory's contents (this emptied the harness on 2026-10-03), so a bare `git worktree remove`
is as destructive as deleting the folder by hand.

This script removes the junction itself first (rmdir without /s deletes only the link), checks that the
shared directory still has its contents, and only then runs `git worktree remove` (from the main
worktree, so the script works from any directory). Before touching the link it refuses what git would
refuse afterwards -- local changes without -Force, a locked worktree -- so a refusal never leaves a
worktree without its tooling. It also refuses the main worktree, and any worktree with another link
at its top level that it would not know how to treat. -Force is passed through to
`git worktree remove` (uncommitted or untracked changes are discarded).

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
$norm = { param($p) ($p -replace '\\', '/').TrimEnd('/').ToLowerInvariant() }

# git runs from the main worktree, so the script works from any directory outside the worktree. Its
# path comes from git itself -- the first `worktree list` entry, the main worktree or a bare
# repository -- not from the metadata layout, which --separate-git-dir and bare repositories change.
# The listing must succeed and must contain $wt before anything is touched.
$listing = @(git -C $wt worktree list --porcelain)
if ($LASTEXITCODE -ne 0 -or -not $listing -or $listing[0] -notlike 'worktree *') {
    throw "git worktree list failed for $wt; nothing was touched"
}
$main = $listing[0].Substring(9)
$entries = @{}   # normalised worktree path -> its porcelain lines
$block = $null
foreach ($line in $listing) {
    if ($line -like 'worktree *') { $block = & $norm $line.Substring(9); $entries[$block] = @() }
    elseif ($block) { $entries[$block] += $line }
}
# git lists resolved paths, so look the worktree up by git's own name for it, not the typed path.
$self = & $norm (git -C $wt rev-parse --show-toplevel)
if ($LASTEXITCODE -ne 0 -or -not $entries.ContainsKey($self)) {
    throw "git does not list $wt as a worktree of this repository; nothing was touched"
}

# Windows cannot delete a directory a process is standing in: git would empty and unregister the
# worktree, then fail on its top folder, after the link was gone. Refuse before touching anything.
foreach ($here in @($PWD.ProviderPath, [Environment]::CurrentDirectory)) {
    $h = & $norm $here
    $w = & $norm $wt
    if ($h -eq $w -or $h.StartsWith("$w/")) {
        throw "refusing: the current directory ($here) is inside $wt; cd out of it first"
    }
}

# Any top-level link other than .claude is unexpected: refuse rather than guess what it points at.
$links = @(Get-ChildItem -LiteralPath $wt -Force | Where-Object { $_.LinkType })
$others = @($links | Where-Object { $_.Name -ne '.claude' })
if ($others) {
    throw (("refusing: $wt has other links at its top level ({0}); remove them by hand first with " +
            "cmd /c rmdir, which deletes only the link") -f ($others.Name -join ', '))
}

# Refuse here what git worktree remove would refuse after the link is gone, so a refusal never
# leaves a worktree without its tooling: local changes (unless -Force) and a lock (always; unlock
# it with git worktree unlock first).
$dirty = @(git -C $wt status --porcelain)
if ($LASTEXITCODE -ne 0) { throw "git status failed in $wt" }
if ($dirty.Count -and -not $Force) {
    throw "refusing: $wt has $($dirty.Count) uncommitted or untracked change(s); commit them, or pass -Force to discard them"
}
if (@($entries[$self] | Where-Object { $_ -like 'locked*' }).Count) {
    throw "refusing: $wt is locked; run git worktree unlock `"$wt`" first if it should go"
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

$gitArgs = @('-C', $main, 'worktree', 'remove')
if ($Force) { $gitArgs += '--force' }
$gitArgs += $wt
if ($PSCmdlet.ShouldProcess($wt, "git $($gitArgs -join ' ')")) {
    git @gitArgs
    if ($LASTEXITCODE -ne 0) {
        throw ("git worktree remove failed after the .claude link was removed. If git worktree list " +
               "still shows $wt, restore the link with New-Worktree.ps1 -Existing `"$wt`"; if not, " +
               "only a leftover folder remains and it holds no link, so it can be deleted.")
    }
    Write-Host "removed: $wt"
}
