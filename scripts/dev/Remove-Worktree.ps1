<#
.SYNOPSIS
Remove a linked git worktree without deleting the shared tooling directory its .claude junction points at.

.DESCRIPTION
Worktrees made by New-Worktree.ps1 contain a .claude directory junction pointing at a directory every
worktree shares. Git for Windows follows that junction during `git worktree remove` and deletes the
shared directory's contents (this emptied the harness on 2026-10-03), so a bare `git worktree remove`
is as destructive as deleting the folder by hand.

This script removes the junction itself first (a non-recursive delete of the link), checks that the
shared directory still has its contents, and only then runs `git worktree remove` (from the main
worktree, so the script works from any directory). Before touching the link it refuses what git would
refuse afterwards -- local changes without -Force, a locked worktree -- so a refusal never leaves a
worktree without its tooling. It also refuses the main worktree, any worktree containing another
directory link at any depth, ignored or not, since git would follow that one too, and a .claude link
whose resolved target lies inside the worktree (git would delete it as ordinary contents) or contains
it. -Force is passed through to `git worktree remove` (uncommitted or untracked changes are
discarded).

Paths are compared by identity (GetFinalPathNameByHandle), so aliases of the worktree or of the
current directory are seen through, and the path given must be the worktree root itself. Before
unlinking, it inventories the shared directory recursively and refuses if any part cannot be read;
after unlinking it checks every path is still there, and stops before git if one is not.
Requires PowerShell 7.

.EXAMPLE
.\scripts\dev\Remove-Worktree.ps1 ..\cfd-x
.\scripts\dev\Remove-Worktree.ps1 ..\cfd-x -WhatIf
.\scripts\dev\Remove-Worktree.ps1 ..\cfd-x -Force
#>
#Requires -Version 7.0
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory, Position = 0)][string]$Path,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'

# Path identity, not spelling: the final path Windows reports for an open handle, with every
# junction, symlink and differently spelled alias -- in the leaf or any ancestor -- resolved.
if (-not ('CfdWorktreePath' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;
public static class CfdWorktreePath {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern SafeFileHandle CreateFileW(string name, uint access, uint share, IntPtr security,
                                             uint disposition, uint flags, IntPtr template);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern uint GetFinalPathNameByHandleW(SafeFileHandle handle, StringBuilder path, uint size,
                                                 uint flags);
    public static string Final(string path) {
        // No access rights needed for the query; BACKUP_SEMANTICS lets CreateFile open a directory.
        using (var h = CreateFileW(path, 0, 7, IntPtr.Zero, 3, 0x02000000, IntPtr.Zero)) {
            if (h.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error(), path);
            var sb = new StringBuilder(32768);
            uint n = GetFinalPathNameByHandleW(h, sb, (uint)sb.Capacity, 0);
            if (n == 0 || n >= sb.Capacity) throw new Win32Exception(Marshal.GetLastWin32Error(), path);
            string s = sb.ToString();
            if (s.StartsWith(@"\\?\UNC\")) return @"\\" + s.Substring(8);
            return s.StartsWith(@"\\?\") ? s.Substring(4) : s;
        }
    }
}
'@
}
$final = { param($p) [CfdWorktreePath]::Final($p) }

$wt = & $final (Resolve-Path -LiteralPath $Path).Path
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
# --show-toplevel also answers for a subdirectory, so $Path must BE that root: otherwise a
# subdirectory's own .claude link could be removed before git rejected the path.
$top = git -C $wt rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0 -or -not $top) { throw "git rev-parse failed for $wt; nothing was touched" }
$self = & $norm $top
if ((& $norm (& $final $top)) -ne (& $norm $wt)) {
    throw "refusing: $Path is not the root of a worktree (that is $top); nothing was touched"
}
if (-not $entries.ContainsKey($self)) {
    throw "git does not list $wt as a worktree of this repository; nothing was touched"
}

# Windows cannot delete a directory a process is standing in: git would empty and unregister the
# worktree, then fail on its top folder, after the link was gone. Refuse before touching anything.
# Both locations are compared by final path, so an alias for the worktree or an ancestor of it
# cannot hide that the shell is inside it.
$w = & $norm $wt
$places = @([Environment]::CurrentDirectory)
if ($PWD.Provider.Name -eq 'FileSystem') { $places += $PWD.ProviderPath }
foreach ($here in $places) {
    $h = & $norm (& $final $here)
    if ($h -eq $w -or $h.StartsWith("$w/")) {
        throw "refusing: the current directory ($here) is inside $wt; cd out of it first"
    }
}

# Every entry under a directory, recursively, including hidden and system ones. Links are returned
# but never descended into, so a walk never leaves the tree it was given. A folder that cannot be
# read throws: a skipped folder could hide a link, or a deletion. Explicit .NET enumeration, because
# Get-ChildItem -Recurse silently skips a folder it may not list.
function Get-Tree([string]$Dir) {
    $opts = [System.IO.EnumerationOptions]::new()
    $opts.IgnoreInaccessible = $false
    $opts.AttributesToSkip = [System.IO.FileAttributes]::None
    $out = [System.Collections.Generic.List[System.IO.FileSystemInfo]]::new()
    $pending = [System.Collections.Generic.Stack[string]]::new()
    $pending.Push($Dir)
    try {
        while ($pending.Count) {
            foreach ($e in [System.IO.DirectoryInfo]::new($pending.Pop()).EnumerateFileSystemInfos('*', $opts)) {
                $out.Add($e)
                $a = $e.Attributes
                if (($a -band [System.IO.FileAttributes]::Directory) -and
                    -not ($a -band [System.IO.FileAttributes]::ReparsePoint)) {
                    $pending.Push($e.FullName)
                }
            }
        }
    }
    catch { throw "could not list all of $Dir ($(($_.Exception.InnerException ?? $_.Exception).Message))" }
    return , $out
}

# git's recursive delete follows a directory link at ANY depth, ignored ones included, so every
# link but the root .claude is refused, wherever it is.
try { $tree = Get-Tree $wt }
catch { throw "refusing: $($_.Exception.Message), so it cannot be checked for links; nothing was touched" }
$rootLink = & $norm (Join-Path $wt '.claude')
$nested = @($tree | Where-Object {
        ($_.Attributes -band [System.IO.FileAttributes]::Directory) -and
        ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -and
        (& $norm $_.FullName) -ne $rootLink })
if ($nested) {
    $rel = $nested | ForEach-Object { $_.FullName.Substring($wt.Length).TrimStart('\', '/') }
    throw (("refusing: $wt contains links besides .claude ({0}); git would delete what they point at. " +
            "Remove each link first with [System.IO.Directory]::Delete('<full path>'), which deletes " +
            "only the link and takes the path literally") -f ($rel -join ', '))
}

# Refuse here what git worktree remove would refuse after the link is gone, so a refusal never
# leaves a worktree without its tooling: local changes (unless -Force) and a lock (always; unlock
# it with git worktree unlock first).
# Explicit flags, so status.showUntrackedFiles=no or a submodule setting cannot hide a change.
$dirty = @(git -C $wt status --porcelain --untracked-files=normal --ignore-submodules=none)
if ($LASTEXITCODE -ne 0) { throw "git status failed in $wt" }
if ($dirty.Count -and -not $Force) {
    throw "refusing: $wt has $($dirty.Count) uncommitted or untracked change(s); commit them, or pass -Force to discard them"
}
if (@($entries[$self] | Where-Object { $_ -like 'locked*' }).Count) {
    throw "refusing: $wt is locked; run git worktree unlock `"$wt`" first if it should go"
}

$link = Join-Path $wt '.claude'
$item = Get-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
$isLink = [bool]($item -and $item.LinkType)
$gitArgs = @('-C', $main, 'worktree', 'remove')
if ($Force) { $gitArgs += '--force' }
$gitArgs += $wt

# Where the link really points, resolved through the link itself, so a relative or differently
# spelled recorded target cannot hide it. Only "not found" means the target is gone; any other
# failure refuses, since an unresolved target could not be protected. A target at or beneath the
# worktree root survives the unlink but not git, which deletes it as ordinary worktree contents
# (New-Worktree.ps1 -HarnessDir accepts such a path, and an ignored one passes the clean check);
# a worktree inside its own target would take part of the target with it.
$target = $null
if ($isLink) {
    $recorded = @($item.Target)[0]
    try { $target = & $final $link }
    catch {
        $e = $_.Exception
        while ($e.InnerException) { $e = $e.InnerException }
        if (-not ($e -is [System.ComponentModel.Win32Exception] -and $e.NativeErrorCode -in 2, 3)) {
            throw "refusing: cannot resolve where $link points ($($e.Message)); nothing was touched"
        }
    }
    if ($target) {
        $t = & $norm $target
        if ($t -eq $w -or $t.StartsWith("$w/")) {
            throw ("refusing: $link points inside the worktree being removed ($target), so git would " +
                   "delete it as ordinary worktree contents; move that directory outside $wt and " +
                   "re-link first. Nothing was touched")
        }
        if ($w.StartsWith("$t/")) {
            throw ("refusing: $wt lies inside the directory its link points at ($target), so removing " +
                   "the worktree would delete part of that directory. Nothing was touched")
        }
    }
}

# One confirmation for the whole operation, before either step: confirming the removal but not the
# unlink would run git through the junction, and the reverse would strand the worktree untooled.
$plan = "git $($gitArgs -join ' ')"
if ($isLink) {
    $kept = if ($target) { "$target is kept" } else { "its target $recorded no longer exists" }
    $plan = "remove the $($item.LinkType) $link (link only; $kept), then $plan"
}
if (-not $PSCmdlet.ShouldProcess($wt, $plan)) { return }

# Every relative path under the shared directory. A count could hide a nested deletion or a swapped
# entry, and Get-Tree throws on any folder it cannot read, so two snapshots are always complete.
function Get-Inventory([string]$Dir) {
    $set = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($e in (Get-Tree $Dir)) { [void]$set.Add($e.FullName.Substring($Dir.Length).TrimStart('\', '/')) }
    return , $set
}

if ($isLink) {
    # A link whose target is gone protects nothing and endangers nothing: just remove it.
    $live = [bool]$target
    if ($live) {
        try { $before = Get-Inventory $target }
        catch { throw "refusing: $($_.Exception.Message); nothing was touched" }
    }
    # Non-recursive RemoveDirectory on the link itself: deletes the reparse point, never the target.
    # Not cmd /c rmdir, which expands %VAR% even inside quotes and could hit another path.
    [System.IO.Directory]::Delete($link, $false)
    if (Test-Path -LiteralPath $link) { throw "could not remove the link $link; nothing else was touched" }
    # From here the link is gone, so git can no longer reach the shared directory; a rerun of this
    # script goes straight to git worktree remove.
    $rerun = "Once the shared directory is confirmed intact, run this script again to finish."
    if ($live) {
        try { $after = Get-Inventory $target }
        catch { throw "unlinked $link but could not re-check the shared directory ($($_.Exception.Message)); stopped before git. $rerun" }
        $missing = @($before | Where-Object { -not $after.Contains($_) })
        if ($missing) {
            throw (("{0} path(s) under the shared directory $target disappeared between the listings " +
                    "taken before and after unlinking ({1}). Another session editing it can do that; so " +
                    "could a damaging unlink. Stopped before git. $rerun") -f
                   $missing.Count, (($missing | Select-Object -First 5) -join ', '))
        }
        Write-Host "unlinked: $link (target $target untouched, all $($before.Count) paths present)"
    }
    else {
        Write-Host "unlinked: $link (its target $recorded no longer exists)"
    }
}

git @gitArgs
if ($LASTEXITCODE -ne 0) {
    throw ("git worktree remove failed after the .claude link was removed. If git worktree list " +
           "still shows $wt, restore the link with New-Worktree.ps1 -Existing `"$wt`"; if not, " +
           "only a leftover folder remains and it holds no link, so it can be deleted.")
}
Write-Host "removed: $wt"
