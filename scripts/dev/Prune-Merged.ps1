<#
.SYNOPSIS
List (and with -Delete remove) local branches whose pull request has been merged.

.DESCRIPTION
PRs are squash-merged, so `git branch --merged` cannot see them. This script asks GitHub for merged PR
head branches and their final commit, and treats a local branch as prunable only when its tip is exactly
the commit the PR merged (a branch with commits after the PR is kept and reported). Branches checked
out in a worktree are reported, not deleted; remove the worktree first.

.EXAMPLE
.\scripts\dev\Prune-Merged.ps1              # report only
.\scripts\dev\Prune-Merged.ps1 -Delete      # delete, asking per branch
.\scripts\dev\Prune-Merged.ps1 -Delete -Confirm:$false
#>
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [switch]$Delete,
    [int]$Limit = 500
)
$ErrorActionPreference = 'Stop'
$merged = gh pr list --state merged --limit $Limit --json headRefName,headRefOid,number | ConvertFrom-Json
$byHead = @{}
foreach ($pr in $merged) { $byHead[$pr.headRefName] = $pr }

$current = (git branch --show-current).Trim()
$inWorktree = @{}
$wtPath = $null
foreach ($line in (git worktree list --porcelain)) {
    if ($line -like 'worktree *') { $wtPath = $line.Substring(9) }
    elseif ($line -like 'branch refs/heads/*') { $inWorktree[$line.Substring(18)] = $wtPath }
}

$prunable = @(); $ahead = @(); $checkedOut = @()
foreach ($b in (git for-each-ref --format='%(refname:short) %(objectname)' refs/heads)) {
    $name, $sha = $b -split ' ', 2
    if ($name -in @('master', 'main', $current)) { continue }
    $pr = $byHead[$name]
    if (-not $pr) { continue }
    if ($inWorktree.ContainsKey($name)) { $checkedOut += "$name  (worktree $($inWorktree[$name]), PR #$($pr.number))"; continue }
    if ($sha -eq $pr.headRefOid) { $prunable += @{ name = $name; pr = $pr.number } }
    else { $ahead += "$name  (PR #$($pr.number) merged $($pr.headRefOid.Substring(0,7)), local tip $($sha.Substring(0,7)))" }
}

Write-Host "prunable (tip == merged PR head): $($prunable.Count)"
$prunable | ForEach-Object { Write-Host ("  {0}  (PR #{1})" -f $_.name, $_.pr) }
if ($ahead.Count -gt 0) { Write-Host ""; Write-Host "kept, commits after the merged PR: $($ahead.Count)"; $ahead | ForEach-Object { Write-Host "  $_" } }
if ($checkedOut.Count -gt 0) { Write-Host ""; Write-Host "kept, checked out in a worktree: $($checkedOut.Count)"; $checkedOut | ForEach-Object { Write-Host "  $_" } }

if (-not $Delete) { Write-Host ""; Write-Host "re-run with -Delete to remove the prunable branches"; return }
foreach ($b in $prunable) {
    if ($PSCmdlet.ShouldProcess($b.name, "git branch -D (PR #$($b.pr) merged)")) {
        git branch -D $b.name
    }
}
