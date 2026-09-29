<#
.SYNOPSIS
Print a short, read-only brief of the workspace: branch state, worktrees, detached runs, stale builds.

.DESCRIPTION
Run it when you want to know where things stand: when asked for status, and before starting a
branch, a long run or a pull request. Without -Remote it only runs git queries and reads files; it
changes nothing. -Remote fetches origin first and lists open pull requests.

.EXAMPLE
.\scripts\dev\Get-SessionBrief.ps1
.\scripts\dev\Get-SessionBrief.ps1 -Remote
.\scripts\dev\Get-SessionBrief.ps1 -Path ..\cfd-validation
#>
param(
    [string]$Path = (Get-Location).Path,
    [switch]$Remote
)
$ErrorActionPreference = 'Continue'

$top = (git -C $Path rev-parse --show-toplevel 2>$null)
if (-not $top) { Write-Output "not a git checkout: $Path"; return }
$top = $top.Trim()
$topResolved = (Resolve-Path -LiteralPath $top).Path
$lines = [System.Collections.Generic.List[string]]::new()
$ok = [System.Collections.Generic.List[string]]::new()

if ($Remote) {
    git -C $top fetch origin --quiet 2>$null
    if ($LASTEXITCODE -ne 0) { $lines.Add('fetch failed; counts below are from the last successful fetch') }
}

# --- this checkout ---
$branch = (git -C $top branch --show-current)
$onBranch = [bool]$branch
$branch = if ($onBranch) { $branch.Trim() } else { 'detached at ' + (git -C $top rev-parse --short HEAD).Trim() }
$dirty = @(git -C $top status --short).Count
$up = (git -C $top rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>$null)
if ($up) {
    $ab = ((git -C $top rev-list --left-right --count "HEAD...$($up.Trim())") -split '\s+')
    $track = "ahead $($ab[0]) behind $($ab[1]) of $($up.Trim())"
} else {
    $track = 'no upstream'
}
$lines.Insert(0, "checkout $top  branch $branch  dirty files $dirty  $track")

$behind = (git -C $top rev-list --count master..origin/master 2>$null)
if ($behind -and $behind.Trim() -ne '0') {
    $asOf = if ($Remote) { '' } else { ' (as of the last fetch)' }
    $lines.Add("local master is $($behind.Trim()) commits behind origin/master$asOf")
} elseif ($behind) {
    $ok.Add('master up to date')
}

# --- other worktrees ---
$total = 0; $clean = @(); $wt = $null
foreach ($l in (git -C $top worktree list --porcelain)) {
    if ($l -like 'worktree *') { $wt = $l.Substring(9); continue }
    $br = $null
    if ($l -like 'branch refs/heads/*') { $br = $l.Substring(18) }
    elseif ($l -eq 'detached') { $br = '(detached)' }
    if (-not ($wt -and $br)) { continue }
    $total++
    $name = Split-Path $wt -Leaf
    if (-not (Test-Path -LiteralPath $wt)) {
        $lines.Add("worktree $name ($br): missing on disk; run git worktree prune")
    } elseif ((Resolve-Path -LiteralPath $wt).Path -ne $topResolved) {
        $flags = @()
        $n = @(git -C $wt status --short 2>$null).Count
        if ($n -gt 0) { $flags += "$n dirty" }
        if (-not (Test-Path -LiteralPath (Join-Path $wt '.claude'))) {
            $flags += "no .claude junction (scripts/dev/New-Worktree.ps1 -Existing `"$wt`")"
        }
        if ($flags.Count -gt 0) { $lines.Add("worktree $name ($br): " + ($flags -join '; ')) } else { $clean += $name }
    }
    $wt = $null
}
if ($total -gt 1) {
    $cleanTxt = if ($clean.Count -gt 0) { '; clean: ' + ($clean -join ', ') } else { '' }
    $lines.Add("worktrees: $total in total$cleanTxt")
}

# --- detached runs (Start-LongRun.ps1) ---
$runDir = Join-Path $top 'output\runs'
if (Test-Path -LiteralPath $runDir) {
    $runs = @(& (Join-Path $PSScriptRoot 'Get-LongRun.ps1') -RunDir $runDir 6>&1 | ForEach-Object { "$_" } |
        Where-Object { $_ -and $_ -notlike 'no runs*' })
    foreach ($r in ($runs | Select-Object -First 5)) { $lines.Add('run ' + ($r -replace '\s+', ' ')) }
    if ($runs.Count -gt 5) { $lines.Add("... and $($runs.Count - 5) older runs (scripts/dev/Get-LongRun.ps1)") }
}

# --- stale build markers (written by the build_state hook) ---
$stale = 0
foreach ($b in (Get-ChildItem -LiteralPath $top -Directory -Filter 'build*' -ErrorAction SilentlyContinue)) {
    $f = Join-Path $b.FullName '.claude-build-failed'
    if (Test-Path -LiteralPath $f) {
        $stale++
        $first = Get-Content -LiteralPath $f -TotalCount 1
        $lines.Add("stale build $($b.Name): $first; rebuild before running its tests")
    }
}
if ($stale -eq 0) { $ok.Add('no stale builds') }

# --- /pr-ready marker for this branch (same rule as the shell guard: HEAD match, 24 h) ---
$hasSkill = Test-Path -LiteralPath (Join-Path $top '.claude\skills\pr-ready\SKILL.md')
if ($hasSkill -and $onBranch -and $branch -notin @('master', 'main')) {
    $mk = Join-Path $env:TEMP ('claude-hooks\pr-ready\' + ($branch -replace '/', '__') + '.json')
    $state = 'missing (run /pr-ready before gh pr create)'
    if (Test-Path -LiteralPath $mk) {
        try {
            $j = Get-Content -LiteralPath $mk -Raw | ConvertFrom-Json
            $head = (git -C $top rev-parse HEAD).Trim()
            $age = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [double]$j.time
            if ($j.head -ne $head) { $state = "stale (written for $($j.head.Substring(0, 8)), HEAD is $($head.Substring(0, 8)))" }
            elseif ($age -gt 86400) { $state = 'stale (older than 24 h)' }
            else { $state = 'valid' }
        } catch {
            $state = 'unreadable'
        }
    }
    $lines.Add("/pr-ready marker: $state")
}

# --- stashes, LSP database ---
$stashes = @(git -C $top stash list).Count
if ($stashes -gt 0) { $lines.Add("stashes: $stashes (git stash list)") }
if (Test-Path -LiteralPath (Join-Path $top '.clangd')) {
    if (Test-Path -LiteralPath (Join-Path $top 'build-ninja\compile_commands.json')) {
        $ok.Add('LSP database present')
    } else {
        $lines.Add('LSP: build-ninja/compile_commands.json is missing; run cmake --preset windows-ninja from a VS developer shell')
    }
}

# --- open pull requests (network; only with -Remote) ---
if ($Remote) {
    Push-Location $top
    try {
        $json = gh pr list --state open --json number,headRefName,isDraft,reviews,title 2>$null
        if ($LASTEXITCODE -eq 0 -and $json) {
            $prs = @($json | ConvertFrom-Json)
            if ($prs.Count -eq 0) { $ok.Add('no open PRs') }
            foreach ($pr in $prs) {
                $title = if ($pr.title.Length -gt 60) { $pr.title.Substring(0, 57) + '...' } else { $pr.title }
                $draft = if ($pr.isDraft) { ' [draft]' } else { '' }
                $lines.Add("open PR #$($pr.number) $($pr.headRefName)${draft}: $(@($pr.reviews).Count) review(s); $title")
            }
        } else {
            $lines.Add('open PRs: gh call failed (offline or not authenticated)')
        }
    } finally {
        Pop-Location
    }
}

if ($ok.Count -gt 0) { $lines.Add('ok: ' + ($ok -join ', ')) }
Write-Output '[workspace brief]'
$shown = @($lines | Select-Object -First 24)
$shown | ForEach-Object { Write-Output "- $_" }
if ($lines.Count -gt $shown.Count) { Write-Output "- ... $($lines.Count - $shown.Count) more lines not shown" }
