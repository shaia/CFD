# Developer scripts

PowerShell 7 helpers for working on this repository on Windows. None of them build the library;
use `build.ps1` / `build.sh` for that. Every script has comment-based help
(`Get-Help .\scripts\dev\<name>.ps1 -Examples`).

| Script | What it does | Typical call |
| --- | --- | --- |
| `Get-SessionBrief.ps1` | Read-only workspace status: branch and dirty state, how far `master` is behind, other worktrees, detached runs, stale build markers, stashes. `-Remote` fetches and lists open pull requests. | `.\scripts\dev\Get-SessionBrief.ps1` |
| `Start-LongRun.ps1` | Launches an executable detached from the current shell and tracks it in `output/runs/<name>.status.json` with its log beside it. Use it for anything longer than about ten minutes. | `.\scripts\dev\Start-LongRun.ps1 -Name rich257 -Exe build-rel\Release\test_cavity_richardson.exe -ExeArgs 257` |
| `Get-LongRun.ps1` | Lists detached runs, or shows one with the tail of its log. | `.\scripts\dev\Get-LongRun.ps1 rich257` |
| `LongRun-Worker.ps1` | Worker process started by `Start-LongRun.ps1`. Not called directly. | |
| `LongRun-Common.ps1` | Default run directory and the liveness check, shared by the two scripts above. Not called directly. | |
| `New-Worktree.ps1` | Creates a git worktree on a new or existing branch and links the local tooling directory into it. `-Existing <path>` retrofits a worktree made with plain `git worktree add`. | `.\scripts\dev\New-Worktree.ps1 -Branch feat/x` |
| `Prune-Merged.ps1` | Lists local branches whose tip is the last commit of a merged pull request, the commit the PR was merged from (with a squash merge this is not the commit that landed on `master`). Deletes them only with `-Delete`. | `.\scripts\dev\Prune-Merged.ps1` |

## Notes

- Pull requests are squash-merged, so `git branch --merged` does not see merged branches.
  `Prune-Merged.ps1` asks GitHub instead and keeps any branch with commits after its pull request.
- Remove a worktree with `git worktree remove <path>`, never by deleting the directory: it contains
  a directory junction.
- `output/` is ignored by git, so run logs and status files never show up as untracked.
