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
| `Remove-Worktree.ps1` | Removes a linked worktree safely: unlinks its `.claude` junction first, checks the shared directory kept its contents, then runs `git worktree remove`. Refuses the main worktree, a locked one, one containing any other directory link (git would follow it too), one whose `.claude` points inside the worktree or at a directory containing it (git would delete the shared directory as ordinary contents), and one with local changes unless `-Force` (which discards them), before touching the link. | `.\scripts\dev\Remove-Worktree.ps1 ..\cfd-x` |
| `Prune-Merged.ps1` | Lists local branches whose tip is the last commit of a merged pull request, the commit the PR was merged from (with a squash merge this is not the commit that landed on `master`). Deletes them only with `-Delete`. | `.\scripts\dev\Prune-Merged.ps1` |

## Notes

- Pull requests are squash-merged, so `git branch --merged` does not see merged branches.
  `Prune-Merged.ps1` asks GitHub instead and keeps any branch with commits after its pull request.
- Remove a worktree with `Remove-Worktree.ps1`. A bare `git worktree remove` is not safe here: Git
  for Windows follows the `.claude` junction and deletes the shared directory behind it, which
  emptied the harness on 2026-10-03. Deleting the directory by hand does the same. To do it by
  hand, first remove the link alone with `[System.IO.Directory]::Delete('<path>\.claude')` in
  PowerShell (`cmd /c rmdir` also works, but expands any `%VAR%` in the path).
- `output/` is ignored by git, so run logs and status files never show up as untracked.
