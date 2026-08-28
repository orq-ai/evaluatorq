# docs-autofill ledger

One row per attempt. The routine's only memory — read from `main` and from every unmerged `docs/autofill-*` branch, so a gap whose PR was rejected is never re-attempted. Never delete a row; a row removed is a gap that comes back.

Outcomes: `prepared` (work committed, PR not yet created) · `opened` · `blocked` (3 rounds, persona still stuck) · `skipped-no-key` · `skipped-code-defect` (the page could not be written honestly around a code/docs inconsistency — see the Slack message for that run).

| date | matrix cell | branch | outcome |
|---|---|---|---|
| 2026-08-23 | `entry point (red_team()) × data source (replay)` | docs/autofill-redteam-replay | opened |
