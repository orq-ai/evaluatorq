# docs-autofill ledger

One row per attempt. The routine's only memory — read from `main` and from every
unmerged `docs/autofill-*` branch, so a gap whose PR was rejected is never
re-attempted. Never delete a row; a row removed is a gap that comes back.

Outcomes: `opened` · `blocked` (3 rounds, persona still stuck) · `skipped-no-key`.

| date | matrix cell | branch | outcome |
|---|---|---|---|
