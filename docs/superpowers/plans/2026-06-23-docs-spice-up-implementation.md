# evaluatorq Docs Spice-Up Implementation Plan

## Branch-local baseline (2026-06-23)

- Strict docs build: `uv run --group docs mkdocs build --strict`
  - Result: PASS on June 23, 2026.
  - Notes: MkDocs reported the expected orphan-page info notices for `docs/dashboard.md` and `docs/redteam-dashboard-only-views.md`. The stricter failures called out in the inventory were not present in this worktree baseline.
- Example smoke gate: `uv run python scripts/smoke_examples.py`
  - Result: PASS on June 23, 2026 (`compiled 80/80 examples`).

## Frozen execution boundary

Implement Tier 0, Tier 1, and Tier 2 only.
Delete `docs/redteam-dashboard-only-views.md` instead of moving it.
Fold `*-openai` guide pages into tabs inside the canonical guide pages, then add redirects.
Leave Tier 3, README slimming, and landing-page hero work for a follow-on plan.
