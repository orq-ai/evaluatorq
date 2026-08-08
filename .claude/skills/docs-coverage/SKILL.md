---
name: docs-coverage
description: Use when checking whether evaluatorq docs cover every usage path — entry points crossed with surfaces, target kinds, modes, data sources and evaluator kinds. Produces a coverage matrix of documented vs GAP vs N/A. Triggers on "what's undocumented", "docs coverage", "which usage paths are missing", and when adding a feature that opens a new mode, backend, surface or entry point.
---

# docs-coverage

The code supports combinations of things. This skill finds combinations a user could
plausibly hit that no page explains.

Not a symbol checklist. "`red_team` is mentioned somewhere" is not coverage — the
interesting question is whether *static mode against a `deployment:` target* is
explained anywhere.

Companion skill: `docs-drift` answers the opposite question — what the docs claim
that is no longer true.

## Axes

Read `axes.md` (next to this file) first. It defines the axis *names*, the tiers, and
the impossible-combination list. Axis *values* come from source at run time, so new
modes and backends appear automatically.

## Step 0 — env guard

```bash
uv sync --all-extras --all-groups
uv run eq --help >/dev/null && uv run eq redteam --help >/dev/null && echo "ENV OK"
```

Without the `redteam` extra, whole axes vanish and the matrix reports phantom gaps.
If `ENV OK` does not print, stop.

## Step 1 — resolve axis values

For each axis in `axes.md`, enumerate current values from source:

```bash
uv run python -c "import evaluatorq; print(evaluatorq.__all__)"
uv run eq --help                       # command groups
uv run eq redteam run --help           # --mode, --target forms
grep -rn 'register_backend' src        # backend registry
grep -rn 'environ\|getenv' src         # env vars
```

## Step 2 — build the pairwise matrix

Cross axes **pairwise**, not as a full cross-product. Six axes fully crossed is
hundreds of cells, most of them nonsense; pairwise is the level at which a gap is
actually actionable.

Pairs worth checking: entry point × surface · entry point × target kind · entry point
× mode · entry point × data source · entry point × evaluator kind · surface × target
kind · mode × target kind.

For each cell:

- **documented** — a hand-written page shows this combination working, or explains it
  explicitly. Record `page.md#anchor`. Generated API pages never satisfy Tier 1.
- **`GAP`** — meaningful, undocumented.
- **`N/A`** — listed in `axes.md` as impossible, or structurally meaningless.

If a cell is `N/A` for a reason not yet in `axes.md`, **add it there** as part of the
run. That is how the list stays useful instead of the same false gap being re-argued
every time.

## Step 3 — report

Write `.context/docs-coverage-matrix.md`: one table per axis pair, then a ranked gap
list — Tier 1 gaps first (entry points, CLI commands, env vars), Tier 2 below.

```markdown
# docs-coverage — <date>

## entry point × mode

| | dynamic | static | hybrid |
|---|---|---|---|
| `red_team()` | guides/red-teaming.md#dynamic | guides/red-teaming.md#static | **GAP** |
| `build_report()` | N/A | N/A | N/A |

## Tier 1 gaps
1. `red_team()` × `hybrid` — no page explains when hybrid beats either pure mode.
```

Summarise to the user, then ask before writing any docs. Gap-filling is prose
authoring, not a mechanical fix — it needs a decision about what to say and where it
belongs.

`.context/` is gitignored.

## Honest limits

- The matrix is only as good as `axes.md`. A dimension nobody added is a dimension
  nobody checks.
- "Documented" is a judgement call. A passing mention and a worked example both
  resolve to a page anchor; prefer requiring a runnable example for Tier 1.
