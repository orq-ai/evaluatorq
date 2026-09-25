# Task 12 report: Insights dashboard shell, Dimensions, Labels and Traces

## Status

DONE_WITH_CONCERNS

## Changes

Added the read-only `/insights` dashboard with a newest-first run rail, the Insights sidebar entry, population and label chips, error-stage banners, failed-trace notes, a two-level Dimensions tree, a cluster detail side panel, label distribution cards, and filterable trace rows with Orq deep links. Routes cover run pages, HTMX tab fragments, cluster detail, filtered traces, JSON export, and the stable `map.json` endpoint. Live `insights` manifests appear as running entries, while truncated and unknown-version run files remain visible as unreadable entries.

The Map, Crosstab, and Priority views render explicit placeholders for Task 13. The `map.json` route currently returns an empty payload for the same reason. This task adds no control that starts an Insights run.

## Files

- `src/evaluatorq/dashboard/insights_routes.py`
- `src/evaluatorq/dashboard/insights_views.py`
- `src/evaluatorq/dashboard/app.py`
- `src/evaluatorq/dashboard/shell.py`
- `src/evaluatorq/dashboard/styles.py`
- `tests/dashboard/test_insights_page.py`

## Verification

- `uv run ruff check src` — passed.
- `uv run ruff format --check src` — passed; 264 files already formatted.
- `uv run basedpyright` — passed; 0 errors, 0 warnings, 0 notes.
- `uv run pytest -m 'not integration' tests/insights tests/test_reuse_guardrails.py tests/dashboard/test_insights_page.py` — passed; 135 passed, 2 expected UMAP warnings.
- `git diff --check` — passed.

## Concerns

The validated-model cache is used for readable run files, as required, but `store.list_runs` validates the directory entries before the cache lookup. This means repeated page requests still perform one uncached validation per run; the existing `list_runs` interface does not expose paths without parsing. The map endpoint and Map, Crosstab, and Priority views are intentionally incomplete until Task 13.
