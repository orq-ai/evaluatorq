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

The Map, Crosstab, and Priority views and map payload remain placeholders for Task 13.


## Review fixes

The header now reads the pipeline echo's top-level query, facets, numeric bounds, start, end, and limit; its fixture uses that persisted shape. Traces displays active dimension/cluster and label/value chips, with individual removal links and a clear-all link that preserve the remaining filters. Ordinary navigation or refresh on a tab URL renders the full shell and selects that tab; requests carrying `HX-Request: true` still receive only the fragment.

The dashboard now lists matching run paths by modification time and passes each readable path directly to `library.load_model_cached(path, InsightsRun.model_validate)`. This avoids parsing every run once through `store.list_runs` and then again through the cache, and it avoids duplicating the store's Pydantic validation rules. Validation and read errors are logged and kept as unreadable entries; unknown schema versions follow the same path. The focused tests verify a single model validation across repeat page requests, unknown-version visibility, and newest-first ordering.

- `uv run pytest tests/dashboard/test_insights_page.py -q` — passed; 10 passed.
- `uv run ruff check src` — passed.
- `uv run ruff format --check src` — passed; 263 files already formatted.
- `uv run basedpyright` — passed; 0 errors, 0 warnings, 0 notes.
- `uv run pytest -m 'not integration' tests/insights tests/test_reuse_guardrails.py tests/dashboard/test_insights_page.py` — passed; 139 passed, 2 expected UMAP warnings.
- `git diff --check` — passed.

Remaining concern: Map, Crosstab, and Priority content and the map JSON payload are still reserved for Task 13.
