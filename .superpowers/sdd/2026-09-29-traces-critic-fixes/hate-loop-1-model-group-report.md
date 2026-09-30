# Hate loop 1: model group summary

Added a compact, accessible model count and cost summary below the `/traces` totals strip. It is rendered from the same `visible_rows()` population, so quick views, AI matches, facets, and within-results narrowing feed the same rows. Costs remain unknown when any row cost or currency is missing, currencies differ, or a trace contains multiple models whose trace-level cost cannot be attributed. Traces without a model are grouped under `Unknown model`. The `.finder:has(> .finder-command)` CSS scope preserves `/find` presentation.

Added coverage for per-model counts, mixed and missing cost data, unknown models, filter response, and `/find` scope. Updated `docs/trace-finder.md` with the grouping behavior and cost limitations.

Verification passed: `uv run pytest tests/dashboard/test_explorer.py -q` (99 passed), `uv run ruff check src`, `uv run ruff format --check src`, `uv run basedpyright` (0 errors, 0 warnings, 0 notes), and `uv run --group docs mkdocs build --strict`. A live dashboard run on port 8128 showed the unfiltered summary and a `gpt-6-luna` model filter; no Ask AI request was submitted.

Screenshots: `.context/critics/after-hate-1-model-group-unfiltered.png` and `.context/critics/after-hate-1-model-group-filter.png`.
