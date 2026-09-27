# Task 2: Unified traces toolbar and behavior

## TDD and implementation

I added focused assertions for toolbar slot order in Table and Trajectories modes, toolbar persistence in idle/loading/loaded states, quick-view filtering and empty states, loaded-population-only AI matches, and relative/exact range controls. The implementation scaffolding preceded those tests, so the quick-view and toolbar work did not follow a strict red-green sequence. The first focused run found a loading-state regression: the toolbar progress count was no longer present. I restored the count, then a later self-review found and fixed a duplicate toolbar in loading states. The final tests assert that the toolbar and load form each render once.

The filter trigger and active filter chips now sit before the All, Errors, and AI matches tabs. The shared toolbar keeps the range and row controls, Columns or Sort, Load, and Table/Trajectories in fixed slots. Columns is before Load, and the view switch remains after Load in both modes. The hidden explorer results form retains its CSRF field and HTMX load action; filter controls remain available to the existing query form and can be opened from the toolbar.

Quick-view selection is part of `ExplorerView` and updates through `/find/rows`. Errors use the loaded rows' error status. AI matches require a classification result for a loaded row, so results from unrelated populations cannot appear. Both filters render an empty state when they have no rows.

Relative ranges update their local date and time fields immediately before Load using the selected duration and current browser time. Exact dates and times live in a collapsed native details panel; opening it selects exact mode, and selecting a preset returns to relative mode. The existing endpoint timezone-offset handling still captures separate offsets for the two endpoints across daylight-saving changes.

## Results

- `uv run pytest tests/dashboard/test_explorer.py -q`: 57 passed.
- `uv run ruff check src/evaluatorq/dashboard/trace_finder src/evaluatorq/trace_finder/explorer.py`: passed.
- `node --check src/evaluatorq/dashboard/static/dashboard.js`: passed.

## Self-review

The toolbar is rendered once inside `#explorer-results`, so `/find/rows` refreshes the contextual Columns/Sort slot and quick-view state without moving Load or the view switch. The date parser already has focused browser-offset and daylight-saving tests. The relative-load timestamp refresh is JavaScript behavior; it was syntax-checked but not exercised in a browser automation test. Existing dirty trajectory edits in `styles.py`, `explorer_views.py`, and `test_explorer.py`, and unrelated dirty files, are intentionally excluded from the Task 2 commit.

## Review follow-up

`/find/poll` now passes the current `ExplorerView` to the fragment renderer, which keeps its active facet chips and filters aligned with the loaded population. The `/traces` filter controls now use a command-strip-scoped CSS rule to stay collapsed despite the base `.finder-controls { display:flex }` rule. The Filters button toggles a class on the stable `.finder` wrapper, so body polls and OOB result swaps preserve the open state. The trace count lives in a compact status row before the toolbar, leaving Filters as the toolbar's first control.

Regression coverage checks that a poll response has one `#finder-controls` row and retains the loaded model filter, that the initial `/traces` response has one filter panel with collapsed/open CSS and delegated toggle wiring, and that the status row precedes the toolbar.

A real Dia browser session loaded the current `dashboard.js` with the toolbar visibility CSS and native form controls. Initially, the filter contents were absent from the accessibility tree; clicking Filters showed them, and clicking it again hid them. Relative Load submitted a fresh one-hour range (`15:52:13` to `16:52:13` local time) with the browser's current `-120` minute offset. Exact Load preserved the selected `2026-10-24 22:00:00` to `2026-10-25 04:00:00` range and sent endpoint offsets `-120` and `-60` across the Amsterdam daylight-saving transition.

Follow-up checks: `uv run pytest tests/dashboard/test_explorer.py -q` passed (58 tests), Ruff passed, and `node --check src/evaluatorq/dashboard/static/dashboard.js` passed. The browser interaction was performed manually because this repo has no installed Playwright, Puppeteer, or jsdom test harness.
