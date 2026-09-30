# Task 7 report: toolbar hierarchy and Filters menu

- Search (Ask AI row) and Load (toolbar) are the only solid buttons. Every other control is a 32px ghost with a 6px radius. Table/Trajectories and Search-in are a quiet grey track with a white active chip.
- Filters popover is an opaque panel (white, 1px border, layered shadow, z-index 60).
- Facet counts: `facet_menu(..., loaded_rows=)` tallies model, provider, status, agent and product from the already-loaded rows (no new API calls), sorts most common first, and says "Counts are of the N loaded traces." When Orq's catalogue is unavailable it lists the values seen in the loaded rows. `/find/facets?counts=loaded` keeps counts after the self-load. /find is unchanged (no rows passed). Project, trace type and tool have no counts (rows do not carry them).
- Toolbar stability: the Ask AI result band, criteria and progress now render under the toolbar (display:contents plus order), and the right control group is narrower so the "AI matches" tab no longer wraps it. Measured toolbar y before and after one Ask AI run: 244 -> 244 (Loaded traces) and 260 -> 260 (All traces). Active filter chips sit on their own row under the controls.
- 900px: question box on a full row; Search in / AI settings / Search share row 2; Filters + tabs on toolbar row 1; time, Rows, Columns, Load, switch on row 2. No lone control.
- Screenshots: .context/critics/after-t7-{1280,filters,narrowed,ask,900,find}.png.
- Note: Orq returned 503 "no healthy upstream" on several loads mid-run; retrying Load worked.

## Follow-up review fixes (2026-09-30)

- Reordered the Traces page so `#explorer-toolbar` is a real DOM sibling before the Ask AI body. Removed CSS flex ordering that disagreed with keyboard order; the results band, criteria and progress remain after the toolbar. Active chips stay on their own line.
- Kept counts tied to `ExplorerView.rows` and clarified the note to say the number of rows currently loaded and that counts change as filters change. Added a focused route regression that narrows the fake loaded population to two rows and checks the initial menu and `/find/facets?counts=loaded` response both render count 2.
- Found that the Traces toolbar opens the shared Ask AI facet menu, while closing it submits `explorer-load-form`. On the Traces surface, facet checkboxes now associate with `explorer-load-form`; `hx-include="#finder-controls"` still carries the same controls into Ask AI. A browser click confirmed the checkbox form association and a selected model chip after loading.
- Added plain-language Orq-unavailable copy with the raw detail available and a Retry button on failed loads.
- Focused test evidence: `uv run pytest tests/dashboard/test_explorer.py -q` passed 84 tests in 2.35s. `uv run ruff check src/evaluatorq/dashboard/trace_finder/views.py` passed. `uv run ruff format --check src/evaluatorq/dashboard/trace_finder/views.py` passed.
- Browser evidence on a clean local server at 127.0.0.1:8126: opened Filters, expanded Model, saw counts for 200 loaded rows, selected `gpt-5.6-luna`, closed the menu, and observed the active chip and the `/find/load` request. Submitted one Ask AI question total; 48 of 483 traces matched. Toolbar remained ahead of the result band. Screenshots: `.context/critics/after-task7-filters.png`, `.context/critics/after-task7-ask-ai.png`, and `.context/critics/after-task7-filter-form.png`.
- Concern: during the live filtered-load recheck, Orq returned 503 `no healthy upstream` for span hydration. The retained table showed 200 rows, but reopening the menu reported 0 current rows; the fake route regression returns correct counts for a narrowed view. Treat the remaining live mismatch as unverified while the provider is unhealthy; no second Ask AI request was sent.

## Facet refresh follow-up (2026-09-30)

- `/find/facets` now allows `explorer-load-form` only on the row-count (`counts=loaded`) path; unscoped or unrecognized form IDs still fall back to `finder-query-form`. This keeps Find's form IDs unchanged while preserving the Traces table form after the async menu refresh.
- Regression coverage verifies the refreshed checkbox still has `form="explorer-load-form"`, an unscoped request falls back to `finder-query-form`, and selecting `gpt-5.6-luna` reaches the fake explorer row source as the submitted facet.
- Final browser check after completion supports the stale-menu-response explanation: the completed table showed “200 traces loaded” while its already-rendered menu still showed zero; requesting `/find/facets?form_id=explorer-load-form&counts=loaded` afterward returned “Counts are of the 200 rows currently loaded” and model count 200. No further Ask AI submission was made.
- Follow-up test evidence: `uv run pytest tests/dashboard/test_explorer.py -q` passed 84 tests in 2.43s; `uv run ruff check src/evaluatorq/dashboard/trace_finder/routes.py` passed; `uv run ruff format --check src/evaluatorq/dashboard/trace_finder/routes.py` passed.

## Row-count menu refresh follow-up (2026-09-30)

- The count mismatch came from the menu's initial asynchronous response being rendered while the explorer still held zero rows. The table later completed its load, but the menu stayed stale in the DOM. `facet_menu` now marks the explorer menu to refresh when opened; the Filters click triggers `refreshFacets`, includes current controls, requests row-scoped counts, and asks the response to remain open. The after-swap handler reanchors the replacement menu.
- Added `test_pending_traces_facet_menu_refreshes_when_loaded_rows_arrive`: render a zero-row menu, load two rows, refresh with a selected model, and verify the updated 2-row note/count, open state, selection, and `explorer-load-form` association. The existing selected-facet route test continues to verify that a checked facet reaches `/find/load`.
- Browser validation used the existing 200-row table and did not submit Ask AI again. Clicking Filters caused `/find/facets?form_id=explorer-load-form&counts=loaded&...&open=1` to return 200; expanding Model displayed “Counts are of the 200 rows currently loaded” with model counts 174 for `gpt-6-luna` and 20 for `gpt-5.6-luna`. Screenshot: `.context/critics/after-task7-refresh-counts.png`.
- Exact test evidence: `uv run pytest tests/dashboard/test_explorer.py -q` passed 85 tests in 2.68s. `uv run ruff check src/evaluatorq/dashboard/trace_finder/views.py src/evaluatorq/dashboard/trace_finder/routes.py` passed. `uv run ruff format --check src/evaluatorq/dashboard/trace_finder/views.py src/evaluatorq/dashboard/trace_finder/routes.py` passed.
