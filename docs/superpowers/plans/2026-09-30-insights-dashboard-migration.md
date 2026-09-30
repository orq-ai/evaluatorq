# Insights dashboard mock migration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the approved Insights redesign mock on the real `/insights/{run_id}` dashboard page, including its Frustration score blocks and Activity view, while using saved `InsightsRun` data and existing dashboard services.

**Architecture:** Keep the existing Insights pipeline, run store, launch handlers, trace links, and shared dashboard shell. Add a scoped Insights shell variant and a typed run-to-review adapter, then port the mock's DOM, CSS, and JavaScript components into dedicated static assets so the production page uses the same visual implementation. Switch the run review page only after all four views and the launch sheet work; retain old tab URLs as compatibility routes.

**Tech Stack:** Python 3.10+, FastHTML/Starlette, existing Pydantic Insights models, HTML/CSS, plain JavaScript, NumPy/UMAP from the Insights extra, pytest, ruff, basedpyright.

## Global constraints

- Match `.context/insights-mock/index.html` at the same viewport, using `.context/insights-mock/template.html` as the visual and interaction reference. Both files are gitignored, so record their checksums and capture a reviewable baseline when implementation begins.
- Scope the new look to Insights. Other dashboard surfaces must keep their current shell, tokens, routes, and behavior.
- The Activity view shows one run and recomputes counts for current run filters. A skill load is a recorded `Skill` tool call, shell calls overlap tool calls, and missing `tool_stats` is unknown rather than zero use.
- Use saved `TraceInsight`, `InsightsRun`, and `ToolStats`; do not fetch live traces or rerun models to display a saved run. Preserve snapshot runs without Orq links and partial/error runs with visible warnings.
- Reuse `dashboard.shell.page`, `dashboard.trace_links`, `insights.models.label_key`, `insights.models.real_assistant_errors`, `insights.transcript.SHELL_TOOLS`, and existing launch, store, auth, and route handlers. Do not copy the mock's static sample JSON into production.
- Every filtered section has an empty state. HTML and JSON embedded in the page must escape user-supplied run names, trace text, labels, and command names.
- Do not run `git stash`, `git reset`, or `git checkout`. Stage only task-owned paths; other sessions have uncommitted production dashboard changes.
- Before each push, run `uv run ruff check src`, `uv run ruff format --check src`, `uv run basedpyright`, and `uv run pytest -m 'not integration'` exactly as CI does.

## Reference and reuse map

| Mock surface | Current source | Production reuse and change |
|---|---|---|
| Run title, stage/cost details, warnings | `template.html:593-615` | Reuse `insights_views.header`, `progress`, `failures`, and `InsightsRun`; render them in the mock layout. |
| Four highlight cards and filtered theme tree | `template.html:617-716` | Reuse cluster assignments, labels, and summary data; add one review aggregate rather than another pipeline stage. |
| Activity, weekly counts, pairings | `template.html:768-905` | Reuse persisted `ToolStats` and `SHELL_TOOLS`; implement the approved Activity spec in a focused aggregate and renderer. |
| Map and Compare | `template.html:906-993` | Reuse run coordinates, cluster/label color semantics, and trace links; replace the current Plotly/Vega presentation on the new page with mock-shaped SVG/table output. |
| Trace list and detail drawer | `template.html:995-1148` | Reuse `_trace_id_link`, `trace_link_button`, label formatting, and the existing trace detail contract; render an in-page drawer plus linked full trace page. |
| New-run sheet | `template.html:1166-1223` | Reuse `/insights/new`, preview, `/insights/runs`, CSRF, validation, and `insights-wizard.js`; adapt the form layout to the mock. |
| Shell and styles | `shell.py:198`, `styles.py:2803`, `theme.py` | Keep shared shell and tokens; add an Insights-only shell variant and stylesheet so other surfaces do not change. |

The mock is a single HTML/CSS/JavaScript file, while the current production review is Python-rendered. Reuse the mock's visual code and the production data/services directly. The adapter and scoped shell are the boundary between them; do not recreate the same cards and charts separately in Python or embed the mock as an iframe. The mock's `Open in Trace search` button is a placeholder: Finder cannot load exact saved Insights members, and snapshot runs do not retain source messages. The approved production action is `View matching traces` inside Insights, using the saved trace list; this is the one intentional copy difference.

## Task 1: Freeze the reference and build the review data adapter

**Files:** Create `src/evaluatorq/dashboard/insights_review_data.py`; create `tests/dashboard/test_insights_review_data.py`; use `.context/insights-mock/template.html`, `.context/insights-mock/index.html`, and `.context/coding-run-mock3.json` as local reference files.

**Interfaces:** Produce `build_review_payload(run: InsightsRun) -> dict[str, object]` and `review_trace_key(trace: TraceInsight) -> str`. The JavaScript views consume this one payload and aggregate its trace records consistently.

- [ ] **Step 1: Capture the reference.** Save baseline screenshots of Themes, Activity, Map, Compare, the run detail drawer, and the new-run sheet at 1440×900 and 390×844 under `.context/insights-parity/`. Include a filtered cohort, selected Activity item, and open trace drawer with their full URL hashes. Record the sample run ID, viewport, and mock `index.html` checksum in a short `.context/insights-parity/README.md`. Keep screenshots out of the package and avoid committing trace content.
- [ ] **Step 2: Write failing data tests.** Build synthetic `InsightsRun` fixtures with two traces sharing a trace ID but different span IDs, a missing `tool_stats`, an empty `ToolStats`, positive tool/skill/command counts, a failed label, and a missing summary. Assert unique trace keys, criteria-aware score values through `label_key` (specifically 1–5 for `user_frustration`), missing-versus-empty tool stats, summary error cleanup through `real_assistant_errors`, and null rather than zero for unmeasured values.
- [ ] **Step 3: Implement the adapter.** Map the saved model to the mock's run, dimension, label, and trace fields. Use both `trace_id` and `span_id` in keys and link parameters. Retain top and base cluster assignments, question errors, source/missing coverage, and optional values explicitly. Call `label_key` for score categories, and expose `SHELL_TOOLS` from the existing source rather than restating shell tool names. Keep the three ToolStats maps separate because their totals overlap.
- [ ] **Step 4: Verify and commit.** Run `uv run pytest tests/dashboard/test_insights_review_data.py -v`, then `uv run ruff check src/evaluatorq/dashboard/insights_review_data.py`; commit only the new adapter and tests with a conventional subject.

## Task 2: Reproduce the scoped shell, header, highlights, Themes, and Frustration blocks

**Files:** Modify `src/evaluatorq/dashboard/shell.py`; create `src/evaluatorq/dashboard/insights_review_views.py`, `src/evaluatorq/dashboard/static/insights-review.css`, and `src/evaluatorq/dashboard/static/insights-review.js`; extend `tests/dashboard/test_insights_page.py`.

**Interfaces:** Add `page(..., shell_variant: str | None = None)` with the current default output unchanged. Add `review_page(run, manifest) -> str` to emit the mock's structural shell and a safely escaped JSON data island; `insights-review.js` renders the initial Themes state. Keep the current `/insights/{run_id}` handler unchanged until Task 6 switches it.

- [ ] **Step 1: Write failing markup tests.** Assert that `review_page` contains the mock's structural IDs for header, highlights, filter bar, tabs, canvas, trace list, drawer, and sheet, plus a safely escaped payload. Assert that a one-dimension run, zero usable labels, partial run, and no-summary run load the same structural page with visible warnings/fallback data. Assert the existing `page()` output remains unchanged without `shell_variant`.
- [ ] **Step 2: Add the scoped shell variant.** Apply a body class only for Insights review, let its header live inside the page body, and keep real sidebar links, assets, CSRF, and authentication notices. Adapt the shared sidebar to the mock's 60px rail only for this variant. Put the mock's spacing, type sizes, borders, colors, and responsive rules in `insights-review.css`; use existing `theme.py` variables where their values match the mock and scoped values where they do not.
- [ ] **Step 3: Port the initial page.** Copy the mock's structural markup into `review_page`, then extract its CSS and JavaScript into the new static assets. Replace the mock's embedded sample data with `build_review_payload(run)`. Keep the mock's headline and Themes components, including one-decimal tinted frustration blocks, measured-count tooltips, whole-number trace blocks, and neutral `—` for unknown. Replace hard-coded completed/cost text with actual saved status and unknown-cost states, and preserve `failures(run)` and progress notices.
- [ ] **Step 4: Verify and commit.** Run `uv run pytest tests/dashboard/test_insights_page.py -v`; compare the initial Themes screenshot with the frozen reference at 1440×900; commit only task-owned files after correcting visible spacing and typography differences.

## Task 3: Add one shared filter and navigation controller

**Files:** Modify `src/evaluatorq/dashboard/static/insights-review.js` and `src/evaluatorq/dashboard/insights_review_views.py`; extend `tests/dashboard/test_insights_page.py`.

**Interfaces:** The server embeds the escaped review payload in a non-executable JSON island. The ported mock controller owns `view`, `dimension`, filter list, selected cluster/trace/activity item, search, color choice, compare axis, and expanded UI sections; the browser URL hash restores them. The same filtered trace set drives every view, headline, trace count, and detail drawer.

- [ ] **Step 1: Write failing render and URL tests.** Assert safe JSON embedding when a trace contains `</script>`, `&`, quotes, and Unicode; assert every trace link includes both identifiers. Add a browser interaction checklist for a theme filter, label filter, high-frustration filter, clear-one/clear-all, free-text request search, Back/Forward, copied URL, Escape, and selecting a trace from two views. Test full trace identifiers for brush selection rather than the mock's first-eight-character shortcut.
- [ ] **Step 2: Implement the controller.** Port `baseVisible`, `visible`, `toUrl`, `fromUrl`, and render dispatch from the mock into the dedicated JS file. Use delegated listeners so rerendered panels keep working. Validate URL values against available dimensions, labels, clusters, item names, and trace keys; ignore invalid selections. Use `history.pushState` for user selections and `replaceState` for typing/toggling transient controls. Avoid a second parallel source of truth in HTMX fragment state.
- [ ] **Step 3: Port the shared trace list and drawer.** Feed the mock components saved summaries, labels, and `ToolStats`; inject safe Orq and full-trace links from the server payload. Keep the mock's columns, score blocks, chips, flags, search, show-more, cluster detail, and trace detail. Omit Orq links for snapshot runs and keep the full `/insights/{run_id}/trace` page as a stable deep link.
- [ ] **Step 4: Verify and commit.** Run `node --check src/evaluatorq/dashboard/static/insights-review.js` and `uv run pytest tests/dashboard/test_insights_page.py -v`; execute the browser checklist on the sample run, then commit task-owned files.

## Task 4: Build the Activity view from saved ToolStats

**Files:** Modify `src/evaluatorq/dashboard/static/insights-review.js` and `src/evaluatorq/dashboard/static/insights-review.css`; create `tests/dashboard/insights_review_activity.cjs`; extend `tests/dashboard/test_insights_page.py`.

**Interfaces:** Port the mock's `activityRows`, `activityWeeks`, and `activityPairs` helpers for `skills`, `tools`, and `commands`, using Task 1's per-trace payload. The global filter set selects the run cohort; the selected item filters only the trace list, while rankings and pairings continue to describe that cohort.

- [ ] **Step 1: Write failing tests.** Use the existing `.context/insights-mock/activity-check.cjs` as a reference for `tests/dashboard/insights_review_activity.cjs`. Assert Skills is the default, all three categories show eligible/missing counts and distinct/using/total summaries, rows sort by coverage or calls, search finds hidden helpers, commands group by program, and selected commands keep their group open. Assert trace bars use the eligible denominator, load/call bars use the largest visible item or program in the current category, and co-occurrence percentages use the selected item's trace count. Test missing stats, empty stats, filtered zero matches, no co-occurrences, sparse weeks, and a single populated week.
- [ ] **Step 2: Port the mock layout.** Move the category tabs, searchable ranking, relative bars under both numeric columns, helper toggle, `Potentially state-changing` heuristic label, selected item detail, UTC Monday coverage chart with numerator/denominator, same/cross-category pairings, and matching trace drilldown into the production JS/CSS. Keep the exact counts and both bar-scale explanations visible. Show each pair's overlap count, selected-item trace denominator, and rounded percentage together. Stack the detail panel and place the two bars beneath each item name at narrow widths. Keep the `Skill`-load and shell-call overlap notes visible. The heuristic labels a command name; it must not claim that the trace changed state.
- [ ] **Step 3: Wire interaction and verify.** Test selection, pairing navigation, helper toggle, search, sort, empty states, and URL restoration in a browser. Run `node tests/dashboard/insights_review_activity.cjs`, `node --check src/evaluatorq/dashboard/static/insights-review.js`, and `uv run pytest tests/dashboard/test_insights_review_data.py tests/dashboard/test_insights_page.py -v`; compare Activity screenshots with the reference; commit task-owned files.

## Task 5: Reproduce Map and Compare without changing persisted run schema

**Files:** Create `src/evaluatorq/dashboard/insights_review_projection.py`; modify `src/evaluatorq/dashboard/insights_review_data.py`, `src/evaluatorq/dashboard/static/insights-review.js`, and `src/evaluatorq/dashboard/static/insights-review.css`; create `tests/dashboard/test_insights_review_projection.py`; extend `tests/dashboard/test_insights_map.py`.

**Interfaces:** `review_xy(run, dimension) -> dict[str, tuple[float, float]]` reduces saved finite 3D coordinates to a deterministic 2D display projection with UMAP `random_state=7`, Euclidean metric, and `n_neighbors=min(15, n-1)`, exactly as the mock extractor does. This matches the mock's display coordinates for the same saved run, but remains a projection of stored 3D coordinates rather than of the original embeddings. Cache by stable run/dimension/coordinate content so tab changes do not rerun UMAP. Preserve the existing 3D `map_payload` and `/map.json` compatibility endpoint.

- [ ] **Step 1: Write failing projection tests.** Assert deterministic trace-to-point mapping, fewer than five usable points returning an explicit no-map state, invalid coordinates being skipped with a warning, and a cache hit on repeated rendering. Verify that the existing 3D map payload still passes its tests.
- [ ] **Step 2: Implement the 2D projection and map.** Follow the mock's SVG canvas sizing, dot radii, muted filtered dots, labels, color choices, tooltip, click-to-drawer, rectangle brush, full-screen control, and no-map state. Use full trace keys in brush filters. Load UMAP only on this Insights path; never on the base dashboard import path.
- [ ] **Step 3: Implement Compare.** Render the mock's heat table for a chosen dimension against error, a label, or another dimension. Count only traces with both values where needed, fold excess categories into `Other`, keep cell denominators visible, and let a cell add its two filters. Reuse palette and `_cross_value` semantics rather than recomputing incompatible label values.
- [ ] **Step 4: Verify and commit.** Run `uv run pytest tests/dashboard/test_insights_review_projection.py tests/dashboard/test_insights_map.py tests/dashboard/test_insights_page.py -v`; compare Map and Compare screenshots and keyboard paths with the reference; commit task-owned files.

## Task 6: Make the mock's new-run controls real

**Files:** Modify `src/evaluatorq/dashboard/insights_launch.py`, `src/evaluatorq/dashboard/insights_worker.py`, `src/evaluatorq/dashboard/insights_routes.py`, `src/evaluatorq/dashboard/insights_review_views.py`, `src/evaluatorq/dashboard/static/insights-review.js`, `src/evaluatorq/dashboard/static/insights-review.css`, `src/evaluatorq/dashboard/static/insights-wizard.js`, `src/evaluatorq/insights/models.py`, `src/evaluatorq/insights/pipeline.py`, and `src/evaluatorq/insights/labeling.py`; create `src/evaluatorq/dashboard/insights_uploads.py`; extend `tests/dashboard/test_insights_launch.py` and `tests/insights/test_pipeline.py`.

**Interfaces:** The sheet posts to the existing CSRF-protected `/insights/runs` handler through an extended `InsightsLaunchSpec`. Existing `coding_analysis=True` still means the full coding bundle; an optional selected coding-label list requests a subset plus the required coding-agent gate. Custom questions become validated `LabelSpec` values. The existing `/insights/new` form remains usable.

- [ ] **Step 1: Write failing launch tests.** Cover all three mock templates, individual question toggles, a valid custom yes/no/choice/score question, duplicate or reserved names, malformed criteria, the coding-agent gate, and unchanged behavior for existing `coding_analysis=True` callers. Test CSRF, upload size/type/cleanup, file-source validation, authentication failures, and server errors retaining user input.
- [ ] **Step 2: Extend the launch contract.** Accept allow-listed general presets already in `LABEL_PRESETS`, selected coding labels from `CODING_LABELS`, and a bounded list of custom `LabelSpec` objects. Pass them through `insights_worker` to `insights()` and `label_traces()` without copying the classifier loop. Preserve `coding_analysis=True` as the full bundle and record the selected specs in the run so Re-run can prefill them. The custom choice editor must collect its answer options, and the score editor must define its five criteria; name/question alone are not enough to construct a valid `LabelSpec`.
- [ ] **Step 3: Connect the sheet.** Reuse the existing source selection, facet loading, snapshot preview, CSRF, validation, and launch route. Replace the mock's hard-coded trace/cost/time estimates with a range based on a priced prior run, or an explicit unavailable state when no reliable basis exists. Make each Browse button open a file input that posts a Finder export or snapshot to a CSRF-protected upload route, validates `RunExport` or `Snapshot`, enforces a 100 MB limit, and stores it under the Insights runs directory until the worker finishes. Pass only the validated server path to the existing launch model, and delete only files that route created after the worker consumes them. Make the sheet keyboard accessible and preserve its visual layout.
- [ ] **Step 4: Verify and commit.** Run `uv run pytest tests/dashboard/test_insights_launch.py tests/insights/test_pipeline.py tests/insights/test_labeling.py -v`, then `node --check src/evaluatorq/dashboard/static/insights-review.js`; use the sheet to start one local snapshot run and confirm its requested questions appear in the saved run. Commit only task-owned files.

## Task 7: Switch real routes and finish visual parity

**Files:** Modify `src/evaluatorq/dashboard/insights_routes.py`, `src/evaluatorq/dashboard/insights_views.py`, `src/evaluatorq/dashboard/insights_review_views.py`, `src/evaluatorq/dashboard/static/insights-review.js`, and `src/evaluatorq/dashboard/static/insights-review.css`; extend `tests/dashboard/test_insights_page.py` and `tests/dashboard/test_insights_map.py`.

**Interfaces:** `/insights/{run_id}` uses `review_page`; `/insights/{run_id}/trace`, `/insights/new`, launch/preview/export, and existing tab fragment routes remain valid. New navigation offers Themes, Activity, Map, and Compare; Priority is absent from the new page and old priority URLs continue to resolve for existing bookmarks.

- [ ] **Step 1: Write failing route tests.** Assert a real saved run opens the redesign, an old tab URL still resolves, missing/unreadable/running runs keep their current handling, export remains available, and snapshots hide Orq links. Assert the breadcrumb, Export, Re-run, Open in Orq, View matching traces, and Turn into a question actions have working routes or forms rather than toast placeholders.
- [ ] **Step 2: Switch and wire actions.** Have `insights_run` call `review_page` for saved `InsightsRun` instances. Map the breadcrumb to `/insights`, Export to existing JSON export, Open in Orq to `trace_span_url`, Re-run to a prefilled launch sheet, and Turn into a question to the new custom-question editor. Replace the mock's placeholder `Open in Trace search` with `View matching traces`, which selects the saved cluster's exact members in the Insights trace list; do not open Finder with a different population. File-backed sources require a fresh selection if the saved source path is unavailable.
- [ ] **Step 3: Complete visual comparison.** Serve this worktree on a verified free 81xx port and restart after edits. Capture production screenshots at the same 1440×900 and 390×844 viewports and the same sample run/URL states as Task 1. Compare shell, typography, color, spacing, card and table geometry, score blocks, all four views, drawer, popover, and sheet. Align key element boxes within 2px and use the same color values for static components; differences in real run status and missing data must remain truthful.
- [ ] **Step 4: Verify and commit.** Run `uv run pytest tests/dashboard/test_insights_page.py tests/dashboard/test_insights_map.py -v` and `node --check src/evaluatorq/dashboard/static/insights-review.js`. Commit only this task's route, view, style, and test files.

## Task 8: Document the shipped behavior and run the release checks

**Files:** Modify `docs/insights.md` and `.claude/skills/docs-coverage/axes.md` for the new question-selection paths; modify `CHANGELOG.md` only if a public default changes; update focused tests as needed.

- [ ] **Step 1: Apply the repo docs skills.** Run `.claude/skills/docs-drift`, `.claude/skills/docs-coverage`, and `.claude/skills/docs-writing` for the diff. Describe Activity counts and missing data, cross-view filters and URL restoration, the display-only 2D map projection, snapshot uploads, selected coding labels, custom questions, and the new-run sheet in `docs/insights.md`. Add the new question-selection dimension to `.claude/skills/docs-coverage/axes.md`. Do not imply that co-occurrence proves causality or that a skill load proves success.
- [ ] **Step 2: Run final checks.** Run `uv run ruff check src`, `uv run ruff format --check src`, `uv run basedpyright`, `uv run pytest -m 'not integration'`, `uv run --group docs mkdocs build --strict`, and `uv run python scripts/validate_mermaid.py`. Recheck the 1440×900 and 390×844 visual comparison after the final edit.
- [ ] **Step 3: Review the diff and commit.** Compare only task-owned paths against `origin/main`, verify there is no priority scatter in the new page and no accidental changes to other dashboard surfaces, then commit the documentation and any final focused fixes with a conventional subject. Keep the PR in draft until the user asks to change its status.

## Acceptance checklist

- [ ] The real saved run matches the mock's shell, run header, highlights, Themes, Activity, Map, Compare, trace list, detail drawer, and new-run sheet at the reference viewports, apart from the documented `View matching traces` copy change for the mock's placeholder action.
- [ ] The Frustration column uses the tinted score blocks in the mock: one decimal for theme averages, whole numbers for traces, neutral `—` for unknown, and a measured-count tooltip.
- [ ] All four views share one filter state; links restore it; counts and visible traces agree; every empty/degraded case explains what is missing.
- [ ] Activity counts, weekly denominators, and pairings agree with the 100-trace sample and synthetic edge-case tests.
- [ ] Existing run creation, export, trace deep links, snapshot behavior, warnings, and old tab URLs still work; no other dashboard surface changes appearance.
