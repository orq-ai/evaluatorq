# Unified Sim Detail Drawer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Agent Sim report’s split modal, transcript foldouts, and Failure anchors with one accessible right-side detail drawer for conversations, personas, and scenarios.

**Architecture:** Keep the native `<dialog>` and existing transcript fragment endpoint. Server renderers emit lightweight, keyboard-operable row triggers and hidden persona/scenario templates; the browser clones templates or lazily fetches conversation bodies into the dialog. A stable full-run conversation index is carried through filtered views, while cohort grouping uses persisted configuration fingerprints rather than display names.

**Tech Stack:** Python 3.13, FastHTML-rendered HTML, HTMX 2, native `<dialog>`, vanilla JavaScript, CSS, pytest, Ruff.

## Global Constraints

- Reuse `GET /r/{rid}/sim/transcript?idx=` and `render_transcript_fragment`; do not add a second conversation-detail endpoint.
- A conversation URL’s `idx` is always its zero-based position in `run.results`, even in a filtered report; the endpoint deliberately resolves against the full run.
- All drawer trigger rows use `role="button"`, `tabindex="0"`, and respond to click, Enter, and Space. A Failures `Why` cell uses `data-no-drawer` so its text can be selected without opening the drawer.
- Keep the native dialog semantics: Escape, focus trap, and clicking the backdrop close it. Pin it with `inset: 0 0 0 auto; margin: 0; width: min(560px, 92vw)`.
- Conversation transcript bodies remain lazy. Persona/scenario templates may embed only compact conversation-list metadata.
- All report-derived strings must still pass through `esc()` / `_esc()`; no raw HTML from a run report may enter markup.
- The standalone export gets the clean four-column Failures table, but no drawer or interactive behavior.
- Keep the current public `individual_entries(results)` payload backward-compatible; do not add dashboard-only identity fields to exported `SimulationEntry` JSON.

---

## File Structure

- `src/evaluatorq/simulation/reports/sections.py` — derives collision-safe persona/scenario cohort keys and includes them in overview and breakdown section rows while retaining display-name fields for existing exports.
- `src/evaluatorq/dashboard/report_tabs.py` — derives stable filtered entries, builds cohort templates, renders dashboard-only Failures and whole-row breakdown triggers, and emits the drawer shell.
- `src/evaluatorq/dashboard/sim_views.py` — changes Transcripts from `<details>` cards to flat conversation triggers; keeps the existing fragment route unchanged.
- `src/evaluatorq/dashboard/static/dashboard.js` — dispatches all entity triggers, lazily loads conversations, implements drawer navigation and drill-stack back, and removes obsolete `#conv-N` behavior.
- `src/evaluatorq/dashboard/styles.py` — converts the centered dialog and foldout-card styles into responsive drawer, row, cohort-list, and focus styles.
- `src/evaluatorq/simulation/reports/export_html.py` — removes the static Failure anchor and criteria-dot column.
- `tests/simulation/reports/test_sections.py` — verifies duplicate display names remain separate cohorts.
- `tests/dashboard/test_report_tabs.py` — verifies dashboard Failures, cohort templates, trigger accessibility, and filtered stable conversation URLs.
- `tests/dashboard/test_sim_transcript.py` — replaces foldout assertions with flat-row trigger/lazy-load assertions and retains endpoint identity coverage.
- `tests/simulation/reports/test_export.py` — verifies the standalone Failures table has exactly Scenario, Persona, Why, and Score.

## Task 1: Preserve stable cohort identity in report sections

**Files:**

- Modify: `src/evaluatorq/simulation/reports/sections.py:41-52,146-190,222-275,441-470`
- Test: `tests/simulation/reports/test_sections.py`

**Interfaces:**

- Consumes: `SimulationResult.metadata` fields already written by the runner: `persona`, `persona_traits`, `scenario`, `scenario_goal`, `scenario_context`, and criteria metadata.
- Produces: `_persona_cohort_id(result: SimulationResult) -> str`, `_scenario_cohort_id(result: SimulationResult) -> str`; overview entities and breakdown rows with an `id` key; heatmap cells keyed by those IDs while preserving display labels.
- Compatibility: existing fields (`persona`, `scenario`, `success_rate`, `avg_goal_completion_score`, `total_tokens`) remain present and unchanged in meaning.

- [ ] **Step 1: Write the failing collision regression tests**

Add tests using two results named `Alex` with different `persona_traits`, and two results named `Billing` with different `scenario_goal` values. Assert that the sections do not merge them:

```python
def test_persona_breakdown_keeps_same_named_distinct_personas_separate():
    sections = build_report_sections([
        _result(persona='Alex', traits={'patience': 0.1}, score=0.2),
        _result(persona='Alex', traits={'patience': 0.9}, score=0.8),
    ])
    rows = _section(sections, 'persona_breakdown').data['rows']
    assert len(rows) == 2
    assert {row['persona'] for row in rows} == {'Alex'}
    assert len({row['id'] for row in rows}) == 2


def test_overview_entities_and_breakdown_rows_share_the_same_cohort_id():
    sections = build_report_sections([_result(persona='Alex', traits={'patience': 0.1})])
    overview = _section(sections, 'overview').data['personas'][0]
    breakdown = _section(sections, 'persona_breakdown').data['rows'][0]
    assert overview['id'] == breakdown['id']
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run: `uv run pytest tests/simulation/reports/test_sections.py -q`

Expected: FAIL because the current name-keyed dictionaries create one `Alex` row and no `id` fields.

- [ ] **Step 3: Implement deterministic cohort IDs and key all aggregations by them**

Add the JSON and hashing imports and use a canonical, persisted configuration snapshot. Prefer an explicit stored ID if a future runner writes one; the canonical fallback keeps existing saved reports usable:

```python
def _cohort_id(result: SimulationResult, kind: Literal['persona', 'scenario']) -> str:
    explicit = result.metadata.get(f'{kind}_id')
    if explicit:
        return f'{kind}:{explicit}'
    if kind == 'persona':
        snapshot = {
            'name': _persona_name(result),
            'traits': result.metadata.get('persona_traits') or {},
        }
    else:
        snapshot = {
            'name': _scenario_name(result),
            'goal': result.metadata.get('scenario_goal'),
            'context': result.metadata.get('scenario_context'),
            'criteria': _criteria_meta(result),
        }
    payload = json.dumps(snapshot, sort_keys=True, separators=(',', ':'), default=str)
    return f'{kind}:{sha256(payload.encode()).hexdigest()[:16]}'


def _persona_cohort_id(result: SimulationResult) -> str:
    return _cohort_id(result, 'persona')


def _scenario_cohort_id(result: SimulationResult) -> str:
    return _cohort_id(result, 'scenario')
```

Replace the name-keyed overview/breakdown dictionaries with dictionaries keyed by the cohort ID. Store `id` on every overview entity and breakdown row, while keeping `name`, `persona`, and `scenario` as the display text. In `_build_persona_scenario_heatmap_section`, aggregate by `(persona_id, scenario_id)` and include `persona_id` / `scenario_id` in each cell; construct labels from the corresponding overview names, appending ` (2)`, ` (3)`, and so on when duplicate labels occur.

- [ ] **Step 4: Run sections and export tests**

Run: `uv run pytest tests/simulation/reports/test_sections.py tests/simulation/reports/test_export.py -q`

Expected: PASS. Existing exports continue to show their original display labels.

- [ ] **Step 5: Commit the section contract**

```bash
git add src/evaluatorq/simulation/reports/sections.py tests/simulation/reports/test_sections.py
git commit -m "feat(sim): retain stable cohort identities in reports"
```

### Task 2: Make filtered conversation entries retain full-run indexes and flatten Transcript rows

**Files:**

- Modify: `src/evaluatorq/dashboard/report_tabs.py:85-134`
- Modify: `src/evaluatorq/dashboard/sim_views.py:87-163`
- Modify: `src/evaluatorq/dashboard/view.py:1097-1125`
- Test: `tests/dashboard/test_sim_transcript.py:402-790`

**Interfaces:**

- Consumes: `run.results`, the filtered `results` argument, and `individual_entries()`.
- Produces: `_stable_entries(run: SimulationRun, rows: list[SimulationResult]) -> list[SimulationEntry]`, where `entry.index` is always the full-run index; `render_sim_row_list()` output with `[data-sim-entity-trigger][data-entity-kind="conversation"]` and `data-drawer-url`.
- Preserves: `GET /r/{rid}/sim/transcript?idx=` continues to use its full-run resolver; no filter parameter changes that behavior.

- [ ] **Step 1: Replace foldout tests with flat-row and stable-index tests**

Replace `TestConversationCards.test_row_list_renders_details_cards_with_tint` with assertions for the new contract:

```python
def test_row_list_renders_clickable_conversation_rows(sim_entries):
    html = render_sim_row_list('rid', sim_entries)
    assert '<details' not in html
    assert 'sim-conv-card' not in html
    assert html.count('data-sim-entity-trigger') == len(sim_entries)
    assert 'data-entity-kind="conversation"' in html
    assert 'data-drawer-url="/r/rid/sim/transcript?idx=0"' in html
    assert 'role="button" tabindex="0"' in html
    assert 'hx-trigger="toggle once' not in html
```

Extend the existing filtered Bob end-to-end test so a filtered `/filter` response’s Failures and Transcript rows both contain `idx=2`, and the transcript endpoint returns only `BOB-ONE` for that URL.

- [ ] **Step 2: Run the focused transcript tests and verify failure**

Run: `uv run pytest tests/dashboard/test_sim_transcript.py -q`

Expected: FAIL because rows are still `<details>` elements and `sim_report_tabs()` still reindexes filtered entries.

- [ ] **Step 3: Implement stable-entry derivation in the report renderer**

In `report_tabs.py`, keep `individual_entries()` unchanged and add a dashboard-only adapter that restores the full-run index after filtering:

```python
def _stable_entries(run: SimulationRun, rows: list[Any]) -> list[Any]:
    from evaluatorq.simulation.reports.sections import individual_entries

    full_indexes = {id(result): index for index, result in enumerate(run.results)}
    return [
        entry.model_copy(update={'index': full_indexes[id(result)]})
        for result, entry in zip(rows, individual_entries(rows), strict=True)
    ]
```

Use `_stable_entries(run, rows)` instead of `individual_entries(rows)` in `sim_report_tabs()`. Pass these entries to both `sim_interactive_panels()` and the new drawer context in Task 3. Update `sim_interactive_panels()` documentation to say it receives stable-index entries; remove the obsolete assertion that the drawer request needs `hx-include="#filter-form"` for index resolution.

- [ ] **Step 4: Render each transcript as one lazy drawer trigger**

Replace the `<details>`/`<summary>`/HTMX-body construction in `render_sim_row_list()` with one row shell. Preserve existing score, status, termination, and trace elements; only the transcript body moves to the drawer:

```python
row_attrs = (
    'class="sim-conv-row {tint}" role="button" tabindex="0" '
    'data-sim-entity-trigger data-entity-kind="conversation" '
    'data-drawer-url="/r/{rid}/sim/transcript?idx={idx}"'
).format(tint=tint, rid=safe_rid, idx=idx)
rows_html.append(
    f'<div {row_attrs}>'
    f'<span class="sim-conv-idx">#{idx + 1}</span>'
    f'<span class="sim-conv-persona">{persona}</span>'
    f'<span class="sim-conv-sep">&middot;</span>'
    f'<span class="sim-conv-scenario">{scenario}</span>'
    f'<span class="sim-conv-right">{right_cluster}</span>'
    '</div>'
)
```

Retain `onclick='event.stopPropagation()'` on `trace_link_button()`. Do not emit `id="conv-N"`, `hx-get`, `hx-trigger`, `hx-target`, or `hx-swap` from this list.

- [ ] **Step 5: Update row-list styles and tests**

Rename the card/summary selectors to `.sim-conv-row`, include `:hover` and `:focus-visible` affordances, and remove `.sim-conv-body` / `::-webkit-details-marker` rules. Update all route count assertions from `sim-conv-card` to `sim-conv-row`.

Run: `uv run pytest tests/dashboard/test_sim_transcript.py tests/dashboard/test_report_tabs.py -q`

Expected: PASS, including the filtered stable-index regression.

- [ ] **Step 6: Commit transcript-row conversion**

```bash
git add src/evaluatorq/dashboard/report_tabs.py src/evaluatorq/dashboard/sim_views.py src/evaluatorq/dashboard/view.py src/evaluatorq/dashboard/styles.py tests/dashboard/test_sim_transcript.py tests/dashboard/test_report_tabs.py
git commit -m "feat(dashboard): open sim transcripts in drawer rows"
```

### Task 3: Render dashboard Failure rows and cohort-detail templates

**Files:**

- Modify: `src/evaluatorq/dashboard/report_tabs.py:137-585`
- Modify: `src/evaluatorq/common/reports/html_helpers.py:119-134`
- Modify: `src/evaluatorq/simulation/reports/export_html.py:279-304`
- Test: `tests/dashboard/test_report_tabs.py`
- Test: `tests/simulation/reports/test_export.py`

**Interfaces:**

- Consumes: section rows with `id`, stable `SimulationEntry` objects, and `rid`.
- Produces: `_sim_entity_context(by_kind, entries, rid) -> dict[str, Any]`, `_sim_failures_table(section, rid) -> str`, and `html_table(headers, rows, row_attrs: list[str] | None = None)`.
- Produces markup: persona/scenario rows carry `data-entity-id`; conversation triggers carry `data-drawer-url`; templates retain `data-sim-entity-template`.

- [ ] **Step 1: Write renderer and export regression tests**

Add direct dashboard assertions:

```python
def test_dashboard_failures_use_four_columns_and_drawer_rows(sim_run):
    html = sim_report_tabs('rid', sim_run)
    failures = html.split('id="section-failures_first"', 1)[1].split('Top failure modes', 1)[0]
    assert ['Scenario', 'Persona', 'Why', 'Score'] == _headers(failures)
    assert 'Criteria' not in failures
    assert 'href="#conv-' not in failures
    assert 'data-entity-kind="conversation"' in failures
    assert 'data-drawer-url="/r/rid/sim/transcript?idx=1"' in failures
    assert 'data-no-drawer' in failures


def test_cohort_template_contains_stats_and_compact_conversation_triggers(sim_run):
    html = sim_report_tabs('rid', sim_run)
    template = _template(html, 'persona-')
    assert 'Goal rate' in template and 'Avg score' in template and 'Tokens' in template
    assert 'sim-cohort-conversations' in template
    assert 'data-entity-kind="conversation"' in template
```

Add an export test that locates the Failures `<thead>` and asserts `Criteria` and `href="#conv-` are absent while `fail-why` remains present.

- [ ] **Step 2: Run renderer/export tests and verify failure**

Run: `uv run pytest tests/dashboard/test_report_tabs.py tests/simulation/reports/test_export.py -q`

Expected: FAIL because dashboard uses the shared static renderer, breakdown labels are inner buttons, and static export still has a Criteria column and anchor.

- [ ] **Step 3: Extend `html_table` for accessible whole-row triggers**

Change the helper signature and row opening tag only; all callers without `row_attrs` preserve byte-for-byte markup:

```python
def html_table(
    headers: list[str], rows: list[list[str]], row_attrs: list[str] | None = None
) -> str:
    if row_attrs is not None and len(row_attrs) != len(rows):
        raise ValueError('row_attrs length must match rows')
    parts = ['<table>', '<thead><tr>']
    parts.extend(f'<th>{esc(header)}</th>' for header in headers)
    parts.append('</tr></thead><tbody>')
    for index, row in enumerate(rows):
        attrs = f' {row_attrs[index]}' if row_attrs and row_attrs[index] else ''
        parts.append(f'<tr{attrs}>')
        parts.extend(
            f'<td data-label="{esc(headers[cell_index])}">{cell}</td>'
            if cell_index < len(headers) else f'<td>{cell}</td>'
            for cell_index, cell in enumerate(row)
        )
        parts.append('</tr>')
    parts.append('</tbody></table>')
    return ''.join(parts)
```

Validate `len(row_attrs) == len(rows)` and raise `ValueError('row_attrs length must match rows')` when a caller supplies a mismatched list. Add a unit test for both the normal and attributed paths.

- [ ] **Step 4: Build a collision-safe drawer context and templates**

Replace name-keyed `persona_ids` / `scenario_ids` with section IDs. Build lists once from stable entries and the matching result metadata:

```python
def _sim_entity_context(by_kind: dict[str, Any], entries: list[Any], rid: str) -> dict[str, Any]:
    personas = _section_rows(by_kind, 'overview', 'personas')
    scenarios = _section_rows(by_kind, 'overview', 'scenarios')
    persona_rows = _section_rows(by_kind, 'persona_breakdown', 'rows')
    scenario_rows = _section_rows(by_kind, 'scenario_breakdown', 'rows')
    cohorts = _group_entries_by_cohort(entries, personas, scenarios)
    return {
        'rid': rid,
        'personas': personas,
        'scenarios': scenarios,
        'persona_stats': {row['id']: row for row in persona_rows},
        'scenario_stats': {row['id']: row for row in scenario_rows},
        'cohorts': cohorts,
    }
```

`_group_entries_by_cohort()` must use the IDs emitted by Task 1 and must append each entry exactly once per entity kind. Its list item renderer emits only `#N`, persona/scenario display names, score, outcome badge, and `/sim/transcript?idx=N`; it must never inline a chat message. For a cohort missing from the filtered entries, render `<p class="sim-cohort-empty">No conversations.</p>`.

Update `_sim_entity_modal()` to pass each entity’s stats and list into `_sim_persona_template()` / `_sim_scenario_template()`. Add a reusable `_sim_cohort_stats()` block and `_sim_cohort_list()` block below the current profile body. Give each template its existing `persona-{index}` / `scenario-{index}` DOM ID, but map that DOM ID from the stable section `id`, not from a display name.

- [ ] **Step 5: Render dashboard-only Failures and whole-row breakdown triggers**

Implement `_sim_failures_table(section, rid)` in `report_tabs.py`; do not call `_SECTION_RENDERERS['failures_first']` for the dashboard. Emit these headers and cells in exactly this order:

```python
headers = ['Scenario', 'Persona', 'Why', 'Score']
row_attrs = (
    'class="sim-drawer-row sim-failure-row" role="button" tabindex="0" '
    'data-sim-entity-trigger data-entity-kind="conversation" '
    f'data-drawer-url="/r/{esc(rid)}/sim/transcript?idx={int(row["index"]) - 1}"'
)
cells = [
    esc(str(row['scenario'])),
    esc(str(row['persona'])),
    f'<span class="fail-why" data-no-drawer title="{esc(str(row.get("reason", "")))}">'
    f'{esc(_cap(str(row.get("reason", ""))))}</span>',
    f'{float(row["score"]):.2f}',
]
```

Pass row attributes to `_sim_breakdown_table()` and use `row['id']` to set `data-entity-id`; render the display label as plain escaped text, deleting `_sim_entity_button()` and its `.sim-entity-link` CSS. Preserve Config’s real button rows, but obtain their entity ID from the indexed overview entity rather than `dict[name]`.

- [ ] **Step 6: Clean the standalone export**

Replace the static table row with the following four cells and matching four headers:

```python
f'<tr><td>{_esc(r["scenario"])}</td>'
f'<td>{_esc(r["persona"])}</td>'
f'<td class="fail-why" title="{_esc(r.get("reason", ""))}">{_esc(_cap(r.get("reason", "")))}</td>'
f'<td>{r["score"]:.2f}</td></tr>'
```

Delete the `_criteria_dots(r['criteria'])` cell and the anchor. Do not change static per-conversation sections; they remain the export’s criteria detail.

- [ ] **Step 7: Run renderer/export tests**

Run: `uv run pytest tests/dashboard/test_report_tabs.py tests/simulation/reports/test_export.py tests/common/reports/test_html_helpers.py -q`

Expected: PASS. The dashboard and export now have the same visible Failures columns, with interaction only in the dashboard.

- [ ] **Step 8: Commit renderer and export work**

```bash
git add src/evaluatorq/dashboard/report_tabs.py src/evaluatorq/common/reports/html_helpers.py src/evaluatorq/simulation/reports/export_html.py tests/dashboard/test_report_tabs.py tests/simulation/reports/test_export.py tests/common/reports/test_html_helpers.py
git commit -m "feat(dashboard): render sim drawer entities and failure rows"
```

### Task 4: Implement drawer dispatch, navigation, and drill-stack behavior

**Files:**

- Modify: `src/evaluatorq/dashboard/static/dashboard.js:114-205,294-314`
- Modify: `src/evaluatorq/dashboard/report_tabs.py:354-378`
- Test: `tests/dashboard/test_report_tabs.py`

**Interfaces:**

- Consumes: `[data-sim-entity-trigger]`, `data-entity-kind`, `data-entity-id`, `data-drawer-url`, `[data-sim-entity-template]`, and `#filter-form`.
- Produces: one drawer containing `[data-sim-entity-content]`, `[data-sim-entity-prev]`, `[data-sim-entity-next]`, `[data-sim-entity-back]`, and `[data-sim-entity-close]`.
- State: `activeState = {kind, id, url, origin}` and `drillStack: DrawerState[]`; an `origin` is the containing conversation list used for j/k order.

- [ ] **Step 1: Add markup tests for actions and keyboard attributes**

```python
def test_sim_drawer_has_back_nav_close_controls(sim_run):
    html = sim_report_tabs('rid', sim_run)
    assert 'data-sim-entity-back' in html
    assert 'data-sim-entity-prev' in html
    assert 'data-sim-entity-next' in html
    assert 'data-sim-entity-close' in html
    assert 'aria-label="Back to cohort"' in html
```

Assert that both a Failure row and a breakdown table row contain `role="button" tabindex="0"`.

- [ ] **Step 2: Run the new markup tests and verify failure**

Run: `uv run pytest tests/dashboard/test_report_tabs.py -q`

Expected: FAIL because the dialog has no back control and breakdown rows still delegate keyboard behavior to inner buttons.

- [ ] **Step 3: Extend the dialog action strip**

Add the initially hidden back control before previous/next in `_sim_entity_modal()`:

```html
<button type="button" class="sim-entity-back" data-sim-entity-back hidden aria-label="Back to cohort">&larr; Back</button>
<button type="button" class="sim-entity-nav" data-sim-entity-prev aria-label="Previous entity (k)" title="Previous entity (k)">&larr;<kbd class="sim-entity-kbd">k</kbd></button>
<button type="button" class="sim-entity-nav" data-sim-entity-next aria-label="Next entity (j)" title="Next entity (j)">&rarr;<kbd class="sim-entity-kbd">j</kbd></button>
<button type="button" class="sim-entity-close" data-sim-entity-close>Close</button>
```

- [ ] **Step 4: Replace the entity-dialog IIFE with unified dispatch**

Replace the current persona/scenario-only IIFE with functions that separate rendering from navigation:

```javascript
function openConversation(trigger, pushCurrent) {
  var url = trigger.getAttribute('data-drawer-url');
  var content = contentNode();
  if (!url || !content || !window.htmx) return;
  if (pushCurrent && activeState) drillStack.push(activeState);
  activeState = { kind: 'conversation', url: url, origin: trigger.parentElement };
  content.innerHTML = '<p class="sim-drawer-loading">Loading conversation…</p>';
  openDialog();
  updateActions();
  window.htmx.ajax('GET', url, { target: content, swap: 'innerHTML', values: formValues() });
}

function openTemplate(kind, id, pushCurrent) {
  var template = document.querySelector('[data-sim-entity-template][data-entity-kind="' + kind + '"][data-entity-id="' + id + '"]');
  if (!template) return;
  if (pushCurrent && activeState) drillStack.push(activeState);
  activeState = { kind: kind, id: id, origin: template.parentElement };
  contentNode().innerHTML = template.innerHTML;
  openDialog();
  updateActions();
}
```

`formValues()` returns `new FormData(document.getElementById('filter-form'))` when the form exists, otherwise `{}`. The endpoint ignores filters for lookup but preserving the form payload maintains the established request shape.

For fresh table-row opens, set `drillStack = []`. For a conversation-list trigger inside a template, call `openConversation(trigger, true)`. The back handler pops the saved state and rerenders it without pushing again. `step(delta)` must query only `activeState.origin.querySelectorAll('[data-sim-entity-trigger][data-entity-kind="' + activeState.kind + '"]')`; this gives j/k DOM-order navigation within the originating table/list for every entity type.

Handle keyboard activation separately from j/k:

```javascript
document.body.addEventListener('keydown', function (evt) {
  var trigger = evt.target.closest('[data-sim-entity-trigger]');
  if (trigger && (evt.key === 'Enter' || evt.key === ' ')) {
    evt.preventDefault();
    activateTrigger(trigger);
    return;
  }
  if (dialogIsOpen() && !isEditable(document.activeElement) && (evt.key === 'j' || evt.key === 'J')) step(1);
  if (dialogIsOpen() && !isEditable(document.activeElement) && (evt.key === 'k' || evt.key === 'K')) step(-1);
});
```

In the click delegate, return immediately when `evt.target.closest('[data-no-drawer]')` is truthy. Delete the separate IIFE that intercepts `a[href^="#conv-"]`; its targets no longer exist.

- [ ] **Step 5: Run dashboard markup tests and a JavaScript syntax check**

Run: `uv run pytest tests/dashboard/test_report_tabs.py tests/dashboard/test_sim_transcript.py -q && node --check src/evaluatorq/dashboard/static/dashboard.js`

Expected: PASS and Node exits 0.

- [ ] **Step 6: Commit drawer interaction code**

```bash
git add src/evaluatorq/dashboard/static/dashboard.js src/evaluatorq/dashboard/report_tabs.py tests/dashboard/test_report_tabs.py
git commit -m "feat(dashboard): add unified sim detail drawer controls"
```

### Task 5: Restyle the native dialog as a responsive right drawer

**Files:**

- Modify: `src/evaluatorq/dashboard/styles.py:1114-1175,1601-1677,1739-1748`
- Test: `tests/dashboard/test_report_tabs.py`

**Interfaces:**

- Consumes: existing `sim-entity-*`, `sim-conv-row`, `sim-transcript-*`, and new `sim-cohort-*` classes.
- Produces: a right-pinned, full-height dialog with a dimmed backdrop, scrollable content, usable mobile width, and visible keyboard focus.

- [ ] **Step 1: Add CSS contract tests**

Add a test that inspects `dashboard_styles()` / the assembled dashboard CSS and asserts the essential declarations are present:

```python
def test_sim_entity_dialog_is_a_right_side_drawer():
    from evaluatorq.dashboard.styles import DASHBOARD_CSS

    css = DASHBOARD_CSS
    assert 'inset: 0 0 0 auto' in css
    assert 'width: min(560px, 92vw)' in css
    assert 'height: 100vh' in css
    assert '.sim-entity-dialog::backdrop' in css
    assert '@media (max-width: 480px)' in css
```

- [ ] **Step 2: Run the CSS contract test and verify failure**

Run: `uv run pytest tests/dashboard/test_report_tabs.py -q`

Expected: FAIL because the dialog is currently centered and capped at 720×780px.

- [ ] **Step 3: Apply drawer, row, and cohort-list styles**

Replace the centered dialog rules with this base geometry:

```css
.sim-report .sim-entity-dialog {
    inset: 0 0 0 auto; margin: 0; width: min(560px, 92vw); height: 100vh; max-height: 100vh;
    padding: 0; border: 0; border-left: 1px solid var(--border-subtle); border-radius: 12px 0 0 12px;
    background: var(--surface-app); color: var(--text-body); box-shadow: var(--shadow-lg);
}
.sim-report .sim-entity-dialog[open] { animation: sim-drawer-in 160ms ease-out; }
@keyframes sim-drawer-in { from { transform: translateX(100%); } to { transform: translateX(0); } }
.sim-report .sim-entity-modal-shell { height: 100%; max-height: none; }
.sim-report .sim-entity-modal-content { flex: 1; min-height: 0; padding: 24px; overflow: auto; }
.sim-report .sim-entity-dialog::backdrop { background: rgb(18 17 15 / 0.42); }
```

Add `.sim-drawer-row` / `.sim-conv-row` hover and `:focus-visible` rules, `cursor: pointer`, and a background transition. Add compact `.sim-cohort-stats`, `.sim-cohort-conversations`, `.sim-cohort-conversation`, and `.sim-cohort-empty` rules. Keep `.sim-transcript-grid` responsive and add a drawer-specific `min-width: 0` to its columns so long chat content wraps rather than creates horizontal scroll.

Add the narrow-screen rule:

```css
@media (max-width: 480px) {
    .sim-report .sim-entity-dialog { width: 92vw; border-radius: 10px 0 0 10px; }
    .sim-report .sim-entity-modal-content { padding: 16px; }
    .sim-report .sim-entity-trait-row { grid-template-columns: 88px minmax(0, 1fr) 36px; gap: 8px; }
    .sim-report .sim-entity-modal-actions { padding: 10px 12px; }
}
```

Delete obsolete `.sim-conv-card`, `.sim-conv-summary`, and `.sim-conv-body` declarations from both transcript CSS blocks; do not leave competing duplicate styles.

- [ ] **Step 4: Run CSS and dashboard tests**

Run: `uv run pytest tests/dashboard/test_report_tabs.py tests/dashboard/test_sim_transcript.py -q`

Expected: PASS.

- [ ] **Step 5: Commit styling**

```bash
git add src/evaluatorq/dashboard/styles.py tests/dashboard/test_report_tabs.py
git commit -m "style(dashboard): present sim details in right drawer"
```

### Task 6: Verify the complete flow and remove obsolete behavior

**Files:**

- Modify: `tests/dashboard/test_report_tabs.py`
- Modify: `tests/dashboard/test_sim_transcript.py`
- Test: `tests/dashboard/test_report_tabs.py`, `tests/dashboard/test_sim_transcript.py`, `tests/simulation/reports/test_sections.py`, `tests/simulation/reports/test_export.py`, `tests/common/reports/test_html_helpers.py`

**Interfaces:**

- Verifies: static exports remain noninteractive; dashboard drawer works from Failures, Transcripts, and both breakdown tables; all old `#conv-N` and `<details>` wiring is gone.

- [ ] **Step 1: Add deletion guards**

Add assertions against generated dashboard HTML and the JavaScript source:

```python
def test_sim_dashboard_no_longer_emits_anchor_or_foldout_drilldown(sim_run):
    html = sim_report_tabs('rid', sim_run)
    assert 'href="#conv-' not in html
    assert 'id="conv-' not in html
    assert '<details class="sim-conv-card"' not in html
    assert 'toggle once from:closest details' not in html


def test_dashboard_runtime_has_no_failure_anchor_handler():
    source = Path('src/evaluatorq/dashboard/static/dashboard.js').read_text()
    assert 'a[href^="#conv-"]' not in source
```

- [ ] **Step 2: Run the full affected test set**

Run:

```bash
uv run pytest \
  tests/simulation/reports/test_sections.py \
  tests/simulation/reports/test_export.py \
  tests/dashboard/test_report_tabs.py \
  tests/dashboard/test_sim_transcript.py \
  tests/common/reports/test_html_helpers.py -q
uv run ruff check src/evaluatorq/simulation/reports/sections.py \
  src/evaluatorq/dashboard/report_tabs.py \
  src/evaluatorq/dashboard/sim_views.py \
  src/evaluatorq/dashboard/view.py \
  src/evaluatorq/dashboard/styles.py \
  src/evaluatorq/simulation/reports/export_html.py
node --check src/evaluatorq/dashboard/static/dashboard.js
```

Expected: all pytest tests pass, Ruff exits 0 for source files, and Node reports no syntax error.

- [ ] **Step 3: Perform the browser acceptance check**

Start the dashboard with a run that has at least two personas, two scenarios, a failed conversation, a reason longer than 90 characters, and one errored conversation. Verify each concrete behavior:

1. Click a Failure row: the right drawer opens with judge reason, bubbles, and color-coded criteria; click-dragging the Why cell does not open it.
2. Press Enter and Space on a focused Failure, Transcript, persona-breakdown, and scenario-breakdown row: each opens its correct entity.
3. Click a persona and scenario breakdown row: profile, three cohort stats, and compact conversation list are shown; an empty filtered cohort shows `No conversations.`.
4. Click a cohort conversation, then Back: the original cohort detail returns. Click j/k in every entity kind and confirm it cycles only the origin table/list.
5. Press Escape and click the dimmed backdrop: both close the drawer. On a viewport under 480px, the drawer and transcript grid have no horizontal overflow.
6. Filter to a conversation whose full-run index is not zero, open it from both Failures and Transcripts, and confirm the transcript matches that row.
7. Download static HTML: Failures has only Scenario, Persona, Why, and Score; no Criteria dots, no conversation link, and the per-conversation export still includes criteria detail.

- [ ] **Step 4: Commit final verification guards**

```bash
git add tests/dashboard/test_report_tabs.py tests/dashboard/test_sim_transcript.py
git commit -m "test(dashboard): guard unified sim drawer behavior"
```

## Self-Review

- Spec coverage: Tasks 2-5 cover all three entity kinds, whole-row activation, a11y, Why selection, lazy conversation loading, j/k navigation, multi-level Back, drawer geometry, mobile layout, and static export cleanup. Task 1 corrects the name-collision defect and Task 2 enforces filtered/full-run index alignment.
- Placeholder scan: no task defers an implementation decision; each changed API, command, and behavior is named.
- Type consistency: the implementation deliberately keeps `SimulationEntry` unchanged; `_stable_entries()` changes only its in-memory `index`. Section-level `id` is a string used by report-tab context and never replaces display `persona` / `scenario` strings.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-20-sim-detail-drawer.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
