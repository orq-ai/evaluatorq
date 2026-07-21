# Unified sim detail drawer

**Date:** 2026-07-20
**Status:** Approved design, ready for implementation plan
**Area:** `src/evaluatorq/dashboard/` (Agent Sim report)

## Problem

The Agent Sim dashboard report has two separate drill-down mechanisms:

1. A centered `<dialog class="sim-entity-dialog">` modal showing persona/scenario
   detail (client-side `<template>` clones, j/k nav), triggered by name-buttons
   in the Breakdown tables and the Config tab.
2. Inline `<details>` expand-cards in the Transcripts tab, plus `#conv-N` anchor
   links from the Failures table that jump to those cards.

They are inconsistent and the Failures table exposes low-signal chrome (a
criteria-dots foldout column, a scenario→anchor link) instead of letting the
user open the full conversation.

## Goal

One **right-side detail drawer** that serves as the detailed view for whichever
entity the user clicks. Whole rows are clickable. The drawer overlays the page
with a dimmed backdrop.

Three entity kinds:

| Kind | Opened from | Body |
|------|-------------|------|
| **conversation** | Failures table row, Transcripts row, or a cohort's conversation-list item | judge callout + chat bubbles + colour-coded criteria |
| **persona** | per-persona Breakdown row | profile (traits/background) + cohort stats + clickable conversation list |
| **scenario** | per-scenario Breakdown row | goal/context/criteria + cohort stats + clickable conversation list |

## Design

### Drawer shell

Reuse the existing `<dialog class="sim-entity-dialog">` and its `showModal()`
call — `<dialog>` already provides a dimmed `::backdrop`, Escape-to-close, and a
focus trap. Restyle from centered modal to right drawer via CSS:

- The UA default centers an open `<dialog>` with `margin: auto; inset: 0`. To
  pin it right, override **all four insets/margins**, not just the left — e.g.
  `inset: 0 0 0 auto; margin: 0;` then `height: 100vh; max-height: 100vh;
  width: min(560px, 92vw);`. (Setting only `margin-left: auto` leaves the UA
  `margin-top/bottom: auto` centering it vertically — the earlier one-line
  snippet was wrong; the implementer must verify the box model, not assume.)
- Left-edge radius; slide-in transform on open.
- Backdrop click and the existing `[data-sim-entity-close]` button close it
  (already wired in `dashboard.js`).

Keep the `sim-entity-*` class/data-attribute names to minimise churn; the
concept is now a drawer, not a modal.

### Entity kinds and loading

- **persona / scenario** — unchanged loading model: hidden `<template>` clones
  rendered once into the page, cloned into the drawer body on trigger. Enriched
  content (see below).
- **conversation** — lazy-loaded. Transcripts are heavy and numerous, so do NOT
  inline them as templates. The trigger carries a `data-drawer-url` pointing at
  the existing `GET /r/{rid}/sim/transcript?idx=` endpoint; the drawer fetches
  via `htmx.ajax` into `[data-sim-entity-content]`, including `#filter-form` so
  `idx` maps into the filtered entry list (same contract the current inline
  expand uses). Body = `render_transcript_fragment` (judge callout + chat
  bubbles + colour-coded criteria) — reused as-is.

### Enriched persona / scenario templates (decision B)

Each persona/scenario template gains, below the existing profile:

- **Cohort stats**: goal rate, avg score, tokens — sourced from the
  `persona_breakdown` / `scenario_breakdown` section rows already in `by_kind`.
- **Conversation list**: that cohort's conversations, each item a conversation
  trigger (`data-entity-kind="conversation"` + `data-drawer-url=…`). Clicking an
  item swaps the drawer body to that transcript and **pushes onto the drill
  stack** (see JavaScript), so **back** returns to the cohort — multi-level, not
  one-shot.

The conversation-list items are cheap (index + score + outcome badge + the
drawer URL), so they render inline in the cohort template. The transcript
bodies stay lazy (hx-get). **This is new grouping work** — `§Data` below is
corrected accordingly; it is not "already computed."

**Grouping by stable id, not display name.** Group entries into cohorts by the
entry's persona/scenario **id/index**, never the display name. Two personas (or
scenarios) can share a display name; keying by name silently merges their
conversations and mis-attributes cohort stats. `persona_ids` / `scenario_ids`
in `_sim_entity_context` are name-keyed dicts with the same collision bug —
build the cohort grouping from the enumerated entry index instead (the same
`persona-{i}` / `scenario-{i}` id the templates already use), and fix or stop
relying on the name-keyed maps for this path.

### Row triggers

- **Failures (dashboard)** — new dashboard-specific renderer. Reads the same
  `failures_first` section rows. Columns: **Scenario · Persona · Why · Score**
  (drop the Criteria column and the scenario anchor link). The whole `<tr>` is a
  conversation trigger; `idx = row["index"] - 1`.
- **Failures (static export)** — the shared `_render_failures_first_html` in
  `export_html.py` **also** drops the criteria-dots foldout column and the
  `#conv-N` scenario anchor (decision 3 — the user's "more broadly" covers the
  downloadable file too). Static columns become **Scenario · Persona · Why ·
  Score**, plain text, no drill (no server → no drawer). Criteria detail there
  still lives in the per-conversation sections the static export already emits.
  This means dashboard and static Failures tables share the same clean column
  set; the dashboard renderer differs only by adding the row trigger.
- **Breakdown tables** — the whole `<tr>` becomes the persona/scenario trigger
  (previously only the name was a `sim-entity-link` button).
- **Transcripts tab** — replace the inline `<details>` expand-cards with a flat
  list of clickable conversation rows. This removes the
  `hx-trigger="toggle once…"` lazy-on-expand code; the drawer's lazy-load
  replaces it. The per-row trace-link button keeps `event.stopPropagation()` so
  it opens the trace without opening the drawer.

**Row-click affordance (all clickable rows).** A `<tr>` is not natively
focusable or keyboard-activatable, and a bare row-click swallows text selection.
So:

- **Keyboard/a11y**: give trigger rows `role="button"` + `tabindex="0"`, and the
  JS click handler also fires on `Enter` / `Space`. (The persona/scenario name
  was previously a real `<button>` — do not lose keyboard access when promoting
  the trigger to the row.)
- **Preserve text selection on the Why cell** (decision 1): the `Why` cell
  carries the judge's reason users will want to copy. Mark it
  `data-no-drawer` (or an equivalent opt-out class); the row-click handler
  ignores clicks whose target is inside a `data-no-drawer` cell, so
  drag-to-select works there. All other cells open the drawer.

### JavaScript (extend the existing IIFE in `dashboard.js`)

- Trigger dispatch: a click (or `Enter`/`Space`) on a `[data-sim-entity-trigger]`
  row, ignoring targets inside `[data-no-drawer]`. If the trigger has
  `data-drawer-url`, `htmx.ajax('GET', url, {target: content, values from
  #filter-form})` and open the drawer; otherwise clone the matching `<template>`
  as today.
- **j/k nav for all kinds** (decision 2): persona/scenario step through their
  templates as today. Conversation kind steps through the **originating list**
  — the set of conversation triggers in the table/cohort the drawer was opened
  from (query the sibling `[data-entity-kind="conversation"]` triggers in that
  container, step by DOM order, load via `data-drawer-url`). Conversation is the
  most-opened kind; it must not be the only one without next/prev.
- **Drill stack** (decision 4): maintain a stack of prior drawer states
  (kind + id/url). Opening a conversation from a cohort pushes; **back** pops.
  Multi-level (cohort → conversation → back → cohort), not one-shot. Opening a
  fresh entity from a table row resets the stack.
- Escape / backdrop / close-button behaviour unchanged (native `<dialog>`).
- Nav chrome is state-dependent: prev/next (j/k) shown for every kind; the
  **back** button shows only when the drill stack is non-empty.

### Data

No new **section builders** — but the cohort conversation lists are **new
grouping work** done at render time (correcting the earlier "already computed"
claim):

- Failures rows: `failures_first` section (carries `index`, `persona`,
  `scenario`, `reason`, `score`) — unchanged.
- Cohort stats: `persona_breakdown` / `scenario_breakdown` sections — unchanged.
- Conversation lists: **new** — group `entries` (`individual_entries`) by
  persona/scenario **id/index** (not display name; see grouping note above).
  Build once and reuse for all cohorts (a single pass keyed by id), not
  per-template re-scans. Only list-item metadata is emitted eagerly; transcript
  bodies stay lazy.

### Edge cases (must be handled)

- **Empty cohort**: a persona/scenario with zero conversations shows the profile
  + cohort stats and an explicit "No conversations" note, not a broken/empty
  list.
- **Filtered `idx` alignment**: worked example — `sim_transcript` indexes into
  `individual_entries(run.results)` and the drawer passes `#filter-form` so the
  server re-derives the same filtered ordering; the conversation `idx` used by
  Failures rows (`index - 1`), Transcripts rows, and cohort list items must all
  come from that **same filtered entry ordering**. Add a test that a filtered
  view's row `idx` resolves to the same conversation the endpoint returns.
- **Mobile width**: the `min(560px, 92vw)` drawer must be usable < 480px;
  verify the transcript grid + cohort list don't overflow horizontally.
- **Error / no-criteria conversations**: drawer body already handled by
  `render_transcript_fragment` (error branch + criteria-optional) — reused
  unchanged; confirm the cohort list still renders their row.

### Deletions (part of this work, not just additions)

- `dashboard.js` — the Failures-anchor drill-down handler (`a[href^="#conv-"]`
  → tab-flip → open `<details>` → scroll, ~line 277) becomes dead once rows are
  drawer triggers. Remove it.
- `sim_views.py` — the `<details class="sim-conv-card" id="conv-{idx+1}">`
  wrapper + its `hx-trigger="toggle once…"` body, replaced by flat rows. The
  `id="conv-N"` anchor target is no longer referenced (anchor dropped in both
  Failures renderers) — remove it.
- `report_tabs.py` — the per-name `_sim_entity_button` wrapping in breakdown
  tables, superseded by the whole-row trigger.

### Files touched

- `dashboard/report_tabs.py` — drawer markup (restyle trigger), enriched
  persona/scenario templates (cohort stats + conversation list), whole-row
  breakdown triggers, new dashboard Failures renderer wired into
  `_sim_breakdown`.
- `dashboard/sim_views.py` — Transcripts row list → flat clickable rows (drop
  `<details>`/toggle + `id=conv-N`); `render_transcript_fragment` reused
  unchanged for the drawer body.
- `dashboard/static/dashboard.js` — conversation-kind dispatch, j/k for
  conversations, drill stack + back button, keyboard activation; **remove** the
  dead Failures-anchor handler.
- `simulation/reports/export_html.py` — static Failures renderer drops the
  criteria-dots column + `#conv-N` anchor (decision 3).
- `common/reports/report.css` — drawer styling (centered `<dialog>` → right
  drawer); `.fail-why` already present.

## Out of scope

- Breakdown static-export **tables** (persona/scenario breakdowns) — only the
  Failures static table is cleaned up here.
- Breakdown-row aggregate charts inside the drawer.
- Deep drill history beyond the cohort→conversation chain (the stack handles
  the chain; no cross-entity history / URL routing).

## Testing

- Unit: new dashboard Failures renderer emits `Scenario · Persona · Why · Score`
  columns, no Criteria/anchor, whole-row conversation trigger with correct
  `idx`. Enriched persona/scenario templates include cohort stats + conversation
  triggers. Transcripts row list emits clickable conversation rows (no
  `<details>`/toggle).
- Existing `render_transcript_fragment` and transcript endpoint tests unchanged
  (reused as-is).
- Manual: open drawer from each of the three surfaces; backdrop dim, Esc,
  backdrop-click close; cohort→conversation→back; filter round-trip keeps
  `idx` aligned.
