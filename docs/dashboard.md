# Dashboard

evaluatorq ships a built-in web dashboard for browsing red team and simulation
reports.  It is powered by **FastHTML** (a lightweight Python web framework)
and served locally via **uvicorn**.  There is no external service dependency —
everything runs on your machine.

## Install

The dashboard is an optional extra (it pulls in `python-fasthtml` and
`uvicorn`):

```bash
pip install "evaluatorq[dashboard]"
# or — if you already have the redteam / simulation extras:
pip install "evaluatorq[redteam,dashboard]"
```

With `uv`:

```bash
uv add "evaluatorq[dashboard]"
```

## Launch

The `eq` CLI exposes three convenience sub-commands:

| Command | What it opens |
|---|---|
| `eq ui [PATH]` | Generic launcher — serves any directory you point it at, or the combined default stores |
| `eq redteam ui` | Serves `.evaluatorq/runs` (red team reports only) |
| `eq sim ui` | Serves `.evaluatorq/sim-runs` (simulation reports only) |

### `eq ui` — generic launcher

```bash
# Browse both default stores at once (red team + simulation)
eq ui

# Point at a custom directory containing exported JSON reports
eq ui /path/to/my/reports

# Specify host / port
eq ui --host 0.0.0.0 --port 8888
```

### `redteam ui` — red team reports

```bash
evaluatorq redteam ui
# or
eq redteam ui
```

Serves `.evaluatorq/runs` from the current working directory (the same
directory that `red_team()` / `eq redteam run` write reports to).

### `sim ui` — simulation reports

```bash
evaluatorq sim ui
# or
eq sim ui
```

Serves `.evaluatorq/sim-runs` from the current working directory.

---

## What the dashboard browses

The dashboard auto-discovers JSON report files in the configured root
directories:

| Default store | Written by |
|---|---|
| `.evaluatorq/runs/*.json` | `red_team()` / `eq redteam run` |
| `.evaluatorq/sim-runs/*.json` | `eq sim run` (auto-saves unless `--no-save`); `simulate()` only when called with `save=True` |

Files are identified by a content-hash of their absolute path — the URL for
each report is stable for the lifetime of the file.

### Supported surfaces

| Surface | JSON discriminator | Rendered by |
|---|---|---|
| Red team | `"pipeline"` key present | `redteam/reports/export_html.py` |
| Simulation | `"mode"` key present (`mode` wins over `pipeline`) | `simulation/reports/export_html.py` |

Files that cannot be parsed (invalid JSON) are silently skipped.  Files that
parse but fail model validation appear in the index as **broken cards** with an
error badge; their detail page shows a non-fatal error message instead of a
traceback.

---

## Report index (GET /)

The index lists all discovered reports sorted by creation time (newest first).
Each card shows:

- Surface type (Red Team / Simulation)
- Report name / description
- Creation timestamp
- Summary headline (attack count or conversation count)
- Error badge when the report JSON is partially valid

Clicking a card opens the embedded report view.  The **export** link on each
card downloads the standalone self-contained HTML for offline sharing.

---

## Filters

Both surfaces expose dimension filters in a sidebar:

### Red team filters (7 dimensions)

| Dimension | Values |
|---|---|
| `result` | VULNERABLE / RESISTANT |
| `severity` | critical / high / medium / low / info |
| `category` | framework category codes (ASI01, LLM01, …) |
| `vulnerability` | vulnerability enum values |
| `attack_technique` | technique identifiers |
| `delivery_method` | delivery method identifiers |
| `source` | dataset source identifiers |

### Simulation filters (4 dimensions)

| Dimension | Values |
|---|---|
| `goal_outcome` | achieved / not achieved |
| `persona` | persona names present in the run |
| `scenario` | scenario names present in the run |
| `evaluator` | evaluator names present in the run |

Filters are applied via HTMX (no page reload).  The report body, summary
aggregates, and download links all update in-place to reflect the active
filter state.

---

## Interactive views (red team)

The red team surface exposes four dashboard-only interactive panels alongside
the static report body:

1. **Interactive breakdown** — pick a group-by and stack-by dimension (7 × 7
   combinations); attack-success rate recomputed per (group, stack) cell.
2. **Agent heatmap** — select the pivot dimension (vulnerability / category /
   technique / severity) for the agent × dimension ASR heatmap.
3. **Conversation viewer** — drill into the full message-by-message transcript
   for any individual attack (system / user / assistant / tool messages plus
   evaluator explanation).
4. **Disagreement viewer** — for multi-agent runs, select any agent pair and
   page through attacks where their results differ (side-by-side transcripts).

### Simulation transcript viewer

Simulation reports expose a conversation transcript panel: select any
conversation entry from the run to see the full multi-turn exchange between the
simulated user and the target agent.

---

## Downloads

Every report page includes a download sidebar with export links:

| Format | Red team | Simulation |
|---|---|---|
| HTML (standalone, self-contained) | yes | yes |
| Markdown | yes | — |
| CSV (filtered result rows) | yes | — |
| JSON (filtered result rows) | yes | yes |

Download links respect the currently active filter state — the CSV/JSON exports
contain only the rows visible in the filtered report body.
