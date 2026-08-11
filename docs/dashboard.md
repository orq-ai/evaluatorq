# Dashboard

!!! note "Primary UI — FastHTML `eq dashboard`"
    The combined `eq dashboard` documented here is the primary way to browse
    saved runs. Its canonical invocation scans a run directory and opens the
    multi-run FastHTML UI — `eq dashboard` (no path) browses both default stores,
    and `eq dashboard .evaluatorq/sim-runs` scopes to simulation. Passing a single
    JSON report file is an optional direct deep-link to that report. The older
    `eq redteam ui` / `eq sim ui` remain callable as deprecated legacy Streamlit
    commands (see the [CLI Reference](cli-reference/overview.md)).

evaluatorq ships a built-in web dashboard for browsing red team and simulation
reports.  It is powered by **FastHTML** (a lightweight Python web framework)
and served locally via **uvicorn**.  There is no external service dependency —
everything runs on your machine.

![The evaluatorq combined dashboard — stat band, runs by type, attack-resistance, findings by severity, token usage, and recent runs.](assets/dashboard-index.png){ .dashboard-shot }

## Install

The dashboard is an optional extra (it pulls in `python-fasthtml` and
`uvicorn`):

```bash
uv add "evaluatorq[dashboard]"
# or — if you already have the redteam / simulation extras:
uv add "evaluatorq[redteam,dashboard]"
```

Prefer pip? Use `python -m pip install "evaluatorq[dashboard]"`, which installs
into the interpreter you just named rather than whichever `pip` happens to be
first on your `PATH`.

## Launch

Launch it with `eq dashboard` (the `evaluatorq` and `eq` entry points are
interchangeable):

```bash
# Canonical — browse both default stores at once (red team + simulation)
eq dashboard

# Canonical — scope to the simulation run store
eq dashboard .evaluatorq/sim-runs

# Scope to any directory of exported reports
eq dashboard /path/to/my/reports

# Optional direct deep-link — open a single report file
eq dashboard .evaluatorq/runs/red-team_20260626_143024.json

# Bind a custom host / port (default 127.0.0.1:8080)
eq dashboard --host 0.0.0.0 --port 8888

# Enable “View Traces” links in reports. Use the workspace slug from the
# Orq URL (https://my.orq.ai/<workspace>/...), not an API key or workspace UUID.
ORQ_WORKSPACE=orq-research eq dashboard
```

| Invocation | What it scans |
|---|---|
| `eq dashboard` | Both default stores: `.evaluatorq/runs` (red team) and `.evaluatorq/sim-runs` (simulation) |
| `eq dashboard <dir>` | Only that directory (e.g. `eq dashboard .evaluatorq/sim-runs`) |
| `eq dashboard <file>.json` | Optional direct deep-link; prints that report's direct URL so you land straight on it |
| `eq redteam ui` / `eq sim ui` | Deprecated legacy Streamlit views, scoped to a single surface (see the note below) |

With no `PATH` the server prints the local URL to open. Pointing at a directory
(`eq dashboard .evaluatorq/sim-runs`) scopes the UI to that store. Passing a
single JSON report file is an optional direct deep-link that prints that
report's direct URL.

### Orq trace links

Set `ORQ_WORKSPACE` when launching the dashboard to show **View Traces** links
for conversations and runs. Its value is the workspace slug in the Orq UI URL;
for example, `https://my.orq.ai/orq-research/traces` uses
`ORQ_WORKSPACE=orq-research`. It is configured explicitly and is not derived
from `ORQ_API_KEY`. If it is unset, trace-link buttons are hidden.

`ORQ_WORKSPACE_SLUG` remains supported as an alias. For a self-hosted or
staging Orq UI, set `ORQ_UI_BASE_URL` as well; otherwise the dashboard uses
`ORQ_BASE_URL`, then `https://my.orq.ai`.

!!! note "Deprecated legacy Streamlit views"
    `eq redteam ui` and `eq sim ui` are deprecated legacy Streamlit commands,
    scoped to a single surface. The FastHTML `eq dashboard` documented here is
    the primary UI that browses both surfaces together (`eq dashboard` for both
    stores, `eq dashboard .evaluatorq/sim-runs` for simulation). See the
    [CLI Reference](cli-reference/overview.md).

---

## What the dashboard browses

The dashboard auto-discovers JSON report files in the configured root
directories:

| Default store | Written by |
|---|---|
| `.evaluatorq/runs/*.json` | `red_team()` / `eq redteam run` |
| `.evaluatorq/sim-runs/*.json` | `eq sim run` (auto-saves unless `--no-save`); `simulate()` only when called with `save=True` |
| `.evaluatorq/pairwise-runs/*.json` | `PairwiseRun.save()` (see [Pairwise Judging](pairwise-judging.md#saving-a-run-and-viewing-it-in-the-dashboard)) |

Each report gets a stable URL for the lifetime of its file, so links you share
keep working.

### Supported surfaces

| Surface | JSON discriminator | Rendered by |
|---|---|---|
| Red team | `"pipeline"` key present | `redteam/reports/export_html.py` |
| Simulation | `"mode"` key present (`mode` wins over `pipeline`) | `simulation/reports/export_html.py` |
| Pairwise | `"judging"` key present | `pairwise_reports/export_html.py` |

Files that cannot be parsed (invalid JSON) are silently skipped.  Files that
parse but fail model validation appear in the index as **broken cards** with an
error badge; their detail page shows a non-fatal error message instead of a
traceback.

---

## Landing (GET /)

`GET /` opens the combined dashboard: a stat band (total runs, per-surface
counts, attack resistance), runs-by-type and attack-resistance breakdowns,
findings by severity, token usage, and a **recent runs** list across both
stores.  The left sidebar switches surface — **Red Team** and **Agent Sim**
open filtered run lists at `?surface=…`, sorted by creation time (newest
first).  Each run row drills into its report view; reports whose JSON is only
partially valid surface an error badge instead of a traceback.  The **export**
action on a report downloads the standalone self-contained HTML for offline
sharing.

### How attack resistance is counted

Resistance is **attack-weighted**: the rate is resisted attacks over *evaluated*
attacks, not over every attack attempted.  An attack only counts as evaluated
once a judge returned a verdict — an attack whose target call failed, or whose
judge crashed, timed out or was skipped, is excluded from both sides of the
ratio rather than counted as resisted.  So a run headlined `100 attacks` can
show a rate measured over 60; the Score tooltip names both numbers.

Older reports predate the summary fields this rolls up (`evaluated_attacks`,
`by_severity`, `token_usage_total`), so the dashboard derives their counts,
severities and token usage from the stored per-attack results instead of
dropping them from the totals.  Where that derivation lands on a different rate
than the one recorded in the report itself — usually because the recorded rate
was computed over a wider denominator — the Score cell is marked with `*` and
the tooltip gives the recorded value, so a row can always be reconciled against
that run's own exported report.

!!! note "Rates may shift for existing runs"

    Attacks with no judge verdict were previously counted as resisted.  They are
    now excluded, so red-team runs recorded before this change can read lower on
    the dashboard than they did before, and lower than their own exported HTML
    report.  Nothing about the runs changed — only what the dashboard counts as
    a measured attack.  Re-exporting a report regenerates it under the current
    rule.

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

### Simulation filters

The sim rail exposes chip toggles, `<details>` dropdowns, and range
controls, rendered directly in the sidebar wherever they always apply and
tucked behind a **More filters** expander (`filter-dd-more`, reusing the
same open-state persistence as the red team rail) when they may be
unavailable for a given run:

| Control | Kind | Direction | Notes |
|---|---|---|---|
| `goal_outcome` | chip (2-value) | — | achieved / not achieved; zero or both selected means "all" |
| `terminated_by` | chip | — | termination reasons present in the run |
| `persona` | dropdown | — | persona names present in the run |
| `scenario` | dropdown | — | scenario names present in the run |
| `rule_broken` | chip (opt-in) | — | `yes` narrows to results with any broken rule; absent (default) shows all |
| `max_goal_score` | range | ceiling (`≤`) | `goal_completion_score`, step `0.05`, range `0..1` |
| `min_turns` | range | floor (`≥`) | raw `turn_count`, step `1`, max = the run's actual longest conversation (never normalized) |
| `min_total_tokens` *(More)* | range | floor (`≥`) | raw `token_usage.total_tokens`, step `1`, max = the run's actual highest token count; hidden when the run recorded zero tokens |
| per-turn metric thresholds *(More)* | range | floor for hallucination risk (`≥`), ceiling for response quality / tone appropriateness / factual accuracy (`≤`) | step `0.05`, range `0..1`; each control is rendered **only when that metric was actually scored somewhere in the run** — unavailable metrics are hidden, not shown disabled |

Every range control renders its default bound numerically (`≤`/`≥ value`) —
when unset it shows the no-op end of the range, so it always reads as a number,
never "all". Min-turns additionally shows the run's max turns beside the
readout. Metric thresholds compare
against a result's **worst scored turn** (`max()` for hallucination risk,
`min()` for the quality metrics); a result with **no scored turns for that
metric stays visible** regardless of the threshold, so unscored results are
never silently dropped from the filtered view.

Filters are applied via HTMX (no page reload).  The report body, summary
aggregates, and download links all update in-place to reflect the active
filter state.

![A red team report filtered by the sidebar dimension filters.](assets/dashboard-redteam-filtered.png){ .dashboard-shot }

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

### Apply recommendations to the agent (red team)

When a **single-agent** run was executed with `generate_recommendations=True`,
the **Focus areas** tab lists each area's actionable recommendations and an apply bar
showing how many are still pending. Every pending recommendation carries its
own **Apply…** button; the bar's **Preview & apply all…** takes the whole
pending set at once. Either way the recommendations are folded into the tested
agent's instructions with an LLM (the agent is only read at this point) and a
right-hand drawer opens with the breakdown: for a single recommendation, its
focus area (priority tier, risk score, traces analyzed, observed patterns),
the recommendation being merged, and a colorized diff of the instructions
change.
Nothing is written until you click **Apply to agent** in the drawer, which
writes the previewed instructions back as a new **minor agent version** —
exactly what you saw in the diff, no second LLM call — and records the applied
recommendations on the report, so applied bullets show a ✓ tick and a later
preview skips them.

The merge model is a dashboard config setting: `EVALUATORQ_APPLY_MODEL`
(default `gpt-5.6-luna`), shown on the Settings page next to the other
runtime config.

Requirements: `ORQ_API_KEY` in the dashboard's environment and the `orq`
extra (`orq-ai-sdk`) installed. Multi-agent runs do not show the flow: they
are aimed at comparing agents, and the recommendations render as plain
bullets there. Without them the drawer explains what is
missing instead of failing. The same flow is available programmatically via
`evaluatorq.redteam.reports.apply.apply_recommendations`.

### Simulation transcript viewer

Simulation reports expose a conversation transcript panel: select any
conversation entry from the run to see the full multi-turn exchange between the
simulated user and the target agent.

![The dashboard conversation transcript viewer, message by message.](assets/dashboard-transcript.png){ .dashboard-shot }

### Pairwise comparison view

Pairwise runs render three sections: the consensus win rate for each side, a
per-judge table (win rates, tie rate, position bias), and the comparison list.

![A pairwise run: consensus win rates per side, the per-judge table, and the comparison list.](assets/dashboard-pairwise.png){ .dashboard-shot }

Each comparison row expands to the two responses side by side with every
judge's vote and rationale.  Rows where the judges split, or where the panel
could not decide, are marked; those are the ones worth opening.

![Two expanded comparisons: a split panel and one the panel could not decide.](assets/dashboard-pairwise-comparison.png){ .dashboard-shot }

The judge table flags a judge whose position bias reaches 0.15.  A judge that
contradicts itself across the two orderings has no real preference, so its
votes are noise.  The column reads `n/a` rather than `0.00` wherever nothing
was flippable, since there was no measurement to make.

Whether swapping happened is read from the votes rather than from the saved
`swap` flag.  A vote is only marked complete when both orderings landed, so the
data settles it and a run saved with the default `swap=True` but executed
single-ordering is labelled `on (never observed)` instead of a bare `on`.  The
distinction carries into the `n/a` tooltips: when no judge in the run completed
a pair that is a run-level fact, and the table says so rather than blaming each
judge in turn.

In the run lists, a pairwise run scores as its **decided rate** — the share of
comparisons the panel could call, or `1 - inconclusive_rate`.  Mean inter-judge
agreement reads like the more natural choice but is a modal vote share, so it
is quantized by panel size: against the shared `≥ 0.80` threshold it silently
means "unanimous" for three judges and "four of five" for five, and it is
undefined for a single-judge run.  Every surface's Score column names its own
metric on hover.

### Additional red team charts

Beyond the four panels above, the red team surface recomputes several charts
live that the static exported report does not carry:

- **Cumulative discovery curve** — vulnerabilities found as a function of
  conversation turn depth.
- **Attack-failure treemap** — vulnerability → technique, sized by attack count.
- **Token histograms** — prompt and completion token distributions per attack.
- **Vulnerability × severity** — a cross-join stacked bar.

---

## Downloads

Every report page includes a download sidebar with export links:

| Format | Red team | Simulation | Pairwise |
|---|---|---|---|
| HTML (standalone, self-contained) | yes | yes | yes |
| Markdown | yes | yes | — |
| CSV (filtered result rows) | yes | — | yes |
| JSON (filtered result rows) | yes | yes | yes |

The pairwise CSV writes one row per comparison: the question, the consensus
winner, and each judge's vote as its own column, all resolved to the run's side
labels rather than the bare `A` / `B` slot letters.

Download links respect the currently active filter state — the CSV/JSON exports
contain only the rows visible in the filtered report body.

## Where to next

- **[Red Teaming](guides/red-teaming.md)** — generate the red-team reports the dashboard browses.
- **[Agent Simulation](guides/agent-simulation.md)** — generate simulation reports.
- **[CLI Reference](cli-reference/overview.md)** — the `eq dashboard` command and options.
