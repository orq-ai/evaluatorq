# Redteam: sanctioned dashboard-only views

The redteam static report is rendered from `redteam/reports/sections.py`
(`build_report_sections` — 19 section kinds). The following views are
**interactive** and do not map onto the static `ReportSection` model. They are
**sanctioned dashboard-only** computations: they live in the FastHTML
`dashboard/` package (Plan C), recomputed per request, and are **not** promoted
to `sections.py` section kinds.

Ruling per the FastHTML dashboard spec
(`docs/superpowers/specs/2026-06-22-fasthtml-dashboard-design.md`, build-order
step 2): the canonical static risk formula is *average* severity weighting
(`sections.py._compute_risk_score`); the dashboard's old *dominant*-severity
focus-area ranking is dropped with the Streamlit UI.

## Dashboard-only interactive views (re-implemented in `dashboard/`)

1. **Interactive breakdown** — user picks group-by and stack-by dimensions
   (7×7 combinations); ASR recomputed per (group, stack) cell. No static
   equivalent.
2. **Agent-heatmap dimension selector** — user picks the pivot dimension
   (vulnerability / category / technique / severity); the agent×dimension ASR
   pivot recomputes. The static report ships the fixed
   vulnerability×technique `attack_heatmap` and the vulnerability×agent
   `agent_comparison` heatmap; the selectable-dimension pivot is dashboard-only.
3. **Conversation viewer** — per-row transcript drill-down (system/user/
   assistant/tool messages, evaluator explanation). The static report ships
   the `individual_results` section (prompt/response/explanation per attack);
   the full message-by-message transcript viewer is dashboard-only.
4. **Disagreement viewer** — agent-pair selector + pagination + side-by-side
   transcripts. The static report ships the `agent_disagreements` section (the
   disagreement list); the interactive pair-viewer is dashboard-only.

## Dashboard-only chart derivations (not promoted to static sections in v1)

- Turn-depth **cumulative** vulnerability-discovery curve (the static
  `turn_depth_analysis` section carries per-turn rates, not the cumulative
  curve).
- Attack-failure **treemap** (vulnerability → technique).
- Per-attack **token histograms** (prompt / completion distributions).
- Vulnerability × severity **cross-join** stacked bar.
