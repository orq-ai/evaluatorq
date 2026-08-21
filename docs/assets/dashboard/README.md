# Dashboard screenshots

Persistent copies of the `eq dashboard` screenshots produced for
[DOC-653](https://linear.app/orqai/issue/DOC-653) (red teaming) and
[DOC-655](https://linear.app/orqai/issue/DOC-655) (agent simulation). They were
posted as Linear comments first; Linear's upload URLs are signed and expire, so
the files live here instead. Docs pages and the README should reference these
paths, not the Linear links.

Captured against the FastHTML dashboard at `main @ fcb525a6`, 1440×900, light
theme, so the whole set reads consistently side by side.

## Red teaming — `redteam-*.png`

Source run: hybrid red team, 2 target agents (one deliberately vulnerable, one
hardened), 40 attacks across 10 OWASP ASI/LLM categories, `gemini-2.5-flash`,
LLM-as-judge scoring. 78% resistance, 9 vulnerabilities, 3 critical — real
findings rather than an all-green wall.

| File | Page |
|---|---|
| `redteam-01-landing.png` | Dashboard landing — jobs, spend, tokens, run mix, findings by severity |
| `redteam-02-run-list.png` | Red Team run list |
| `redteam-03-overview.png` | Report → Overview — executive summary, ASR/resistance, outcome donut, per-agent success |
| `redteam-04-agents.png` | Report → Agents — per-agent ASR, model, tools, skills, knowledge |
| `redteam-05-focus-areas.png` | Report → Focus areas — fixes ranked by `risk = success rate × avg severity` |
| `redteam-06-breakdowns.png` | Report → Breakdowns — attack success per OWASP category |
| `redteam-07-attacks.png` | Report → Attacks — all 40 attacks, filterable |
| `redteam-08-attack-detail.png` | Report → Attack detail — evaluator verdict plus full transcript (memory poisoning) |
| `redteam-09-usage.png` | Report → Usage — token spend per agent |
| `redteam-10-config.png` | Report → Config — pipeline, framework, scoring, severity weights |

## Agent simulation — `sim-*.png`

Source run: `sim:refund-agent-fixed` — 10 personas × 5 scenarios = 50
conversations against a refund agent, judged per turn. 74% goal completion, mean
score 0.86, with one scenario column that clearly fails.

| File | Page |
|---|---|
| `sim-01-landing.png` | Dashboard landing, agent sim alongside red team |
| `sim-02-run-list.png` | Agent Sim run list — goal completion, avg turns, cost per sim |
| `sim-03-overview.png` | Report → Overview — best/worst persona × scenario, quality metrics |
| `sim-04-breakdown-heatmap.png` | Report → Breakdown — the persona × scenario heatmap |
| `sim-05-transcripts.png` | Report → Transcripts — all 50 conversations |
| `sim-06-conversation-detail.png` | Report → Conversation detail — criteria, judge rationale, transcript |
| `sim-07-turn-quality.png` | Report → Turn quality — per-turn trend, turn-count distribution |
| `sim-08-config.png` | Report → Config — run metadata and the persona dials |
| `sim-09-compare.png` | Run comparison — same agent before vs after a fix |

## Pending re-shoot

Five shots do not show the thing their caption teaches. The decision is to
re-shoot the set in one purpose-built capture session rather than keep
documenting around it. Until then the four that exist are **unreferenced** —
their `![...]` lines were removed from
[`docs/dashboard.md`](../../dashboard.md); the prose stayed. Re-add each one in
the same commit that replaces the file, not before. The prose caveats standing
in for them are the only thing keeping the pages honest, so **do not delete
them either**.

| Shot | What's wrong | What the capture run needs |
|---|---|---|
| `redteam-05-focus-areas.png` | Predates the apply flow — no **Apply…** buttons or apply bar | A single-agent run against a real Orq agent, recommendations enabled; capture the preview drawer with its diff too |
| *(missing)* sim Recommendations | Tab landed after capture; no shot exists | Any sim run that generated recommendations — the tab drops out otherwise |
| `sim-05-transcripts.png` | **TRACES** column empty | `ORQ_WORKSPACE=<slug>` set when launching `eq dashboard` |
| `sim-07-turn-quality.png` | Judge ended most conversations after one turn, so the trend spans two points | Scenarios whose goals need several exchanges, and a higher `max_turns` |
| `sim-09-compare.png` | Red "low overlap (0%)" banner — the two runs share no (persona, scenario) pairs | Two runs replayed from the same datapoints file, with **distinct run names** (the compare picker shows names only, no timestamps) |

The screenshots that do show a report tab strip (`sim-03` … `sim-08`) are one
tab short of the current dashboard, since Recommendations sits between
Breakdown and Transcripts now. `sim-01`, `sim-02` and `sim-09` are outside a
report and unaffected.

To re-shoot: run `eq dashboard` in a checkout with runs under
`.evaluatorq/runs/` and `.evaluatorq/sim-runs/`, 1440×900, light theme, to
match the existing set.
