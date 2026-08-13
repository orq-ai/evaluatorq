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

Known issue with `sim-09-compare.png`: the header carries a red "low overlap
(0%)" banner because the two runs do not share identical (persona, scenario)
pairs, so per-conversation matching finds nothing. The KPI deltas are still
correct — crop the banner, or re-shoot against two runs generated from the same
persona/scenario set.

To re-shoot: run `eq dashboard` in a checkout with runs under
`.evaluatorq/runs/` and `.evaluatorq/sim-runs/`.
