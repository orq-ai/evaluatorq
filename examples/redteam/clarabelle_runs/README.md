# Clarabelle red-team runs

Real `eq redteam run` reports against Clarabelle, a demo orq.ai agent that is told to be a cow and
nothing else. Every run targets goal hijacking (OWASP ASI01): the attacker tries to get her to drop
the cow persona and do something else. They come from the "Red Team Your Agent" talk at AICA Summit
2026.

Each file is a full report: every attack, every agent response, the judge verdict and token counts.
Internal orq handles (workspace, project, deployment, evaluator and trace IDs, `my.orq.ai` URLs) are
replaced with `[redacted]`.

Open them in the dashboard:

```bash
eq dashboard examples/redteam/clarabelle_runs
```

| File | Agent | Attacks | Hijacked | Resisted |
|---|---|---:|---:|---:|
| `smoke_20260606_160449.json` | clarabelle-cow | 2 | 0 | 100% |
| `Clarabelle-Goal-Hijacking_20260606_161537.json` | clarabelle-cow | 3 | 0 | 100% |
| `Clarabelle-Goal-Hijacking_20260606_162101.json` | clarabelle-cow | 3 | 0 | 100% |
| `Clarabelle-Goal-Hijacking_20260606_165005.json` | clarabelle-cow | 20 | 2 | 90% |
| `Clarabelle-Goal-Hijacking_20260606_190842.json` | clarabelle-cow | 9 | 0 | 100% |
| `Clarabelle-Goal-Hijacking_20260606_191000.json` | clarabelle-cow | 9 | 0 | 100% |
| `Clarabelle-Goal-Hijacking_20260606_191633.json` | clarabelle-cow | 9 | 0 | 100% |
| `Clarabelle-Goal-Hijacking_20260607_123646.json` | clarabelle-cow | 40 | 4 | 90% |
| `red-team_20260608_224502.json` | clarabelle-cow | 3 | 0 | 100% |
| `red-team_20260609_191716.json` | clarabelle-cow | 3 | 0 | 100% |
| `Clarabelle-Goal-Hijacking_20260616_225858.json` | clarabelle-cow (gpt-5.4-mini) | 100 | 11 | 89% |
| `Clarabelle-Goal-Hijacking_gpt-6-luna_20261124_104517.json` | clarabelle-cow-gpt-6-luna | 100 | 12 | 88% |
| `Clarabelle-Goal-Hijacking_gpt-6-1-terra_20261124_104348.json` | clarabelle-cow-gpt-6-1-terra | 100 | 2 | 98% |

The last three are the 100-attack campaigns from the talk. The two November runs used
`google/gemini-3.5-flash` as the attacker and `openai/gpt-5.4` as the judge; only Clarabelle's model
differs between them. The run files themselves don't record the attacker or judge model (see
RES-1964), so that comes from the talk notes. The smaller June runs are development and smoke runs.
