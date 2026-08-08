# Coverage axes and impossible combinations

The interaction surface of evaluatorq, expressed as axes. `docs-coverage` crosses
these pairwise and checks each meaningful pair appears in prose.

**Hand-maintained.** Axis *names* live here; axis *values* are pulled from source at
run time, so a new mode or backend appears without editing this file. Edit this file
only when a genuinely new **dimension** appears — a new entry point, a new surface, a
new kind of thing a user chooses between.

If this file rots, `docs-coverage` reports stale gaps and people stop reading it.
Treat it as part of the public surface: changing it belongs in the same PR as the
feature that changed it.

## Axes

| Axis | Where values come from | Notes |
|---|---|---|
| **entry point** | `evaluatorq.__all__` + `evaluatorq.simulation.__all__` + `evaluatorq.redteam.__all__` | `evaluatorq()`, `red_team()`, `simulate()`, `generate_and_simulate()`, `wrap_simulation_agent()`, pairwise `build_report()`, `deployment()` / `invoke()` |
| **surface** | fixed | Python API · CLI (`eq`) · dashboard (`eq dashboard` / `eq ui`) |
| **target kind** | `_BACKEND_REGISTRY` in `redteam/backends/registry.py` + CLI `--target` prefixes | `agent:<key>`, `deployment:<key>`, direct OpenAI backend, custom `AgentTarget` / `CallableTarget` |
| **mode** | `--mode` on `eq redteam run` | `dynamic`, `static`, `hybrid` |
| **data source** | `evaluatorq()` / `red_team()` dataset params | inline `DataPoint`s, ORQ dataset id, HuggingFace dataset, generated |
| **evaluator kind** | `VULNERABILITY_EVALUATOR_REGISTRY`, `SIMULATION_EVALUATORS`, pairwise types | built-in scorer, LLM jury, pairwise jury, custom `Evaluator` |

## Impossible or meaningless combinations

Marked `N/A` in the matrix, never reported as a gap.

| Pair | Why |
|---|---|
| pairwise jury × `mode` | `--mode` is red-team only |
| pairwise jury × target kind | operates on collected outputs, not a live target |
| `deployment()` / `invoke()` × `mode` | deployment invocation has no red-team mode |
| `deployment()` / `invoke()` × evaluator kind | it fetches a response; scoring is a separate step |
| dashboard × data source | the dashboard reads run artifacts from disk; it does not select a dataset |
| dashboard × evaluator kind | renders scores, does not choose evaluators |
| `wrap_simulation_agent()` × data source | wraps a target for reuse; does not consume a dataset |
| `generate_and_simulate()` × data source `ORQ dataset id` | generation *is* the data source |
| `static` mode × generated data source | static mode consumes a fixed dataset by definition |
| CLI × custom `AgentTarget` | custom targets are constructed in Python; the CLI resolves string identifiers |

## Tiers

**Tier 1 — needs prose.** Top-level `evaluatorq.__all__` entry points, every CLI
command and subcommand, every env var. The generated API reference does **not**
count: a docstring is not discovery.

**Tier 2 — API reference suffices.** Supporting types, contracts, backends, and
subpackage `__all__` members. Flag only when there is no docstring at all.
