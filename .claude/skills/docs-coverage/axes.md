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
| **mode** | `--mode` on `eq redteam run`, **plus `replay`** (see below) | `dynamic`, `static`, `hybrid`, `replay` |
| **data source** | `evaluatorq()` / `red_team()` dataset params | inline `DataPoint`s, ORQ dataset id, HuggingFace dataset, generated |
| **evaluator kind** | `VULNERABILITY_EVALUATOR_REGISTRY`, `SIMULATION_EVALUATORS`, pairwise types | built-in scorer, LLM jury, pairwise jury, custom `Evaluator` |
| **reasoning-effort scope** | fixed (see below) | target under test · pipeline attacker/judge · simulator's own calls |

### `reasoning-effort scope` is a choice, not a value

Three settings carry the words "reasoning effort" and they apply to three
different models: `target_reasoning_effort` (the agent under test),
`LLMCallConfig.reasoning_effort` on `attacker=` / `evaluator=` (red team's own
calls), and `EVALUATORQ_REASONING_EFFORT` (the simulator's user-simulator and
judge). Picking the wrong one is silent — the call just runs at the default — so
this is a dimension a user chooses along, not a tuning number.

Values are fixed here because no registry enumerates them; the *accepted efforts*
per model come from the model catalogue and are a different thing entirely.

### `replay` is a mode value that no enum contains

`--mode` accepts only `dynamic`, `static` and `hybrid` (validated as a plain
string in `redteam/runner.py`). Replay is reached by a *different* flag,
`--from-run`, which is explicitly **incompatible** with `--mode` — it re-runs a
previous run's exact attacks, so only the target and models may differ.

Behaviourally that makes replay a fourth mode: it is the same choice a user makes
at the same point, expressed through another flag. Source-derived values can
never see it, so it is hardcoded here on purpose. **Do not resolve this by adding
`replay` to the accepted `--mode` values in code** — that would be a public
surface change, and it contradicts the two flags being mutually exclusive.

Replay exists on both `eq redteam run` and `eq sim simulate`; treat it as a mode
value for both when building the matrix.

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
| Python-only entry points × surface `CLI` | `build_report()`, `wrap_simulation_agent()`, `deployment()` / `invoke()` have no `eq` command |
| `evaluatorq()` × target kind | it scores rows you already have; the target is your own task function |
| dashboard × target kind | it reads saved artifacts; no target is invoked |
| dashboard × entry points that write no artifacts | nothing lands in the run store, so there is nothing to browse |
| `red_team()` × target kind `Vercel` | Vercel AI SDK agents are a simulation target kind only |
| `red_team()` × data source `inline DataPoint`s | `dataset` takes a `Path` or specifier string; there is no inline-datapoint parameter |
| `static` mode × generated data source | static mode consumes a fixed dataset by definition |
| CLI × custom `AgentTarget` | custom targets are constructed in Python; the CLI resolves string identifiers |
| reasoning-effort scope `simulator's own calls` × `red_team()` | red teaming has no user simulator; its own calls are the attacker/judge scope |
| reasoning-effort scope `pipeline attacker/judge` × `simulate()` / `generate_and_simulate()` | simulation has no attacker; its own calls are the simulator scope |
| reasoning-effort scope × dashboard | the dashboard reads saved artifacts; it invokes no model under test |
| reasoning-effort scope `target under test` × target kinds without a Responses endpoint | effort reaches the target only on Responses-capable targets (`agent:<key>`); a direct OpenAI backend, callable target or Vercel endpoint has nowhere to put it |

## Tiers

**Tier 1 — needs prose.** Top-level `evaluatorq.__all__` entry points, every CLI
command and subcommand, every env var. The generated API reference does **not**
count: a docstring is not discovery.

**Tier 2 — API reference suffices.** Supporting types, contracts, backends, and
subpackage `__all__` members. Flag only when there is no docstring at all.
