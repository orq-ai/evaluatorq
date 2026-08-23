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
| **data source** | `evaluatorq()` / `red_team()` dataset params, **plus `replay`** (see below) | inline `DataPoint`s, ORQ dataset id, HuggingFace dataset, generated, replay of a stored run |
| **evaluator kind** | `VULNERABILITY_EVALUATOR_REGISTRY`, `SIMULATION_EVALUATORS`, pairwise types | built-in scorer, LLM jury, pairwise jury, custom `Evaluator` |

### `replay` is a data source, not a fourth mode

Replay is reached by `previous_run=` / `--from-run`, which is explicitly
**incompatible** with `--mode`. The reason is not that they are rival ways of
saying the same thing — it is that a replayed run *already has* a mode. Replay
loads a stored run and does `mode = replay.pipeline` (`redteam/runner.py`), so
replaying a run that was `static` runs static, and replaying a `hybrid` run runs
hybrid. Passing `--mode` alongside would be supplying a value that is about to be
overwritten, which is why it raises instead of silently losing.

So replay does not sit *among* the modes, it crosses *with* them. What it
actually replaces is the data source: instead of generating attacks or reading a
dataset, the attack set comes from a run you already did. That is the axis it
belongs on, and it is the axis that makes its value obvious — holding the attacks
fixed is the only way a moved resistance rate means the agent moved.

Source-derived values can never see it (there is no enum to read), so it is
hardcoded on this axis on purpose. **Do not resolve this by adding `replay` to
the accepted `--mode` values in code** — it is not a mode, and the docs-autofill
routine may not touch `src/` regardless.

Replay exists on both `eq redteam run` and `eq sim simulate`; treat it as a data
source for both when building the matrix.

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

## Tiers

**Tier 1 — needs prose.** Top-level `evaluatorq.__all__` entry points, every CLI
command and subcommand, every env var. The generated API reference does **not**
count: a docstring is not discovery.

**Tier 2 — API reference suffices.** Supporting types, contracts, backends, and
subpackage `__all__` members. Flag only when there is no docstring at all.

**Not a gap at all — a symbol reached only as a parameter value, whose output is
documented.** A name with no prose is not automatically Tier 1. Ask how a user
reaches it: if the answer is "they pass a string to some other function's
argument" and the *result* they get back is already explained in prose, the
symbol is an implementation detail and prose about it would document something
nobody types. `bt_sigma_aggregation` is the worked example — it is reached only
via `build_report(aggregation='bt-sigma')`, and what a reader actually needs, the
`report.bt_sigma` output and how to read it, is covered at
`pairwise-judging.md:168-245`. A symbol-name grep flags it every time; it has
never been a real gap. Resolve these `N/A`, not `GAP`.
