# Coverage axes and impossible combinations

The interaction surface of evaluatorq, expressed as axes. `docs-coverage` crosses these pairwise and checks each meaningful pair appears in prose.

**Hand-maintained.** Axis *names* live here; axis *values* are pulled from source at run time, so a new mode or backend appears without editing this file. Edit this file only when a genuinely new **dimension** appears — a new entry point, a new surface, a new kind of thing a user chooses between.

If this file rots, `docs-coverage` reports stale gaps and people stop reading it. Treat it as part of the public surface: changing it belongs in the same PR as the feature that changed it.

## Axes

| Axis | Where values come from | Notes |
|---|---|---|
| **entry point** | `evaluatorq.__all__` + `evaluatorq.simulation.__all__` + `evaluatorq.redteam.__all__` | `evaluatorq()`, `red_team()`, `simulate()`, `generate_and_simulate()`, `wrap_simulation_agent()`, pairwise `build_report()`, `deployment()` / `invoke()` |
| **surface** | fixed | Python API · CLI (`eq`) · dashboard (`eq dashboard`) |
| **target kind** | `_BACKEND_REGISTRY` in `redteam/backends/registry.py` + CLI `--target` prefixes | `agent:<key>`, `deployment:<key>`, direct OpenAI backend, custom `AgentTarget` / `CallableTarget` |
| **mode** | `--mode` on `eq redteam run` | `dynamic`, `static`, `hybrid` |
| **data source** | `evaluatorq()` / `red_team()` dataset params, **plus `replay`** (see below) | inline `DataPoint`s, ORQ dataset id, HuggingFace dataset, generated, replay of a stored run |
| **evaluator kind** | `VULNERABILITY_EVALUATOR_REGISTRY`, `SIMULATION_EVALUATORS`, pairwise types | built-in scorer, LLM jury, pairwise jury, custom `Evaluator` |
| **reasoning-effort scope** | fixed (see below) | target under test · pipeline attacker/judge · simulator's own calls · core-evaluation judge |
| **API endpoint** | `LLMCallConfig.api` / `EvaluatorConfig.api` (`contracts.py`, `redteam/contracts.py`) | `chat_completions` · `responses` |
| **own-calls LLM config** | `llm_config=` on `simulate()` / `generate_and_simulate()` / `generate()` / the trace helpers, `llm_config=` on `red_team()`, plus the `sim_model=` / `model=` shorthands | full `LLMCallConfig` · model-name shorthand · neither (per-call-site defaults) |

### `reasoning-effort scope` is a choice, not a value

Four settings carry the words "reasoning effort" and they apply to four different models: `target_reasoning_effort` (the agent under test), `LLMCallConfig.reasoning_effort` on `attacker=` / `evaluator=` (red team's own calls), `EVALUATORQ_REASONING_EFFORT` (the simulator's user-simulator and judge), and `llm_jury(reasoning_effort=...)` / `llm_jury_pairwise` / `PairwiseComparator` (the judge in core `evaluatorq()`). Picking the wrong one is silent — the call just runs at the default — so this is a dimension a user chooses along, not a tuning number.

Values are fixed here because no registry enumerates them; the *accepted efforts* per model come from the model catalogue and are a different thing entirely.

### `API endpoint` is a per-role choice with a per-role default

`LLMCallConfig.api` defaults to `chat_completions`; `EvaluatorConfig.api` defaults to `responses`, because that is the endpoint the Orq router prices. The endpoint decides the spelling of every knob (`max_completion_tokens` vs `max_output_tokens`, flat `reasoning_effort` vs a `reasoning` block), which keys `check_reserved_keys` rejects inside `extra_kwargs`, and whether the call records cost at all.

It is honoured by the judge (`common/judge.py`) and by simulation agents (`simulation/agents/base.py`). Red team's attacker call sites pass `api='chat_completions'` explicitly, so the field is inert there — treat that as a gap to document, not an `N/A`, until the code either honours it or warns.

Simulation agents carry a third default, distinct from both: `BaseAgent.DEFAULT_API` is `responses`, not `LLMCallConfig`'s own `chat_completions`, because the judge sends function tools and `reasoning_effort` in the same request and chat completions answers that combination with a 400. `BaseAgent.__init__` applies it only when the caller's `llm_config` did not set `api` itself (the `model_fields_set` gate, same mechanism as the rest of the `own-calls LLM config` axis), so `LLMCallConfig(api='chat_completions')` opts a simulation run back out. The deprecated `AgentConfig` path pins `api` to `chat_completions` unconditionally instead of leaving it unset, so it never picks up this default — document that as the legacy path's own behaviour, not a bug. There is deliberately no environment variable for this default.

### `own-calls LLM config` is where a surface's own sampling settings come from

Every surface makes LLM calls that are not the target under test — red team's attacker and evaluator, simulation's user simulator, judge and generators, the executive summary. Each of those has a config object the caller can supply: `red_team(llm_config=LLMConfig(attacker=..., evaluator=...))` and `simulate(llm_config=LLMCallConfig(...))`. The model-name shorthands (`sim_model=`, `model=`) set only the model on the same object; when both are given the full config wins and the contradiction is logged.

Supplying neither is the common case and is not a gap: a field the caller never touched is not forwarded at all (the check is `model_fields_set`, never the value), so each call site's own literal applies and `temperature` is omitted from the request entirely. That is why the check is on the set rather than on `is None`: `model`, `max_tokens`, `timeout_ms` and `retry_count` carry real defaults, and `None` is itself a meaningful explicit value for the rest. That last part is load-bearing — reasoning-class models answer `400` to `temperature` at any value.

Never confuse this axis with the target under test. The target is the thing being measured; it is configured where it is constructed (`target_reasoning_effort`, the backend's own settings), never through this config.

### `replay` is a data source, not a fourth mode

Replay is reached by `previous_run=` / `--from-run`, which is explicitly **incompatible** with `--mode`. The reason is not that they are rival ways of saying the same thing — it is that a replayed run *already has* a mode. Replay loads a stored run and does `mode = replay.pipeline` (`redteam/runner.py`), so replaying a run that was `static` runs static, and replaying a `hybrid` run runs hybrid. Passing `--mode` alongside would be supplying a value that is about to be overwritten, which is why it raises instead of silently losing.

So replay does not sit *among* the modes, it crosses *with* them. What it actually replaces is the data source: instead of generating attacks or reading a dataset, the attack set comes from a run you already did. That is the axis it belongs on, and it is the axis that makes its value obvious — holding the attacks fixed is the only way a moved resistance rate means the agent moved.

Source-derived values can never see it (there is no enum to read), so it is hardcoded on this axis on purpose. **Do not resolve this by adding `replay` to the accepted `--mode` values in code** — it is not a mode, and the docs-autofill routine may not touch `src/` regardless.

Replay exists on both `eq redteam run` and `eq sim simulate`; treat it as a data source for both when building the matrix.

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
| any `mode` × data source `replay` | `previous_run=` restores the stored run's pipeline and raises if `mode` is also supplied; the pair cannot be expressed |
| CLI × custom `AgentTarget` | custom targets are constructed in Python; the CLI resolves string identifiers |
| reasoning-effort scope `simulator's own calls` × `red_team()` | red teaming has no user simulator; its own calls are the attacker/judge scope |
| reasoning-effort scope `pipeline attacker/judge` × `simulate()` / `generate_and_simulate()` | simulation has no attacker; its own calls are the simulator scope |
| reasoning-effort scope × dashboard | the dashboard reads saved artifacts; it invokes no model under test |
| reasoning-effort scope `target under test` × `deployment:`, callable and Vercel targets | the ORQ SDK agents `invoke_async` payload has no reasoning field, and a callable or Vercel endpoint has nowhere to put it. Only `agent:<key>` carries it, as a `reasoning` block |
| reasoning-effort scope `target under test` × direct OpenAI backend | `OpenAIBackend` does accept the effort, but nothing reaches it: every `resolve_backend(...)` in `src/` passes `'orq'` or `'openresponses'`, and `parse_target` yields only `AGENT` or `DEPLOYMENT`, so the `'openai'` registration has no caller on a `red_team()` path |
| reasoning-effort scope `core-evaluation judge` × `red_team()` / `simulate()` | those surfaces score through their own judge roles, not `llm_jury()` |
| API endpoint × `evaluatorq()` / `llm_jury()` | juries pin `api='responses'` internally; there is no parameter to choose |
| API endpoint × dashboard | reads saved artifacts; issues no request |
| API endpoint × target kind `deployment:` / callable / Vercel | these do not go through `request_params`; the endpoint is fixed by the transport |
| `eq redteam ui` / `eq sim ui` × anything | retired Streamlit viewers. Still registered for compatibility, deliberately de-documented in favour of `eq dashboard` (stated in `docs/cli-reference/dashboard.md`). A deliberate de-documentation is a decision, not a coverage gap |
| own-calls LLM config × dashboard | it reads saved artifacts; it makes no call of its own to configure |
| own-calls LLM config × `evaluatorq()` | the core loop runs your task function and your evaluators; it has no own-calls role to configure. Judge settings go to `llm_jury(...)` |
| own-calls LLM config `model-name shorthand` × `red_team()` | red team has no `sim_model` equivalent; both roles are named on `LLMConfig` |

## Tiers

**Tier 1 — needs prose.** Top-level `evaluatorq.__all__` entry points, every CLI command and subcommand, every env var. The generated API reference does **not** count: a docstring is not discovery.

Tier-1 items are **not** matrix cells — there is no `env var` axis, and `surface` has exactly the three values above. Record a Tier-1 gap in the `docs-autofill` ledger as `tier 1: <kind> (<name>, …)` with no second axis, e.g. `tier 1: env var (ORQ_OTEL_MAX_QUEUE_SIZE, ORQ_OTEL_MAX_BATCH_SIZE)`. The ledger dedupe compares that cell as free text, so naming the items is what stops the same gap being re-derived next week; an invented axis pair never matches.

**A symbol whose *output* is documented is not a Tier-1 gap.** A symbol-name grep does not know whether a reader reaches the concept by another name. `bt_sigma_aggregation` is the standing example: it is internal, called only from `build_report(aggregation='bt-sigma')`, and readers meet it as `report.bt_sigma`, documented at `pairwise-judging.md`. Before recording a Tier-1 gap for a symbol, check whether its result is already documented under the name users actually type; if it is, the gap is a false positive, not prose to write.

**Tier 2 — API reference suffices.** Supporting types, contracts, backends, and subpackage `__all__` members. Flag only when there is no docstring at all.
