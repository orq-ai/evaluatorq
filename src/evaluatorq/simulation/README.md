# Agent Simulation

Multi-turn conversational testing for agents. A **user-simulator LLM** plays a persona pursuing a goal across a conversation; a **judge LLM** scores the result against your criteria. Runs through the `evaluatorq()` framework, so you get parallelism, OTel tracing, Orq experiment upload, and CI gating for free.

It is the non-adversarial counterpart to [red teaming](../redteam/README.md): red teaming asks *"does it break under attack?"*, simulation asks *"does it work for real users?"*.

## What it does

1. Builds **datapoints** from personas × scenarios (or takes them inline, from an Orq dataset, or from Orq production traces).
2. For each datapoint, runs a turn-by-turn conversation between the user-simulator and your agent.
3. The judge scores each run: goal achieved, criteria met, rules broken, termination reason.
4. Returns `SimulationResult` objects and (by default) uploads an Experiment to Orq.

## Entry points

Two async functions, same target shapes and knobs:

| Function | Use when |
|----------|----------|
| `simulate(...)` | You already have personas/scenarios/datapoints (or an Orq `dataset_id`). |
| `generate_and_simulate(agent_description=..., num_personas=..., num_scenarios=...)` | You have nothing yet — the LLM invents personas and scenarios from a description of the agent. With `target="agent:<key>"`, it fetches the Orq agent description when omitted. |

```python
from evaluatorq.simulation import simulate

results = await simulate(
    evaluation_name="support-agent-sim",
    target=my_async_agent,            # or target="agent:<key>" / target=AgentTarget
    personas=[persona],
    scenarios=[scenario],
    max_turns=8,
    evaluator_names=["goal_achieved", "criteria_met"],
)
```

A runnable, narrated walkthrough lives in [`examples/agent_simulation_intro.ipynb`](../../../examples/agent_simulation_intro.ipynb).

## Targets

The agent under test is supplied one of these ways (mutually exclusive):

- **`target="agent:<key>"`** (or bare **`target="<key>"`**) — a hosted **orq.ai** agent, invoked through the stateless Responses API target. Requires `ORQ_API_KEY`. The primary path.
- **`target="deployment:<key>"`** — bridges to an **orq.ai** deployment. Requires `ORQ_API_KEY`.
- **`target=<AgentTarget>`** — an `AgentTarget` instance (e.g. `OrqResponsesTarget`) that speaks `respond(messages)`.
- **`target=<callable>`** — any `async`/sync callable `(list[Message]) -> str`. Great for local mocks and quick checks.

On the CLI, prefer `--target` to mirror red teaming:

- **`--target agent:<key>`** or bare **`--target <key>`** — invokes a hosted Orq agent through the stateless Responses API target.
- **`--target deployment:<key>`** — uses the deployment callback bridge.

## Personas & scenarios

- **`Persona`** — *who* the user is: `patience`, `assertiveness`, `politeness`, `technical_level`, `communication_style`, `background`, and an optional `emotional_arc` (tone shifts across turns).
- **`Scenario`** — *what* they want: `goal`, `context`, `starting_emotion`, and a list of `Criterion` (`must_happen` / `must_not_happen`) that become the judge's checklist. Flag adversarial cases with `is_edge_case=True`.

`simulate()` takes the cartesian product (every persona × every scenario).

For `generate_and_simulate()`, pass `agent_description` for any local, deployment, or callable target. When the target is a hosted Orq agent (`target="agent:<key>"` or a bare key), the SDK uses that agent's stored description if you omit it. An explicit description always takes precedence. The call raises `ValueError` before generation if neither source supplies a non-empty description.

## LLM configuration

`llm_config` (an `LLMCallConfig`) configures every simulation-side LLM call: the user-simulator, the judge, the persona / scenario / first-message generators, the recommendations pass and the executive summary. It never reaches the target under test. `sim_model` (default `openai/gpt-5.6-luna`) is the shorthand for setting only the model; when both are given, `llm_config.model` wins and the contradiction is logged.

Provider resolution mirrors red teaming: an injected `generation_client` → `llm_config.client` → `ORQ_API_KEY` (Orq router) → `OPENAI_API_KEY` (with optional `OPENAI_BASE_URL`).

The default `openai/gpt-5.6-luna` assumes the Orq router. If you target OpenAI directly (only `OPENAI_API_KEY` set), drop the prefix: `sim_model="gpt-5.6-luna"`.

Override the user-simulator or judge entirely by passing pre-built `BaseAgent` instances via `user_simulator=` / `judge=`. An injected agent arrives already built, so `llm_config` does not reach it — the runner warns once when you set both.

## Results & CI gating

Each result carries `goal_achieved`, `goal_completion_score`, `turn_count`, `terminated_by`, `rules_broken`, `criteria_results`, and the full `messages` transcript.

`exit_on_failure=True` (default) makes a run raise `SimulationDroppedError` when a datapoint is dropped — drop it straight into a CI step. Evaluator score failures are returned in the results for callers to inspect. Pass `exit_on_failure=False` for interactive runs where dropped rows should surface as warnings instead.

## Datasets

Set `dataset_id="..."` to pull simulation datapoints from a named Orq dataset instead of inline personas/scenarios. Each row's `inputs` must already match a simulation input shape (`datapoint`, or `persona` + `scenario`).

## Experiments

A prior Orq experiment run can seed simulations two ways (requires `ORQ_API_KEY`):

- **Direct** — set `experiment_id="..."` (optionally `experiment_run_id="..."`; latest run when omitted) to replay the run's rows as datapoints. Same row-shape rules as `dataset_id`; experiments uploaded by a previous simulation run round-trip as-is. CLI: `eq sim simulate --experiment-id`.
- **Extension** — `extend_from_experiment()` feeds the run's personas and scenarios to the standard generators as seeds and returns *new* similar-but-not-duplicate datapoints.

```python
from evaluatorq.simulation import extend_from_experiment, simulate

# direct: replay the experiment's rows
results = await simulate(evaluation_name="replay", experiment_id="ex_abc", target=...)

# extension: generate fresh datapoints seeded by the run
extra = await extend_from_experiment("ex_abc", num_personas=3, num_scenarios=5)
results = await simulate(evaluation_name="extended", datapoints=extra, target=...)
```

## Data sources

Where cases come from, and what you can do with each. **Replay** re-runs the exact same cases (reproducible compare across agent versions/evaluators); **Seed new cases** mines a source for archetypes that generate *fresh* personas/scenarios (extends the dataset).

| Source | Replay (re-use exact cases) | Seed new cases (extend) |
|--------|:---------------------------:|:-----------------------:|
| Inline `personas` + `scenarios` | ✅ | — (they *are* the new cases) |
| JSONL datapoints (`--datapoints` / `load_datapoints_from_jsonl()`) | ✅ | ⚠️ manual (hand-pick seeds) |
| Orq dataset (`dataset_id=`) | ✅ | ⚠️ manual |
| Previous run (`previous_run="<id>"` / `--from-run`, or export to JSONL via `eq sim generate --datapoints`) | ✅ | ⚠️ manual |
| Orq experiment (`experiment_id=`) | ✅ | ✅ `extend_from_experiment()` |
| Production traces (`datapoints_from_traces` / `eq sim from-traces`) | ✅ | ✅ `extend_from_traces()` |

Legend: ✅ built-in · ⚠️ possible but manual · ❌ not supported yet.

`previous_run`, `dataset_id`, `experiment_id`, `datapoints`, and `personas` + `scenarios` are mutually exclusive — pass exactly one source per run.

### Replaying a previous run

Saved runs record the cases they simulated, so a run can be repeated against a new agent version without regenerating anything:

```bash
eq sim simulate --from-run latest --target agent:my-agent-v2
```

```python
results = await simulate(target='agent:my-agent-v2', previous_run='latest')
```

`--from-run` accepts `latest`, the run name or file name `eq sim runs` prints, a run id (or an unambiguous 8+ character prefix), or a path to a saved run JSON, resolved against `.evaluatorq/sim-runs/`. The stored personas, scenarios, and first messages are re-used exactly — no persona/scenario generation, no first-message generation, no dataset fetch — and the run's turn cap is restored unless you pass `--max-turns`. What you vary between runs is the target and the evaluators. Runs saved before this shipped carry no datapoints and are rejected with an explanatory error, as are runs stamped with a replay format newer than the installed version understands.

## Traces as input

Production traces from Orq's observability product can seed simulations (requires `ORQ_API_KEY`). Two modes:

- **Direct** — one datapoint per fetched trace: an LLM infers the persona and scenario from the transcript; the first message is the real user's opening message, verbatim.
- **Extension** — an LLM distills the fetched traffic into a distribution profile (topic mix, tones, technical levels), then generates *new* distribution-matched datapoints through the standard generators.

```python
from evaluatorq.simulation import (
    datapoints_from_traces, extend_from_traces, fetch_trace_conversations, simulate,
)

conversations = await fetch_trace_conversations(limit=20)
datapoints = await datapoints_from_traces(conversations)          # direct
datapoints += await extend_from_traces(conversations, num_datapoints=10)  # extension
results = await simulate(evaluation_name="from-traces", datapoints=datapoints, target=...)
```

On the CLI:

```bash
eq sim from-traces --limit 20 --lookback-hours 24 --extend 10 --output dp.jsonl
eq sim simulate --input dp.jsonl --target agent:my-agent
```

## Tracing & PII

Runs emit OTel spans under `Evaluatorq - Agent Simulation` (auto-visible in orq.ai when `ORQ_API_KEY` is set). Two shared env vars control message capture, identical to red teaming:

- `EVALUATORQ_CAPTURE_MESSAGE_CONTENT` — set `false`/`0` to keep raw message text (incl. PII) off spans while still recording tokens/model/latency. Defaults `true`.
- `EVALUATORQ_SPAN_MAX_TEXT_CHARS` — cap stored text per span attribute. Defaults to no truncation.

## CLI

The same capability is exposed as `eq sim run` / `eq sim generate`. Install and usage:

```bash
uv add "evaluatorq[simulation]"
uv run eq sim run --help
uv run eq sim generate --help
```

`uv run evaluatorq` is the same entry point under its long name. Avoid `uv tool install` here: it builds an isolated environment that exposes the CLI but leaves `evaluatorq` unimportable from your own scripts.

Prefer pip? `python -m pip install "evaluatorq[simulation]"` installs into the interpreter you just named, and `eq` lands on that environment's `PATH`.

Examples:

```bash
eq sim run --target agent:refund-agent-fixed
eq sim generate --target refund-agent-fixed --datapoints dp.jsonl
eq sim simulate --datapoints dp.jsonl --target deployment:refund-agent
```

See [`examples/agent_simulation/README.md`](../../../examples/agent_simulation/README.md) for the full set of runnable scripts (orq deployment, tool-using agents, hardening loop, the `wrap_simulation_agent()` + `evaluatorq()` production pattern).
