# Agent Simulation

Multi-turn conversational testing for agents. A **user-simulator LLM** plays a
persona pursuing a goal across a conversation; a **judge LLM** scores the result
against your criteria. Runs through the `evaluatorq()` framework, so you get
parallelism, OTel tracing, Orq experiment upload, and CI gating for free.

It is the non-adversarial counterpart to [red teaming](../redteam/README.md):
red teaming asks *"does it break under attack?"*, simulation asks *"does it work
for real users?"*.

## What it does

1. Builds **datapoints** from personas × scenarios (or takes them inline / from an Orq dataset).
2. For each datapoint, runs a turn-by-turn conversation between the user-simulator and your agent.
3. The judge scores each run: goal achieved, criteria met, rules broken, termination reason.
4. Returns `SimulationResult` objects and (by default) uploads an Experiment to Orq.

## Entry points

Two async functions, same target shapes and knobs:

| Function | Use when |
|----------|----------|
| `simulate(...)` | You already have personas/scenarios/datapoints (or an Orq `dataset_id`). |
| `generate_and_simulate(agent_description=..., num_personas=..., num_scenarios=...)` | You have nothing yet — the LLM invents personas and scenarios from a description of the agent. |

```python
from evaluatorq.simulation import simulate

results = await simulate(
    evaluation_name="support-agent-sim",
    target_callback=my_async_agent,   # or target="agent:<key>" / target=AgentTarget
    personas=[persona],
    scenarios=[scenario],
    max_turns=8,
    evaluator_names=["goal_achieved", "criteria_met"],
)
```

A runnable, narrated walkthrough lives in
[`examples/agent_simulation_intro.ipynb`](../../../examples/agent_simulation_intro.ipynb).

## Targets

The agent under test is supplied one of these ways (mutually exclusive):

- **`target="agent:<key>"`** (or bare **`target="<key>"`**) — a hosted **orq.ai** agent, invoked through the stateless Responses API target. Requires `ORQ_API_KEY`. The primary path.
- **`target="deployment:<key>"`** — bridges to an **orq.ai** deployment. Requires `ORQ_API_KEY`.
- **`target=<AgentTarget>`** — an `AgentTarget` instance (e.g. `OrqResponsesTarget`) that speaks `respond(messages)`.
- **`target_callback=` / `target=<callable>`** — any `async`/sync callable `(list[Message]) -> str`. Great for local mocks and quick checks.

On the CLI, prefer `--target` to mirror red teaming:

- **`--target agent:<key>`** or bare **`--target <key>`** — invokes a hosted Orq agent through the stateless Responses API target.
- **`--target deployment:<key>`** — uses the deployment callback bridge.

## Personas & scenarios

- **`Persona`** — *who* the user is: `patience`, `assertiveness`, `politeness`, `technical_level`, `communication_style`, `background`, and an optional `emotional_arc` (tone shifts across turns).
- **`Scenario`** — *what* they want: `goal`, `context`, `starting_emotion`, and a list of `Criterion` (`must_happen` / `must_not_happen`) that become the judge's checklist. Flag adversarial cases with `is_edge_case=True`.

`simulate()` takes the cartesian product (every persona × every scenario).

## LLM configuration

`sim_model` (default `openai/gpt-5.4-mini`) drives the user-simulator, the judge,
and — for `generate_and_simulate` — persona/scenario generation. Provider
resolution mirrors red teaming: an injected `generation_client` →
`ORQ_API_KEY` (Orq router) → `OPENAI_API_KEY` (with optional `OPENAI_BASE_URL`).

The default `openai/gpt-5.4-mini` assumes the Orq router. If you target OpenAI
directly (only `OPENAI_API_KEY` set), drop the prefix: `sim_model="gpt-5.4-mini"`.

Override the user-simulator or judge entirely by passing pre-built `BaseAgent`
instances via `user_simulator=` / `judge=`.

## Results & CI gating

Each result carries `goal_achieved`, `goal_completion_score`, `turn_count`,
`terminated_by`, `rules_broken`, `criteria_results`, and the full `messages`
transcript.

`exit_on_failure=True` (default) makes a run exit non-zero when any datapoint or
evaluator fails — drop it straight into a CI step. Score-based failures go
through evaluatorq's own gate; dropped jobs raise `SimulationDroppedError`. Pass
`exit_on_failure=False` for interactive runs where failures should surface as
warnings instead.

## Datasets

Set `dataset_id="..."` to pull simulation datapoints from a named Orq dataset
instead of inline personas/scenarios. Each row's `inputs` must already match a
simulation input shape (`datapoint`, or `persona` + `scenario`).

## Tracing & PII

Runs emit OTel spans under `orq.simulation.pipeline` (auto-visible in orq.ai when
`ORQ_API_KEY` is set). Two shared env vars control message capture, identical to
red teaming:

- `EVALUATORQ_CAPTURE_MESSAGE_CONTENT` — set `false`/`0` to keep raw message text (incl. PII) off spans while still recording tokens/model/latency. Defaults `true`.
- `EVALUATORQ_SPAN_MAX_TEXT_CHARS` — cap stored text per span attribute. Defaults to no truncation.

## CLI

The same capability is exposed as `eq sim run` / `eq sim generate`. Install and
usage:

```bash
uv tool install "evaluatorq[simulation]"
eq sim run --help
eq sim generate --help
```

Examples:

```bash
eq sim run --target agent:refund-agent-fixed
eq sim generate --target refund-agent-fixed --output dp.jsonl
eq sim simulate --datapoints dp.jsonl --target deployment:refund-agent
```

See [`examples/agent_simulation/README.md`](../../../examples/agent_simulation/README.md)
for the full set of runnable scripts (orq deployment, tool-using agents,
hardening loop, the `wrap_simulation_agent()` + `evaluatorq()` production pattern).
