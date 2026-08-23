# Tuning: timeouts, retries and reasoning effort

Every knob on this page has a working default. Reach for it when a run is slow,
flaky, or spending more than you want — not before.

The knobs split into three groups, and confusing them is the usual source of a
setting that "does nothing":

| Group | What it governs | Where it lives |
|---|---|---|
| **Target calls** | The agent *under test* — its timeout, its retries, its reasoning effort | `LLMConfig` (red team) / `simulate()` kwargs (simulation) |
| **Pipeline calls** | evaluatorq's own LLMs — the attacker, the judge, the user simulator, the generators | `LLMCallConfig` / `EvaluatorConfig` / `EVALUATORQ_*` env vars |
| **Provider options** | Anything the provider accepts that evaluatorq does not model | `extra_kwargs` and `extra_body` |

## Reasoning effort: three different things

Four separate settings carry the words "reasoning effort", and they apply to
four different models. Setting the wrong one is silent — the call simply runs at
the default.

| You want to change | Use | Applies to |
|---|---|---|
| The **target agent** under test | `--target-reasoning-effort`, or `LLMConfig(target_reasoning_effort=...)` / `simulate(target_reasoning_effort=...)` | The agent being red-teamed or simulated |
| The **attacker or judge** in red teaming | `LLMCallConfig(reasoning_effort=...)` on `attacker=` / `evaluator=` | evaluatorq's own pipeline calls |
| The **user simulator and judge** in simulation | `EVALUATORQ_REASONING_EFFORT`, or `LLMCallConfig(reasoning_effort=...)` on a custom `user_simulator=` / `judge=` | evaluatorq's own simulation calls |
| The **jury** in core evaluation | `reasoning_effort=` on `llm_jury()`, `llm_jury_pairwise()` or `PairwiseComparator` | The verdict calls those evaluators make under `evaluatorq()` |

```python
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.redteam import EvaluatorConfig, LLMConfig, red_team

await red_team(
    target="agent:my-agent",
    llm_config=LLMConfig(
        # the attacker's own thinking budget
        attacker=LLMCallConfig(model="openai/gpt-5.6-luna", reasoning_effort="high"),
        # the judge's own thinking budget
        evaluator=EvaluatorConfig(model="openai/gpt-5.6-luna", reasoning_effort="low"),
        # the thinking budget of the agent being attacked
        target_reasoning_effort="medium",
    ),
)
```

Accepted values differ per model, and evaluatorq does not guess: the value is
forwarded verbatim and the **provider** is the authority. An unsupported value
comes back as a 400, which the executors turn into a drop-the-block-and-retry-once
— the call succeeds without reasoning rather than failing, and the rejection is
remembered for the rest of the process so later calls to the same model skip the
block up front.

Reasoning effort reaches the target only on **Responses-capable** targets
(`agent:<key>` on the Orq router). A `deployment:` target executes via
the ORQ SDK agents endpoint, which has no reasoning parameter; a callable target
or a Vercel endpoint has nowhere to put it either. In each of those cases the
setting is accepted, a warning names the drop, and the run proceeds.

!!! tip "Pre-validate against the catalogue"
    The model catalogue publishes the accepted reasoning-effort values per model.
    A model the catalogue does not list cannot be pre-validated — register it with
    `register_model()` to fix that. See
    [Configuration › Model catalogue overrides](configuration.md#model-catalogue-overrides).

`EVALUATORQ_REASONING_EFFORT` is **unset by default**, and unset means the parameter
is not sent at all — the model applies its own default. There is deliberately no
global effort: sending one costs a rejected request plus a retry on every model that
does not accept the parameter, and overrides the tuned default on every model that
does. Set it only when you want a specific effort across the simulator's own calls.

It is read **at call time**, not at import, and it is only a fallback: an explicitly
set `LLMCallConfig.reasoning_effort` on the agent's config always wins — including an
explicit `None`, which opts that agent out of the env value on purpose. The same is
true of `EVALUATORQ_LLM_TIMEOUT_S` and `EVALUATORQ_LLM_MAX_TOKENS`.

## Target-call reliability

The target is the slow, flaky part of any run: it is a real agent running real
tools. Four knobs bound it.

=== "Red teaming"

    ```bash
    eq redteam run -t agent:my-agent \
      --target-timeout-ms 480000 \
      --max-target-retries 3 \
      --max-tool-continuations 8
    ```

    ```python
    from evaluatorq.redteam import LLMConfig, red_team

    await red_team(
        target="agent:my-agent",
        llm_config=LLMConfig(
            target_agent_timeout_ms=480_000,
            max_target_retries=3,
            max_tool_continuations=8,
        ),
    )
    ```

=== "Simulation"

    ```bash
    eq sim run --target agent:my-agent --target-reasoning-effort medium
    ```

    ```python
    from evaluatorq.simulation import simulate

    await simulate(
        target="agent:my-agent",
        datapoints=datapoints,
        target_agent_timeout_ms=480_000,
        max_target_retries=3,
        per_simulation_timeout_s=900,
        max_tool_result_chars=2000,
    )
    ```

| Knob | Default | Raise it when |
|---|---|---|
| `target_agent_timeout_ms` / `--target-timeout-ms` | `240000` (4 min) | A single target call legitimately takes minutes — a self-hosted endpoint, or an agent that chains several tools per turn |
| `max_target_retries` / `--max-target-retries` | `2` (0–10) | The target's transport is flaky. A retry never consumes a new attacker turn and never changes the transcript |
| `max_tool_continuations` / `--max-tool-continuations` | `5` | An Orq agent needs more client-driven tool-result rounds than five to finish a turn |
| `per_simulation_timeout_s` (simulation, Python only) | `None` — unbounded | A conversation can stall in a loop. Without it, only the per-call timeouts apply |

`per_simulation_timeout_s` is a wall clock over **one whole datapoint** — every
turn, plus the user-simulator and judge calls. On expiry the simulation does not
raise: it returns a partial result with `terminated_by="timeout"` and the turns it
completed, and logs a warning naming the budget. Set it on any unattended run;
`simulate()` calls the runner once per row and is otherwise unbounded. `None` is
the only spelling of unbounded — `0` is rejected at construction rather than read
as "no bound".

`max_tool_result_chars` (default `500`) caps each tool result rendered into the
text the user simulator sees on a tool-only turn. Raise it for a tool-heavy agent
whose results are being cut before the simulator — and the judge, which scores the
same transcript — can react to them.

## Pipeline-call reliability

These bound evaluatorq's own LLM calls, not the target's.

| Knob | Default | What it covers |
|---|---|---|
| `LLMConfig.retry_count` / `--retry-count` | `3` (0–10) | Pipeline-owned LLM calls and Orq context/enrichment/cleanup |
| `LLMConfig.retry_on_codes` | `[429, 500, 502, 503, 504]` | Which HTTP statuses count as retryable |
| `LLMConfig.max_content_filter_retries` | `2` | Attacker turns the attack model content-filtered or self-censored |
| `LLMConfig.max_consecutive_adversarial_timeouts` | `2` | Consecutive attacker-LLM timeouts before the orchestrator abandons the attack |
| `EvaluatorConfig.retry_count` | `1` | Retries per judge call. `0` falls straight through to `replacement_judges` / `min_successful_judges` |
| `LLMCallConfig.timeout_ms` | `90000` | Per-call timeout for a pipeline role |

!!! warning "One retry layer, including on a client you inject"
    SDK-level retries and evaluatorq's own retries would **multiply** — five
    evaluatorq attempts over a client doing two SDK retries is fifteen requests,
    not five. So the helpers that wrap a call in evaluatorq's retry loop disarm
    the SDK budget first: both `common.judge` and `common.structured_output`'s
    `generate_structured` clone the client with `max_retries=0` before their
    first attempt. The clone reuses your transport, auth, base URL, headers and
    timeout, and your own client object is never mutated — so you do not need to
    pass `max_retries=0` on a client you inject, and configuring SDK retries on
    it does not stack.

    What you configure instead is the evaluatorq layer: `LLMConfig.retry_count`
    and `LLMConfig.retry_on_codes` in the table above.

Two knobs govern cost rather than reliability, because each unit is a live call:

- `LLMConfig.max_objectives_per_llm_call` (default `8`) — objectives requested per
  attacker call before the generator batches into multiple calls. Raising it asks
  for more output tokens per call; roughly 150 tokens per objective is what keeps
  the default from truncating.
- `LLMConfig.max_probe_turns` (default `8`) — probe turns the black-box classifier
  may send before giving up on capability inference. Each turn is a live call
  **against the target**.

## Provider options: `extra_kwargs` vs `extra_body`

`extra_kwargs` is for provider options evaluatorq does not model — `top_p`,
`store`, `truncation`, `tool_choice`. It is merged **last**, so a caller-supplied
value overrides evaluatorq's computed one. That ordering is also the escape hatch
for reasoning-class models that reject a lowered temperature:

```python
from evaluatorq.contracts import LLMCallConfig

cfg = LLMCallConfig(model="openai/gpt-5.6-luna", extra_kwargs={"top_p": 0.9})
```

Structural request fields are **reserved** and raise `ValueError` if they appear in
`extra_kwargs`: `model`, `messages` / `input`, `response_format` / `text`, and
`extra_body`. Letting `extra_kwargs` silently replace one of those would break the
call it rides on — dropping a required JSON schema, or replacing the Orq router's
thread and memory body wholesale.

```python
# WRONG — raises ValueError: extra_kwargs cannot override structural request fields
LLMCallConfig(extra_kwargs={"extra_body": {"my": "option"}}).request_params()
```

### Adding fields to the request body

`extra_body` is the seam for options that belong in the HTTP **body** rather than
as a top-level SDK argument — what the Orq router uses for its retry policy and
for thread and memory ids. It is a first-class `LLMCallConfig` field and a
dedicated parameter on the `common.llm_call` executors; it is never an
`extra_kwargs` key.

It **merges into** the body the call site built, rather than replacing it. Your
keys win per key, and the keys you did not set survive:

```python
from evaluatorq.contracts import LLMCallConfig

cfg = LLMCallConfig(
    model="openai/gpt-5.6-luna",
    extra_body={"cache": {"ttl": 60}},
)
```

A call that would otherwise send `{"retry": {"count": 3}, "thread": {"id": "..."}}`
now sends that plus your `cache` key. Set `retry` yourself and yours wins, while
`thread` is still there.

That merge is the whole point. `extra_kwargs={"extra_body": ...}` used to *replace*
the body, so adding one unrelated field silently dropped the router's retry hints
with no error and no log line — which is why the key is reserved there and raises.

So there are exactly two injection seams, and they target different parts of the
request: `extra_kwargs` for top-level call arguments, `extra_body` for body fields.
Anything the provider accepts is reachable through one of them — wherever you can
hand evaluatorq an `LLMCallConfig`. The jury evaluators (`llm_jury()`,
`llm_jury_pairwise()`, `PairwiseComparator`) build their config internally, so
they take both as explicit parameters instead.

Sampling fields have first-class parameters now — prefer them over routing the
same key through `extra_kwargs`:

| Instead of | Use |
|---|---|
| `extra_kwargs={"reasoning_effort": "medium"}` | `LLMCallConfig(reasoning_effort="medium")` |
| `extra_kwargs={"max_completion_tokens": 4000}` | `LLMCallConfig(max_tokens=4000)` |
| `extra_kwargs={"temperature": 0.2}` | `LLMCallConfig(temperature=0.2)` |

`LLMCallConfig` renders those per endpoint for you through a single builder,
`request_params(api=None, **params)`: it defaults to `self.api`, or renders
whichever endpoint you pass explicitly (a call site that only speaks one
endpoint passes `api="chat_completions"` or `api="responses"`). On chat
completions it emits `max_completion_tokens` and a flat `reasoning_effort`; on
Responses it emits `max_output_tokens` and `reasoning={"effort": ...}`.

## Where to next

- **[Configuration](configuration.md)** — environment variables and model catalogue overrides.
- **[Red Teaming](guides/red-teaming.md)** — what the knobs above are tuning.
- **[Agent Simulation](guides/agent-simulation.md)** — multi-turn conversation testing.
