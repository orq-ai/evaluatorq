# Tracing

evaluatorq ships optional OpenTelemetry tracing. When enabled, every evaluation
run, job, evaluator, and LLM call becomes a span you can view in the Orq
dashboard or any OTLP-compatible backend.

## How tracing is enabled

Tracing initialises lazily on the first evaluation run. It turns on automatically
when either condition is true:

- `ORQ_API_KEY` is set — the OTLP base endpoint is `https://my.orq.ai/v2/otel`
  (or `<ORQ_BASE_URL>/v2/otel` if `ORQ_BASE_URL` is set); the exporter appends
  `/v1/traces`, so spans POST to `…/v2/otel/v1/traces`.
- `OTEL_EXPORTER_OTLP_ENDPOINT` is set — that endpoint is used as the OTLP base.

If neither variable is set, no tracer is created and all span context managers
are no-ops.

Set `ORQ_DISABLE_TRACING=1` or `ORQ_DISABLE_TRACING=true` to suppress tracing
even when the above variables are present.

## Install the OTEL packages

Tracing depends on optional packages that are not installed by default:

```bash
uv add opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-http \
    opentelemetry-semantic-conventions
# or via the extras bundle:
uv add "evaluatorq[otel]"
```

Prefer pip? Use `python -m pip install "evaluatorq[otel]"`, which installs into
the interpreter you just named rather than whichever `pip` happens to be first
on your `PATH`.

If these packages are absent the SDK silently skips initialisation — no error is
raised.

## Minimal enable example

```python
import os
import asyncio

os.environ["ORQ_API_KEY"] = "your_orq_api_key"   # tracing auto-enables

from evaluatorq import DataPoint, evaluatorq, job, string_contains_evaluator


@job("echo")
async def echo_job(data: DataPoint, _row: int) -> str:
    return str(data.inputs.get("text", ""))


asyncio.run(
    evaluatorq(
        "my-eval",
        data=[DataPoint(inputs={"text": "hello"}, expected_output="hello")],
        jobs=[echo_job],
        evaluators=[string_contains_evaluator()],
    )
)
```

To send traces to a custom OTLP endpoint instead:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uv run my_eval.py
```

To debug tracing setup:

```bash
ORQ_DEBUG=1 uv run my_eval.py
```

This prints the resolved endpoint, auth header presence, and any initialisation
errors to stdout.

## OTLP exporter details

- **Protocol**: HTTP/protobuf (`OTLPSpanExporter` from
  `opentelemetry-exporter-otlp-proto-http`)
- **Export mode**: `BatchSpanProcessor` (asynchronous batching)
- **Timeout**: 5 seconds per export request
- **Auth**: `Authorization: Bearer <ORQ_API_KEY>` is added automatically when
  the resolved endpoint's hostname ends in `.orq.ai` or is exactly `orq.ai`.
  For any other endpoint the header is not added; use `OTEL_EXPORTER_OTLP_HEADERS`
  to supply auth manually.
- **Custom headers**: parsed from `OTEL_EXPORTER_OTLP_HEADERS` as
  `key1=value1,key2=value2`.

## Span hierarchy

### Evaluation runner spans

```
orq.job                          # one per DataPoint — root when no ambient trace is active,
  ├── <your job code>            #   otherwise a child of the caller's span
  └── orq.evaluation             # one per evaluator applied to this job
```

All `orq.job` spans from a single `evaluatorq()` call share the same `orq.run_id`
attribute, which ties them together as a logical run without requiring a common
parent span. Because there is no common parent, though, an N-row run arrives as
**N separate traces** — one rooted at each `orq.job`.

#### One trace per run: `single_trace=True`

Pass `single_trace=True` to bracket the whole run in one `evaluatorq.run` span,
so every row lands in a single trace:

```python
await evaluatorq("my-eval", data=rows, jobs=[my_job], single_trace=True)
```

```
evaluatorq.run                   # one per evaluatorq() call — the root
  └── orq.job                    # one per DataPoint, now a child rather than a root
      └── orq.evaluation
```

It defaults to `False` so existing traces keep their shape. Red teaming and
simulation do not need the flag — they already open their own root spans
(`Evaluatorq - Red Teaming` / `Evaluatorq - Agent Simulation`), and `orq.job`
nests under those.

Span attributes on `evaluatorq.run`:

| Attribute | Value |
|---|---|
| `orq.trace_type` | `"evaluatorq"` |
| `orq.run_id` | UUID for this evaluation run — the same one every `orq.job` carries |
| `orq.run_name` | The `name` passed to `evaluatorq()` |
| `orq.evaluatorq_run_id` | Same UUID again, under the key every evaluatorq root span uses, so one query finds a run's root whatever the surface |

Span attributes on `orq.job`:

| Attribute | Value |
|---|---|
| `orq.trace_type` | `"evaluatorq"` |
| `orq.run_id` | UUID for this evaluation run |
| `orq.row_index` | Zero-based row number |
| `orq.job_name` | Job name (if set via `@job("name")`) |

Span attributes on `orq.evaluation`:

| Attribute | Value |
|---|---|
| `orq.run_id` | Same UUID as the parent job span |
| `orq.evaluator_name` | Name of the evaluator |
| `orq.score` | JSON-serialised score value |
| `orq.explanation` | Explanation string (if the evaluator provides one) |
| `orq.pass` | Boolean pass/fail result |

### Red teaming spans

```
Evaluatorq - Red Teaming         # root — one per red_team() call
  ├── orq.redteam.context_retrieval
  ├── orq.redteam.datapoint_generation
  │     ├── orq.redteam.capability_classification
  │     │     ├── chat (llm_purpose=classify_tools)
  │     │     └── chat (llm_purpose=infer_resources)
  │     └── orq.redteam.strategy_planning
  │           └── chat (llm_purpose=generate_strategies)
  ├── orq.job                    # one per attack datapoint
  │     └── orq.redteam.attack
  │           ├── orq.redteam.target_call
  │           └── orq.redteam.attack_turn  (x N turns)
  │                 ├── orq.redteam.adversarial_generation
  │                 │     └── chat (llm_purpose=adversarial)
  │                 └── orq.redteam.target_call
  ├── orq.evaluation             # security evaluator result
  │     └── orq.redteam.security_evaluation
  │           └── chat (llm_purpose=evaluation)
  └── orq.redteam.memory_cleanup # post-run agent memory entity cleanup (only when cleanup is enabled, entities exist, and the target has configured memory stores)
```

LLM spans (`chat ...`) carry standard GenAI attributes:

| Attribute | Value |
|---|---|
| `gen_ai.operation.name` | Operation name (e.g. `"chat"`) |
| `gen_ai.system` | Provider name |
| `gen_ai.request.model` | Model identifier |
| `gen_ai.usage.input_tokens` | Prompt token count |
| `gen_ai.usage.output_tokens` | Completion token count |
| `gen_ai.usage.total_tokens` | Total token count |
| `gen_ai.usage.calls` | Number of LLM calls rolled into this span (omitted when zero) |
| `gen_ai.usage.cost` | Total cost in USD, only when the provider reported one (also emitted as `gen_ai.usage.total_cost`; `gen_ai.usage.input_cost` / `gen_ai.usage.output_cost` when the provider breaks it down) |
| `gen_ai.usage.cache_read.input_tokens` | Cached prompt tokens, when the provider reports them |
| `gen_ai.usage.cache_creation.input_tokens` | Cache-write prompt tokens, when the provider reports them |
| `gen_ai.usage.reasoning.output_tokens` | Reasoning tokens, when the provider reports them |
| `gen_ai.input.messages` | JSON serialised input messages (gated by `EVALUATORQ_CAPTURE_MESSAGE_CONTENT`) |
| `gen_ai.output.messages` | JSON serialised output messages (gated by `EVALUATORQ_CAPTURE_MESSAGE_CONTENT`) |
| `orq.llm.purpose` | Cross-domain purpose tag (e.g. `"adversarial"`, `"evaluation"`, `"target"`) |

!!! note "Attribute aliases removed (August 2026, RES-985)"
    Earlier releases emitted every token count under up to three names: the
    canonical `gen_ai.usage.*` key above, a legacy alias
    (`gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`,
    `gen_ai.usage.prompt_tokens_details.cached_tokens`), and a bare
    un-namespaced key (`prompt_tokens`, `completion_tokens`, `input_tokens`,
    `output_tokens`, `total_tokens`, `calls`). The aliases and bare keys are no
    longer emitted. This was verified against the Orq platform's OTel ingest
    (`extractCommonUsage` in `orquesta-web` `apps/traces-api`): its attribute
    pattern lists try the canonical `gen_ai.usage.*` spellings first, cache
    counts are read from the `cache_read.input_tokens` /
    `cache_creation.input_tokens` keys kept here, and the bare keys and `calls`
    are read nowhere. Reasoning tokens moved from
    `gen_ai.usage.completion_tokens_details.reasoning_tokens` (a spelling the
    platform never read) to `gen_ai.usage.reasoning.output_tokens`, the one it
    does. Third-party OTLP consumers that matched the removed aliases must
    switch to the canonical keys.

The root `Evaluatorq - Red Teaming` span additionally carries:

| Attribute | Value |
|---|---|
| `orq.evaluatorq_run_id` | This run's id — see [Run correlation](#run-correlation) |

### Simulation spans

```
Evaluatorq - Agent Simulation    # root — one per simulate() / generate_and_simulate() call
  ├── chat/responses {model}     # persona/scenario generation calls
  ├── orq.simulation.first_message_generation   # ONE span for the whole persona x scenario sweep
  │     └── chat/responses {model} (x N pairs)
  └── orq.simulation.run         # one per datapoint
        ├── orq.simulation.first_message_generation   # only when no first message was pre-generated
        │     └── chat/responses {model} (orq.llm.purpose="first_message")
        └── orq.simulation.turn  (x N turns)
              ├── orq.simulation.target_call           # calls the agent under test; no span attrs of its own
              ├── orq.simulation.judge_evaluation
              │     └── chat/responses {model} (orq.llm.purpose="judge")
              └── orq.simulation.user_simulator_call
                    └── chat/responses {model} (orq.llm.purpose="user_simulator")

orq.simulation.generate          # root — one per standalone generate() call
  └── chat/responses {model}     # persona/scenario/first-message generation calls
```

`generate_personas()` and `generate_scenarios()` don't open a synthetic root span
when invoked standalone. They do create `orq.simulation.persona_generation` /
`orq.simulation.scenario_generation` spans around their LLM calls. Those generation
spans carry the active run metadata when called inside an outer simulation or
red-team scope; standalone helpers intentionally have no synthetic run id to stamp.

Span attributes on `Evaluatorq - Agent Simulation` / `orq.simulation.generate`:

| Attribute | Value | Present on |
|---|---|---|
| `orq.simulation.evaluation_name` | Evaluation name passed to `simulate()` / `generate_and_simulate()` | `Evaluatorq - Agent Simulation` |
| `orq.simulation.max_turns` | Configured max turns | `Evaluatorq - Agent Simulation` |
| `orq.simulation.parallelism` | Configured parallelism | `Evaluatorq - Agent Simulation` |
| `orq.simulation.mode` | `"generate_and_simulate"` or `"generate"` | `Evaluatorq - Agent Simulation` (generate_and_simulate only), `orq.simulation.generate` |
| `orq.simulation.num_personas` | Requested persona count | `Evaluatorq - Agent Simulation` (generate_and_simulate only), `orq.simulation.generate` |
| `orq.simulation.num_scenarios` | Requested scenario count | `Evaluatorq - Agent Simulation` (generate_and_simulate only), `orq.simulation.generate` |
| `orq.simulation.datapoints_count` | Resolved datapoint count | `Evaluatorq - Agent Simulation` only |
| `orq.evaluatorq_run_id` | This run's id — see [Run correlation](#run-correlation) | `Evaluatorq - Agent Simulation`, `orq.simulation.generate` |

Span attributes on `orq.simulation.run`:

| Attribute | Value |
|---|---|
| `orq.simulation.persona` | Persona name for this datapoint |
| `orq.simulation.scenario` | Scenario name for this datapoint |
| `orq.simulation.max_turns` | Effective max turns for this run |
| `orq.simulation.model` | Model driving the user-simulator/judge |
| `orq.thread_id` | Orq thread id (`{run_id}:{index}`) grouping this conversation's calls |
| `orq.simulation.terminated_by` | How the conversation ended (set on the error exit path, e.g. `"error"`) |
| `orq.simulation.goal_achieved` | Whether the judge scored the goal as achieved |
| `orq.simulation.turn_count` | Number of turns completed |

Span attributes on `orq.simulation.first_message_generation`:

Under the root (one span covering the whole persona x scenario sweep):

| Attribute | Value |
|---|---|
| `orq.simulation.model` | Model used for generation |
| `orq.simulation.pair_count` | persona x scenario pairs attempted |
| `orq.simulation.persona_count` / `orq.simulation.scenario_count` | Input counts |
| `orq.simulation.generated_count` | Datapoints successfully generated |
| `orq.simulation.failed_count` | Pairs that failed (each also gets an `orq.simulation.first_message_generation_failed` span event with persona, scenario, and error) |

Under `orq.simulation.run` (only when a datapoint carried no pre-generated first message):

| Attribute | Value |
|---|---|
| `orq.simulation.persona` | Persona name for this datapoint |
| `orq.simulation.scenario` | Scenario name for this datapoint |
| `orq.simulation.model` | Model used for generation |

Span attributes on `orq.simulation.turn`:

| Attribute | Value |
|---|---|
| `orq.simulation.turn` | 1-based turn number |
| `orq.simulation.max_turns` | Effective max turns for this run |
| `orq.simulation.goal_achieved` | Whether the judge scored the goal as achieved this turn |
| `orq.simulation.goal_completion_score` | Judge's goal-completion score |
| `orq.simulation.should_terminate` | Whether the judge signalled the conversation should end |

`orq.simulation.target_call`, `orq.simulation.judge_evaluation`, and
`orq.simulation.user_simulator_call` carry no span attributes of their own — they
exist purely to scope the nested LLM call (and, for `target_call`, the target's own
input/output recording). LLM spans nested under `judge_evaluation` and
`user_simulator_call` carry the same GenAI attributes as the red teaming LLM spans
above, tagged via `orq.llm.purpose`.

### Judge-panel spans

`llm_jury()`, `run_jury()`, and `run_pairwise()` (`src/evaluatorq/common/jury.py`,
`src/evaluatorq/pairwise.py`) run a panel of judges under a shared span
hierarchy:

```
orq.evaluation {evaluator}         # from the core runner, when a jury backs an evaluator
  └── orq.jury                     # one per deliberation (orq.pairwise_jury in comparative mode)
        └── orq.judge              # one per judge (x2 in comparative mode — see below)
              └── chat {model}     # the judge's own LLM call(s), tagged orq.llm.purpose="judge"
```

The panel opens no span of its own outside `orq.jury` — it can equally be
called standalone (not nested under `orq.evaluation`), in which case `orq.jury`
is the root. All jury/judge spans are opened via `evaluatorq.common.tracing`'s
`with_span()`, so — like every other span in this document — they are a no-op
when tracing is disabled; verdicts and aggregation are unaffected either way.

A judge whose call failed leaves its `orq.judge` span with OTel status `ERROR`
(via `set_span_error`), but the failure is swallowed at the panel level — the
jury carries on with whatever judges succeeded (or promotes a replacement) and
the parent `orq.jury` span stays OK.

Span attributes on `orq.judge`:

| Attribute | Value |
|---|---|
| `judge.name` | Judge model ID |
| `judge.model` | Judge model ID (same value as `judge.name`) |
| `judge.verdict` | Stringified verdict, always in the canonical frame (bool / float / str all coerce to `str`); unset when the vote has no value |
| `judge.success` | Whether the judge produced a usable outcome (decisive or abstained) |
| `judge.abstained` | Whether the judge explicitly abstained |
| `judge.replacement` | Whether this judge stood in for a failed configured judge |
| `judge.label_swapped` | Comparative (pairwise) mode only — which ordering this vote was cast in |
| `judge.latency_ms` | Wall-clock time for this judge's repetitions |
| `judge.error` | Error string when the judge failed (truncated per `EVALUATORQ_SPAN_MAX_TEXT_CHARS`) |
| `judge.repetitions_failed` | Count of repetitions that failed to produce a usable verdict (an error, or a non-decisive non-abstained pass; a clean abstention is not counted), out of the configured repetition count |

No token usage or cost here: those are recorded once, on the `chat` spans
underneath, and rolled up by the consumer. Stamping them on every ancestor as
well made the same tokens appear three times in one trace.

`judge.label_swapped` is only ever set (`True`/`False`) in comparative mode —
in plain `run_jury()` deliberations it is absent, since each judge votes once.

Span attributes on `orq.jury`:

| Attribute | Value |
|---|---|
| `jury.verdict` | Stringified panel verdict |
| `jury.aggregator` | Consensus rule name: one of the `aggregator=` keywords (`mode`, `majority`, `mean_std`, `median`, `min`, `max`), `custom` for a caller-supplied callable, or `pairwise_plurality` — see the note below |
| `jury.min_successful_judges` | Configured quorum |
| `jury.raw_agreement` | Modal-vote share among decisive votes; unset when inconclusive |
| `jury.judges_configured` | Panel size |
| `jury.judges_succeeded` | Judges that cast a decisive vote |
| `jury.judges_failed` | Judges that failed outright |
| `jury.replacements_used` | Number of stand-in judges promoted |
| `jury.tie` | Whether the verdict came from a tie-break |
| `jury.inconclusive` | Whether the panel failed to reach quorum |

`pairwise_plurality` is a **reported value, not an accepted argument** — you
cannot pass it to `aggregator=`, and `validate_aggregator()` rejects it. It
names the rule `run_pairwise()` applies internally: `pairwise_consensus()`,
a strict plurality over *reconciled pair* votes, run after judges that flipped
across the two orderings have already been dropped to abstentions. The six
`aggregator=` keywords reduce raw per-judge votes instead, so labelling this
one `mode` would name it after a function it does not call.

#### Comparative (pairwise) mode

`run_pairwise()` compares two responses (A vs. B) and, to control for position
bias, runs every judge in **both** label orderings. This changes the span
shape from the plain jury case:

- **One `orq.pairwise_jury` span covers the whole comparison** — both orderings
  drive the same span rather than each minting its own; `run_pairwise` calls the
  internal `_run_jury_core` directly (not `run_jury`) so it doesn't open a
  second jury span per ordering.
- **Each judge appears twice** under that one `orq.pairwise_jury` span — one
  `orq.judge` span per ordering, distinguished by `judge.label_swapped`
  (`False` for the A/B ordering, `True` for the swapped B/A ordering).
- **The span is named `orq.pairwise_jury`, not `orq.jury`** — it aggregates
  reconciled *pair* votes rather than raw per-judge votes, so it gets its own
  name rather than masquerading as a plain jury. Its attributes stay in the
  `jury.*` namespace, plus these comparative-only extras:

| Attribute | Value |
|---|---|
| `jury.flipped` | Count of judges that contradicted themselves across the two orderings (position bias) |
| `jury.flipped_judges` | Comma-separated model names of the flipped judges |
| `jury.swap` | Whether the comparison ran both orderings (`swap=True`, the default) or only one |

**`judge.verdict` is already un-swapped in comparative mode.** The labels a judge
returns there name a *position*, not a response: a judge that picks the same
response both times says `A` in one ordering and `B` in the other, which reads as
a self-contradiction and is in fact the opposite, a perfectly consistent judge.
`label_swapped=True` spans are mapped back to the canonical frame before the
attribute is written, so "how often did this judge pick response A" is answerable
from `judge.verdict` alone, with no join against `judge.label_swapped`.

There is deliberately no raw-frame twin. The text the verdict was parsed from is
one level down, on the `chat` child's `gen_ai.output.messages`, so a second
attribute here would only restate what the trace already holds — the same
reasoning as the alias removal noted above.

Un-swapping is per-ordering and needs nothing but `label_swapped`. *Flip
detection* is what needs both orderings, and it stays on the parent —
`jury.flipped_judges` names the judges that really did follow slot order.

`jury.flipped` counts judges that answered in both orderings but disagreed
with themselves — that is position bias, not a failure, so a flipped judge is
deliberately excluded from `jury.judges_failed`: `judges_failed` counts only
judges with no reconciled vote *and* no flip (i.e. one or both orderings
raised an error). A judge can be flipped, failed, or a normal decisive vote,
but never counted under more than one of those buckets.

`orq.evaluation`, `orq.jury` / `orq.pairwise_jury`, `orq.judge`, and the nested `chat {model}` LLM
spans follow the ambient OTel context — nothing threads an explicit parent
across the `orq.evaluation` → `orq.jury` seam, so a jury backing a custom
evaluator's scorer nests correctly without extra plumbing.

## Run correlation

Every LLM invocation issued during a `red_team()` or simulation run (`simulate()`,
`generate_and_simulate()`, or `generate()`) is tagged so an operator can filter
Orq's trace UI down to exactly the model calls belonging to one run. The same
metadata is inherited by `generate_personas()` and `generate_scenarios()` when
they are called inside an outer simulation or red-team scope; standalone calls
have no synthetic root run id.

| Surface | Key | Where |
|---|---|---|
| Request `metadata` on every LLM invocation | `evaluatorq_run_id` | red-team + simulation runs, including inherited nested work |
| Root span attribute | `orq.evaluatorq_run_id` | `Evaluatorq - Red Teaming` root span; `Evaluatorq - Agent Simulation` / `orq.simulation.generate` root spans |

A companion key rides the same rail: `evaluatorq_pipeline`, whose value is
`"red_teaming"` or `"agent_simulation"`. It identifies which surface issued the call
and is sent as request metadata alongside `evaluatorq_run_id` — filter on it to
separate red-team traffic from simulation traffic regardless of run. Both
`evaluatorq_run_id` and `evaluatorq_pipeline` are native request `metadata` fields
on Chat Completions and Responses calls. They are sent to direct
OpenAI-compatible endpoints as well as through the Orq router.

### How it reaches every call

Both red-team and simulation route their datapoints through a nested `evaluatorq()`
call. The run id isn't threaded through function arguments — it's bound to a
`contextvars.ContextVar` (`src/evaluatorq/common/thread_context.py`) at the run's
entrypoint and read back at the call site. Because a `ContextVar` set in an ancestor
scope is visible to nested calls (and copied into child `asyncio` tasks), every LLM
call issued from inside the nested `evaluatorq()` run automatically carries the SAME
`evaluatorq_run_id` as the outer red-team/sim run — no explicit plumbing required.

Call sites read it back one of two ways, and the difference matters when you are
tracking down a missing tag:

- **Chat Completions** (`create` / `.parse`) and **Responses** calls read the same
  context and send it as native request `metadata`.
- The router-specific `thread` body parameter is separate and remains endpoint-
  gated: it is included only when the client routes through Orq and a conversation
  thread is active. It is never required for run correlation.

Separate root invocations receive separate ids: two calls to `simulate()`,
`generate_and_simulate()`, or `generate()` each get a distinct
`evaluatorq_run_id`, even if called back-to-back in the same process. Nested
`evaluatorq()` work within one red-team or simulation root receives that root's id,
and nested generation helpers inherit it. Standalone `generate_personas()` and
`generate_scenarios()` do not mint ids of their own. The evaluatorq-core
`orq.run_id` attributes continue to describe evaluatorq evaluation runs and are
unchanged by this correlation mechanism.

### Using it

In Orq's trace UI, filter spans/traces on the `evaluatorq_run_id` request-metadata
value (copy it from the `orq.evaluatorq_run_id` attribute on the run's root span, or
from your own logs/hooks that captured the run id) to see every model call — target,
judge, user-simulator, attacker, evaluator, generation — that belongs to one
`red_team()` or `simulate()`/`generate_and_simulate()`/`generate()` invocation, including calls
made through the nested `evaluatorq()` run. Add `evaluatorq_pipeline` to the filter to
scope further to just red-team or just simulation traffic.

## Content capture and truncation

Two env vars control how much text is stored on spans:

- **`EVALUATORQ_CAPTURE_MESSAGE_CONTENT`** (default `true`): set to `false` or
  `0` to keep LLM message content out of traces entirely. Token counts and
  model name are still recorded.
- **`EVALUATORQ_SPAN_MAX_TEXT_CHARS`** (default: no limit): set to a positive
  integer to truncate span text attributes. Truncated strings end with
  `... [truncated]`.

## W3C trace context propagation

To propagate trace context across service boundaries, inject the active span's
W3C `traceparent`/`tracestate` headers into your outgoing HTTP requests. Use the
OpenTelemetry SDK's public `inject()` helper — a stable, supported API:

```python
from opentelemetry.propagate import inject

headers: dict[str, str] = {}
inject(headers)          # writes `traceparent` (+ `tracestate`) for the active span
# pass `headers` into your outgoing request, e.g. httpx.get(url, headers=headers)
```

`inject()` is a no-op when no span is active, so it is safe to call whenever
OpenTelemetry is installed. (The `from opentelemetry.propagate import inject`
import itself requires OTel; if you need code that also runs without it
installed, use the internal helper below, which degrades to an empty dict.)

!!! note "Internal convenience helper"
    evaluatorq also ships `get_trace_context_headers()` in
    `evaluatorq.common.tracing`, an `async` helper you `await` for the same
    headers as a dict (empty when OTel is unavailable). It is an internal
    utility — **not** re-exported
    from the public `evaluatorq.tracing` namespace, and its import path may
    change without a deprecation cycle. Prefer the OpenTelemetry `inject()` path
    above for anything stable.

## Where to next

- **[Configuration](configuration.md)** — API keys and environment variables.
- **[CLI Reference](cli-reference/overview.md)** — run evaluations and red-team/sim from the terminal.
- **[Orq Deployment](orq-deployment.md)** — trace invocations against an Orq-hosted deployment.
