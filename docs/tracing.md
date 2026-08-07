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
parent span.

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
| `gen_ai.input.messages` | JSON serialised input messages (gated by `EVALUATORQ_CAPTURE_MESSAGE_CONTENT`) |
| `gen_ai.output.messages` | JSON serialised output messages (gated by `EVALUATORQ_CAPTURE_MESSAGE_CONTENT`) |
| `orq.llm.purpose` | Cross-domain purpose tag (e.g. `"adversarial"`, `"evaluation"`, `"target"`) |

The root `Evaluatorq - Red Teaming` span additionally carries:

| Attribute | Value |
|---|---|
| `orq.evaluatorq_run_id` | This run's id — see [Run correlation](#run-correlation) |

### Simulation spans

```
Evaluatorq - Agent Simulation    # root — one per simulate() / generate_and_simulate() call
  ├── chat/responses {model}     # persona/scenario/first-message generation calls
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
