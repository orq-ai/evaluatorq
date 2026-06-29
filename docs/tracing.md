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
pip install opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-http \
    opentelemetry-semantic-conventions
# or via the extras bundle:
pip install "evaluatorq[otel]"
```

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
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 python my_eval.py
```

To debug tracing setup:

```bash
ORQ_DEBUG=1 python my_eval.py
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
orq.redteam.pipeline             # root — one per red_team() call
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
  └── orq.redteam.memory_cleanup # post-run agent memory entity cleanup
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
    `evaluatorq.common.tracing`, which returns the same headers as a dict (empty
    when OTel is unavailable). It is an internal utility — **not** re-exported
    from the public `evaluatorq.tracing` namespace, and its import path may
    change without a deprecation cycle. Prefer the OpenTelemetry `inject()` path
    above for anything stable.
