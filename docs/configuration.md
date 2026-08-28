# Configuration

All configuration is via environment variables. No config file is required.

## Environment variables

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `ORQ_API_KEY` | Required for Orq features | — | Authenticates against the Orq platform. Required to fetch datasets, upload results, and invoke deployments. Also auto-enables OpenTelemetry tracing (spans are sent to `https://my.orq.ai/v2/otel`). |
| `ORQ_BASE_URL` | No | `https://my.orq.ai` | Overrides the Orq API base URL. Affects Orq SDK calls (dataset fetch, deployment invocation) and the derived OTLP tracing endpoint (`<ORQ_BASE_URL>/v2/otel`). Does **not** redirect OpenAI-compatible LLM calls — use `OPENAI_BASE_URL` for that. |
| `ROUTER_BASE_URL` | Deprecated | — | Predecessor to `ORQ_BASE_URL`, no longer honoured. If set (and `ORQ_BASE_URL` is unset) it only logs a warning. Use `ORQ_BASE_URL` instead. |
| `OPENAI_API_KEY` | Red-team / sim only, if not using Orq | — | API key for the OpenAI (or compatible) backend. Used by the red teaming pipeline and agent simulation when `ORQ_API_KEY` is absent. Not required for core `evaluatorq()` evaluation. |
| `OPENAI_BASE_URL` | No | OpenAI default | Redirect OpenAI-compatible calls to a different host (vLLM, OpenRouter, Azure, local). Honoured by the red teaming and simulation LLM client. |
| `ORQ_DISABLE_TRACING` | No | unset | Set to `1` or `true` to suppress all OpenTelemetry spans even when `ORQ_API_KEY` or `OTEL_EXPORTER_OTLP_ENDPOINT` is present. |
| `ORQ_DEBUG` | No | unset | Set to any non-empty value to print tracing setup diagnostics to stdout (endpoint, auth headers, initialization errors). |
| `EQ_DEBUG` | No | unset | Set to any non-empty value to show the full traceback on CLI errors instead of the one-line message. CLI-wide; distinct from `ORQ_DEBUG`, which only affects tracing diagnostics. |
| `EVALUATORQ_DIR` | No | `.evaluatorq` in the current directory | Base directory for the run store, where both red teaming (`runs/`) and simulation (`sim-runs/`) persist reports. Must point at the store directory itself (e.g. `/tmp/x/.evaluatorq`), not its parent — only the working-directory fallback appends `.evaluatorq`. Empty is treated as unset. |
| `EVALUATORQ_LOG_LEVEL` | No | `INFO` | Log level for the dashboard server. Accepts any level name (e.g. `DEBUG`). |
| `ORQ_WORKSPACE` / `ORQ_WORKSPACE_SLUG` | No | unset | Workspace slug used to build dashboard deep-links into the Orq UI. `ORQ_WORKSPACE` wins when both are set. When neither is set, the deep-link buttons are hidden. See [Dashboard](dashboard.md). |
| `ORQ_UI_BASE_URL` | No | `ORQ_BASE_URL`, else `https://my.orq.ai` | Base URL for dashboard deep-links into the Orq UI. Set this when the UI host differs from the API host. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | — | Explicit OTLP HTTP endpoint. Takes precedence over the `ORQ_BASE_URL`-derived endpoint. See [Tracing](tracing.md). |
| `OTEL_EXPORTER_OTLP_HEADERS` | No | — | Comma-separated `key=value` pairs added to every OTLP export request. Format: `key1=value1,key2=value2`. |
| `OTEL_SERVICE_NAME` | No | `evaluatorq` | Service name recorded on every span's `service.name` resource attribute. |
| `OTEL_SERVICE_VERSION` | No | `1.0.0` | Service version recorded on every span's `service.version` resource attribute. |
| `ORQ_OTEL_MAX_QUEUE_SIZE` | No | `4096` | Maximum spans buffered by the `BatchSpanProcessor`. When it is full the SDK evicts the oldest buffered span and logs a warning on the stdlib `opentelemetry` logger. See [Tracing → Batching and flush](tracing.md#batching-and-flush). |
| `ORQ_OTEL_MAX_BATCH_SIZE` | No | `512` | Maximum spans per OTLP export request. A value larger than the queue size is clamped down to it with a `WARNING`. |
| `ORQ_OTEL_SCHEDULE_DELAY_MS` | No | `5000` | Milliseconds between scheduled batch exports. Lower it to drain the queue more eagerly. |
| `ORQ_OTEL_FLUSH_TIMEOUT_MS` | No | `5000` | Milliseconds the end-of-run force-flush blocks before giving up and logging a warning. Read per run, so a long-lived host can raise it before a big run. Bounds the final flush only — the per-request export timeout is a fixed 5s. |
| `EVALUATORQ_CAPTURE_MESSAGE_CONTENT` | No | `true` | Set to `false` or `0` to strip LLM message content (prompts and responses) from spans. Token counts, model name, and latency are still recorded. Useful when exporting to third-party backends or to avoid capturing PII. |
| `EVALUATORQ_SPAN_MAX_TEXT_CHARS` | No | unset (no limit) | Maximum characters per span text attribute. Set a positive integer (e.g. `8192`) to truncate long strings. Unset or `0` / `-1` means capture all. |
| `EVALUATORQ_LLM_TIMEOUT_S` | No | `60.0` | Per-LLM-call timeout in seconds. **Simulation only** — has no effect on red teaming or core evaluation. A fallback default: `LLMCallConfig.timeout_ms` on the agent's config wins when set. Read at call time, so setting it after import takes effect. Increase for slow self-hosted endpoints; for the *target's* timeout rather than the simulator's, pass `target_agent_timeout_ms` to `simulate()`. |
| `EVALUATORQ_LLM_MAX_TOKENS` | No | `10000` | Maximum completion tokens per LLM call. **Simulation only** — red teaming shares the same default (`DEFAULT_TARGET_MAX_TOKENS`) but is tuned per role via `LLMConfig.max_tokens` / `EvaluatorConfig.max_tokens`, not by this variable. A fallback default: `LLMCallConfig.max_tokens` on the agent's config wins when set. Read at call time, so setting it after import takes effect. Increase for reasoning models that exhaust the default budget before emitting a tool call. |
| `EVALUATORQ_REASONING_EFFORT` | No | *(unset — parameter not sent)* | Reasoning effort hint for the **simulator's own** LLM calls (user simulator, judge) — not for the agent under test, which takes `target_reasoning_effort` on `simulate()` / `red_team()`. **Simulation only.** There is no global default — unset means the parameter is omitted and the model uses its own. A fallback: `LLMCallConfig.reasoning_effort` on the agent's config wins when set. Read at call time, so setting it after import takes effect. Set to `""`, `none`, or `off` to omit the parameter entirely. |
| `EVALUATORQ_APPLY_MODEL` | No | `openai/gpt-5.6-luna` | Model used by the dashboard's apply-recommendations merge. Shown in the dashboard config panel. See [Dashboard](dashboard.md). |
| `EVALUATORQ_DASHBOARD_ROOTS` | No | unset | Colon-separated run-store roots the dashboard serves. Set automatically for the reloader subprocess; set it yourself only to point the dashboard at stores outside `EVALUATORQ_DIR`. |
| `EVALUATORQ_CATALOGUE_TIMEOUT_S` | No | `30` | HTTP timeout in seconds for the `GET /v2/models` model-catalogue fetch, read once at import (unlike the three `EVALUATORQ_LLM_*` vars above, which are read at call time). The catalogue supplies per-model prices, Responses support, and accepted reasoning-effort values. A failed fetch is **not** cached immediately: the first two failures leave the catalogue unloaded so a later call can still succeed, and only the third gives up and degrades the rest of the process to unpriced and chat-completions-only. Each failure logs a warning naming the attempt number. |

## Model catalogue overrides

Prices, provider ids, Responses support and accepted reasoning-effort values come
from Orq's `GET /v2/models`, fetched once per process. A model that catalogue does
not list — a self-hosted deployment, or one newer than your workspace's catalogue —
degrades silently in three ways: the call stays unpriced, `qualified_model()` sends
it to Chat Completions instead of Responses, and its reasoning effort cannot be
pre-validated.

Register an entry to fix that. Registered entries take priority over the fetched
catalogue, so this also corrects an entry that is wrong:

```python
from evaluatorq.common.model_catalogue import ModelInfo, get_model_info, register_model

register_model(
    'my-self-hosted-llama',
    ModelInfo(
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.0008,
        provider='self',
        supports_responses=False,
        reasoning_efforts=None,  # None = "unknown", never "nothing allowed"
    ),
)

info = await get_model_info('my-self-hosted-llama')
```

Costs are USD **per 1000 tokens**, matching what `/v2/models` publishes.

The id is stored unprefixed, so `'openai/gpt-x'` and `'gpt-x'` register and resolve
the same entry — register either spelling and both lookups find it. Registering
both replaces rather than duplicates: there is one model. `reasoning_efforts=None`
means "unknown, cannot pre-validate"; an empty set means the same thing and is
normalized to `None`, because a literally-empty accepted-values list would reject
every effort including the defaults.

## `.env` file

The library itself does not call `load_dotenv()`. The examples ship with `python-dotenv` calls in their scripts. To load a `.env` file in your own code, call `load_dotenv()` before importing evaluatorq:

```python
from dotenv import load_dotenv

load_dotenv()  # must run before evaluatorq reads env vars

from evaluatorq import evaluatorq, DataPoint
```

A minimal `.env` for Orq platform use:

```ini
ORQ_API_KEY=your_orq_api_key_here
```

With OpenAI as the LLM backend (red teaming / simulation, no Orq):

```ini
OPENAI_API_KEY=sk-...
```

Self-hosted LLM endpoint:

```ini
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:8000/v1
```

To send traces to a custom OTLP collector instead of Orq:

```ini
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

## Where to next

- **[Getting Started](guides/getting-started.md)** — run your first evaluation.
- **[Orq Deployment](orq-deployment.md)** — target an Orq-hosted deployment.
- **[Tracing](tracing.md)** — enable OpenTelemetry tracing.
