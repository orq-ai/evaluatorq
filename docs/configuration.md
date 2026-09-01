# Configuration

Everything is configured through environment variables. There is no config file, and nothing to set up before your first run: evaluatorq with no environment at all still runs local jobs and local evaluators.

You only add variables when you want something more — a hosted model, a dataset from Orq, traces, a dashboard.

## Get started in two steps

**1. Pick where your LLM calls go.** One of these three, in a `.env` or in your shell:

=== "Orq platform"

    ```ini
    ORQ_API_KEY=your_orq_api_key_here
    ```

    Gives you Orq datasets, deployments, the model catalogue, and tracing, which switches on by itself.

=== "OpenAI directly"

    ```ini
    OPENAI_API_KEY=sk-...
    ```

    Enough for red teaming and simulation without an Orq account.

=== "Self-hosted / any OpenAI-compatible host"

    ```ini
    OPENAI_API_KEY=dummy
    OPENAI_BASE_URL=http://localhost:8000/v1
    ```

    vLLM, OpenRouter, Azure, Ollama — anything that speaks the OpenAI API. Red teaming honours `OPENAI_BASE_URL`; simulation does not, so pass a pre-built client there instead.

**2. Get it into the process.** Exporting the variables in your shell is enough, and needs nothing installed. evaluatorq never reads a `.env` file itself — if you keep one, load it yourself, and install `python-dotenv` first (`uv add python-dotenv`; it is not a dependency of evaluatorq):

```python
from dotenv import load_dotenv

load_dotenv()  # must run before evaluatorq reads env vars

from evaluatorq import DataPoint, evaluatorq
```

Head to [Getting Started](guides/getting-started.md) and run something.

## What to set first

Four decisions cover almost every real configuration. The rest of this page is reference material you can read when you hit a specific need.

| Decision | Variable | Default | Notes |
|---|---|---|---|
| **Which backend runs my LLM calls?** | `ORQ_API_KEY` or `OPENAI_API_KEY` | — | Set one. `ORQ_API_KEY` unlocks datasets, deployments and tracing, and wins when both are set; `OPENAI_API_KEY` (plus `OPENAI_BASE_URL` for a non-OpenAI host) is the standalone route. |
| **Where do run reports land?** | `EVALUATORQ_DIR` | `.evaluatorq` in the current directory | The run store that red teaming (`runs/`) and simulation (`sim-runs/`) write to, and that the [dashboard](dashboard.md) reads. |
| **Do I want traces?** | `ORQ_DISABLE_TRACING` | off (traces enabled when `ORQ_API_KEY` **or** `OTEL_EXPORTER_OTLP_ENDPOINT` is set) | Set to `1` to send nothing. Point traces elsewhere with `OTEL_EXPORTER_OTLP_ENDPOINT`. See [Tracing](tracing.md). |
| **Do dashboard links open my Orq workspace?** | `ORQ_WORKSPACE` | unset | Your workspace slug. Unset hides the deep-link buttons — nothing else breaks. |

Two more worth knowing before you need them: `EQ_DEBUG=1` turns a one-line CLI error into a full traceback, and `EVALUATORQ_CAPTURE_MESSAGE_CONTENT=false` keeps prompts and responses out of your spans.

## Full environment variable reference

### Backend, credentials and model catalogue

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `ORQ_API_KEY` | Required for Orq features | — | Authenticates against the Orq platform. Required to fetch datasets, upload results, and invoke deployments. Also auto-enables OpenTelemetry tracing (spans go to `<ORQ_BASE_URL>/v2/otel`, so `https://my.orq.ai/v2/otel` by default). |
| `ORQ_BASE_URL` | No | `https://my.orq.ai` | Overrides the Orq API base URL. Affects Orq SDK calls (dataset fetch, deployment invocation) and the derived OTLP tracing endpoint (`<ORQ_BASE_URL>/v2/otel`). Does **not** redirect OpenAI-compatible LLM calls — use `OPENAI_BASE_URL` for that. |
| `OPENAI_API_KEY` | Red-team / sim only, if not using Orq | — | API key for the OpenAI (or compatible) backend. Used by the red teaming pipeline and agent simulation when `ORQ_API_KEY` is absent. Not required for core `evaluatorq()` evaluation. |
| `OPENAI_BASE_URL` | No | OpenAI default | Redirect OpenAI-compatible calls to a different host (vLLM, OpenRouter, Azure, local). Honoured by the red teaming LLM client. **Not** honoured on the simulation path, which resolves its client with `honor_openai_base_url=False` — inject a pre-built client there instead. |
| `EVALUATORQ_CATALOGUE_TIMEOUT_S` | No | `30` | HTTP timeout in seconds for the `GET /v2/models` model-catalogue fetch, read once at import (unlike the three `EVALUATORQ_LLM_*` vars under [Simulation LLM defaults](#simulation-llm-defaults), which are read at call time). The catalogue supplies per-model prices, Responses support, and accepted reasoning-effort values. A failed fetch is **not** cached immediately: the first two failures leave the catalogue unloaded so a later call can still succeed, and only the third gives up and degrades the rest of the process to unpriced and chat-completions-only. Each failure logs a warning naming the attempt number. |

### Runs, storage and output

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `EVALUATORQ_DIR` | No | `.evaluatorq` in the current directory | Base directory for the run store, where both red teaming (`runs/`) and simulation (`sim-runs/`) persist reports. Must point at the store directory itself (e.g. `/tmp/x/.evaluatorq`), not its parent — only the working-directory fallback appends `.evaluatorq`. Empty is treated as unset. |
| `EQ_DEBUG` | No | unset | Set to any non-empty value to show the full traceback on CLI errors instead of the one-line message. CLI-wide; distinct from `ORQ_DEBUG`, which only affects tracing diagnostics. |
| `EVALUATORQ_LOG_LEVEL` | No | `INFO` | Log level for the dashboard server. Accepts any level name (e.g. `DEBUG`). |

### Dashboard

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `ORQ_WORKSPACE` / `ORQ_WORKSPACE_SLUG` | No | unset | Workspace slug used to build dashboard deep-links into the Orq UI. `ORQ_WORKSPACE` wins when both are set. When neither is set, the deep-link buttons are hidden. See [Dashboard](dashboard.md). |
| `ORQ_UI_BASE_URL` | No | `ORQ_BASE_URL`, else `https://my.orq.ai` | Base URL for dashboard deep-links into the Orq UI. Set this when the UI host differs from the API host. |
| `EVALUATORQ_APPLY_MODEL` | No | `openai/gpt-5.6-luna` | Model used by the dashboard's apply-recommendations merge. Shown in the dashboard config panel. See [Dashboard](dashboard.md). |
| `EVALUATORQ_DASHBOARD_ROOTS` | No | unset | Internal. A JSON array of run-store roots, set by `eq dashboard` for its reloader subprocess and parsed back with `json.loads`. Pass extra stores as positional CLI paths rather than setting this by hand. |

### Tracing

See [Tracing](tracing.md) for the full picture.

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `ORQ_DISABLE_TRACING` | No | unset | Set to `1` or `true` to suppress all OpenTelemetry spans even when `ORQ_API_KEY` or `OTEL_EXPORTER_OTLP_ENDPOINT` is present. |
| `ORQ_DEBUG` | No | unset | Set to any non-empty value to print tracing setup diagnostics to stdout (endpoint, auth headers, initialization errors). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | — | Explicit OTLP HTTP endpoint. Takes precedence over the `ORQ_BASE_URL`-derived endpoint. |
| `OTEL_EXPORTER_OTLP_HEADERS` | No | — | Comma-separated `key=value` pairs added to every OTLP export request. Format: `key1=value1,key2=value2`. |
| `OTEL_SERVICE_NAME` | No | `evaluatorq` | Service name recorded on every span's `service.name` resource attribute. |
| `OTEL_SERVICE_VERSION` | No | `1.0.0` | Service version recorded on every span's `service.version` resource attribute. |
| `ORQ_OTEL_MAX_QUEUE_SIZE` | No | `4096` | Maximum spans buffered by the `BatchSpanProcessor`. When it is full the SDK evicts the oldest buffered span and logs `Queue full, dropping Span.` on the stdlib `opentelemetry.sdk._shared_internal` logger. See [Tracing › Batching and flush](tracing.md#batching-and-flush). |
| `ORQ_OTEL_MAX_BATCH_SIZE` | No | `512` | Maximum spans per OTLP export request. Reaching it wakes the exporter immediately. A value larger than the queue size is clamped to the queue size, with a warning. |
| `ORQ_OTEL_SCHEDULE_DELAY_MS` | No | `5000` | Milliseconds a partial batch waits before it is exported. It does not throttle a full batch. |
| `ORQ_OTEL_FLUSH_TIMEOUT_MS` | No | `5000` | Milliseconds the end-of-run force-flush waits before giving up and logging a warning. Read per run; enforced by evaluatorq rather than the SDK. Bounds the final flush only — the per-request export timeout is a fixed 5s. |
| `EVALUATORQ_CAPTURE_MESSAGE_CONTENT` | No | `true` | Set to `false` or `0` to strip LLM message content (prompts and responses) from spans. Token counts, model name, and latency are still recorded. Useful when exporting to third-party backends or to avoid capturing PII. |
| `EVALUATORQ_PROPAGATE_TRACE_CONTEXT` | No | `true` | Set to `false` or `0` to stop evaluatorq sending W3C `traceparent`/`tracestate` headers on its outgoing LLM and target calls. Default on, so an Orq-hosted deployment or agent nests its server-side spans under the calling span. Turn it off when the receiving side should trace independently, or when a gateway rejects an unexpected `traceparent`. |
| `EVALUATORQ_SPAN_MAX_TEXT_CHARS` | No | unset (no limit) | Maximum characters per span text attribute. Set a positive integer (e.g. `8192`) to truncate long strings. Unset or `0` / `-1` means capture all. |

### Simulation LLM defaults

All three are fallback defaults only: the matching field on the agent's `LLMCallConfig` wins when set, and all three are read at call time, so setting them after import still takes effect.

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `EVALUATORQ_LLM_TIMEOUT_S` | No | `60.0` | Per-LLM-call timeout in seconds. **Simulation only** — has no effect on red teaming or core evaluation. `LLMCallConfig.timeout_ms` wins when set. Increase for slow self-hosted endpoints; for the *target's* timeout rather than the simulator's, pass `target_agent_timeout_ms` to `simulate()`. |
| `EVALUATORQ_LLM_MAX_TOKENS` | No | `10000` | Maximum completion tokens per LLM call. **Simulation only** — red teaming shares the same default (`DEFAULT_TARGET_MAX_TOKENS`) but is tuned per role via `LLMConfig.max_tokens` / `EvaluatorConfig.max_tokens`, not by this variable. `LLMCallConfig.max_tokens` wins when set. Increase for reasoning models that exhaust the default budget before emitting a tool call. |
| `EVALUATORQ_REASONING_EFFORT` | No | *(unset — parameter not sent)* | Reasoning effort hint for the **simulator's own** LLM calls (user simulator, judge) — not for the agent under test, which takes `target_reasoning_effort` on `simulate()` / `red_team()`. **Simulation only.** There is no global default — unset means the parameter is omitted and the model uses its own. `LLMCallConfig.reasoning_effort` wins when set. Set to `""`, `none`, or `off` to omit the parameter entirely. |

## Model catalogue overrides

Prices, provider ids, Responses support and accepted reasoning-effort values come from Orq's `GET /v2/models`, fetched once per process. A model that catalogue does not list — a self-hosted deployment, or one newer than your workspace's catalogue — degrades silently in three ways: the call stays unpriced, `qualified_model()` sends it to Chat Completions instead of Responses, and its reasoning effort cannot be pre-validated.

Register an entry to fix that. Registered entries take priority over the fetched catalogue, so this also corrects an entry that is wrong:

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

The id is stored unprefixed, so `'openai/gpt-x'` and `'gpt-x'` register and resolve the same entry — register either spelling and both lookups find it. Registering both replaces rather than duplicates: there is one model. `reasoning_efforts=None` means "unknown, cannot pre-validate"; an empty set means the same thing and is normalized to `None`, because a literally-empty accepted-values list would reject every effort including the defaults.

## Where to next

- **[Getting Started](guides/getting-started.md)** — run your first evaluation.
- **[Targets](guides/targets.md#orq-hosted-agents-and-deployments)** — point a run at an Orq-hosted agent or deployment.
- **[Tracing](tracing.md)** — enable OpenTelemetry tracing.
