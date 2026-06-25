# Configuration

All configuration is via environment variables. No config file is required.

## Environment variables

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `ORQ_API_KEY` | Required for Orq features | — | Authenticates against the Orq platform. Required to fetch datasets, upload results, and invoke deployments. Also auto-enables OpenTelemetry tracing (spans are sent to `https://my.orq.ai/v2/otel`). |
| `ORQ_BASE_URL` | No | `https://my.orq.ai` | Overrides the Orq API base URL. Affects Orq SDK calls (dataset fetch, deployment invocation) and the derived OTLP tracing endpoint (`<ORQ_BASE_URL>/v2/otel`). Does **not** redirect OpenAI-compatible LLM calls — use `OPENAI_BASE_URL` for that. |
| `OPENAI_API_KEY` | Required if not using Orq | — | API key for the OpenAI (or compatible) backend. Used by the red teaming pipeline and agent simulation when `ORQ_API_KEY` is absent. |
| `OPENAI_BASE_URL` | No | OpenAI default | Redirect OpenAI-compatible calls to a different host (vLLM, OpenRouter, Azure, local). Honoured by the red teaming and simulation LLM client. |
| `ORQ_DISABLE_TRACING` | No | unset | Set to `1` or `true` to suppress all OpenTelemetry spans even when `ORQ_API_KEY` or `OTEL_EXPORTER_OTLP_ENDPOINT` is present. |
| `ORQ_DEBUG` | No | unset | Set to any non-empty value to print tracing setup diagnostics to stdout (endpoint, auth headers, initialization errors). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | — | Explicit OTLP HTTP endpoint. Takes precedence over the `ORQ_BASE_URL`-derived endpoint. See [Tracing](tracing.md). |
| `OTEL_EXPORTER_OTLP_HEADERS` | No | — | Comma-separated `key=value` pairs added to every OTLP export request. Format: `key1=value1,key2=value2`. |
| `OTEL_SERVICE_NAME` | No | `evaluatorq` | Service name recorded on every span's `service.name` resource attribute. |
| `OTEL_SERVICE_VERSION` | No | `1.0.0` | Service version recorded on every span's `service.version` resource attribute. |
| `EVALUATORQ_CAPTURE_MESSAGE_CONTENT` | No | `true` | Set to `false` or `0` to strip LLM message content (prompts and responses) from spans. Token counts, model name, and latency are still recorded. Useful when exporting to third-party backends or to avoid capturing PII. |
| `EVALUATORQ_SPAN_MAX_TEXT_CHARS` | No | unset (no limit) | Maximum characters per span text attribute. Set a positive integer (e.g. `8192`) to truncate long strings. Unset or `0` / `-1` means capture all. |
| `EVALUATORQ_LLM_TIMEOUT_S` | No | `60.0` | Per-LLM-call timeout in seconds. **Simulation only** — has no effect on red teaming or core evaluation. Increase for slow self-hosted endpoints. |
| `EVALUATORQ_LLM_MAX_TOKENS` | No | `8192` | Maximum completion tokens per LLM call. **Simulation only** — has no effect on red teaming or core evaluation. Increase for reasoning models that exhaust the default budget before emitting a tool call. |
| `EVALUATORQ_REASONING_EFFORT` | No | `medium` | Reasoning effort hint passed to reasoning-capable models. **Simulation only** — has no effect on red teaming or core evaluation. Set to `""`, `none`, or `off` to omit the parameter entirely. |

## `.env` file

The library itself does not call `load_dotenv()`. The examples ship with `python-dotenv` calls in their scripts. To load a `.env` file in your own code, call `load_dotenv()` before importing evaluatorq:

```python
from dotenv import load_dotenv

load_dotenv()  # must run before evaluatorq reads env vars

from evaluatorq import evaluatorq, DataPoint
```

A minimal `.env` for Orq platform use:

```dotenv
ORQ_API_KEY=your_orq_api_key_here
```

With OpenAI as the LLM backend (red teaming / simulation, no Orq):

```dotenv
OPENAI_API_KEY=sk-...
```

Self-hosted LLM endpoint:

```dotenv
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:8000/v1
```

To send traces to a custom OTLP collector instead of Orq:

```dotenv
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

## Notes

- `EVALUATORQ_OWASP_DATASET_ID` is listed in `CLAUDE.md` but is not present in the current source tree; it was not found in any `os.getenv` / `os.environ.get` call and is not documented here.
- `ROUTER_BASE_URL` was a predecessor to `ORQ_BASE_URL`. Setting it now triggers a deprecation warning; use `ORQ_BASE_URL` instead.
