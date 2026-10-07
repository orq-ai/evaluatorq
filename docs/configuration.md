# Configuration

Most settings use environment variables. The dashboard and trace finder can also save defaults in `.evaluatorq/dashboard-settings.json`; nothing needs setting up before your first local job or local evaluator.

You only add variables when you want something more — a hosted model, a dataset from Orq, traces, a dashboard.

## Get started in two steps

**1. Pick where your LLM calls go.** One of these three, in a `.env` or in your shell:

=== "Orq platform"

    ```ini
    ORQ_API_KEY=your_orq_api_key_here
    ```

    Gives you Orq datasets, deployments, the model catalogue, and tracing, which switches on by itself once the OpenTelemetry packages are installed (`uv add "evaluatorq[otel]"`; without them tracing is a silent no-op). See [Tracing](tracing.md).

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

    vLLM, OpenRouter, Azure, Ollama — anything that speaks the OpenAI API. `OPENAI_BASE_URL` is honoured by red teaming's attacker and judge calls and by `OpenAIModelTarget`. It is **not** honoured by `OrqResponsesTarget` or by simulation, which build their client with the host pinned; pass a pre-built client to those instead.

**2. Get it into the process.** Exporting the variables in your shell is enough, and needs nothing installed. `eq dashboard` reads `.env` from the directory where you launch it, without replacing exported values. For Python code and other evaluatorq commands, load the file yourself and install `python-dotenv` first (`uv add python-dotenv`):

```python
from dotenv import load_dotenv

load_dotenv()  # must run before evaluatorq reads env vars

from evaluatorq import DataPoint, evaluatorq
```

Head to [Getting Started](guides/getting-started.md) and run something.

## What to set first

Five decisions cover almost every real configuration. The rest of this page is reference material you can read when you hit a specific need.

| Decision | Variable | Default | Notes |
|---|---|---|---|
| **Which backend runs my LLM calls?** | `ORQ_API_KEY` or `OPENAI_API_KEY` | — | Set one. `ORQ_API_KEY` unlocks datasets, deployments and tracing, and wins when both are set; `OPENAI_API_KEY` (plus `OPENAI_BASE_URL` for a non-OpenAI host) is the standalone route. |
| **Where do run reports land?** | `EVALUATORQ_DIR` | `.evaluatorq` in the current directory | The run store that red teaming (`runs/`) and simulation (`sim-runs/`) write to, and that the [dashboard](dashboard.md) reads. |
| **Do I want traces?** | `ORQ_DISABLE_TRACING` | off (traces enabled when `ORQ_API_KEY` **or** `OTEL_EXPORTER_OTLP_ENDPOINT` is set, and the `otel` extra is installed) | Set to `1`, `true`, `yes` or `on` to send nothing. Point traces elsewhere with `OTEL_EXPORTER_OTLP_ENDPOINT`. See [Tracing](tracing.md). |
| **Which models run each step?** | `EVALUATORQ_FAST_MODEL`, `EVALUATORQ_SMART_MODEL` | `openai/gpt-6-luna`, `openai/gpt-6-sol` | Four model roles cover every LLM call evaluatorq makes. See [Models](#models). |
| **Do dashboard links open my Orq workspace?** | `ORQ_WORKSPACE` | unset | Your workspace slug for runs without an experiment URL. Without one, their deep-link buttons are hidden. |

Two more worth knowing before you need them: `EQ_DEBUG=1` turns a one-line CLI error into a full traceback, and `EVALUATORQ_CAPTURE_MESSAGE_CONTENT=false` keeps prompts and responses out of your spans.

## Models

A **model role** names a kind of work, and each role has one configured model. evaluatorq has four roles, and every LLM call it makes on your behalf belongs to one of them. Every role has a working default, so reach for this section when a run costs more than you want, when you are going OpenAI-direct, or when you want one model everywhere. A call that is handed an explicit model (`LLMCallConfig(model=...)`, `--attack-model`) ignores roles entirely.

| Role | Built-in default | What runs on it |
|---|---|---|
| `fast` | `openai/gpt-6-luna` | The simulated user, simulation generators (persona, scenario, first message, datapoints, from-traces), the simulation recommendations pass and executive summary, and the trace-finder query compiler. |
| `smart` | `openai/gpt-6-sol` | Red-team attack generation (attacker, objective and attack generators, the adaptive orchestrator, capability and blackbox classifiers) and the red-team executive summary, every judge and evaluator (red-team and OWASP evaluators, the simulation judge, the default `llm_jury()` panel), apply-recommendations, and Insights summaries. |
| `classifier` | `typesafe/jev-latest` | Ask AI and trace-finder classification, Insights labels, and signal tool-role classification. |
| `embedding` | `openai/text-embedding-3-small` | Insights embeddings. |

The smart default is a larger model than the fast one, so red-team, simulation-judge and apply runs cost more than they did before roles existed. To keep the old model everywhere, set both `EVALUATORQ_FAST_MODEL` and `EVALUATORQ_SMART_MODEL` to `openai/gpt-5.6-luna`.

### Pin one task

A **task** is one named use of a role. A task override pins that single use to a model and leaves the rest of the role alone. The keys are fixed.

| Task | Role | What it covers |
|---|---|---|
| `redteam.attacker` | `smart` | Attack generation, including the adaptive orchestrator. |
| `redteam.evaluator` | `smart` | Red-team and OWASP evaluators. |
| `sim.user` | `fast` | The simulated user. |
| `sim.judge` | `smart` | The simulation judge. |
| `sim.generator` | `fast` | Persona, scenario, first-message and datapoint generation, simulation recommendations and executive summary. |
| `finder.compiler` | `fast` | The trace-finder query compiler. |
| `finder.classifier` | `classifier` | Ask AI and trace-finder classification. |
| `apply` | `smart` | The dashboard's apply-recommendations merge. |
| `insights.summary` | `smart` | Insights per-trace summaries. |
| `insights.labels` | `classifier` | Insights label questions. |
| `insights.embedding` | `embedding` | Insights embeddings. |
| `signals` | `classifier` | Signal tool-role classification. |

### How a model is chosen

evaluatorq takes the first of these that is set, highest first:

1. A per-command flag or an explicit `model=` argument, such as `--attack-model` or `LLMCallConfig(model=...)`.
2. `--model-override TASK=MODEL` on the command line.
3. A role flag: `--fast-model`, `--smart-model`, `--classifier-model` or `--embedding-model`.
4. The deprecated task variables `EVALUATORQ_COMPILER_MODEL` (task `finder.compiler`) and `EVALUATORQ_APPLY_MODEL` (task `apply`).
5. `model_overrides` for the task in the settings file.
6. The role variable: `EVALUATORQ_FAST_MODEL`, `EVALUATORQ_SMART_MODEL`, `EVALUATORQ_CLASSIFIER_MODEL` or `EVALUATORQ_EMBEDDING_MODEL`.
7. The role field in the settings file.
8. The built-in default.

A task override is more specific than a role, so a task override in the settings file beats a role variable. A variable that is empty or only whitespace counts as unset.

### Set models in the settings file

The dashboard Settings page and the trace finder read and write `.evaluatorq/dashboard-settings.json`, or the path in `EVALUATORQ_DASHBOARD_SETTINGS`. The same file feeds the library and the CLI. A role field left out, or `null`, means use the default.

```json
{
  "smart_model": "openai/gpt-5.6-luna",
  "embedding_model": "openai/text-embedding-3-large",
  "model_overrides": {
    "apply": "openai/gpt-6-luna",
    "finder.compiler": "openai/gpt-5.6-luna"
  }
}
```

A file saved before roles existed still loads. Its `compiler_model` and `apply_model` become the `finder.compiler` and `apply` overrides when they differ from the old default `openai/gpt-5.6-luna`, and are dropped when they match it, so the old default is not frozen into your file. A `classifier_model` equal to `typesafe/jev-latest` is dropped the same way. An override for an unknown task or with a blank model is dropped with a warning, and the rest of the file still loads. The dashboard saves only what you chose in the form: a model that came from an environment variable or a CLI flag is never written back to the file.

### Set models on the command line

These options go before the subcommand, for example `eq --smart-model gpt-6-sol redteam run ...`.

| Option | Sets |
|---|---|
| `--fast-model MODEL` | The `fast` role. |
| `--smart-model MODEL` | The `smart` role. |
| `--classifier-model MODEL` | The `classifier` role. |
| `--embedding-model MODEL` | The `embedding` role. |
| `--model-override TASK=MODEL` | One task from the table above. Repeatable. |

An unknown task exits with an error that lists the valid tasks, and a pair without `=` exits with an `expected task=model` error. Subcommands keep their own model flags (`--attack-model`, `--sim-model`, `--summary-model` and the others), which rank above everything here. `eq dashboard` and `eq find` also take `--compiler-model`, a `finder.compiler` override, and `--classifier-model`.

### Go OpenAI-direct

The built-in defaults are provider-prefixed because the default route is the Orq router, which resolves `provider/model`. With only `OPENAI_API_KEY` set, calls go straight to OpenAI, which rejects `openai/gpt-6-sol`. Set the bare ids on the two roles that run there, and the failure goes away for every command at once:

```bash
export OPENAI_API_KEY=...
export EVALUATORQ_FAST_MODEL=gpt-6-luna
export EVALUATORQ_SMART_MODEL=gpt-6-sol
eq redteam run -t agent:my-agent
```

The classifier role defaults to a model served only by the Orq router, so Ask AI, `eq find`, Insights and the classify judge need `ORQ_API_KEY`.

## Full environment variable reference

### Backend, credentials and model catalogue

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `ORQ_API_KEY` | Required for Orq features | — | Authenticates against the Orq platform. Required to fetch datasets, upload results, and invoke deployments. Also auto-enables OpenTelemetry tracing when the `otel` extra is installed (spans go to `<ORQ_BASE_URL>/v2/otel`, so `https://my.orq.ai/v2/otel` by default). |
| `ORQ_BASE_URL` | No | `https://my.orq.ai` | Overrides the Orq API base URL. Affects Orq SDK calls (dataset fetch, deployment invocation) and the derived OTLP tracing endpoint (`<ORQ_BASE_URL>/v2/otel`). Does **not** redirect OpenAI-compatible LLM calls — use `OPENAI_BASE_URL` for that. |
| `OPENAI_API_KEY` | Red-team / sim only, if not using Orq | — | API key for the OpenAI (or compatible) backend. Used by the red teaming pipeline and agent simulation when `ORQ_API_KEY` is absent. Not required for core `evaluatorq()` evaluation. |
| `OPENAI_BASE_URL` | No | OpenAI default | Redirect OpenAI-compatible calls to a different host (vLLM, OpenRouter, Azure, local). Honoured by the red teaming LLM client. **Not** honoured on the simulation conversation path, which resolves its client with `honor_openai_base_url=False` — inject a pre-built client there instead. Simulation's post-run calls (recommendations and the executive summary) do honour it, because they resolve their own client at the default. |
| `EVALUATORQ_CATALOGUE_TIMEOUT_S` | No | `30` | HTTP timeout in seconds for the `GET /v2/models` model-catalogue fetch, read once at import (unlike the three `EVALUATORQ_LLM_*` vars under [Simulation LLM defaults](#simulation-llm-defaults), which are read at call time). The catalogue supplies per-model prices, Responses support, and accepted reasoning-effort values. A failed fetch is **not** cached immediately: the first two failures leave the catalogue unloaded so a later call can still succeed, and only the third gives up and degrades the rest of the process to unpriced and chat-completions-only. Each failure logs a warning naming the attempt number. |

### Model roles

See [Models](#models) for what each role runs and the order they are resolved in.

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `EVALUATORQ_FAST_MODEL` | No | `openai/gpt-6-luna` | Model for the `fast` role: the simulated user, simulation generators, and the trace-finder compiler. |
| `EVALUATORQ_SMART_MODEL` | No | `openai/gpt-6-sol` | Model for the `smart` role: red-team attacks, every judge and evaluator, apply-recommendations, and Insights summaries. Set it to `openai/gpt-5.6-luna` to restore the previous default. |
| `EVALUATORQ_CLASSIFIER_MODEL` | No | `typesafe/jev-latest` | Model for the `classifier` role: Ask AI and trace-finder classification, Insights labels, and signal tool-role classification. See [Trace finder](trace-finder.md) and [Signals](signals.md). |
| `EVALUATORQ_EMBEDDING_MODEL` | No | `openai/text-embedding-3-small` | Model for the `embedding` role: Insights embeddings. |
| `EVALUATORQ_COMPILER_MODEL` | No | unset | **Deprecated.** Pins the `finder.compiler` task. Logs one warning per process. Use `--model-override finder.compiler=MODEL`, `model_overrides` in the settings file, or `EVALUATORQ_FAST_MODEL`. |
| `EVALUATORQ_APPLY_MODEL` | No | unset | **Deprecated.** Pins the `apply` task. Logs one warning per process. Use `--model-override apply=MODEL`, `model_overrides` in the settings file, or `EVALUATORQ_SMART_MODEL`. |

Set the embedding role with `EVALUATORQ_EMBEDDING_MODEL`, `--embedding-model`, the dashboard Settings page or `embedding_model` in the settings file.

### Runs, storage and output

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `EVALUATORQ_DIR` | No | `.evaluatorq` in the current directory | Base directory for the run store, where both red teaming (`runs/`) and simulation (`sim-runs/`) persist reports. Must point at the store directory itself (e.g. `/tmp/x/.evaluatorq`), not its parent — only the working-directory fallback appends `.evaluatorq`. Empty is treated as unset. |
| `EQ_DEBUG` | No | unset | Set to any non-empty value to show the full traceback on CLI errors instead of the one-line message. CLI-wide; distinct from `ORQ_DEBUG`, which only affects tracing diagnostics. |
| `EVALUATORQ_LOG_LEVEL` | No | `INFO` | Dashboard server log level. `DEBUG` logs finder model requests and responses, including trace content, in CLI and dashboard runs; `eq find` also prints changed progress. The default suppresses httpx's per-request INFO line; explicit `INFO` or `DEBUG` allows it. |

### Dashboard

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `ORQ_WORKSPACE` / `ORQ_WORKSPACE_SLUG` | No | unset | Workspace slug used to build dashboard deep-links into the Orq UI for runs without an experiment URL. `ORQ_WORKSPACE` wins over `ORQ_WORKSPACE_SLUG`. When neither is set, those deep-link buttons are hidden. See [Dashboard](dashboard.md). |
| `ORQ_UI_BASE_URL` | No | Saved profile host, then `ORQ_BASE_URL`, then `https://my.orq.ai` | Base URL for dashboard deep-links into the Orq UI. Set this when the UI host differs from the selected profile's API host. |
| `EVALUATORQ_DASHBOARD_SETTINGS` | No | `.evaluatorq/dashboard-settings.json` | Path to the JSON file used for dashboard and trace-finder models and the selected authentication method. |
| `EVALUATORQ_FINDER_WINDOW_DAYS` | No | `7` | Default number of recent days searched by the trace finder. Valid values are `1` through `90`. |
| `EVALUATORQ_FINDER_LIMIT` | No | `500` | Default maximum number of traces selected for trace-finder classification. Valid values are `1` through `5000`. |
| `EVALUATORQ_FINDER_PARALLELISM` | No | `100` | Default number of concurrent per-trace classify calls. Valid values are `1` through `200`. |
| `EVALUATORQ_DASHBOARD_ROOTS` | No | unset | Internal. A JSON array of run-store roots, set by `eq dashboard` for its reloader subprocess and parsed back with `json.loads`. Pass extra stores as positional CLI paths rather than setting this by hand. |

### Tracing

See [Tracing](tracing.md) for the full picture.

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `ORQ_DISABLE_TRACING` | No | unset | Set to `1`, `true`, `yes` or `on` to suppress all OpenTelemetry spans even when `ORQ_API_KEY` or `OTEL_EXPORTER_OTLP_ENDPOINT` is present. Case-insensitive; see [When a value is wrong](#when-a-value-is-wrong). |
| `ORQ_DEBUG` | No | unset | Set to any non-empty value to print tracing setup diagnostics to stdout (endpoint, auth headers, initialization errors). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | — | Explicit OTLP HTTP endpoint. Takes precedence over the `ORQ_BASE_URL`-derived endpoint. |
| `OTEL_EXPORTER_OTLP_HEADERS` | No | — | Comma-separated `key=value` pairs added to every OTLP export request. Format: `key1=value1,key2=value2`. |
| `OTEL_SERVICE_NAME` | No | `evaluatorq` | Service name recorded on every span's `service.name` resource attribute. |
| `OTEL_SERVICE_VERSION` | No | `1.0.0` | Service version recorded on every span's `service.version` resource attribute. |
| `ORQ_OTEL_MAX_QUEUE_SIZE` | No | `4096` | Maximum spans buffered by the `BatchSpanProcessor`. When it is full the SDK evicts the oldest buffered span and logs `Queue full, dropping Span.` on the stdlib `opentelemetry.sdk._shared_internal` logger. See [Tracing › Batching and flush](tracing.md#batching-and-flush). |
| `ORQ_OTEL_MAX_BATCH_SIZE` | No | `512` | Maximum spans per OTLP export request. Reaching it wakes the exporter immediately. A value larger than the queue size is clamped to the queue size, with a warning. |
| `ORQ_OTEL_SCHEDULE_DELAY_MS` | No | `5000` | Milliseconds a partial batch waits before it is exported. It does not throttle a full batch. |
| `ORQ_OTEL_FLUSH_TIMEOUT_MS` | No | `5000` | Milliseconds the end-of-run force-flush waits before giving up and logging a warning. Read per run; enforced by evaluatorq rather than the SDK. Bounds the final flush only — the per-request export timeout is a fixed 5s. |
| `EVALUATORQ_CAPTURE_MESSAGE_CONTENT` | No | `true` | Set to `false`, `0`, `no` or `off` to strip LLM message content (prompts and responses) from spans. Case-insensitive; see [When a value is wrong](#when-a-value-is-wrong). Token counts, model name, and latency are still recorded. Useful when exporting to third-party backends or to avoid capturing PII. |
| `EVALUATORQ_PROPAGATE_TRACE_CONTEXT` | No | `true` | Set to `false`, `0`, `no` or `off` to stop evaluatorq sending W3C `traceparent`/`tracestate` headers on its outgoing LLM and target calls. Case-insensitive; see [When a value is wrong](#when-a-value-is-wrong). Default on, so an Orq-hosted deployment or agent nests its server-side spans under the calling span. Turn it off when the receiving side should trace independently, or when a gateway rejects an unexpected `traceparent`. |
| `EVALUATORQ_SPAN_MAX_TEXT_CHARS` | No | unset (no limit) | Maximum characters per span text attribute. Set a positive integer (e.g. `8192`) to truncate long strings. Unset or `0` / `-1` means capture all. |

### Simulation LLM defaults

All three are fallback defaults only: the matching field on the agent's `LLMCallConfig` wins when set, and all three are read at call time, so setting them after import still takes effect.

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `EVALUATORQ_LLM_TIMEOUT_S` | No | `60.0` | Per-LLM-call timeout in seconds. **Simulation only** — has no effect on red teaming or core evaluation. `LLMCallConfig.timeout_ms` wins when set. Increase for slow self-hosted endpoints; for the *target's* timeout rather than the simulator's, pass `target_agent_timeout_ms` to `simulate()`. |
| `EVALUATORQ_LLM_MAX_TOKENS` | No | `10000` | Maximum completion tokens per LLM call. **Simulation only** — red teaming shares the same default (`DEFAULT_TARGET_MAX_TOKENS`) but is tuned per role via `LLMConfig.max_tokens` / `EvaluatorConfig.max_tokens`, not by this variable. `LLMCallConfig.max_tokens` wins when set. Increase for reasoning models that exhaust the default budget before emitting a tool call. |
| `EVALUATORQ_REASONING_EFFORT` | No | *(unset — parameter not sent)* | Reasoning effort hint for the **simulator's own** LLM calls (user simulator, judge) — not for the agent under test, which takes `target_reasoning_effort` on `simulate()` / `red_team()`. **Simulation only.** There is no global default — unset means the parameter is omitted and the model uses its own. `LLMCallConfig.reasoning_effort` wins when set. Set to `""`, `none`, or `off` to omit the parameter entirely. |

## When a value is wrong

Ten of the variables on this page are validated when they are read, and a wrong value in any of them never stops a run: evaluatorq logs a `WARNING`, uses the documented default, and carries on.

A variable that looks like it did nothing was rejected, outranked, or never read, and only the first of the three says so. A rejected value leaves the `WARNING` below. An outranked one leaves nothing: the three [Simulation LLM defaults](#simulation-llm-defaults) are fallbacks, and an `LLMCallConfig` field beats them whenever it is set at all — including when it is set to the same number as the default. A knob nobody read leaves nothing either, because these are read where they are used rather than at startup: `ORQ_OTEL_MAX_QUEUE_SIZE` is read when tracing initialises, so with tracing off a bad value passes without a word. Read the warnings first, and treat their absence as "no opinion" rather than as "fine".

**Which variables this covers.** Ten knobs are parsed and validated — the numbers against a minimum, the booleans against an accepted vocabulary: `ORQ_DISABLE_TRACING`, the four `ORQ_OTEL_*` variables, `EVALUATORQ_CAPTURE_MESSAGE_CONTENT`, `EVALUATORQ_PROPAGATE_TRACE_CONTEXT`, `EVALUATORQ_CATALOGUE_TIMEOUT_S`, `EVALUATORQ_LLM_TIMEOUT_S` and `EVALUATORQ_LLM_MAX_TOKENS`.

The minimum is `1` for the four `ORQ_OTEL_*` variables and for `EVALUATORQ_LLM_MAX_TOKENS`, `1.0` for `EVALUATORQ_LLM_TIMEOUT_S`, and `0.1` for `EVALUATORQ_CATALOGUE_TIMEOUT_S`. None of them has a maximum.

Everything else on this page is a string evaluatorq hands on as it finds it — a key, a base URL, a directory, `EVALUATORQ_REASONING_EFFORT` — and a wrong value there surfaces where it is used, as a failed request or an empty run store, not as one of these warnings. `EVALUATORQ_LOG_LEVEL` is the one that does not land softly: a name Python's `logging` module does not recognise raises `ValueError: Unknown level` out of the dashboard's logging setup, and the dashboard never starts.

`EVALUATORQ_SPAN_MAX_TEXT_CHARS` is parsed on its own terms: a value that is not an integer warns and falls back to capturing everything.

| What you set | What happens |
|---|---|
| Nothing — the variable is absent | The default, with no log line. |
| A good value with whitespace around it, such as `" 45 "` | Stripped and applied. Not an error. |
| A number that does not parse, such as `abc` | `WARNING`, then the default. |
| A number below the knob's minimum, such as `0` where the minimum is `1` | `WARNING` naming the bound, then the default. |
| `ORQ_OTEL_MAX_BATCH_SIZE` larger than `ORQ_OTEL_MAX_QUEUE_SIZE` | `WARNING`, then the batch size clamped down to the queue size — the one case that lands on a value you did not set rather than on the default. |
| `nan` or `inf` where a float is expected | `WARNING`, then the default. |
| An empty or whitespace-only value, for a **number** | `WARNING`, then the default. An unresolved `${{ vars.X }}` in a CI `env:` block expands to the empty string, and that is worth a signal rather than a silent default. |
| An empty or whitespace-only value, for a **boolean** | The default, with **no log line**. |
| A word a boolean does not recognise, such as `maybe` | `WARNING`, then the default. |

The two empty-value rows differ, and the difference decides what an unresolved CI variable costs you. `EVALUATORQ_CAPTURE_MESSAGE_CONTENT: ${{ vars.CAPTURE_CONTENT }}` with no such repository variable expands to the empty string, which reads as unset, which means the default — and the default is `true`, so every prompt and response goes to your tracing backend with nothing in the log and exit `0`. The boolean knobs fail open.

Assert the values arrived instead of watching for a warning. This distinguishes "set but empty" from "absent", because absent is legitimate:

```bash
rc=0
for v in ORQ_DISABLE_TRACING ORQ_OTEL_MAX_QUEUE_SIZE ORQ_OTEL_MAX_BATCH_SIZE ORQ_OTEL_SCHEDULE_DELAY_MS ORQ_OTEL_FLUSH_TIMEOUT_MS EVALUATORQ_CAPTURE_MESSAGE_CONTENT EVALUATORQ_PROPAGATE_TRACE_CONTEXT EVALUATORQ_CATALOGUE_TIMEOUT_S EVALUATORQ_LLM_TIMEOUT_S EVALUATORQ_LLM_MAX_TOKENS; do
  if [ -n "${!v+set}" ] && [ -z "${!v}" ]; then
    echo "::error::$v is set but empty — a repository variable did not resolve."
    rc=1
  fi
done
exit $rc
```

Booleans accept `1`, `true`, `yes` and `on`, and `0`, `false`, `no` and `off`, in any capitalisation.

The warnings go to stderr, so a step that redirects only stdout loses them. To see one:

```bash
EVALUATORQ_CATALOGUE_TIMEOUT_S=abc eq redteam runs
```

```console
2026-09-14 06:32:21.018 | WARNING  | evaluatorq.common.env_config:env_float:72 - EVALUATORQ_CATALOGUE_TIMEOUT_S is not a number ('abc'); using default 30.0.
```

The command then lists your runs as usual and exits `0` — the rejected timeout costs it nothing but that line. Expect one line per read, not one per process: a knob read on every LLM call warns on every LLM call. The same contract on a boolean:

```bash
ORQ_DISABLE_TRACING=maybe python -c "from evaluatorq.tracing import is_tracing_enabled; is_tracing_enabled()"
```

```console
2026-09-14 06:19:52.231 | WARNING  | evaluatorq.common.env_config:env_bool:95 - ORQ_DISABLE_TRACING is not a boolean ('maybe'); using default False.
```

None of the ten raise. A typo costs you a default and a log line, never a stopped process.

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

## Routers (`orq/*`)

Orq's model garden can carry router ids, listed under the `orq/` provider, that resolve to a real model per request rather than naming one. Pass one anywhere a model id goes:

```python
evaluator = llm_judge(name="quality", criteria="Is the answer correct?", model="orq/autorouter-anthropic-balanced")
```

Nothing extra is needed to reach them: a run holding `ORQ_API_KEY` already talks to the router. Two things worth knowing about what comes back:

- **Cost is read off the model that answered, not the router you asked for.** A router does carry a price in the catalogue, but it is one headline rate standing in for every model the router can pick, so charging a run at it is wrong by whatever the gap happens to be that day — a measured example ran 4.55x over. A call is priced against the id the response came back under instead, and when the catalogue does not list that model the call is left unpriced with a warning rather than billed at the router's rate. A run therefore reports what it actually spent, but two runs of the same evaluation can be priced differently because they were served differently.
- **Which model answered is not knowable in advance**, and changes on every catalog sync, model toggle and key change in the workspace. That is the point of the router, and the reason a comparison you intend to reproduce should name a model instead.

## Where to next

- **[Getting Started](guides/getting-started.md)** — run your first evaluation.
- **[Targets](guides/targets.md#orq-hosted-agents-and-deployments)** — point a run at an Orq-hosted agent or deployment.
- **[Tracing](tracing.md)** — enable OpenTelemetry tracing.
