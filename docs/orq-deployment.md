# Orq Deployment Integration

`evaluatorq.deployment` provides async helpers for calling
[Orq deployments](https://docs.orq.ai/docs/deployment) from within evaluation
jobs or any async Python context.

## What it does

The module wraps the `orq-ai-sdk` client with two thin async functions:

- **`deployment(key, ...)`** — invoke a deployment and return a
  `DeploymentResponse` with both `content` (extracted text) and `raw` (the
  unmodified SDK response).
- **`invoke(key, ...)`** — convenience wrapper that calls `deployment()` and
  returns just the text string.

A single `Orq` client instance is created lazily on first use and reused for
the lifetime of the process.

## Requirements

| Requirement | Detail |
|---|---|
| Package | `orq-ai-sdk>=4.4.7` — install directly with `pip install orq-ai-sdk` or via `pip install evaluatorq[orq]` |
| `ORQ_API_KEY` | Required. Set in environment or `.env` file. |
| `ORQ_BASE_URL` | Optional. Defaults to `https://my.orq.ai`. Override for self-hosted or staging. |

## Usage

### Basic invocation

```python
from evaluatorq.deployment import invoke

async def run():
    text = await invoke("my-deployment")
    print(text)
```

### With template inputs

```python
from evaluatorq.deployment import invoke

async def run():
    text = await invoke("summarizer", inputs={"text": "Long article..."})
    print(text)
```

### Chat-style deployments

```python
from evaluatorq.deployment import deployment

async def run():
    response = await deployment(
        "chatbot",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.content)   # extracted text
    print(response.raw)       # full SDK response object
```

### Thread tracking

```python
from evaluatorq.deployment import deployment

async def run():
    response = await deployment(
        "assistant",
        inputs={"query": "What is AI?"},
        thread={"id": "conversation-123"},
    )
```

### Inside an evaluation job

```python
from evaluatorq import DataPoint, job
from evaluatorq.deployment import invoke

@job("orq-deployment-job")
async def my_job(data: DataPoint, _row: int) -> str:
    return await invoke("my-deployment", inputs=data.inputs)
```

## Replaying an experiment's responses (no-inference mode)

Sometimes you want to score responses that an Orq experiment already produced
instead of generating fresh ones — to try new evaluators against a past run, or
to re-grade without paying for another round of generation. That is what
**no-inference mode** does: pass `inference=False` and evaluators run against the
recorded response in each row rather than calling any job.

The response source is chosen by the `data` argument to `evaluatorq()`:

| `data` value | What it loads |
|---|---|
| `DatasetIdInput(id=...)` | Rows from an Orq dataset (you supply/generate the responses). |
| `ExperimentInput(experiment_id=..., run_id=...)` | The recorded responses from a past experiment run. Requires `inference=False`. |
| `list[DataPoint]` | In-memory datapoints. |

`ExperimentInput` sits alongside `DatasetIdInput` in the `data` union — it is not
a dataset, it is a completed experiment run whose outputs get replayed.

### Finding the IDs

Both IDs are read off the Orq UI:

- **`experiment_id`** — the ID in the experiment URL, `/experiments/<experiment_id>`.
  The REST API calls experiments "spreadsheets", so the same ID appears in
  `/v2/spreadsheets/<id>` routes.
- **`run_id`** — optional. Every execution of an experiment creates a new run (a
  "manifest" in the API). Open a run from the experiment's run history to read its
  ID from the URL. Omit it to replay the latest run.

### Usage

```python
from evaluatorq import evaluatorq, ExperimentInput

async def run():
    await evaluatorq(
        "replay-past-experiment",
        data=ExperimentInput(experiment_id="<experiment_id>"),  # latest run
        evaluators=[my_evaluator],
        inference=False,
    )
```

Pin a specific run with `run_id`:

```python
data=ExperimentInput(experiment_id="<experiment_id>", run_id="<run_id>")
```

`ORQ_API_KEY` must be set — the recorded rows are fetched from the Orq API. When
`inference=False`, `jobs` is optional and ignored. Any row whose recorded
response is missing or blank fails loudly rather than being silently skipped.

## API reference

### `deployment()`

```python
async def deployment(
    key: str,
    inputs: dict[str, object] | None = None,
    context: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    thread: ThreadConfig | None = None,
    messages: list[MessageDict] | None = None,
) -> DeploymentResponse
```

| Parameter | Type | Description |
|---|---|---|
| `key` | `str` | Deployment key (name) as configured in Orq. |
| `inputs` | `dict \| None` | Template input variables. |
| `context` | `dict \| None` | Context attributes for routing. |
| `metadata` | `dict \| None` | Metadata to attach to the request. |
| `thread` | `ThreadConfig \| None` | Thread config for conversation tracking. Include `id` key. |
| `messages` | `list[MessageDict] \| None` | Chat messages for conversational deployments. |

Returns `DeploymentResponse`:

| Attribute | Type | Description |
|---|---|---|
| `content` | `str` | Extracted text content from the response. |
| `raw` | `object` | Raw SDK response object. |

### `invoke()`

Same signature as `deployment()`. Returns `str` (the `content` field only).

### `ThreadConfig`

```python
class ThreadConfig(TypedDict, total=False):
    id: str
    tags: list[str] | None
```

### `MessageDict`

```python
class MessageDict(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: str
    name: str | None
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ORQ_API_KEY` | Yes | — | Orq platform API key. |
| `ORQ_BASE_URL` | No | `https://my.orq.ai` | Override the Orq API base URL. |

`ORQ_API_KEY` must be set before the first call. A missing key raises
`ValueError` at runtime with a descriptive message. A missing `orq-ai-sdk`
installation raises `ImportError` with install instructions.
