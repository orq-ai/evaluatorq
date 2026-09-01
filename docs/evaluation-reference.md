# Evaluation Reference

Everything `evaluatorq()` accepts, and the patterns built on top of it. If you have not run an evaluation yet, start with [Getting Started](guides/getting-started.md).

## `evaluatorq()`

<!-- check-examples: skip (signature display, not a module) -->
```python
async def evaluatorq(
    name: str,
    params: EvaluatorParams | dict[str, Any] | None = None,
    *,
    data: DatasetIdInput | ExperimentInput | Sequence[Awaitable[DataPoint] | DataPointInput] | None = None,
    jobs: list[Job] | None = None,
    evaluators: list[Evaluator] | None = None,
    datapoint_parallelism: int = 10,
    llm_parallelism: int | None = None,
    print_results: bool = True,
    description: str | None = None,
    path: str | None = None,
    inference: bool = True,
) -> EvaluatorqResult
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `list[DataPoint \| dict]` \| `list[Awaitable[DataPoint]]` \| `DatasetIdInput` \| `ExperimentInput` | **required** | Data to evaluate — local rows (a `DataPoint` or a plain dict with the same keys), an Orq dataset, or an existing experiment |
| `jobs` | `list[Job]` | **required** | Jobs to run on each data point |
| `evaluators` | `list[Evaluator]` \| `None` | `None` | Evaluators that score job outputs |
| `datapoint_parallelism` | `int` (≥1) | `10` | Number of concurrent datapoints. The former name `parallelism` still works, deprecated |
| `llm_parallelism` | `int` (≥1) \| `None` | `None` | Ceiling on in-flight LLM requests for the whole run. Unbounded when unset |
| `print_results` | `bool` | `True` | Display the progress and results table |
| `description` | `str` \| `None` | `None` | Optional evaluation description |
| `path` | `str` \| `None` | `None` | Path for organizing results on the Orq dashboard (e.g. `"Project/Category"`) |
| `inference` | `bool` | `True` | Run the jobs; set `False` to score data that already has outputs |

Parameters can also be passed positionally as an `EvaluatorParams` model or a plain dict — the three forms below are equivalent:

```python
await evaluatorq("my-eval", data=[...], jobs=[...], datapoint_parallelism=5)
await evaluatorq("my-eval", {"data": [...], "jobs": [...], "datapoint_parallelism": 5})
await evaluatorq("my-eval", EvaluatorParams(data=[...], jobs=[...], datapoint_parallelism=5))
```

Full type signatures live in the [API Reference](reference/evaluatorq.md).

## Jobs

### The `@job()` decorator

`@job()` names a job. The name shows up in the results table, in traces, and — crucially — in error messages:

```python
from evaluatorq import job

@job("risky-job")
async def risky_operation(data: DataPoint, row: int):
    return await potentially_failing_operation(data)

# Error output: "Job 'risky-job' failed: <error details>"
# Without @job:  "<error details>"
```

It also wraps plain callables, which is handy for one-liners:

```python
uppercase_job = job("uppercase", lambda data, row: data.inputs["text"].upper())
word_count_job = job("word-count", lambda data, row: len(data.inputs["text"].split()))
```

### Multiple jobs per data point

Every job runs against every data point, which is how you compare variants (two prompts, two models, preprocessing on and off) on identical inputs:

```python
await evaluatorq(
    "multi-job-eval",
    data=[...],
    jobs=[preprocessor, analyzer, transformer],
    evaluators=[...],
)
```

### Reporting a failure the job handled

A job that lets a failure raise needs nothing: `process_job` records it and the row counts as failed. A job that *catches* its failure to keep the rest of the batch alive — a target that answered `401`, a simulation the runner ended in `error` — must say so, because a returned dict looks like a clean run:

```python
async def resilient_job(data: DataPoint, row: int) -> dict:
    try:
        answer = await call_my_agent(data.inputs["text"])
    except MyAgentError as exc:
        # Keeps the partial output for diagnosis, and still fails the row.
        return {"name": "my-agent", "output": None, "error": str(exc)}
    return {"name": "my-agent", "output": answer, "error": None}
```

Emit `error` on every path, `None` on success: an omitted key and a clean run are indistinguishable, so a job that forgets the key on one branch reports a dead target as a passing one. A row with a non-empty `error` is counted in the summary table's `Failed Jobs` and keeps its output for diagnosis, but its evaluators are **skipped** — scoring a transcript you already know is dead buys nothing and costs an LLM judge call per row. `check_pass_failures(results, treat_errors_as_failure=True)` is what turns it into a CI failure; the default (`False`) gates on evaluator `pass_` alone.

This is the raw-dict job contract. `@job()` wraps a function's return value into `{"name", "output"}`, so an `error` key returned from a decorated function lands *inside* `output`, where a judge reads it as the target's failure rather than the runner reading it as the row's. A decorated job reports a row failure by raising.
### Calling an Orq deployment from a job

When the thing you are evaluating is an Orq deployment, call it from inside the job. `evaluatorq.deployment` wraps the Orq SDK in two async functions: `deployment()` returns a `DeploymentResponse` carrying both `content` (the extracted text) and `raw` (the untouched SDK response), and `invoke()` is the same call returning just the text. One `Orq` client is created lazily on first use and reused for the process.

This needs `ORQ_API_KEY`. A missing key raises `ValueError` at the first call, naming the variable.

```python
from evaluatorq import DataPoint, job
from evaluatorq.deployment import invoke


@job("orq-deployment-job")
async def my_job(data: DataPoint, _row: int) -> str:
    return await invoke("my-deployment", inputs=data.inputs)
```

`inputs` fills the deployment's template variables, so a summarizer whose prompt references `{{text}}` takes `inputs={"text": "Long article..."}`.

For a chat-style deployment, pass `messages` — and reach for `deployment()` when you want the raw response as well:

```python
from evaluatorq.deployment import deployment

response = await deployment("chatbot", messages=[{"role": "user", "content": "Hello!"}])
print(response.content)  # extracted text
print(response.raw)      # full SDK response object
```

`thread={"id": "conversation-123"}` groups several calls into one conversation on the platform. `context` (routing attributes) and `metadata` are also accepted — see [`deployment()` in the API reference](reference/evaluatorq.md) for the full signature.

To red-team or simulate that same deployment rather than evaluate it, name it as a target instead: [Targets › Orq-hosted](guides/targets.md#orq-hosted-agents-and-deployments).

## Data sources

`data` accepts inline `DataPoint`s, an Orq dataset, or awaitables that resolve to `DataPoint`s — the last of which lets you stream rows in from a slow source without blocking the run:

```python
async def get_data_point(i: int) -> DataPoint:
    await asyncio.sleep(0.01)  # e.g. a network fetch
    return DataPoint(inputs={"value": i})

await evaluatorq(
    "async-eval",
    data=[get_data_point(i) for i in range(1000)],
    jobs=[...],
)
```

For Orq-hosted datasets, pass `data=DatasetIdInput(dataset_id="...")`. That path requires `ORQ_API_KEY` — see [Configuration](configuration.md). To score responses a past Orq experiment already produced, rather than generating new ones, pass `ExperimentInput` — see [Replaying a past experiment](#replaying-a-past-experiment).

## Replaying a past experiment

Sometimes you want to score responses that an Orq experiment already produced instead of generating fresh ones — to try new evaluators against a past run, or to re-grade without paying for another round of generation. That is what **no-inference mode** does: pass `inference=False` and evaluators run against the recorded response in each row rather than calling any job.

The response source is chosen by the `data` argument to `evaluatorq()`:

| `data` value | What it loads |
|---|---|
| `list[DataPoint]` | In-memory datapoints. |
| `DatasetIdInput(dataset_id=...)` | Rows from an Orq dataset (you supply or generate the responses). |
| `ExperimentInput(experiment_id=..., run_id=...)` | The recorded responses from a past experiment run. Requires `inference=False`. |

`ExperimentInput` sits alongside `DatasetIdInput` in the `data` union — it is not a dataset, it is a completed experiment run whose outputs get replayed.

### Finding the IDs

Both IDs are read off the Orq UI:

- **`experiment_id`** — the ID in the experiment URL, `/experiments/<experiment_id>`. The REST API calls experiments "spreadsheets", so the same ID appears in `/v2/spreadsheets/<id>` routes.
- **`run_id`** — optional. Every execution of an experiment creates a new run (a "manifest" in the API). Open a run from the experiment's run history to read its ID from the URL. Omit it to replay the latest run.

### Example

```python
from evaluatorq import ExperimentInput, evaluatorq


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

`ORQ_API_KEY` must be set — the recorded rows are fetched from the Orq API. When `inference=False`, `jobs` is optional and ignored. Any row whose recorded response is missing or blank fails loudly rather than being silently skipped.

## Built-in evaluators

```python
from evaluatorq import exact_match_evaluator, string_contains_evaluator

string_contains_evaluator()                        # case-insensitive by default
string_contains_evaluator(case_insensitive=False)  # case-sensitive
string_contains_evaluator(name="my-contains-check")  # custom name in the table
exact_match_evaluator()                            # case-sensitive by default
```

Both compare the job output against the data point's `expected_output`. For LLM-graded evaluators see [LLM as a Jury](llm-as-a-jury.md); for structured, multi-dimensional scores see [Structured Results](structured-results.md).

## Custom evaluators

An evaluator is a `{"name": ..., "scorer": ...}` pair whose scorer receives the data point and the job output and returns a score:

```python
async def accuracy_scorer(params):
    data, output = params["data"], params["output"]
    score = calculate_score(output, data.expected_output)
    return {"value": score, "explanation": "High accuracy match" if score > 0.8 else "Partial match"}


await evaluatorq(
    "dataset-evaluation",
    data=DatasetIdInput(dataset_id="your-dataset-id"),
    jobs=[processor],
    evaluators=[{"name": "accuracy", "scorer": accuracy_scorer}],
)
```

## Pass/fail and CI

An evaluator that returns `pass_` turns the run into a gate:

```python
async def quality_scorer(params):
    score = calculate_quality(params["output"])
    return {
        "value": score,
        "pass_": score >= 0.8,
        "explanation": f"Quality score: {score}",
    }
```

When any evaluator returns `pass_: False`, `evaluatorq()` returns the results; the library never exits the process. To make a script a CI gate, inspect the results and exit explicitly:

```python
from evaluatorq.evaluatorq import check_pass_failures

results = await evaluatorq(...)
if check_pass_failures(results, treat_errors_as_failure=True):
    raise SystemExit(1)
```

`treat_errors_as_failure=True` also gates on rows that errored — a job that raised, a job that reported its own failure, and an evaluator whose every call failed. It defaults to `False`, which gates on evaluator `pass_` alone, so a run whose target was dead throughout can pass a gate that leaves it off.

The results table gains a pass rate row — `Pass Rate | 75% (3/4)`.

## Controlling the run

### Parallelism

```python
await evaluatorq("parallel-eval", data=[...], jobs=[...], datapoint_parallelism=10)
```

`datapoint_parallelism` counts **tasks**, and the bounds nest: at most `datapoint_parallelism` datapoints run at once, and within each one a separate budget of the same size covers its jobs and then its evaluators. Ten datapoints each running ten evaluators is a hundred concurrent tasks, not ten.

### Bounding LLM requests

Against a provider concurrency limit, size the request ceiling instead:

```python
await evaluatorq("bounded-eval", data=[...], jobs=[...], llm_parallelism=10)
```

This counts requests, not tasks, so it holds however the fan-out nests. It is a concurrency bound rather than a rate limit — ten slots against 10s calls is about 60 requests/minute, but the same ten slots become 300/minute if the provider speeds up to 2s.

Requests evaluatorq issues itself (judges, juries, simulation agents, the red-team pipeline) take a slot automatically. A job that calls a provider SDK directly is invisible to the budget unless you wrap it:

```python
from evaluatorq.common.llm_limit import llm_slot

async def my_job(data_point, row_index):
    async with llm_slot():
        response = await client.chat.completions.create(...)
    return {"name": "my-job", "output": response.choices[0].message.content}
```

Wrap only the request — holding a slot across parsing shrinks the budget without reducing load on the provider.

`red_team()`, `simulate()`, `generate_and_simulate()` and `generate()` take the same argument, with the same meaning.

### Organizing results on Orq

```python
await evaluatorq(
    "my-evaluation",
    data=[...],
    jobs=[...],
    path="MyProject/Evaluations/Unit Tests",
)
```

`path` groups runs in the Orq dashboard — e.g. `"Team/Sprint-42/Feature-X"`.

### Documenting a run

```python
await evaluatorq(
    "model-comparison",
    description="Compare GPT-4o vs Claude on customer support responses",
    data=[...],
    jobs=[...],
)
```

### Suppressing terminal output

```python
results = await evaluatorq("silent-eval", data=[...], jobs=[...], print_results=False)

for result in results:
    for job_result in result.job_results or []:
        print(f"{job_result.job_name}: {job_result.output}")
```

## Where to next

- **[Structured Results](structured-results.md)** — multi-dimensional scores.
- **[Simulation in an evaluatorq Run](guides/simulation-in-evaluatorq.md)** — a multi-turn conversation as a job, scored by your own evaluators.
- **[Framework Integrations](framework-integrations.md)** — LangChain, LangGraph, OpenResponses.
- **[Targets](guides/targets.md)** — red-team or simulate the same deployment instead of evaluating it.
- **[Configuration](configuration.md)** — environment variables.
