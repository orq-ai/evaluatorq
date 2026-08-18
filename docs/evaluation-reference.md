# Evaluation Reference

Everything `evaluatorq()` accepts, and the patterns built on top of it. If you
have not run an evaluation yet, start with
[Getting Started](guides/getting-started.md).

## `evaluatorq()`

```python
async def evaluatorq(
    name: str,
    params: EvaluatorParams | dict[str, Any] | None = None,
    *,
    data: DatasetIdInput | ExperimentInput | Sequence[Awaitable[DataPoint] | DataPointInput] | None = None,
    jobs: list[Job] | None = None,
    evaluators: list[Evaluator] | None = None,
    parallelism: int = 10,
    max_concurrent_llm_calls: int | None = None,
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
| `parallelism` | `int` (≥1) | `10` | Number of concurrent jobs |
| `max_concurrent_llm_calls` | `int` (≥1) \| `None` | `None` | Ceiling on in-flight LLM requests for the whole run. Unbounded when unset |
| `print_results` | `bool` | `True` | Display the progress and results table |
| `description` | `str` \| `None` | `None` | Optional evaluation description |
| `path` | `str` \| `None` | `None` | Path for organizing results on the Orq dashboard (e.g. `"Project/Category"`) |
| `inference` | `bool` | `True` | Run the jobs; set `False` to score data that already has outputs |

Parameters can also be passed positionally as an `EvaluatorParams` model or a
plain dict — the three forms below are equivalent:

```python
await evaluatorq("my-eval", data=[...], jobs=[...], parallelism=5)
await evaluatorq("my-eval", {"data": [...], "jobs": [...], "parallelism": 5})
await evaluatorq("my-eval", EvaluatorParams(data=[...], jobs=[...], parallelism=5))
```

Full type signatures live in the [API Reference](reference/evaluatorq.md).

## Jobs

### The `@job()` decorator

`@job()` names a job. The name shows up in the results table, in traces, and —
crucially — in error messages:

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

Every job runs against every data point, which is how you compare variants
(two prompts, two models, preprocessing on and off) on identical inputs:

```python
await evaluatorq(
    "multi-job-eval",
    data=[...],
    jobs=[preprocessor, analyzer, transformer],
    evaluators=[...],
)
```

## Data sources

`data` accepts inline `DataPoint`s, an Orq dataset, or awaitables that resolve
to `DataPoint`s — the last of which lets you stream rows in from a slow source
without blocking the run:

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

For Orq-hosted datasets, pass `data=DatasetIdInput(dataset_id="...")`. That
path requires `ORQ_API_KEY` — see [Configuration](configuration.md).

## Built-in evaluators

```python
from evaluatorq import exact_match_evaluator, string_contains_evaluator

string_contains_evaluator()                        # case-insensitive by default
string_contains_evaluator(case_insensitive=False)  # case-sensitive
string_contains_evaluator(name="my-contains-check")  # custom name in the table
exact_match_evaluator()                            # case-sensitive by default
```

Both compare the job output against the data point's `expected_output`. For
LLM-graded evaluators see [LLM as a Jury](llm-as-a-jury.md); for structured,
multi-dimensional scores see [Structured Results](structured-results.md).

## Custom evaluators

An evaluator is a `{"name": ..., "scorer": ...}` pair whose scorer receives the
data point and the job output and returns a score:

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

When any evaluator returns `pass_: False`, `evaluatorq()` returns the results;
the library never exits the process. To make a script a CI gate, inspect the
results and exit explicitly:

```python
from evaluatorq.evaluatorq import check_pass_failures

results = await evaluatorq(...)
if check_pass_failures(results):
    raise SystemExit(1)
```

The results table gains a pass rate row — `Pass Rate | 75% (3/4)`.

## Controlling the run

### Parallelism

```python
await evaluatorq("parallel-eval", data=[...], jobs=[...], parallelism=10)
```

`parallelism` counts **tasks**, and the bounds nest: at most `parallelism`
datapoints run at once, and within each one a separate budget of the same size
covers its jobs and then its evaluators. Ten datapoints each running ten
evaluators is a hundred concurrent tasks, not ten.

### Bounding LLM requests

Against a provider concurrency limit, size the request ceiling instead:

```python
await evaluatorq("bounded-eval", data=[...], jobs=[...], max_concurrent_llm_calls=10)
```

This counts requests, not tasks, so it holds however the fan-out nests. It is a
concurrency bound rather than a rate limit — ten slots against 10s calls is
about 60 requests/minute, but the same ten slots become 300/minute if the
provider speeds up to 2s.

Requests evaluatorq issues itself (judges, juries, simulation agents, the
red-team pipeline) take a slot automatically. A job that calls a provider SDK
directly is invisible to the budget unless you wrap it:

```python
from evaluatorq.common.llm_limit import llm_slot

async def my_job(data_point, row_index):
    async with llm_slot():
        response = await client.chat.completions.create(...)
    return {"name": "my-job", "output": response.choices[0].message.content}
```

Wrap only the request — holding a slot across parsing shrinks the budget
without reducing load on the provider.

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
- **[Framework Integrations](framework-integrations.md)** — LangChain, LangGraph, OpenResponses.
- **[Orq Deployment](orq-deployment.md)** — call Orq deployments from a job.
- **[Configuration](configuration.md)** — environment variables.
