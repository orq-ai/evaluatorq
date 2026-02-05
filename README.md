# evaluatorq-py

An evaluation framework library for Python that provides a flexible way to run parallel evaluations and optionally integrate with the Orq AI platform.

## 🎯 Features

- **Parallel Execution**: Run multiple evaluation jobs concurrently with progress tracking
- **Flexible Data Sources**: Support for inline data, async iterables, and Orq platform datasets
- **Type-safe**: Fully typed with Python type hints and Pydantic models with runtime validation
- **Rich Terminal UI**: Beautiful progress indicators and result tables powered by Rich
- **Orq Platform Integration**: Seamlessly fetch and evaluate datasets from Orq AI (optional)
- **OpenTelemetry Tracing**: Built-in observability with automatic span creation for jobs and evaluators
- **Pass/Fail Tracking**: Evaluators can return pass/fail status for CI/CD integration
- **Built-in Evaluators**: Common evaluators like `string_contains_evaluator` included

## 📥 Installation

```bash
pip install evaluatorq
# or
uv add evaluatorq
# or
poetry add evaluatorq
```

### Optional Dependencies

If you want to use the Orq platform integration:

```bash
pip install orq-ai-sdk
# or
pip install evaluatorq[orq]
```

For OpenTelemetry tracing (optional):

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http opentelemetry-semantic-conventions
# or
pip install evaluatorq[otel]
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from evaluatorq import evaluatorq, job, DataPoint, EvaluationResult

@job("text-analyzer")
async def text_analyzer(data: DataPoint, row: int):
    """Analyze text data and return analysis results."""
    text = data.inputs["text"]
    analysis = {
        "length": len(text),
        "word_count": len(text.split()),
        "uppercase": text.upper(),
    }

    return analysis

async def length_check_scorer(params):
    """Evaluate if output length is sufficient."""
    output = params["output"]
    passes_check = output["length"] > 10

    return EvaluationResult(
        value=1 if passes_check else 0,
        explanation=(
            "Output length is sufficient"
            if passes_check
            else f"Output too short ({output['length']} chars, need >10)"
        )
    )

async def main():
    await evaluatorq(
        "text-analysis",
        data=[
            DataPoint(inputs={"text": "Hello world"}),
            DataPoint(inputs={"text": "Testing evaluation"}),
        ],
        jobs=[text_analyzer],
        evaluators=[
            {
                "name": "length-check",
                "scorer": length_check_scorer,
            }
        ],
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Orq Platform Datasets

```python
import asyncio
from evaluatorq import evaluatorq, job, DataPoint, DatasetIdInput

@job("processor")
async def processor(data: DataPoint, row: int):
    """Process each data point from the dataset."""
    result = await process_data(data)
    return result

async def accuracy_scorer(params):
    """Calculate accuracy by comparing output with expected results."""
    data = params["data"]
    output = params["output"]

    score = calculate_score(output, data.expected_output)

    if score > 0.8:
        explanation = "High accuracy match"
    elif score > 0.5:
        explanation = "Partial match"
    else:
        explanation = "Low accuracy match"

    return {"value": score, "explanation": explanation}

async def main():
    # Requires ORQ_API_KEY environment variable
    await evaluatorq(
        "dataset-evaluation",
        data=DatasetIdInput(dataset_id="your-dataset-id"),  # From Orq platform
        jobs=[processor],
        evaluators=[
            {
                "name": "accuracy",
                "scorer": accuracy_scorer,
            }
        ],
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Features

#### Multiple Jobs

Run multiple jobs in parallel for each data point:

```python
from evaluatorq import job

@job("preprocessor")
async def preprocessor(data: DataPoint, row: int):
    result = await preprocess(data)
    return result

@job("analyzer")
async def analyzer(data: DataPoint, row: int):
    result = await analyze(data)
    return result

@job("transformer")
async def transformer(data: DataPoint, row: int):
    result = await transform(data)
    return result

await evaluatorq(
    "multi-job-eval",
    data=[...],
    jobs=[preprocessor, analyzer, transformer],
    evaluators=[...],
)
```

#### The `@job()` Decorator

The `@job()` decorator provides two key benefits:

1. **Eliminates boilerplate** - No need to manually wrap returns with `{"name": ..., "output": ...}`
2. **Preserves job names in errors** - When a job fails, the error will include the job name for better debugging

**Decorator pattern (recommended):**
```python
from evaluatorq import job

@job("text-processor")
async def process_text(data: DataPoint, row: int):
    # Clean return - just the data!
    return {"result": data.inputs["text"].upper()}
```

**Functional pattern (for lambdas):**
```python
from evaluatorq import job

# Simple transformations with lambda
uppercase_job = job("uppercase", lambda data, row: data.inputs["text"].upper())
word_count_job = job("word-count", lambda data, row: len(data.inputs["text"].split()))
```

#### Deployment Helper

Easily invoke Orq deployments within your evaluation jobs:

```python
from evaluatorq import evaluatorq, job, invoke, deployment, DatasetIdInput

# Simple one-liner with invoke()
@job("summarizer")
async def summarize_job(data, row):
    text = data.inputs["text"]
    return await invoke("my-deployment", inputs={"text": text})

# Full response with deployment()
@job("analyzer")
async def analyze_job(data, row):
    response = await deployment(
        "my-deployment",
        inputs={"text": data.inputs["text"]},
        metadata={"source": "evaluatorq"},
    )
    print("Raw:", response.raw)
    return response.content

# Chat-style with messages
@job("chatbot")
async def chat_job(data, row):
    return await invoke(
        "chatbot",
        messages=[{"role": "user", "content": data.inputs["question"]}],
    )

# Thread tracking for conversations
@job("assistant")
async def conversation_job(data, row):
    return await invoke(
        "assistant",
        inputs={"query": data.inputs["query"]},
        thread={"id": "conversation-123"},
    )
```

The `invoke()` function returns the text content directly, while `deployment()` returns an object with both `content` and `raw` response for more control.

#### Built-in Evaluators

Use the included evaluators for common use cases:

```python
from evaluatorq import evaluatorq, job, string_contains_evaluator, DatasetIdInput

@job("country-lookup")
async def country_lookup_job(data, row):
    country = data.inputs["country"]
    return await invoke("country-capitals", inputs={"country": country})

await evaluatorq(
    "country-unit-test",
    data=DatasetIdInput(dataset_id="your-dataset-id"),
    jobs=[country_lookup_job],
    evaluators=[string_contains_evaluator()],  # Checks if output contains expected_output
    parallelism=6,
)
```

Available built-in evaluators:

- **`string_contains_evaluator()`** - Checks if output contains expected_output (case-insensitive by default)
- **`exact_match_evaluator()`** - Checks if output exactly matches expected_output

```python
# Case-sensitive matching
strict_evaluator = string_contains_evaluator(case_insensitive=False)

# Custom name
my_evaluator = string_contains_evaluator(name="my-contains-check")
```

#### Automatic Error Handling

The `@job()` decorator automatically preserves job names even when errors occur:

```python
from evaluatorq import job

@job("risky-job")
async def risky_operation(data: DataPoint, row: int):
    # If this raises an error, the job name "risky-job" will be preserved
    result = await potentially_failing_operation(data)
    return result

await evaluatorq(
    "error-handling",
    data=[...],
    jobs=[risky_operation],
    evaluators=[...],
)

# Error output will show: "Job 'risky-job' failed: <error details>"
# Without @job decorator, you'd only see: "<error details>"
```

#### Async Data Sources

```python
import asyncio

# Create an array of coroutines for async data
async def get_data_point(i: int) -> DataPoint:
    await asyncio.sleep(0.01)  # Simulate async data fetching
    return DataPoint(inputs={"value": i})

data_promises = [get_data_point(i) for i in range(1000)]

await evaluatorq(
    "async-eval",
    data=data_promises,
    jobs=[...],
    evaluators=[...],
)
```

#### Structured Evaluation Results

Evaluators can return structured, multi-dimensional metrics using `EvaluationResultCell`. This is useful for metrics like BERT scores, ROUGE-N scores, or any evaluation that produces multiple sub-scores.

##### Multi-criteria Rubric

Return multiple quality sub-scores in a single evaluator:

```python
from evaluatorq import evaluatorq, job, DataPoint, EvaluationResult, EvaluationResultCell

@job("echo")
async def echo_job(data: DataPoint, row: int):
    return data.inputs["text"]

async def rubric_scorer(params):
    text = str(params["output"])
    return EvaluationResult(
        value=EvaluationResultCell(
            type="rubric",
            value={
                "relevance": min(len(text) / 100, 1),
                "coherence": 0.9 if "." in text else 0.4,
                "fluency": 0.85 if len(text.split()) > 5 else 0.5,
            },
        ),
        explanation="Multi-criteria quality rubric",
    )

await evaluatorq(
    "structured-rubric",
    data=[
        DataPoint(inputs={"text": "The quick brown fox jumps over the lazy dog."}),
        DataPoint(inputs={"text": "Hi"}),
    ],
    jobs=[echo_job],
    evaluators=[{"name": "rubric", "scorer": rubric_scorer}],
)
```

##### Sentiment Distribution

Break down sentiment across categories:

```python
async def sentiment_scorer(params):
    text = str(params["output"]).lower()
    positive_words = ["good", "great", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "sad", "hate"]
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    total = max(pos_count + neg_count, 1)

    return EvaluationResult(
        value=EvaluationResultCell(
            type="sentiment",
            value={
                "positive": pos_count / total,
                "negative": neg_count / total,
                "neutral": 1 - (pos_count + neg_count) / total,
            },
        ),
        explanation="Sentiment distribution across categories",
    )
```

##### Safety Scores with Pass/Fail

Combine structured scores with pass/fail tracking for CI/CD:

```python
async def safety_scorer(params):
    text = str(params["output"]).lower()
    categories = {
        "hate_speech": 0.8 if "hate" in text else 0.1,
        "violence": 0.7 if ("kill" in text or "fight" in text) else 0.05,
        "profanity": 0.5 if "damn" in text else 0.02,
    }

    return EvaluationResult(
        value=EvaluationResultCell(
            type="safety",
            value=categories,
        ),
        pass_=all(score < 0.5 for score in categories.values()),
        explanation="Content safety severity scores per category",
    )
```

See the runnable Python examples in the `examples/` directory:

- [`structured_rubric_eval.py`](examples/structured_rubric_eval.py) - Multi-criteria quality rubric
- [`structured_sentiment_eval.py`](examples/structured_sentiment_eval.py) - Sentiment distribution breakdown
- [`structured_safety_eval.py`](examples/structured_safety_eval.py) - Safety scores with pass/fail tracking

> **Note:** Structured results display as `[structured]` in the terminal summary table but are preserved in full when sent to the Orq platform and OpenTelemetry spans.

#### Controlling Parallelism

```python
await evaluatorq(
    "parallel-eval",
    data=[...],
    jobs=[...],
    evaluators=[...],
    parallelism=10,  # Run up to 10 jobs concurrently
)
```

#### Disable Progress Display

```python
# Get raw results without terminal output
results = await evaluatorq(
    "silent-eval",
    data=[...],
    jobs=[...],
    evaluators=[...],
    print_results=False,  # Disable progress and table display
)

# Process results programmatically
for result in results:
    print(result.data_point.inputs)
    for job_result in result.job_results:
        print(f"{job_result.job_name}: {job_result.output}")
```

## 🔧 Configuration

### Environment Variables

- `ORQ_API_KEY`: API key for Orq platform integration (required for dataset access and sending results). Also enables automatic OTEL tracing to Orq.
- `ORQ_BASE_URL`: Base URL for Orq platform (default: `https://my.orq.ai`)
- `OTEL_EXPORTER_OTLP_ENDPOINT`: Custom OpenTelemetry collector endpoint (overrides default Orq endpoint)
- `OTEL_EXPORTER_OTLP_HEADERS`: Headers for OTEL exporter (format: `key1=value1,key2=value2`)
- `ORQ_DISABLE_TRACING`: Set to `1` or `true` to disable automatic tracing
- `ORQ_DEBUG`: Enable debug logging for tracing setup

### Evaluation Parameters

Parameters are validated at runtime using Pydantic. The `evaluatorq` function supports three calling styles:

```python
from evaluatorq import evaluatorq, EvaluatorParams

# 1. Keyword arguments (recommended)
await evaluatorq(
    "my-eval",
    data=[...],
    jobs=[...],
    parallelism=5,
)

# 2. Dict style
await evaluatorq("my-eval", {
    "data": [...],
    "jobs": [...],
    "parallelism": 5,
})

# 3. EvaluatorParams instance
await evaluatorq("my-eval", EvaluatorParams(
    data=[...],
    jobs=[...],
    parallelism=5,
))
```

#### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `list[DataPoint]` \| `list[Awaitable[DataPoint]]` \| `DatasetIdInput` | **required** | Data to evaluate |
| `jobs` | `list[Job]` | **required** | Jobs to run on each data point |
| `evaluators` | `list[Evaluator]` \| `None` | `None` | Evaluators to score job outputs |
| `parallelism` | `int` (≥1) | `1` | Number of concurrent jobs |
| `print_results` | `bool` | `True` | Display progress and results table |
| `description` | `str` \| `None` | `None` | Optional evaluation description |

## 📊 Orq Platform Integration

### Automatic Result Sending

When the `ORQ_API_KEY` environment variable is set, evaluatorq automatically sends evaluation results to the Orq platform for visualization and analysis.

```python
# Results are automatically sent when ORQ_API_KEY is set
await evaluatorq(
    "my-evaluation",
    data=[...],
    jobs=[...],
    evaluators=[...],
)
```

#### What Gets Sent

When the `ORQ_API_KEY` is set, the following information is sent to Orq:
- Evaluation name
- Dataset ID (when using Orq datasets)
- Job results with outputs and errors
- Evaluator scores with values and explanations
- Execution timing information

Note: Evaluator explanations are included in the data sent to Orq but are not displayed in the terminal output to keep the console clean.

#### Result Visualization

After successful submission, you'll see a console message with a link to view your results:

```
📊 View your evaluation results at: <url to the evaluation>
```

The Orq platform provides:
- Interactive result tables
- Score statistics
- Performance metrics
- Historical comparisons

## 🔍 OpenTelemetry Tracing

Evaluatorq automatically creates OpenTelemetry spans for observability into your evaluation runs.

### Span Hierarchy

```
orq.job (independent root per job execution)
└── orq.evaluation (child span per evaluator)
```

### Auto-Enable with Orq

When `ORQ_API_KEY` is set, traces are automatically sent to the Orq platform:

```bash
ORQ_API_KEY=your-api-key python my_eval.py
```

### Custom OTEL Endpoint

Send traces to any OpenTelemetry-compatible backend:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-collector:4318 \
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer token" \
python my_eval.py
```

### Disable Tracing

If you want to disable tracing even when `ORQ_API_KEY` is set:

```bash
ORQ_DISABLE_TRACING=1 python my_eval.py
```

## ✅ Pass/Fail Tracking

Evaluators can return a `pass_` field to indicate pass/fail status:

```python
async def quality_scorer(params):
    """Quality check evaluator with pass/fail."""
    output = params["output"]
    score = calculate_quality(output)

    return {
        "value": score,
        "pass_": score >= 0.8,  # Pass if meets threshold
        "explanation": f"Quality score: {score}",
    }
```

**CI/CD Integration:** When any evaluator returns `pass_: False`, the process exits with code 1. This enables fail-fast behavior in CI/CD pipelines.

**Pass Rate Display:** The summary table shows pass rate when evaluators use the `pass_` field:

```
┌──────────────────────┬─────────────────┐
│ Pass Rate            │ 75% (3/4)       │
└──────────────────────┴─────────────────┘
```

## 📚 API Reference

### `evaluatorq(name, params?, *, data?, jobs?, evaluators?, parallelism?, print_results?, description?) -> EvaluatorqResult`

Main async function to run evaluations.

#### Signature:

```python
async def evaluatorq(
    name: str,
    params: EvaluatorParams | dict[str, Any] | None = None,
    *,
    data: DatasetIdInput | Sequence[Awaitable[DataPoint] | DataPoint] | None = None,
    jobs: list[Job] | None = None,
    evaluators: list[Evaluator] | None = None,
    parallelism: int = 1,
    print_results: bool = True,
    description: str | None = None,
) -> EvaluatorqResult
```

#### Parameters:

- `name`: String identifier for the evaluation run
- `params`: (Optional) `EvaluatorParams` instance or dict with evaluation parameters
- `data`: List of DataPoint objects, awaitables, or `DatasetIdInput`
- `jobs`: List of job functions to run on each data point
- `evaluators`: Optional list of evaluator configurations
- `parallelism`: Number of concurrent jobs (default: 1, must be ≥1)
- `print_results`: Whether to display progress and results (default: True)
- `description`: Optional description for the evaluation run

> **Note:** Parameters can be passed either via the `params` argument (as dict or `EvaluatorParams`) or as keyword arguments. Keyword arguments take precedence over `params` values.

#### Returns:

`EvaluatorqResult` - List of `DataPointResult` objects containing job outputs and evaluator scores.

### Types

```python
from typing import Any, Callable, Awaitable
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Output type alias
Output = str | int | float | bool | dict[str, Any] | None

class DataPoint(BaseModel):
    """A data point for evaluation."""
    inputs: dict[str, Any]
    expected_output: Output | None = None

EvaluationResultCellValue = str | float | dict[str, "str | float | dict[str, str | float]"]

class EvaluationResultCell(BaseModel):
    """Structured evaluation result with multi-dimensional metrics."""
    type: str
    value: dict[str, EvaluationResultCellValue]

class EvaluationResult(BaseModel):
    """Result from an evaluator."""
    value: str | float | bool | EvaluationResultCell
    explanation: str | None = None
    pass_: bool | None = None  # Optional pass/fail indicator for CI/CD integration

class EvaluatorScore(BaseModel):
    """Score from an evaluator for a job output."""
    evaluator_name: str
    score: EvaluationResult
    error: str | None = None

class JobResult(BaseModel):
    """Result from a job execution."""
    job_name: str
    output: Output
    error: str | None = None
    evaluator_scores: list[EvaluatorScore] | None = None

class DataPointResult(BaseModel):
    """Result for a single data point."""
    data_point: DataPoint
    error: str | None = None
    job_results: list[JobResult] | None = None

# Type aliases
EvaluatorqResult = list[DataPointResult]

class DatasetIdInput(BaseModel):
    """Input for fetching a dataset from Orq platform."""
    dataset_id: str

class EvaluatorParams(BaseModel):
    """Parameters for running an evaluation (validated at runtime)."""
    data: DatasetIdInput | Sequence[Awaitable[DataPoint] | DataPoint]
    jobs: list[Job]
    evaluators: list[Evaluator] | None = None
    parallelism: int = Field(default=1, ge=1)
    print_results: bool = True
    description: str | None = None

class JobReturn(TypedDict):
    """Job return structure."""
    name: str
    output: Output

Job = Callable[[DataPoint, int], Awaitable[JobReturn]]

class ScorerParameter(TypedDict):
    """Parameters passed to scorer functions."""
    data: DataPoint
    output: Output

Scorer = Callable[[ScorerParameter], Awaitable[EvaluationResult | dict[str, Any]]]

class Evaluator(TypedDict):
    """Evaluator configuration."""
    name: str
    scorer: Scorer

# Deployment helper types
@dataclass
class DeploymentResponse:
    """Response from a deployment invocation."""
    content: str  # Text content of the response
    raw: Any      # Raw API response

# Invoke deployment and get text content
async def invoke(
    key: str,
    inputs: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    thread: dict[str, Any] | None = None,  # Must include 'id' key
    messages: list[dict[str, str]] | None = None,
) -> str: ...

# Invoke deployment and get full response
async def deployment(
    key: str,
    inputs: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    thread: dict[str, Any] | None = None,  # Must include 'id' key
    messages: list[dict[str, str]] | None = None,
) -> DeploymentResponse: ...

# Built-in evaluators
def string_contains_evaluator(
    case_insensitive: bool = True,
    name: str = "string-contains",
) -> Evaluator: ...

def exact_match_evaluator(
    case_insensitive: bool = False,
    name: str = "exact-match",
) -> Evaluator: ...
```

## 🛠️ Development

```bash
# Install dependencies
uv sync

# Run type checking
uv run basedpyright

# Format code
uv run ruff format

# Lint code
uv run ruff check
```
