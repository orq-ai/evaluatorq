# evaluatorq-py

An evaluation framework library for Python that provides a flexible way to run parallel evaluations and optionally integrate with the Orq AI platform.

## 🎯 Features

- **Parallel Execution**: Run multiple evaluation jobs concurrently with progress tracking
- **Flexible Data Sources**: Support for inline data, async iterables, and Orq platform datasets
- **Type-safe**: Fully typed with Python type hints and Pydantic models with runtime validation
- **Rich Terminal UI**: Beautiful progress indicators and result tables powered by Rich
- **Orq Platform Integration**: Seamlessly fetch and evaluate datasets from Orq AI (optional)

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
from evaluatorq import evaluatorq, job, DataPoint, EvaluationResult

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

    return EvaluationResult(value=score, explanation=explanation)

async def main():
    # Requires ORQ_API_KEY environment variable
    await evaluatorq(
        "dataset-evaluation",
        data={"dataset_id": "your-dataset-id"},  # From Orq platform
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

**Manual pattern (not recommended):**
```python
# Without decorator - requires manual wrapper every time
async def process_text(data: DataPoint, row: int):
    return {"name": "text-processor", "output": {"result": data.inputs["text"].upper()}}
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

- `ORQ_API_KEY`: API key for Orq platform integration (required for dataset access and sending results)

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

class EvaluationResult(BaseModel):
    """Result from an evaluator."""
    value: str | float | bool
    explanation: str | None = None

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

Scorer = Callable[[ScorerParameter], Awaitable[EvaluationResult]]

class Evaluator(TypedDict):
    """Evaluator configuration."""
    name: str
    scorer: Scorer
```

## 🛠️ Development

```bash
# Install dependencies
uv sync

# Run type checking
pyright

# Format code
ruff format

# Lint code
ruff check
```
