from collections.abc import Awaitable, Sequence
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

Output = str | int | float | bool | dict[str, Any] | None
"""Output type alias"""


class EvaluationResult(BaseModel):
    value: str | float | bool
    explanation: str | None = None


class EvaluatorScore(BaseModel):
    evaluator_name: str = Field(serialization_alias="evaluatorName")
    score: EvaluationResult
    error: str | None = None


class JobResult(BaseModel):
    job_name: str = Field(serialization_alias="jobName")
    output: Output
    error: str | None = None
    evaluator_scores: list[EvaluatorScore] | None = Field(
        default=None, serialization_alias="evaluatorScores"
    )


class DataPoint(BaseModel):
    """
    A data point for evaluation.

    Args:
        inputs: The inputs to pass to the job.
        expected_output: The expected output of the data point.
                        Used for evaluation and comparing the output of the job.
    """

    inputs: dict[str, Any]
    expected_output: Output | None = None


class DataPointResult(BaseModel):
    data_point: DataPoint = Field(serialization_alias="dataPoint")
    error: str | None = None
    job_results: list[JobResult] | None = Field(
        default=None, serialization_alias="jobResults"
    )


EvaluatorqResult = list[DataPointResult]
"""Type alias for evaluation results"""


class JobReturn(TypedDict):
    """Job return structure"""

    name: str
    output: Output


Job = Callable[[DataPoint, int], Awaitable[dict[str, Any]]]
"""Job function type - returns a dict with 'name' and 'output' keys"""


class ScorerParameter(TypedDict):
    data: DataPoint
    output: Output


Scorer = Callable[[ScorerParameter], Awaitable[EvaluationResult | dict[str, Any]]]


class Evaluator(TypedDict):
    name: str
    scorer: Scorer


class DatasetIdInput(BaseModel):
    """Input for fetching a dataset from Orq platform."""

    dataset_id: str


class EvaluatorParams(BaseModel):
    """
    Parameters for running an evaluation.

    Args:
        data: The data to evaluate. Either a DatasetIdInput to fetch from Orq platform,
              or a list of DataPoint instances/awaitables.
        jobs: The jobs to run on the data.
        evaluators: The evaluators to use. If not provided, only jobs will run.
        parallelism: Number of jobs to run in parallel. Defaults to 1 (sequential).
        print_results: Whether to print results table to console. Defaults to True.
        description: Optional description for the evaluation run.
    """

    model_config: ConfigDict = {"arbitrary_types_allowed": True}

    data: DatasetIdInput | Sequence[Awaitable[DataPoint] | DataPoint]
    jobs: list[Job]
    evaluators: list[Evaluator] | None = None
    parallelism: int = Field(default=1, ge=1)
    print_results: bool = Field(default=True)
    description: str | None = None
