from collections.abc import Awaitable, Sequence
from typing import Any, Callable, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from evaluatorq.openresponses import ResponseResourceDict

Output = str | int | float | bool | ResponseResourceDict | dict[str, Any] | None
"""Output type alias"""


EvaluationResultCellValue = str | int | float | dict[str, "str | float | dict[str, str | float]"]


class EvaluationResultCell(BaseModel):
    type: str
    value: dict[str, EvaluationResultCellValue]


class EvaluationResult(BaseModel):
    value: str | int | float | bool | EvaluationResultCell
    explanation: str | None = None
    pass_: bool | None = Field(default=None, serialization_alias="pass")


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


class _DataPointDictRequired(TypedDict):
    """Required fields for DataPointDict."""

    inputs: dict[str, Any]


class DataPointDict(_DataPointDictRequired, total=False):
    """Dict representation of a DataPoint for type checking."""

    expected_output: Output | None


class DataPoint(BaseModel):
    """
    A data point for evaluation.

    Args:
        inputs: The inputs to pass to the job.
        expected_output: The expected output of the data point.
                        Used for evaluation and comparing the output of the job.
    """

    inputs: dict[str, Any]
    expected_output: Output | None = Field(
        default=None, serialization_alias="expectedOutput"
    )


DataPointInput = DataPoint | DataPointDict
"""Type alias for DataPoint that accepts both model instances and dicts."""


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
    """Parameters passed to a scorer function  
    Args:
        data: The data point being evaluated.
        output: The output produced by the job for the data point.
    """

    data: DataPoint
    output: Output


Scorer = Callable[[ScorerParameter], Awaitable[EvaluationResult | dict[str, Any]]]


class Evaluator(TypedDict):
    name: str
    scorer: Scorer


class DatasetIdInput(BaseModel):
    """Input for fetching a dataset from Orq platform."""

    dataset_id: str
    include_messages: bool = False


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
        path: Optional path (e.g. "MyProject/MyFolder") to place the experiment
              in a specific project and folder on the Orq platform.
    """

    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}

    data: DatasetIdInput | Sequence[Awaitable[DataPoint] | DataPointInput]
    jobs: list[Job]
    evaluators: list[Evaluator] | None = None
    parallelism: int = Field(default=1, ge=1)
    print_results: bool = Field(default=True)
    description: str | None = None
    path: str | None = None
