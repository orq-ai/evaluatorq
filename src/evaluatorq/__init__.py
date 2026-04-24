"""EvaluatorQ Python - An evaluation framework for LLM applications."""

from .deployment import (
    DeploymentResponse,
    MessageDict,
    ThreadConfig,
    deployment,
    invoke,
)
from .evaluatorq import evaluatorq
from .evaluators import (
    exact_match_evaluator,
    string_contains_evaluator,
)
from .job_helper import job
from .openresponses import ResponseResourceDict
from .types import (
    DataPoint,
    DataPointDict,
    DataPointInput,
    DataPointResult,
    DatasetIdInput,
    EvaluationResult,
    EvaluationResultCell,
    EvaluationResultCellValue,
    Evaluator,
    EvaluatorParams,
    EvaluatorqResult,
    EvaluatorScore,
    Job,
    JobResult,
    JobReturn,
    Output,
    Scorer,
    ScorerParameter,
)

__all__ = [
    # Types
    "DataPoint",
    "DataPointDict",
    "DataPointInput",
    "DataPointResult",
    "DatasetIdInput",
    "DeploymentResponse",
    "EvaluationResult",
    "EvaluationResultCell",
    "EvaluationResultCellValue",
    "Evaluator",
    "EvaluatorParams",
    "EvaluatorScore",
    "EvaluatorqResult",
    "Job",
    "JobResult",
    "JobReturn",
    "MessageDict",
    "Output",
    "ResponseResourceDict",
    "Scorer",
    "ScorerParameter",
    "ThreadConfig",
    # Deployment helpers
    "deployment",
    # Main function
    "evaluatorq",
    "exact_match_evaluator",
    "invoke",
    # Helper functions
    "job",
    # Built-in evaluators
    "string_contains_evaluator",
]
