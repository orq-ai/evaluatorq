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
from .types import (
    DataPoint,
    DataPointResult,
    DatasetIdInput,
    EvaluationResult,
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
    # Main function
    "evaluatorq",
    # Helper functions
    "job",
    # Deployment helpers
    "deployment",
    "invoke",
    "DeploymentResponse",
    "ThreadConfig",
    "MessageDict",
    # Built-in evaluators
    "string_contains_evaluator",
    "exact_match_evaluator",
    # Types
    "DataPoint",
    "DataPointResult",
    "DatasetIdInput",
    "EvaluationResult",
    "Evaluator",
    "EvaluatorParams",
    "EvaluatorqResult",
    "EvaluatorScore",
    "Job",
    "JobResult",
    "JobReturn",
    "Output",
    "Scorer",
    "ScorerParameter",
]
