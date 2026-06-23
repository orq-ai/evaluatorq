"""EvaluatorQ Python - An evaluation framework for LLM applications."""

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    # Lazily exported at runtime via __getattr__ (keeps the optional ``openai``
    # dependency out of the import path); declared here so type checkers and
    # ``__all__`` resolve the symbol.
    from .llm_jury import llm_jury as llm_jury
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
    # LLM jury evaluator
    "llm_jury",
    # Built-in evaluators
    "string_contains_evaluator",
]


def __getattr__(name: str) -> object:
    # ``llm_jury`` pulls in ``openai`` (an optional dependency). Keep it lazy so
    # ``import evaluatorq`` stays cheap and works on core-only installs; surface
    # an actionable hint only when someone actually reaches for the jury.
    if name == "llm_jury":
        try:
            from .llm_jury import llm_jury
        except ImportError as exc:
            raise ImportError(
                "`llm_jury` requires the `openai` package, which is not installed. "
                "Install the judge extra: pip install 'evaluatorq[judge]' "
                "(or: uv pip install 'evaluatorq[judge]')."
            ) from exc
        return llm_jury
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
