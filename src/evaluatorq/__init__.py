"""EvaluatorQ Python - An evaluation framework for LLM applications."""

import importlib.util
import sys

_CORE_DEPS = ('openai', 'typer')  # noqa: RUF067


def _require_core_deps() -> None:  # noqa: RUF067
    """Fail with an actionable message when core deps are missing.

    ``openai`` and ``typer`` are hard dependencies, so a missing one almost
    always means the installer targeted a different interpreter than the one
    running. Naming the interpreter is the whole point of this check.
    """
    missing = [name for name in _CORE_DEPS if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            f'evaluatorq requires {", ".join(missing)}, but it is not importable from '
            f'`{sys.executable}`. Install like so: '
            f'`{sys.executable} -m pip install evaluatorq`'
        )


_require_core_deps()  # noqa: RUF067

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
from .llm_jury import PairwiseComparator, llm_jury, llm_jury_pairwise
from .openresponses import ResponseResourceDict
from .pairwise import (
    JudgeStats,
    PairwiseComparison,
    PairwiseReport,
    PairwiseVote,
    build_report,
    run_pairwise,
)
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
    ExperimentInput,
    Job,
    JobResult,
    JobReturn,
    Output,
    Scorer,
    ScorerParameter,
)

__all__ = [
    # Types
    'DataPoint',
    'DataPointDict',
    'DataPointInput',
    'DataPointResult',
    'DatasetIdInput',
    'DeploymentResponse',
    'EvaluationResult',
    'EvaluationResultCell',
    'EvaluationResultCellValue',
    'Evaluator',
    'EvaluatorParams',
    'EvaluatorScore',
    'EvaluatorqResult',
    'ExperimentInput',
    'Job',
    'JobResult',
    'JobReturn',
    'JudgeStats',
    'MessageDict',
    'Output',
    # Pairwise (preference) jury
    'PairwiseComparator',
    'PairwiseComparison',
    'PairwiseReport',
    'PairwiseVote',
    'ResponseResourceDict',
    'Scorer',
    'ScorerParameter',
    'ThreadConfig',
    'build_report',
    # Deployment helpers
    'deployment',
    # Main function
    'evaluatorq',
    'exact_match_evaluator',
    'invoke',
    # Helper functions
    'job',
    # LLM jury evaluator
    'llm_jury',
    'llm_jury_pairwise',
    'run_pairwise',
    # Built-in evaluators
    'string_contains_evaluator',
]
