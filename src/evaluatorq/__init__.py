"""EvaluatorQ Python - An evaluation framework for LLM applications."""

from .contracts import AgentResponse
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
    'AgentResponse',
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
