"""Red teaming subpackage for evaluatorq.

Provides adaptive red teaming capabilities for testing AI agents against
OWASP security frameworks (ASI and LLM Top 10).

Public API:
    red_team(...)  — Unified entry point for dynamic/static/hybrid red teaming.

Semantic convention throughout this package:
    ``passed=True``  → the agent is RESISTANT (attack failed)
    ``passed=False`` → the agent is VULNERABLE (attack succeeded)
"""

from __future__ import annotations

from typing import Any

from evaluatorq.redteam.contracts import (
    AgentContext,
    AgentInfo,
    AttackEvaluationRow,
    AttackInfo,
    AttackStrategy,
    AttackTechnique,
    CategorySummary,
    DatasetInferenceRow,
    DeliveryMethod,
    DynamicAttackResultRow,
    DynamicCategorySummaryRow,
    DynamicErrorAnalysisRow,
    DynamicRunMetadata,
    DynamicSummaryReportRow,
    ErrorTypeDetails,
    EvaluatedRow,
    EvaluatedRowBase,
    EvaluationPayload,
    EvaluationResult,
    ExecutionDetails,
    Framework,
    FunctionCall,
    GeneratedPromptRow,
    JobOutputPayload,
    KnowledgeBaseInfo,
    MemoryStoreInfo,
    Message,
    OrchestratorResult,
    Pipeline,
    PipelineLLMConfig,
    RedTeamInput,
    RedTeamReport,
    RedTeamResult,
    ReportSnapshot,
    ReportSummary,
    Scope,
    Severity,
    StrategySelectionRow,
    TokenUsage,
    ToolCall,
    ToolInfo,
    TurnType,
    UnifiedEvaluationResult,
    infer_framework,
    normalize_category,
    normalize_framework,
)

__all__ = [
    # Core public types
    "RedTeamReport",
    "RedTeamResult",
    "AgentContext",
    "TokenUsage",
    # Enums
    "TurnType",
    "AttackTechnique",
    "DeliveryMethod",
    "Severity",
    "Scope",
    "Framework",
    "Pipeline",
    # Message models
    "FunctionCall",
    "ToolCall",
    "Message",
    # Input models
    "RedTeamInput",
    # Agent context models
    "ToolInfo",
    "MemoryStoreInfo",
    "KnowledgeBaseInfo",
    # Pipeline config
    "PipelineLLMConfig",
    # Attack models
    "AttackStrategy",
    "AttackInfo",
    # Result models
    "OrchestratorResult",
    "EvaluationResult",
    "UnifiedEvaluationResult",
    "EvaluationPayload",
    "JobOutputPayload",
    "ExecutionDetails",
    "AgentInfo",
    # Report models
    "CategorySummary",
    "ReportSummary",
    "ReportSnapshot",
    # Dynamic pipeline row models
    "EvaluatedRowBase",
    "EvaluatedRow",
    "AttackEvaluationRow",
    "DynamicAttackResultRow",
    "DynamicCategorySummaryRow",
    "ErrorTypeDetails",
    "DynamicErrorAnalysisRow",
    "DynamicRunMetadata",
    "StrategySelectionRow",
    "GeneratedPromptRow",
    "DynamicSummaryReportRow",
    "DatasetInferenceRow",
    # Helper functions
    "normalize_framework",
    "normalize_category",
    "infer_framework",
    # Entry point stub
    "red_team",
]


async def red_team(
    target: str,
    *,
    mode: str = "dynamic",
    categories: list[str] | None = None,
    dataset_path: Any = None,
    **kwargs: Any,
) -> RedTeamReport:
    """Unified entry point for red teaming.

    Args:
        target: Target identifier (e.g., "agent:my-agent-key").
        mode: Execution mode — "dynamic", "static", or "hybrid".
        categories: OWASP categories to test (e.g., ["ASI01", "ASI03"]).
        dataset_path: Path to static dataset (required for static/hybrid modes).
        **kwargs: Additional backend-specific options.

    Returns:
        RedTeamReport with results and summary.

    Raises:
        NotImplementedError: Until Phase 3 implementation is complete.
    """
    # Phase 3 stub — parameters are part of the public API contract.
    _ = (target, mode, categories, dataset_path, kwargs)
    raise NotImplementedError(
        "red_team() is not yet implemented. This will be available after Phase 3 of the migration."
    )
