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

from evaluatorq.redteam._runner import red_team
from evaluatorq.redteam.adaptive.strategy_registry import (
    get_category_info,
    list_available_categories as list_categories,
)

__all__ = [
    # Entry point
    "red_team",
    # Category introspection
    "list_categories",
    "get_category_info",
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
]
