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


def _check_redteam_deps() -> None:
    missing = []
    for mod in ('openai', 'loguru', 'typer'):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(
            f"Red teaming requires optional dependencies: {', '.join(missing)}. "
            f"Install with: pip install 'evaluatorq[redteam]'"
        )


_check_redteam_deps()

from typing import Any

from evaluatorq.redteam.contracts import (
    AgentCapability,
    AgentContext,
    TargetConfig,
    AgentInfo,
    AttackInfo,
    AttackSource,
    AttackStrategy,
    AttackTechnique,
    CategorySummary,
    DeliveryMethod,
    DeliveryMethodSummary,
    DimensionSummary,
    ErrorInfo,
    EvaluationPayload,
    AttackEvaluationResult,
    ExecutionDetails,
    FocusAreaRecommendation,
    Framework,
    FrameworkSummary,
    FunctionCall,
    JobOutputPayload,
    KnowledgeBaseInfo,
    LLMConfig,
    MemoryStoreInfo,
    Message,
    OrchestratorResult,
    Pipeline,
    PipelineStage,
    RedTeamInput,
    RedTeamReport,
    RedTeamResult,
    ReportSnapshot,
    ReportSummary,
    DomainSummary,
    SEVERITY_DEFINITIONS,
    Severity,
    SeveritySummary,
    TechniqueSummary,
    TokenUsage,
    ToolCall,
    ToolInfo,
    TurnType,
    TurnTypeSummary,
    UnifiedEvaluationResult,
    Vulnerability,
    VulnerabilityDef,
    VulnerabilityDomain,
    VulnerabilitySummary,
    normalize_category,
    normalize_framework,
)
from evaluatorq.redteam.backends.openai import OpenAIModelTarget

from evaluatorq.redteam.runner import red_team
from evaluatorq.redteam.reports.converters import merge_reports
from evaluatorq.redteam.reports.display import print_report_summary
from evaluatorq.redteam.adaptive.strategy_registry import (
    get_category_info,
    list_available_categories as list_categories,
)
from evaluatorq.redteam.vulnerability_registry import (
    VULNERABILITY_DEFS,
    get_vulnerability_name,
    list_available_vulnerabilities,
    resolve_category,
    resolve_vulnerabilities,
)
from evaluatorq.redteam.exceptions import BackendError, CancelledError, CredentialError, RedTeamError
from evaluatorq.redteam.backends.registry import register_backend
from evaluatorq.redteam.backends.base import (
    AgentTarget,
    DirectTargetFactory,
    SupportsAgentContext,
    SupportsErrorMapping,
    SupportsMemoryCleanup,
    SupportsTargetFactory,
    SupportsTokenUsage,
    is_agent_target,
)
from evaluatorq.redteam.hooks import (
    ConfirmPayload,
    DefaultHooks,
    PipelineHooks,
    RichHooks,
)

__all__ = [
    # Entry points
    "red_team",
    "merge_reports",
    "print_report_summary",
    # Category introspection
    "list_categories",
    "get_category_info",
    # Vulnerability introspection
    "list_available_vulnerabilities",
    "get_vulnerability_name",
    "resolve_vulnerabilities",
    "resolve_category",
    "VULNERABILITY_DEFS",
    # Hook system
    "PipelineHooks",
    "DefaultHooks",
    "RichHooks",
    "ConfirmPayload",
    # Core public types
    "RedTeamReport",
    "RedTeamResult",
    "AgentContext",
    "TargetConfig",
    "TokenUsage",
    # Enums
    "TurnType",
    "AttackSource",
    "AttackTechnique",
    "DeliveryMethod",
    "Severity",
    "Framework",
    "Pipeline",
    "PipelineStage",
    "AgentCapability",
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
    "LLMConfig",
    "OpenAIModelTarget",
    # Attack models
    "AttackStrategy",
    "AttackInfo",
    # Result models
    "OrchestratorResult",
    "AttackEvaluationResult",
    "UnifiedEvaluationResult",
    "EvaluationPayload",
    "JobOutputPayload",
    "ExecutionDetails",
    "AgentInfo",
    # Vulnerability types
    "Vulnerability",
    "VulnerabilityDef",
    "VulnerabilityDomain",
    "VulnerabilitySummary",
    # Error model
    "ErrorInfo",
    # Report models
    "CategorySummary",
    "DimensionSummary",
    "ReportSummary",
    "ReportSnapshot",
    "TechniqueSummary",
    "SeveritySummary",
    "DeliveryMethodSummary",
    "TurnTypeSummary",
    "DomainSummary",
    "FrameworkSummary",
    # Helper functions
    "normalize_framework",
    "normalize_category",
    # Exceptions
    "RedTeamError",
    "CredentialError",
    "BackendError",
    "CancelledError",
    # Backend extension
    "register_backend",
    # Target protocols
    "AgentTarget",
    "SupportsTokenUsage",
    "SupportsAgentContext",
    "SupportsTargetFactory",
    "SupportsMemoryCleanup",
    "SupportsErrorMapping",
    "DirectTargetFactory",
    "is_agent_target",
]


_deprecated_warned: set[str] = set()


def __getattr__(name: str):
    if name in ("RedTeamConfig", "PipelineLLMConfig"):
        if name not in _deprecated_warned:
            import warnings
            _deprecated_warned.add(name)
            warnings.warn(
                f"{name} has been renamed to LLMConfig. Update your imports.",
                DeprecationWarning,
                stacklevel=2,
            )
        from evaluatorq.redteam.contracts import LLMConfig
        return LLMConfig
    if name == "EvaluationResult":
        if name not in _deprecated_warned:
            import warnings
            _deprecated_warned.add(name)
            warnings.warn(
                "EvaluationResult is deprecated in evaluatorq.redteam. "
                "Use AttackEvaluationResult instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return AttackEvaluationResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
