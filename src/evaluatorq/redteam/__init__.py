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


def _check_redteam_deps() -> None:  # noqa: RUF067
    missing = []
    for mod in ('openai', 'loguru', 'typer'):
        try:
            __import__(mod)
        except ImportError:  # noqa: PERF203
            missing.append(mod)
    if missing:
        raise ImportError(
            f"Red teaming requires optional dependencies: {', '.join(missing)}. "
            f"Install with: pip install 'evaluatorq[redteam]'"
        )


_check_redteam_deps()  # noqa: RUF067


from evaluatorq.redteam.adaptive.strategy_registry import (
    get_category_info,
)
from evaluatorq.redteam.adaptive.strategy_registry import (
    list_available_categories as list_categories,
)
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
from evaluatorq.redteam.backends.openai import OpenAIModelTarget
from evaluatorq.redteam.backends.registry import register_backend
from evaluatorq.redteam.contracts import (
    SEVERITY_DEFINITIONS,
    AgentCapability,
    AgentContext,
    AgentInfo,
    AttackEvaluationResult,
    AttackInfo,
    AttackSource,
    AttackStrategy,
    AttackTechnique,
    CategorySummary,
    DeliveryMethod,
    DeliveryMethodSummary,
    DimensionSummary,
    DomainSummary,
    ErrorInfo,
    EvaluationPayload,
    ExecutionDetails,
    FocusAreaRecommendation,
    Framework,
    FrameworkSummary,
    FunctionCall,
    JobOutputPayload,
    KnowledgeBaseInfo,
    LLMCallConfig,
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
    Severity,
    SeveritySummary,
    TargetConfig,
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
from evaluatorq.redteam.exceptions import BackendError, CancelledError, CredentialError, RedTeamError
from evaluatorq.redteam.hooks import (
    ConfirmPayload,
    DefaultHooks,
    PipelineHooks,
    RichHooks,
)
from evaluatorq.redteam.reports.converters import merge_reports
from evaluatorq.redteam.reports.display import print_report_summary
from evaluatorq.redteam.runner import red_team
from evaluatorq.redteam.vulnerability_registry import (
    VULNERABILITY_DEFS,
    get_vulnerability_name,
    list_available_vulnerabilities,
    resolve_category,
    resolve_vulnerabilities,
)

__all__ = [
    'SEVERITY_DEFINITIONS',
    "VULNERABILITY_DEFS",
    "AgentCapability",
    "AgentContext",
    "AgentInfo",
    # Target protocols
    "AgentTarget",
    "AttackEvaluationResult",
    "AttackInfo",
    "AttackSource",
    # Attack models
    "AttackStrategy",
    "AttackTechnique",
    "BackendError",
    "CancelledError",
    # Report models
    "CategorySummary",
    "ConfirmPayload",
    "CredentialError",
    "DefaultHooks",
    "DeliveryMethod",
    "DeliveryMethodSummary",
    "DimensionSummary",
    "DirectTargetFactory",
    "DomainSummary",
    # Error model
    "ErrorInfo",
    "EvaluationPayload",
    "ExecutionDetails",
    'FocusAreaRecommendation',
    "Framework",
    "FrameworkSummary",
    # Message models
    "FunctionCall",
    "JobOutputPayload",
    "KnowledgeBaseInfo",
    # Pipeline config
    "LLMCallConfig",
    "LLMConfig",
    "MemoryStoreInfo",
    "Message",
    "OpenAIModelTarget",
    # Result models
    "OrchestratorResult",
    "Pipeline",
    # Hook system
    "PipelineHooks",
    "PipelineStage",
    # Exceptions
    "RedTeamError",
    # Input models
    "RedTeamInput",
    # Core public types
    "RedTeamReport",
    "RedTeamResult",
    "ReportSnapshot",
    "ReportSummary",
    "RichHooks",
    "Severity",
    "SeveritySummary",
    "SupportsAgentContext",
    "SupportsErrorMapping",
    "SupportsMemoryCleanup",
    "SupportsTargetFactory",
    "SupportsTokenUsage",
    "TargetConfig",
    "TechniqueSummary",
    "TokenUsage",
    "ToolCall",
    # Agent context models
    "ToolInfo",
    # Enums
    "TurnType",
    "TurnTypeSummary",
    "UnifiedEvaluationResult",
    # Vulnerability types
    "Vulnerability",
    "VulnerabilityDef",
    "VulnerabilityDomain",
    "VulnerabilitySummary",
    "get_category_info",
    "get_vulnerability_name",
    "is_agent_target",
    # Vulnerability introspection
    "list_available_vulnerabilities",
    # Category introspection
    "list_categories",
    "merge_reports",
    "normalize_category",
    # Helper functions
    "normalize_framework",
    "print_report_summary",
    # Entry points
    "red_team",
    # Backend extension
    "register_backend",
    "resolve_category",
    "resolve_vulnerabilities",
]


_deprecated_warned: set[str] = set()  # noqa: RUF067


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
