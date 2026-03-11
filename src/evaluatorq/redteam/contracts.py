"""Frozen contract schemas for the evaluatorq.redteam package.

This module is self-contained (stdlib + pydantic only) and defines all
shared data models for both the research repo and evaluatorq consumers.

Semantic convention:
    ``passed=True``  → the agent is RESISTANT (attack failed)
    ``passed=False`` → the agent is VULNERABLE (attack succeeded)
"""

import sys
from datetime import datetime
from typing import Any, ClassVar, Literal

if sys.version_info >= (3, 11):
    from enum import StrEnum  # pyright: ignore[reportUnreachable]
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """String enum compatible with Python 3.10."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TurnType(StrEnum):
    """Conversation turn type."""

    SINGLE = 'single'
    MULTI = 'multi'


class AttackTechnique(StrEnum):
    """Specific attack techniques used in red teaming attacks."""

    INDIRECT_INJECTION = 'indirect-injection'
    DIRECT_INJECTION = 'direct-injection'
    DAN = 'DAN'
    CREDENTIAL_CACHING = 'credential-caching'
    PRIVILEGE_ESCALATION = 'privilege-escalation'
    CONFUSED_DEPUTY = 'confused-deputy'
    RCE_EXPLOIT = 'rce-exploit'
    MESSAGE_SPOOFING = 'message-spoofing'
    SOCIAL_ENGINEERING = 'social-engineering'
    TOOL_ABUSE = 'tool-abuse'
    SUPPLY_CHAIN = 'supply-chain'
    CONTEXT_POISONING = 'context-poisoning'
    CASCADE_TRIGGER = 'cascade-trigger'
    BOUNDARY_VIOLATION = 'boundary-violation'


class DeliveryMethod(StrEnum):
    """Jailbreak and delivery techniques."""

    # Persona/Role-play
    DAN = 'DAN'
    ROLE_PLAY = 'role-play'
    SKELETON_KEY = 'skeleton-key'
    # Obfuscation
    BASE64 = 'base64'
    LEETSPEAK = 'leetspeak'
    MULTILINGUAL = 'multilingual'
    CHARACTER_SPACING = 'character-spacing'
    # Multi-turn
    CRESCENDO = 'crescendo'
    MANY_SHOT = 'many-shot'
    # Social Engineering
    AUTHORITY_IMPERSONATION = 'authority-impersonation'
    REFUSAL_SUPPRESSION = 'refusal-suppression'
    # Direct
    DIRECT_REQUEST = 'direct-request'
    CODE_ELICITATION = 'code-elicitation'
    CODE_ASSISTANCE = 'code-assistance'
    # Tool/Agent-based
    TOOL_RESPONSE = 'tool-response'


class Severity(StrEnum):
    """Attack severity levels."""

    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class Scope(StrEnum):
    """Attack target scope."""

    MODEL = 'model'  # Targets LLM itself (jailbreaks, prompt injection)
    APPLICATION = 'application'  # Targets agent system (tools, memory, workflows)


class Framework(StrEnum):
    """Supported framework identifiers for datasets and reports."""

    OWASP_LLM = 'OWASP-LLM'
    OWASP_ASI = 'OWASP-ASI'
    OWASP_AGENTIC = 'OWASP-AGENTIC'
    CONTENT = 'CONTENT'
    LIABILITY = 'LIABILITY'
    FAIRNESS = 'FAIRNESS'
    GDPR = 'GDPR'
    AI_ACT = 'AI-ACT'


class Pipeline(StrEnum):
    """Supported unified report pipeline identifiers."""

    STATIC = 'static'
    DYNAMIC = 'dynamic'
    MIXED = 'mixed'


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def normalize_framework(framework: 'Framework | str') -> 'Framework':
    """Normalize framework aliases to a canonical value.

    OWASP-AGENTIC is an alias for OWASP-ASI.
    """
    value = str(framework)
    if value == Framework.OWASP_AGENTIC:
        return Framework.OWASP_ASI
    return Framework(value)


def normalize_category(category: str) -> str:
    """Strip 'OWASP-' prefix from category codes.

    Examples:
        >>> normalize_category("OWASP-ASI01")
        'ASI01'
        >>> normalize_category("ASI01")
        'ASI01'
        >>> normalize_category("OWASP-LLM01")
        'LLM01'
    """
    return category.removeprefix('OWASP-')


def infer_framework(category: str) -> str:
    """Infer OWASP framework from category code.

    Args:
        category: Category code (e.g., 'ASI01', 'LLM07', 'OWASP-ASI01')

    Returns:
        'OWASP-ASI' or 'OWASP-LLM'
    """
    normalized = normalize_category(category)
    if normalized.startswith('ASI'):
        return 'OWASP-ASI'
    if normalized.startswith('LLM'):
        return 'OWASP-LLM'
    return 'unknown'


# ---------------------------------------------------------------------------
# Core message models (OpenAI format)
# ---------------------------------------------------------------------------


class FunctionCall(BaseModel):
    """Function call details in OpenAI tool call format."""

    name: str = Field(description='Function name to call')
    arguments: str = Field(description='JSON string of function arguments')


class ToolCall(BaseModel):
    """Tool call in assistant message (OpenAI format)."""

    id: str = Field(description='Unique tool call ID (e.g., call_abc123)')
    type: Literal['function'] = Field(default='function', description='Tool type')
    function: FunctionCall = Field(description='Function call details')


class Message(BaseModel):
    """Single message in conversation history (OpenAI format with tool support).

    Supports both simple messages and tool calls:
    - Simple: {"role": "user", "content": "Hello"}
    - Tool call: {"role": "assistant", "tool_calls": [...]}
    - Tool response: {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
    """

    role: Literal['user', 'assistant', 'tool', 'system']
    content: str | None = Field(
        default=None,
        description='Message content (required for user/system, optional for assistant with tool_calls)',
    )

    # Tool call fields (OpenAI format)
    tool_calls: list[ToolCall] | None = Field(default=None, description='Tool calls made by assistant')
    tool_call_id: str | None = Field(default=None, description='ID linking tool response to call (for role=tool)')
    name: str | None = Field(default=None, description='Function name (for role=tool)')


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class RedTeamInput(BaseModel):
    """Input metadata for a red teaming attack sample."""

    id: str = Field(min_length=1, description='Unique sample identifier')
    category: str = Field(description='Attack category (framework-specific)')
    attack_technique: AttackTechnique = Field(description='Specific attack technique used')
    delivery_method: DeliveryMethod = Field(description='Jailbreak/delivery technique used')
    severity: Severity = Field(description='Attack severity level')
    scope: Scope = Field(description='Attack target scope')
    framework: Framework = Field(description='Security framework identifier')
    evaluator_id: str = Field(default='', description='Evaluator identifier (optional)')
    evaluator_name: str = Field(default='', description='Evaluator name (optional)')
    turn_type: TurnType = Field(description='Conversation turn type')
    source: str = Field(min_length=1, description='Dataset source')
    additional_metadata: dict[str, Any] | None = Field(
        default=None,
        description='Additional metadata that does not fit standard fields (e.g., AgentDojo-specific context)',
    )

    @field_validator('framework', mode='before')
    @classmethod
    def _normalize_framework(cls, value: Any) -> Any:
        if isinstance(value, str):
            return normalize_framework(value)
        return value


# ---------------------------------------------------------------------------
# Agent context models
# ---------------------------------------------------------------------------


class ToolInfo(BaseModel):
    """Information about an agent's tool."""

    name: str = Field(description='Tool name/identifier')
    description: str | None = Field(default=None, description='Tool description')
    parameters: dict[str, Any] | None = Field(default=None, description='Tool parameter schema')


class MemoryStoreInfo(BaseModel):
    """Information about an agent's memory store."""

    id: str = Field(description='Memory store ID')
    key: str | None = Field(default=None, description='Memory store key/slug')
    description: str | None = Field(default=None, description='Memory store description')


class KnowledgeBaseInfo(BaseModel):
    """Information about an agent's knowledge base."""

    id: str = Field(description='Knowledge base ID')
    key: str | None = Field(default=None, description='Knowledge base key/slug')
    name: str | None = Field(default=None, description='Knowledge base name')
    description: str | None = Field(default=None, description='Knowledge base description')


class AgentContext(BaseModel):
    """Extended agent information for context-aware attacks.

    Retrieved from ORQ API via orq_client.agents.retrieve().
    """

    key: str = Field(description='ORQ agent key')
    display_name: str | None = Field(default=None, description='Agent display name')
    description: str | None = Field(default=None, description='Agent description')
    system_prompt: str | None = Field(default=None, description='Agent system prompt (used by deployments)')
    instructions: str | None = Field(default=None, description='Agent behavioral instructions')
    tools: list[ToolInfo] = Field(default_factory=list, description='Available tools')
    memory_stores: list[MemoryStoreInfo] = Field(default_factory=list, description='Memory stores')
    knowledge_bases: list[KnowledgeBaseInfo] = Field(default_factory=list, description='Knowledge bases')
    model: str | None = Field(default=None, description='Primary model')

    @property
    def has_tools(self) -> bool:
        """Check if agent has any tools configured."""
        return len(self.tools) > 0

    @property
    def has_memory(self) -> bool:
        """Check if agent has any memory stores configured."""
        return len(self.memory_stores) > 0

    @property
    def has_knowledge(self) -> bool:
        """Check if agent has any knowledge bases configured."""
        return len(self.knowledge_bases) > 0

    def get_tool_names(self) -> list[str]:
        """Get list of tool names."""
        return [t.name for t in self.tools]


# ---------------------------------------------------------------------------
# Pipeline configuration model
# ---------------------------------------------------------------------------


class PipelineLLMConfig(BaseModel):
    """Centralized LLM call configuration for the dynamic red teaming pipeline.

    All magic numbers (max_tokens, temperature, retries, timeouts) live here.
    Instantiate with overrides or use the defaults via ``PipelineLLMConfig()``.

    Pipeline steps:

    1. **Capability classification** — Analyzes agent tools/resources and tags them
       with semantic capabilities (e.g. code_execution, web_request).
    2. **Strategy generation** — LLM generates novel attack strategies + objectives
       tailored to the agent's capabilities and OWASP category.
    3. **Tool adaptation** — Rewrites a generic attack prompt to reference specific
       tools the agent has, making the attack more targeted.
    4. **Adversarial prompt generation** — The red-team LLM crafts the actual attack
       messages sent to the target agent (single-turn or per-turn in multi-turn).
    """

    # Retry configuration (passed as extra_body to OpenAI calls)
    retry_count: int = 3
    retry_on_codes: list[int] = Field(default=[429, 500, 502, 503, 504])

    # Step 1: Capability classification — deterministic analysis of agent tools
    capability_classification_max_tokens: int = 4000
    capability_classification_temperature: float = 0.0

    # Step 2: Strategy generation — creative but structured strategy/objective creation
    strategy_generation_max_tokens: int = 4000
    strategy_generation_temperature: float = 0.7

    # Step 3: Tool adaptation — deterministic rewrite of prompts to target specific tools
    tool_adaptation_max_tokens: int = 4000
    tool_adaptation_temperature: float = 0.0

    # Step 4: Adversarial prompt generation — creative attack message crafting
    adversarial_max_tokens: int = 4000
    adversarial_temperature: float = 0.8

    # Target agent timeout (ms) for ORQ SDK calls
    target_agent_timeout_ms: int = 120_000

    # Logging level for the red teaming pipeline
    log_level: str = 'INFO'

    @property
    def retry_config(self) -> dict[str, Any]:
        """ORQ retry config dict for ``extra_body``."""
        return {'retry': {'count': self.retry_count, 'on_codes': self.retry_on_codes}}


# Module-level default instance used across the pipeline.
# Import this in other modules; tests can monkeypatch or pass overrides.
PIPELINE_CONFIG = PipelineLLMConfig()


# ---------------------------------------------------------------------------
# OWASP category names (self-contained, no external dependency)
# ---------------------------------------------------------------------------

_ASI_CATEGORY_NAMES: dict[str, str] = {
    'ASI01': 'Agent Goal Hijacking',
    'ASI02': 'Tool Misuse & Exploitation',
    'ASI03': 'Identity & Privilege Abuse',
    'ASI04': 'Supply Chain Vulnerabilities',
    'ASI05': 'Unexpected Code Execution',
    'ASI06': 'Memory & Context Poisoning',
    'ASI07': 'Insecure Inter-Agent Communication',
    'ASI08': 'Cascading Failures',
    'ASI09': 'Human-Agent Trust Exploitation',
    'ASI10': 'Rogue Agents',
}

_LLM_CATEGORY_NAMES: dict[str, str] = {
    'LLM01': 'Prompt Injection',
    'LLM02': 'Sensitive Information Disclosure',
    'LLM03': 'Supply Chain',
    'LLM04': 'Data and Model Poisoning',
    'LLM05': 'Improper Output Handling',
    'LLM06': 'Excessive Agency',
    'LLM07': 'System Prompt Leakage',
    'LLM08': 'Vector and Embedding Weaknesses',
    'LLM09': 'Misinformation',
    'LLM10': 'Unbounded Consumption',
}

OWASP_CATEGORY_NAMES: dict[str, str] = {
    **_ASI_CATEGORY_NAMES,
    **{f'OWASP-{k}': v for k, v in _ASI_CATEGORY_NAMES.items()},
    **_LLM_CATEGORY_NAMES,
    **{f'OWASP-{k}': v for k, v in _LLM_CATEGORY_NAMES.items()},
}


# ---------------------------------------------------------------------------
# Attack strategy model
# ---------------------------------------------------------------------------


class AttackStrategy(BaseModel):
    """Defines a specific attack strategy for an OWASP category.

    Strategies can be hardcoded or generated based on agent context.
    """

    category: str = Field(description='OWASP category code (e.g., ASI01, LLM01)')
    name: str = Field(description='Strategy identifier (e.g., indirect_injection_via_email)')
    description: str = Field(description='Human-readable description of the attack')
    attack_technique: AttackTechnique = Field(description='Primary attack technique')
    delivery_methods: list[DeliveryMethod] = Field(description='Applicable delivery methods')
    turn_type: TurnType = Field(description='Single or multi-turn attack')
    severity: Severity = Field(default=Severity.MEDIUM, description='Expected severity if successful')

    # Context requirements for filtering
    requires_tools: bool = Field(default=False, description='Requires agent to have tools')
    required_capabilities: list[str] = Field(
        default_factory=list,
        description='Capability tags required for this strategy (any match = eligible)',
    )

    # Templates
    objective_template: str = Field(
        description='Objective template for adversarial LLM (can use {tool_name}, {agent_description}, etc.)'
    )
    prompt_template: str | None = Field(default=None, description='Initial prompt template for single-turn attacks')

    # Metadata
    is_generated: bool = Field(default=False, description='Whether this strategy was LLM-generated')


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token usage and cost for an LLM call or aggregation of calls."""

    total_tokens: int = Field(default=0, description='Total tokens used')
    prompt_tokens: int = Field(default=0, description='Prompt/input tokens')
    completion_tokens: int = Field(default=0, description='Completion/output tokens')
    total_cost_usd: float = Field(default=0.0, description='Total cost in USD')
    calls: int = Field(default=0, description='Number of LLM API calls')


# ---------------------------------------------------------------------------
# Orchestrator and evaluation result models
# ---------------------------------------------------------------------------


class OrchestratorResult(BaseModel):
    """Result from multi-turn attack orchestration."""

    conversation: list[Message] = Field(description='Full conversation history (OpenAI message format)')
    turns: int = Field(description='Number of turns executed')
    objective_achieved: bool = Field(
        default=False, description='Whether adversarial LLM self-reported objective achieved'
    )
    final_response: str = Field(default='', description="Target agent's last response")
    duration_seconds: float = Field(default=0.0, description='Elapsed time in seconds')
    token_usage: TokenUsage | None = Field(default=None, description='Aggregated token usage across attack execution')
    token_usage_adversarial: TokenUsage | None = Field(
        default=None, description='Token usage for adversarial LLM calls only'
    )
    token_usage_target: TokenUsage | None = Field(default=None, description='Token usage for target agent calls only')
    system_prompt: str | None = Field(default=None, description='Rendered adversarial system prompt')
    error: str | None = Field(default=None, description='Error message if attack was aborted')
    error_type: str | None = Field(
        default=None, description='Error classification (e.g. content_filter, llm_error, target_error)'
    )
    error_stage: str | None = Field(default=None, description='Pipeline stage where error occurred')
    error_code: str | None = Field(default=None, description='Provider/stage-specific error code')
    error_details: dict[str, Any] | None = Field(default=None, description='Structured backend error details')
    error_turn: int | None = Field(default=None, description='1-based turn number where the error occurred')
    truncated_turns: list[int] = Field(
        default_factory=list, description='Turn numbers where adversarial LLM hit max_tokens'
    )


class EvaluationResult(BaseModel):
    """Result from OWASP evaluator.

    Semantic convention:
        passed=True  → RESISTANT (attack failed)
        passed=False → VULNERABLE (attack succeeded)
    """

    passed: bool | None = Field(default=None, description='True=RESISTANT (attack failed), False=VULNERABLE (attack succeeded), None=error/unevaluated')
    explanation: str = Field(description='Evaluator explanation')
    evaluator_id: str = Field(description='Evaluator identifier used')
    token_usage: TokenUsage | None = Field(default=None, description='Token usage and cost for this evaluation call')
    raw_output: dict[str, Any] | None = Field(default=None, description='Raw evaluator output')


# ---------------------------------------------------------------------------
# Result and report models
# ---------------------------------------------------------------------------


class AttackInfo(BaseModel):
    """Unified attack metadata.

    Superset of static RedTeamInput and dynamic AttackStrategy fields.
    """

    id: str = Field(description="Unique identifier (e.g., 'OWASP-ASI01-0153' or 'ASI01-role_confusion-001')")
    category: str = Field(description="Short-form category code (e.g., 'ASI01', 'LLM01')")
    framework: Framework = Field(description="OWASP framework ('OWASP-ASI' or 'OWASP-LLM')")
    attack_technique: AttackTechnique
    delivery_methods: list[DeliveryMethod] = Field(description='Delivery methods (always a list)')
    turn_type: TurnType
    severity: Severity
    scope: Scope | None = Field(default=None, description='Attack target scope (null for dynamic)')
    source: str = Field(
        description="Origin: 'AgentDojo', 'orq_generated', 'hardcoded_strategy', 'llm_generated_strategy'"
    )
    strategy_name: str | None = Field(default=None, description="Dynamic strategy name (e.g., 'tool_output_hijack')")
    objective: str | None = Field(default=None, description='Filled objective template (dynamic only)')
    evaluator_id: str | None = None
    evaluator_name: str | None = None
    additional_metadata: dict[str, Any] | None = None

    @field_validator('framework', mode='before')
    @classmethod
    def _normalize_framework(cls, value: Any) -> Any:
        if isinstance(value, str):
            return normalize_framework(value)
        return value


class UnifiedEvaluationResult(BaseModel):
    """Typed evaluation result (replaces dict[str, Any]).

    Semantic convention:
        passed=True  → RESISTANT (attack failed)
        passed=False → VULNERABLE (attack succeeded)
        passed=None  → error/unevaluated
    """

    value: Any = None
    passed: bool | None = Field(
        default=None,
        description='True=RESISTANT, False=VULNERABLE, None=error/unevaluated',
    )
    explanation: str = ''
    evaluator_id: str | None = None
    evaluator_name: str | None = None
    token_usage: TokenUsage | None = None
    raw_output: dict[str, Any] | None = None


class EvaluationPayload(BaseModel):
    """Shared evaluation payload schema for staged/static/hybrid outputs."""

    evaluator_id: str | None = None
    evaluator_name: str | None = None
    value: Any = None
    explanation: str = ''
    passed: bool | None = None
    error: str | None = None
    token_usage: TokenUsage | None = None


class JobOutputPayload(BaseModel):
    """Normalized evaluatorq job output payload."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')

    conversation: list[Message] = Field(default_factory=list)
    final_response: str | None = None
    response: str | None = None
    output: str | None = None
    turns: int | None = None
    objective_achieved: bool | None = None
    duration_seconds: float | None = None
    token_usage: TokenUsage | None = None
    token_usage_adversarial: TokenUsage | None = None
    token_usage_target: TokenUsage | None = None
    system_prompt: str | None = None
    error: str | None = None
    error_type: str | None = None
    error_stage: str | None = None
    error_code: str | None = None
    error_details: dict[str, Any] | None = None
    error_turn: int | None = None
    truncated_turns: list[int] = Field(default_factory=list)
    finish_reason: str | None = None

    @property
    def response_text(self) -> str:
        """Best-effort text response extraction."""
        if self.final_response and self.final_response.strip():
            return self.final_response
        if self.response is not None:
            return self.response
        if self.output is not None:
            return self.output
        return ''


class ExecutionDetails(BaseModel):
    """Execution metadata (dynamic pipeline only)."""

    turns: int = 1
    duration_seconds: float | None = None
    objective_achieved: bool | None = Field(
        default=None,
        description='Adversarial LLM self-report (multi-turn)',
    )
    token_usage: TokenUsage | None = None


class AgentInfo(BaseModel):
    """Target agent metadata."""

    key: str | None = None
    model: str | None = None
    display_name: str | None = None


class RedTeamResult(BaseModel):
    """Single unified result item from either pipeline."""

    attack: AttackInfo
    agent: AgentInfo
    messages: list[Message] = Field(description='OpenAI-format conversation')
    response: str | None = None
    evaluation: UnifiedEvaluationResult | None = None
    vulnerable: bool
    execution: ExecutionDetails | None = Field(default=None, description='Null for static pipeline')
    error: str | None = None
    error_type: str | None = None
    error_stage: str | None = None
    error_code: str | None = None
    error_details: dict[str, Any] | None = None


class CategorySummary(BaseModel):
    """Per-category summary statistics."""

    category: str
    category_name: str
    total_attacks: int
    total_conversations: int = 0
    total_turns: int = 0
    vulnerabilities_found: int
    resistance_rate: float
    strategies_used: list[str] = Field(default_factory=list)


class ReportSummary(BaseModel):
    """Aggregate summary statistics for a report."""

    total_attacks: int = 0
    evaluated_attacks: int = 0
    unevaluated_attacks: int = 0
    evaluation_coverage: float = 0.0
    total_conversations: int = 0
    total_turns: int = 0
    vulnerabilities_found: int = 0
    resistance_rate: float = 1.0
    total_errors: int = 0
    errors_by_type: dict[str, int] = Field(default_factory=dict, description='Error counts grouped by type')
    by_category: dict[str, CategorySummary] = Field(default_factory=dict)
    by_technique: dict[str, int] = Field(default_factory=dict)


class RedTeamReport(BaseModel):
    """Top-level unified report wrapping all results."""

    version: str = '2.0.0'
    created_at: datetime
    description: str | None = None
    pipeline: Pipeline = Field(description="'static', 'dynamic', or 'mixed'")
    framework: Framework | None = None

    categories_tested: list[str]
    tested_agents: list[str] = Field(default_factory=list, description='Names/keys of tested agents in this report')
    total_results: int

    agent_context: AgentContext | None = None

    results: list[RedTeamResult]

    summary: ReportSummary

    token_usage_summary: TokenUsage | None = None
    duration_seconds: float | None = None

    @field_validator('framework', mode='before')
    @classmethod
    def _normalize_framework(cls, value: Any) -> Any:
        if isinstance(value, str):
            return normalize_framework(value)
        return value


# ---------------------------------------------------------------------------
# Dynamic pipeline row models
# ---------------------------------------------------------------------------


class EvaluatedRowBase(BaseModel):
    """Common row shape for evaluated JSONL outputs."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')

    input: RedTeamInput
    messages: list[Message] = Field(default_factory=list)
    response: str | None = None
    error: str | None = None
    error_type: str | None = None
    error_stage: str | None = None
    error_code: str | None = None
    error_details: dict[str, Any] | None = None


class EvaluatedRow(EvaluatedRowBase):
    """Serialized JSONL row for evaluated outputs."""

    evaluation_result: EvaluationPayload = Field(default_factory=EvaluationPayload)


class AttackEvaluationRow(BaseModel):
    """Staged attack-level evaluation block."""

    value: Any = None
    passed: bool | None = None
    explanation: str = ''
    skipped: bool = False


class DynamicAttackResultRow(BaseModel):
    """Serialized staged dynamic attack result row."""

    model_config: ClassVar[ConfigDict] = ConfigDict(validate_assignment=True)

    id: str | None = None
    category: str = 'unknown'
    strategy: AttackStrategy | None = None
    objective: str | None = None
    error: str | None = None
    conversation: list[Message] = Field(default_factory=list)
    final_response: str = ''
    turns: int = 1
    objective_achieved: bool | None = None
    duration_seconds: float | None = None
    token_usage: TokenUsage | None = None
    token_usage_adversarial: TokenUsage | None = None
    token_usage_target: TokenUsage | None = None
    system_prompt: str | None = None
    job_error: str | None = None
    error_type: str | None = None
    error_stage: str | None = None
    error_code: str | None = None
    error_details: dict[str, Any] | None = None
    evaluation: AttackEvaluationRow | None = None
    vulnerable: bool | None = None


class DynamicCategorySummaryRow(BaseModel):
    """Per-category staged summary row."""

    total: int = 0
    vulnerable: int = 0
    resistant: int = 0
    errors: int = 0
    strategies: list[str] = Field(default_factory=list)
    resistance_rate: float = 0.0


class ErrorTypeDetails(BaseModel):
    """Typed breakdown for one error type."""

    count: int = 0
    categories: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    sample_error: str = ''


class DynamicErrorAnalysisRow(BaseModel):
    """Error analysis block in staged summary."""

    error_rate: float = 0.0
    by_error_type: dict[str, ErrorTypeDetails] = Field(default_factory=dict)


class DynamicRunMetadata(BaseModel):
    """Run metadata block persisted in staged summaries."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')

    mode: str | None = None
    target: str | None = None
    agent_key: str | None = None
    attack_model: str | None = None
    evaluator_model: str | None = None
    categories_tested: list[str] | None = None
    max_datapoints: int | None = None
    max_turns: int | None = None
    max_per_category: int | None = None
    generated_strategy_count: int | None = None
    parallelism: int | None = None
    timestamp: str | None = None
    duration_seconds: float | None = None


class StrategySelectionRow(BaseModel):
    """Per-category strategy selection details persisted in staged outputs."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')

    count: int = 0
    strategies: list[AttackStrategy] = Field(default_factory=list)


class GeneratedPromptRow(BaseModel):
    """Generated prompt metadata row persisted in staged outputs."""

    id: str
    category: str
    strategy_name: str
    turn_type: str
    is_generated: bool = False
    objective: str
    prompt_template: str | None = None


class DynamicSummaryReportRow(BaseModel):
    """Serialized staged summary report."""

    run_metadata: DynamicRunMetadata
    total_attacks: int = 0
    evaluated_attacks: int = 0
    vulnerabilities_found: int = 0
    resistance_rate: float = 0.0
    errors: int = 0
    error_analysis: DynamicErrorAnalysisRow = Field(default_factory=DynamicErrorAnalysisRow)
    token_usage_by_source: dict[str, TokenUsage] = Field(default_factory=dict)
    by_category: dict[str, DynamicCategorySummaryRow] = Field(default_factory=dict)


class DatasetInferenceRow(BaseModel):
    """Inference output row persisted by dataset mode."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')

    response: str | None = None
    agent_type: str
    agent_model: str
    error: str | None = None
    error_type: str | None = None


class ReportSnapshot(BaseModel):
    """Concise report summary used by CLI report commands."""

    pipeline: Pipeline
    framework: Framework | None = None
    total_results: int
    categories_tested: list[str]
    resistance_rate: float
    vulnerabilities_found: int
    top_techniques: dict[str, int] = Field(default_factory=dict)
