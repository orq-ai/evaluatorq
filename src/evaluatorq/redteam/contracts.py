"""Frozen contract schemas for the evaluatorq.redteam package.

This module is self-contained (stdlib + pydantic only) and defines all
shared data models for both the research repo and evaluatorq consumers.

Semantic convention:
    ``passed=True``  → the agent is RESISTANT (attack failed)
    ``passed=False`` → the agent is VULNERABLE (attack succeeded)
"""

import os
import sys
from datetime import datetime
from typing import Any, Literal, TypedDict

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """String enum compatible with Python 3.10."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TurnType(StrEnum):
    """Conversation turn type."""

    SINGLE = 'single'
    MULTI = 'multi'


class AttackTechnique(StrEnum):
    """Known attack techniques used in red teaming attacks.

    This enum tracks *known* techniques for internal validation and reporting.
    External datasets may contain additional values — use
    :func:`is_known_attack_technique` to check membership without failing.
    """

    # Injection
    INDIRECT_INJECTION = 'indirect-injection'
    DIRECT_INJECTION = 'direct-injection'
    DAN = 'DAN'
    AUTODAN = 'autodan'
    DEVELOPER_MODE = 'developer-mode'
    # Privilege / access
    CREDENTIAL_CACHING = 'credential-caching'
    PRIVILEGE_ESCALATION = 'privilege-escalation'
    CONFUSED_DEPUTY = 'confused-deputy'
    EXCESSIVE_AGENCY = 'excessive-agency'
    PLUGIN_ABUSE = 'plugin-abuse'
    # Infrastructure
    RCE_EXPLOIT = 'rce-exploit'
    SUPPLY_CHAIN = 'supply-chain'
    RESOURCE_EXHAUSTION = 'resource-exhaustion'
    # Social / manipulation
    MESSAGE_SPOOFING = 'message-spoofing'
    SOCIAL_ENGINEERING = 'social-engineering'
    EMOTIONAL_MANIPULATION = 'emotional-manipulation'
    # Evasion / obfuscation
    PAYLOAD_SPLITTING = 'payload-splitting'
    ALTERNATIVE_ENCODING = 'alternative-encoding'
    UNICODE_OBFUSCATION = 'unicode-obfuscation'
    # Data extraction
    MODEL_EXTRACTION = 'model-extraction'
    PII_EXTRACTION = 'pii-extraction'
    PII_INPUT = 'pii-input'
    SYSTEM_PROMPT_EXTRACTION = 'system-prompt-extraction'
    # Agent / tool
    TOOL_ABUSE = 'tool-abuse'
    CONTEXT_POISONING = 'context-poisoning'
    CASCADE_TRIGGER = 'cascade-trigger'
    BOUNDARY_VIOLATION = 'boundary-violation'


def is_known_attack_technique(value: str) -> bool:
    """Check whether *value* is a known :class:`AttackTechnique` member."""
    return value in AttackTechnique.__members__.values()


class DeliveryMethod(StrEnum):
    """Known jailbreak and delivery techniques.

    See :class:`AttackTechnique` — the same open-set policy applies.
    """

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
    WORD_SUBSTITUTION = 'word-substitution'


def is_known_delivery_method(value: str) -> bool:
    """Check whether *value* is a known :class:`DeliveryMethod` member."""
    return value in DeliveryMethod.__members__.values()


class Severity(StrEnum):
    """Attack severity levels."""

    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


SEVERITY_DEFINITIONS: dict[str, str] = {
    'critical': (
        'Attacks that, if successful, could lead to complete system compromise, '
        'unauthorized access to sensitive data, or execution of arbitrary actions '
        'with full privileges. Immediate remediation required.'
    ),
    'high': (
        'Attacks that could cause significant harm such as partial data exfiltration, '
        'privilege escalation, or bypassing critical safety guardrails. '
        'Should be prioritized for remediation.'
    ),
    'medium': (
        'Attacks that could lead to moderate impact such as information leakage of '
        'non-sensitive data, minor policy violations, or degraded safety behavior. '
        'Should be addressed in normal development cycles.'
    ),
    'low': (
        'Attacks with limited impact such as minor guideline deviations, '
        'edge-case behaviors, or cosmetic safety issues. '
        'Address as time permits or during routine hardening.'
    ),
}


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


class VulnerabilityDomain(StrEnum):
    """Where in the stack the vulnerability fix belongs.

    - AGENT: orchestration layer (tools, memory, workflows, permissions)
    - MODEL: LLM inference layer (prompts, guardrails, output filters)
    - DATA: data/retrieval layer (training data, embeddings, RAG)
    """

    AGENT = 'agent'
    MODEL = 'model'
    DATA = 'data'


class Vulnerability(StrEnum):
    """Atomic vulnerability identifiers.

    Each value is a stable, framework-agnostic ID. Framework-specific codes
    (OWASP ASI01, LLM01, etc.) are mapped via the vulnerability registry.
    """

    # Agentic security (maps to OWASP ASI)
    GOAL_HIJACKING = 'goal_hijacking'
    TOOL_MISUSE = 'tool_misuse'
    IDENTITY_PRIVILEGE_ABUSE = 'identity_privilege_abuse'
    SUPPLY_CHAIN = 'supply_chain'
    CODE_EXECUTION = 'code_execution'
    MEMORY_POISONING = 'memory_poisoning'
    INTER_AGENT_COMMS = 'inter_agent_comms'
    CASCADING_FAILURES = 'cascading_failures'
    TRUST_EXPLOITATION = 'trust_exploitation'
    ROGUE_AGENTS = 'rogue_agents'

    # Agent security (maps to OWASP LLM Top 10 LLM06)
    EXCESSIVE_AGENCY = 'excessive_agency'

    # LLM security (maps to OWASP LLM Top 10)
    PROMPT_INJECTION = 'prompt_injection'
    SENSITIVE_INFO_DISCLOSURE = 'sensitive_info_disclosure'
    DATA_POISONING = 'data_poisoning'
    IMPROPER_OUTPUT = 'improper_output'
    SYSTEM_PROMPT_LEAKAGE = 'system_prompt_leakage'
    VECTOR_EMBEDDING_WEAKNESS = 'vector_embedding_weakness'
    MISINFORMATION = 'misinformation'


class TargetKind(StrEnum):
    """Kind of target being red-teamed."""

    AGENT = 'agent'
    LLM = 'llm'
    OPENAI = 'openai'
    DEPLOYMENT = 'deployment'
    DIRECT = 'direct'

    @property
    def is_model(self) -> bool:
        """Return True if this target kind represents a model (not an agent)."""
        return self in (TargetKind.LLM, TargetKind.OPENAI)


class Pipeline(StrEnum):
    """Supported unified report pipeline identifiers."""

    STATIC = 'static'
    DYNAMIC = 'dynamic'
    HYBRID = 'hybrid'


class PipelineStage(StrEnum):
    """Pipeline stage identifiers used in hook callbacks."""

    CONTEXT_RETRIEVAL = 'context_retrieval'
    DATAPOINT_GENERATION = 'datapoint_generation'
    ATTACK_EXECUTION = 'attack_execution'
    REPORT_GENERATION = 'report_generation'
    CLEANUP = 'cleanup'
    TARGET_START = 'target_start'
    TARGET_COMPLETE = 'target_complete'


class AgentCapability(StrEnum):
    """Capability tags for agent resources.

    Moved from ``capability_classifier.py`` so that ``AttackStrategy`` and other
    contract models can reference it without introducing a circular import.
    """

    CODE_EXECUTION = 'code_execution'
    SHELL_ACCESS = 'shell_access'
    FILE_SYSTEM = 'file_system'
    DATABASE = 'database'
    WEB_REQUEST = 'web_request'
    MEMORY_READ = 'memory_read'
    MEMORY_WRITE = 'memory_write'
    KNOWLEDGE_RETRIEVAL = 'knowledge_retrieval'
    EMAIL = 'email'
    MESSAGING = 'messaging'
    PAYMENT = 'payment'
    USER_DATA = 'user_data'


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def normalize_framework(framework: 'Framework | str') -> 'Framework':
    """Normalize framework aliases to a canonical value.

    OWASP-AGENTIC is an alias for OWASP-ASI.
    """
    if isinstance(framework, Framework):
        value = framework.value
    else:
        value = framework
    if value == Framework.OWASP_AGENTIC.value:
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
    vulnerability: str = Field(default='', description='Vulnerability identifier (primary key in new format)')
    category: str = Field(default='', description='Attack category (framework-specific) — kept for backwards compat')
    attack_technique: str = Field(default='', description='Specific attack technique used')
    delivery_method: str = Field(default='', description='Jailbreak/delivery technique used')
    severity: Severity = Field(description='Attack severity level')
    vulnerability_domain: VulnerabilityDomain = Field(description='Vulnerability domain (where the fix belongs)')
    framework: Framework = Field(default=Framework.OWASP_AGENTIC, description='Security framework identifier')
    evaluator_id: str = Field(default='', description='Evaluator identifier (optional)')
    evaluator_name: str = Field(default='', description='Evaluator name (optional)')
    turn_type: TurnType = Field(description='Conversation turn type')
    source: str = Field(min_length=1, description='Dataset source')
    additional_metadata: dict[str, Any] | None = Field(
        default=None,
        description='Additional metadata that does not fit standard fields (e.g., AgentDojo-specific context)',
    )

    @model_validator(mode='before')
    @classmethod
    def _migrate_scope_to_vulnerability_domain(cls, data: Any) -> Any:
        """Migrate legacy ``scope`` field to ``vulnerability_domain``.

        Accepts the old ``scope`` key in input data and maps it:
        ``application`` → ``agent``, ``model`` → ``model``.
        """
        if isinstance(data, dict) and 'scope' in data and 'vulnerability_domain' not in data:
            scope_val = data.pop('scope')
            mapping = {'application': 'agent', 'model': 'model'}
            data['vulnerability_domain'] = mapping.get(scope_val, scope_val)
        return data

    @model_validator(mode='before')
    @classmethod
    def _cross_resolve_vulnerability_and_category(cls, data: Any) -> Any:
        """Cross-resolve vulnerability ↔ category so either field can be omitted."""
        if not isinstance(data, dict):
            return data
        # Lazy import: vulnerability_registry imports enums from this module at
        # the top level, so importing it here avoids a circular import at class
        # definition time.  By the time this validator runs (instance creation)
        # both modules are fully initialised.
        from evaluatorq.redteam.vulnerability_registry import get_primary_category, resolve_category_safe
        has_vuln = bool(data.get('vulnerability'))
        has_cat = bool(data.get('category'))
        if has_vuln and not has_cat:
            try:
                vuln_enum = Vulnerability(data['vulnerability'])
                data['category'] = get_primary_category(vuln_enum)
            except ValueError:
                data['category'] = data['vulnerability']
        elif has_cat and not has_vuln:
            vuln = resolve_category_safe(data['category'])
            if vuln is not None:
                data['vulnerability'] = vuln.value
        return data

    @field_validator('framework', mode='before')
    @classmethod
    def _normalize_framework(cls, value: Any) -> Any:
        """Normalize framework string aliases to canonical Framework enum values."""
        if isinstance(value, str):
            return normalize_framework(value)
        return value

    @model_validator(mode='after')
    def _require_vulnerability_or_category(self) -> 'RedTeamInput':
        """Ensure at least one of vulnerability or category is populated."""
        if not self.vulnerability and not self.category:
            raise ValueError('RedTeamInput requires at least one of vulnerability or category')
        return self


# ---------------------------------------------------------------------------
# Dataset validation models
# ---------------------------------------------------------------------------


class RedTeamSample(BaseModel):
    """Complete red teaming sample in ORQ dataset format."""

    input: RedTeamInput = Field(description='Attack metadata for this sample')
    messages: list[Message] = Field(min_length=1, description='Attack conversation history')


class StaticDataset(BaseModel):
    """Top-level schema for static red teaming dataset files."""

    model_config = ConfigDict(extra='ignore')

    samples: list[RedTeamSample] = Field(min_length=1, description='Attack samples')


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


class TargetConfig(BaseModel):
    """Backend-agnostic target configuration.

    Passed opaquely through the runner to backends/factories.
    """

    system_prompt: str | None = Field(
        default=None,
        description="System prompt for the target model/agent",
    )


class AgentContext(BaseModel):
    """Extended agent information for context-aware attacks.

    Describes the target agent's configuration and capabilities.
    """

    key: str = Field(description='Agent identifier key')
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
    capability_classification_max_tokens: int = 5000
    capability_classification_temperature: float = 1.0

    # Step 2: Strategy generation — creative but structured strategy/objective creation
    strategy_generation_max_tokens: int = 5000
    strategy_generation_temperature: float = 1.0

    # Step 3: Tool adaptation — deterministic rewrite of prompts to target specific tools
    tool_adaptation_max_tokens: int = 5000
    tool_adaptation_temperature: float = 1.0

    # Step 4: Adversarial prompt generation — creative attack message crafting
    adversarial_max_tokens: int = 5000
    adversarial_temperature: float = 1.0

    # Target model max_tokens for static/hybrid model job responses
    target_max_tokens: int = 5000

    # Target agent timeout (ms) for ORQ SDK calls
    target_agent_timeout_ms: int = 240_000

    # Adversarial LLM call timeout (ms) — separate from target agent timeout
    llm_call_timeout_ms: int = 60_000

    # Overall cleanup timeout (ms) — best-effort, should never block the pipeline
    cleanup_timeout_ms: int = 60_000

    # Logging level for the red teaming pipeline
    log_level: str = 'INFO'

    @property
    def retry_config(self) -> dict[str, Any]:
        """ORQ retry config dict for ``extra_body``.

        Returns an empty dict when using the OpenAI API directly (the
        ``retry`` parameter is ORQ-specific and rejected by OpenAI).
        """
        if os.getenv('OPENAI_API_KEY') or not os.getenv('ORQ_API_KEY'):
            return {}
        return {'retry': {'count': self.retry_count, 'on_codes': self.retry_on_codes}}


# Module-level default instance used across the pipeline.
# Import this in other modules; tests can monkeypatch or pass overrides.
PIPELINE_CONFIG = PipelineLLMConfig()

# Default model used across the red teaming pipeline for adversarial generation
# and evaluation. Uses the OpenAI model name directly (works with both the
# openai and orq backends — the orq backend accepts this format as-is).
DEFAULT_PIPELINE_MODEL: str = 'gpt-5-mini'


class RedTeamConfig(BaseModel):
    """Top-level configuration for the red teaming pipeline.

    Centralizes backend routing, model resolution, and LLM call settings
    into a single object.  When ``backend="auto"`` (the default), the
    backend is inferred from the target prefix and available credentials:

    * ``agent:`` / ``deployment:`` targets → ``orq``
    * ``llm:`` targets → ``openai`` (or ``orq`` if only ``ORQ_API_KEY`` is set)

    Models are automatically prefixed (e.g. ``"openai/gpt-5-mini"``) when
    routing through the orq router so callers never have to think about it.
    """

    # --- Backend / routing ---------------------------------------------------
    backend: Literal['auto', 'openai', 'orq'] = 'auto'

    # --- Models --------------------------------------------------------------
    attack_model: str = DEFAULT_PIPELINE_MODEL
    evaluator_model: str = DEFAULT_PIPELINE_MODEL

    # --- LLM call tuning (sub-config) ----------------------------------------
    llm: PipelineLLMConfig = Field(default_factory=PipelineLLMConfig)

    # --- Extra kwargs forwarded to every chat.completions.create() call ------
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)

    def resolve_backend(self, targets: list[str]) -> str:
        """Return the concrete backend name (``"orq"`` or ``"openai"``)."""
        if self.backend != 'auto':
            return self.backend
        orq_prefixes = ('agent:', 'deployment:')
        has_orq_target = any(
            any(t.startswith(p) for p in orq_prefixes) or ':' not in t
            for t in targets
        )
        return 'orq' if has_orq_target else 'openai'

    def resolve_model(self, model: str, *, uses_orq_router: bool) -> str:
        """Add a provider prefix when the model goes through the orq router.

        If the model already contains a ``/`` (e.g. ``"openai/gpt-5-mini"``)
        it is returned unchanged.
        """
        if uses_orq_router and '/' not in model:
            return f'openai/{model}'
        return model

    @property
    def uses_orq_router(self) -> bool:
        """True when the LLM client will route through the orq router.

        This is the case when no custom ``llm_client`` / ``OPENAI_API_KEY``
        is set and ``ORQ_API_KEY`` is available.
        """
        return not os.getenv('OPENAI_API_KEY') and bool(os.getenv('ORQ_API_KEY'))


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
}

OWASP_CATEGORY_NAMES: dict[str, str] = {
    **_ASI_CATEGORY_NAMES,
    **{f'OWASP-{k}': v for k, v in _ASI_CATEGORY_NAMES.items()},
    **_LLM_CATEGORY_NAMES,
    **{f'OWASP-{k}': v for k, v in _LLM_CATEGORY_NAMES.items()},
}


# ---------------------------------------------------------------------------
# Vulnerability definition model
# ---------------------------------------------------------------------------


class VulnerabilityDef(BaseModel):
    """Definition of a vulnerability with metadata and framework mappings."""

    id: Vulnerability
    name: str = Field(description='Human-readable name, e.g. "Agent Goal Hijacking"')
    domain: VulnerabilityDomain
    description: str = ''
    default_attack_technique: AttackTechnique
    framework_mappings: dict[str, list[str]] = Field(
        default_factory=dict,
        description='Framework -> category codes, e.g. {"OWASP-ASI": ["ASI01"]}',
    )


# ---------------------------------------------------------------------------
# Attack strategy model
# ---------------------------------------------------------------------------


class AttackStrategy(BaseModel):
    """Defines a specific attack strategy for a vulnerability.

    Strategies can be hardcoded or generated based on agent context.
    """

    model_config = ConfigDict(frozen=True)

    vulnerability: Vulnerability | None = Field(default=None, description='Primary vulnerability identifier')
    category: str = Field(description='OWASP category code (e.g., ASI01, LLM01) — kept for backwards compat')
    name: str = Field(description='Strategy identifier (e.g., indirect_injection_via_email)')
    description: str = Field(description='Human-readable description of the attack')
    attack_technique: AttackTechnique = Field(description='Primary attack technique')
    delivery_methods: list[DeliveryMethod] = Field(description='Applicable delivery methods')
    turn_type: TurnType = Field(description='Single or multi-turn attack')
    severity: Severity = Field(default=Severity.MEDIUM, description='Expected severity if successful')

    # Context requirements for filtering
    requires_tools: bool = Field(default=False, description='Requires agent to have tools')
    required_capabilities: list[AgentCapability] = Field(
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
    calls: int = Field(default=0, description='Number of LLM API calls')

    @classmethod
    def from_completion(cls, response: Any) -> 'TokenUsage | None':
        """Extract token usage from an OpenAI-compatible completion response."""
        usage = getattr(response, 'usage', None)
        if usage is None:
            return None
        prompt = int(getattr(usage, 'prompt_tokens', 0) or 0)
        completion = int(getattr(usage, 'completion_tokens', 0) or 0)
        total = int(getattr(usage, 'total_tokens', prompt + completion) or 0)
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total, calls=1)


# ---------------------------------------------------------------------------
# Error info model
# ---------------------------------------------------------------------------


class ErrorInfo(BaseModel):
    """Structured error information for attack/evaluation results."""

    message: str
    error_type: str
    stage: str | None = None
    code: str | None = None
    details: dict[str, Any] | None = None
    turn: int | None = None


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

    @property
    def error_info(self) -> 'ErrorInfo | None':
        """Structured view of the flat error fields."""
        if self.error is None:
            return None
        return ErrorInfo(
            message=self.error,
            error_type=self.error_type or 'unknown',
            stage=self.error_stage,
            code=self.error_code,
            details=self.error_details,
            turn=self.error_turn,
        )


class AttackOutput(OrchestratorResult):
    """Typed output from a dynamic red team attack job.

    Extends OrchestratorResult with category and vulnerability fields
    that are merged in during job execution.
    """

    category: str = ''
    vulnerability: str = ''


class AttackEvaluationResult(BaseModel):
    """Result from OWASP evaluator for a single attack.

    Semantic convention:
        passed=True  → RESISTANT (attack failed)
        passed=False → VULNERABLE (attack succeeded)

    .. note:: Named ``AttackEvaluationResult`` to avoid collision with the root
       ``evaluatorq.EvaluationResult`` which is used for generic evaluator scores.
    """

    passed: bool | None = Field(description='True=RESISTANT (attack failed), False=VULNERABLE (attack succeeded), None=error/unevaluated')
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
    vulnerability: str = Field(default='', description='Vulnerability identifier (primary key in new format)')
    category: str = Field(description="Short-form category code (e.g., 'ASI01', 'LLM01') — kept for backwards compat")
    framework: Framework = Field(description="OWASP framework ('OWASP-ASI' or 'OWASP-LLM')")
    attack_technique: AttackTechnique
    delivery_methods: list[DeliveryMethod] = Field(description='Delivery methods (always a list)')
    turn_type: TurnType
    severity: Severity
    vulnerability_domain: VulnerabilityDomain | None = Field(
        default=None, description='Vulnerability domain (null for dynamic attacks without a fixed domain)'
    )
    source: str = Field(
        description="Origin: 'AgentDojo', 'orq_generated', 'hardcoded_strategy', 'llm_generated_strategy'"
    )
    strategy_name: str | None = Field(default=None, description="Dynamic strategy name (e.g., 'tool_output_hijack')")
    objective: str | None = Field(default=None, description='Filled objective template (dynamic only)')
    evaluator_id: str | None = None
    evaluator_name: str | None = None
    additional_metadata: dict[str, Any] | None = None

    @model_validator(mode='before')
    @classmethod
    def _migrate_scope_to_vulnerability_domain(cls, data: Any) -> Any:
        """Migrate legacy ``scope`` field to ``vulnerability_domain``.

        Accepts the old ``scope`` key in input data and maps it:
        ``application`` → ``agent``, ``model`` → ``model``.
        """
        if isinstance(data, dict) and 'scope' in data and 'vulnerability_domain' not in data:
            scope_val = data.pop('scope')
            mapping = {'application': 'agent', 'model': 'model'}
            data['vulnerability_domain'] = mapping.get(scope_val, scope_val)
        return data

    @field_validator('framework', mode='before')
    @classmethod
    def _normalize_framework(cls, value: Any) -> Any:
        """Normalize framework string aliases to canonical Framework enum values."""
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

    model_config = ConfigDict(extra='allow')

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
    max_turns: int | None = None
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

    @property
    def error_info(self) -> 'ErrorInfo | None':
        """Structured view of the flat error fields."""
        if self.error is None:
            return None
        return ErrorInfo(
            message=self.error,
            error_type=self.error_type or 'unknown',
            stage=self.error_stage,
            code=self.error_code,
            details=self.error_details,
        )


class VulnerabilitySummary(BaseModel):
    """Per-vulnerability summary statistics."""

    vulnerability: str
    vulnerability_name: str
    domain: str
    total_attacks: int
    vulnerabilities_found: int
    resistance_rate: float = Field(ge=0.0, le=1.0)
    strategies_used: list[str] = Field(default_factory=list)
    framework_categories: dict[str, list[str]] = Field(
        default_factory=dict,
        description='Framework -> category codes this vulnerability maps to',
    )


class CategorySummary(BaseModel):
    """Per-category summary statistics."""

    category: str
    category_name: str
    total_attacks: int
    evaluated_attacks: int = 0
    unevaluated_attacks: int = 0
    evaluation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    total_conversations: int = 0
    total_turns: int = 0
    vulnerabilities_found: int
    vulnerability_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    resistance_rate: float = Field(ge=0.0, le=1.0)
    total_errors: int = 0
    strategies_used: list[str] = Field(default_factory=list)


class DimensionSummary(BaseModel):
    """Base class for per-dimension summary statistics."""

    total_attacks: int = 0
    vulnerabilities_found: int = 0
    resistance_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    vulnerability_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class TechniqueSummary(DimensionSummary):
    """Per-technique summary statistics."""


class SeveritySummary(DimensionSummary):
    """Per-severity summary statistics."""


class DeliveryMethodSummary(DimensionSummary):
    """Per-delivery-method summary statistics."""


class TurnTypeSummary(DimensionSummary):
    """Per-turn-type (single vs multi) summary statistics."""

    average_turns: float = 0.0


class DomainSummary(DimensionSummary):
    """Per-domain (agent / model / data) summary statistics."""


class FrameworkSummary(DimensionSummary):
    """Per-framework summary statistics (useful for mixed reports)."""


class ReportSummary(BaseModel):
    """Aggregate summary statistics for a report."""

    total_attacks: int = 0
    evaluated_attacks: int = 0
    unevaluated_attacks: int = 0
    evaluation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    total_conversations: int = 0
    total_turns: int = 0
    average_turns_per_attack: float = 0.0
    vulnerabilities_found: int = 0
    vulnerability_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    resistance_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    total_errors: int = 0
    errors_by_type: dict[str, int] = Field(default_factory=dict, description='Error counts grouped by type')
    token_usage_total: TokenUsage | None = Field(default=None, description='Aggregated token usage across all results')
    by_vulnerability: dict[str, VulnerabilitySummary] = Field(default_factory=dict)
    by_category: dict[str, CategorySummary] = Field(default_factory=dict)
    by_technique: dict[str, TechniqueSummary] = Field(default_factory=dict)
    by_severity: dict[str, SeveritySummary] = Field(default_factory=dict)
    by_delivery_method: dict[str, DeliveryMethodSummary] = Field(default_factory=dict)
    by_turn_type: dict[str, TurnTypeSummary] = Field(default_factory=dict)
    by_domain: dict[str, DomainSummary] = Field(default_factory=dict)
    by_framework: dict[str, FrameworkSummary] = Field(default_factory=dict)
    datapoint_breakdown: dict[str, int] | None = Field(
        default=None,
        description='Datapoint counts by source: static, template_dynamic, generated_dynamic (hybrid runs only)',
    )


class FocusAreaRecommendation(BaseModel):
    """LLM-generated actionable recommendation for a focus area."""

    category: str = Field(description='OWASP category code (e.g. ASI01, LLM07)')
    category_name: str = Field(description='Human-readable category name')
    risk_score: float = Field(description='Risk score used for ranking')
    traces_analyzed: int = Field(description='Number of failed traces analyzed')
    recommendations: list[str] = Field(description='Actionable bullet-point recommendations')
    patterns_observed: str = Field(description='Brief summary of patterns the LLM spotted in traces')


class RedTeamReport(BaseModel):
    """Top-level unified report wrapping all results."""

    version: str = '2.0.0'
    created_at: datetime
    description: str | None = None
    pipeline: Pipeline = Field(description="'static', 'dynamic', or 'hybrid'")
    framework: Framework | None = None

    categories_tested: list[str]
    tested_agents: list[str] = Field(default_factory=list, description='Names/keys of tested agents in this report')
    total_results: int

    agent_context: AgentContext | None = None
    agent_contexts: dict[str, AgentContext] = Field(default_factory=dict, description='Per-agent context keyed by agent key')

    results: list[RedTeamResult]

    summary: ReportSummary

    focus_area_recommendations: list[FocusAreaRecommendation] | None = Field(
        default=None,
        description='LLM-generated actionable recommendations for top risk areas (populated when generate_recommendations=True)',
    )

    token_usage_summary: TokenUsage | None = None
    duration_seconds: float | None = None
    pipeline_warnings: list[str] = Field(default_factory=list)

    @field_validator('pipeline', mode='before')
    @classmethod
    def _normalize_pipeline(cls, value: Any) -> Any:
        """Normalize legacy 'mixed' pipeline value to 'hybrid'."""
        if value == 'mixed':
            return 'hybrid'
        return value

    @field_validator('framework', mode='before')
    @classmethod
    def _normalize_framework(cls, value: Any) -> Any:
        """Normalize framework string aliases to canonical Framework enum values."""
        if isinstance(value, str):
            return normalize_framework(value)
        return value


# ---------------------------------------------------------------------------
# Dynamic pipeline row models
# ---------------------------------------------------------------------------


class EvaluatedRowBase(BaseModel):
    """Common row shape for evaluated JSONL outputs."""

    model_config = ConfigDict(extra='allow')

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

    model_config = ConfigDict(validate_assignment=True)

    id: str | None = None
    vulnerability: str = ''
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

    model_config = ConfigDict(extra='allow')

    mode: str | None = None
    target: str | None = None
    agent_key: str | None = None
    attack_model: str | None = None
    evaluator_model: str | None = None
    categories_tested: list[str] | None = None
    max_dynamic_datapoints: int | None = None
    max_static_datapoints: int | None = None
    max_turns: int | None = None
    max_per_category: int | None = None
    generated_strategy_count: int | None = None
    parallelism: int | None = None
    timestamp: str | None = None
    duration_seconds: float | None = None


class StrategySelectionRow(BaseModel):
    """Per-category strategy selection details persisted in staged outputs."""

    model_config = ConfigDict(extra='allow')

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

    model_config = ConfigDict(extra='allow')

    response: str | None = None
    agent_type: str
    agent_model: str
    error: str | None = None
    error_type: str | None = None


class EvaluatorConfig(TypedDict):
    """Minimal evaluatorq evaluator contract."""

    name: str
    scorer: Any


class ReportSnapshot(BaseModel):
    """Concise report summary used by CLI report commands."""

    pipeline: Pipeline
    framework: Framework | None = None
    total_results: int
    categories_tested: list[str]
    resistance_rate: float
    vulnerabilities_found: int
    top_techniques: dict[str, int] = Field(default_factory=dict)


_deprecated_warned: set[str] = set()


def __getattr__(name: str):
    if name == "EvaluationResult":
        if name not in _deprecated_warned:
            import warnings
            _deprecated_warned.add(name)
            warnings.warn(
                "EvaluationResult is deprecated in evaluatorq.redteam.contracts. "
                "Use AttackEvaluationResult instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return AttackEvaluationResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
