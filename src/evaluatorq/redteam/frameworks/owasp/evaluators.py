"""
OWASP evaluator registry.

Maps OWASP category codes to their corresponding evaluator getter functions.
This registry enables dynamic evaluator selection based on sample category.

Supports two OWASP frameworks:
1. OWASP Agentic Security Initiative (ASI) - Agent-specific vulnerabilities
2. OWASP LLM Top 10 - LLM-specific vulnerabilities

All ten ASI categories have a registered prompt-based evaluator. ASI03/04/07/08/10
score the agent's *response text* as a single-agent proxy for behaviour that ideally
warrants live-system testing (permission systems, component integration, runtime
observation). ASI07/08 assume multi-agent architectures, so they are gated to
multi-agent targets at strategy-selection time (see AgentContext.is_multi_agent);
against a single-agent target they are skipped rather than scored on a weak proxy.

Excluded from the category registry (not evaluable via prompt-based red teaming):
- LLM10: Unbounded Consumption — resource exhaustion; not a category code in this registry.
  Passing categories=['LLM10'] raises ValueError('Unrecognized category').

Note: LLM03 (Supply Chain) is NOT excluded — it resolves to the supply_chain vulnerability
and is scored by the ASI04 evaluator. The runner logs a relabel notice at INFO level.
"""

from collections.abc import Callable
from typing import TypedDict

from evaluatorq.redteam.contracts import AnyVulnerability, Vulnerability, resolve_open_value
from evaluatorq.redteam.frameworks.owasp.agent_evaluators import (
    get_asi01_goal_hijacking_evaluator,
    get_asi02_tool_misuse_evaluator,
    get_asi03_privilege_abuse_evaluator,
    get_asi04_supply_chain_evaluator,
    get_asi05_code_execution_evaluator,
    get_asi06_memory_poisoning_evaluator,
    get_asi07_interagent_comm_evaluator,
    get_asi08_cascading_failures_evaluator,
    get_asi09_trust_exploitation_evaluator,
    get_asi10_rogue_agents_evaluator,
)
from evaluatorq.redteam.frameworks.owasp.llm_evaluators import (
    get_llm01_prompt_injection_evaluator,
    get_llm02_sensitive_info_disclosure_evaluator,
    get_llm04_data_model_poisoning_evaluator,
    get_llm05_improper_output_handling_evaluator,
    get_llm06_excessive_agency_evaluator,
    get_llm07_system_prompt_leakage_evaluator,
    get_llm08_vector_embedding_weaknesses_evaluator,
    get_llm09_misinformation_evaluator,
)
from evaluatorq.redteam.frameworks.owasp.models import LlmEvaluatorEntity
from evaluatorq.redteam.registry_scope import ScopedDict
from evaluatorq.redteam.vulnerability_registry import (
    CANONICAL_EVALUATOR_METADATA,
    CATEGORY_TO_VULNERABILITY,
    resolve_category_safe,
)

# Type alias for evaluator getter functions
EvaluatorGetter = Callable[[str | None], LlmEvaluatorEntity]

# Registry mapping OWASP ASI category codes to evaluator getter functions
_ASI_REGISTRY: dict[str, EvaluatorGetter] = {
    'ASI01': get_asi01_goal_hijacking_evaluator,
    'ASI02': get_asi02_tool_misuse_evaluator,
    'ASI03': get_asi03_privilege_abuse_evaluator,
    'ASI04': get_asi04_supply_chain_evaluator,
    'ASI05': get_asi05_code_execution_evaluator,
    'ASI06': get_asi06_memory_poisoning_evaluator,
    'ASI07': get_asi07_interagent_comm_evaluator,
    'ASI08': get_asi08_cascading_failures_evaluator,
    'ASI09': get_asi09_trust_exploitation_evaluator,
    'ASI10': get_asi10_rogue_agents_evaluator,
}

# Registry mapping OWASP LLM Top 10 category codes to evaluator getter functions
_LLM_REGISTRY: dict[str, EvaluatorGetter] = {
    'LLM01': get_llm01_prompt_injection_evaluator,
    'LLM02': get_llm02_sensitive_info_disclosure_evaluator,
    'LLM04': get_llm04_data_model_poisoning_evaluator,
    'LLM05': get_llm05_improper_output_handling_evaluator,
    'LLM06': get_llm06_excessive_agency_evaluator,
    'LLM07': get_llm07_system_prompt_leakage_evaluator,
    'LLM08': get_llm08_vector_embedding_weaknesses_evaluator,
    'LLM09': get_llm09_misinformation_evaluator,
}

# Unified registry supporting 'ASI01', 'OWASP-ASI01', 'LLM01', 'OWASP-LLM01' formats
OWASP_EVALUATOR_REGISTRY: ScopedDict[str, EvaluatorGetter] = ScopedDict(
    'owasp_evaluators',
    {
        **_ASI_REGISTRY,
        **{f'OWASP-{k}': v for k, v in _ASI_REGISTRY.items()},
        **_LLM_REGISTRY,
        **{f'OWASP-{k}': v for k, v in _LLM_REGISTRY.items()},
    },
)

# Backwards compatibility alias
OWASP_EVALUATOR_REGISTRY_FULL: dict[str, EvaluatorGetter] = dict(OWASP_EVALUATOR_REGISTRY)

# Primary registry keyed by vulnerability
VULNERABILITY_EVALUATOR_REGISTRY: ScopedDict[AnyVulnerability, EvaluatorGetter] = ScopedDict('vulnerability_evaluators')
for _cat, _getter in _ASI_REGISTRY.items():
    _vuln = CATEGORY_TO_VULNERABILITY.get(_cat)
    if _vuln is not None:
        VULNERABILITY_EVALUATOR_REGISTRY[_vuln] = _getter
for _cat, _getter in _LLM_REGISTRY.items():
    _vuln = CATEGORY_TO_VULNERABILITY.get(_cat)
    if _vuln is not None:
        VULNERABILITY_EVALUATOR_REGISTRY[_vuln] = _getter

_custom_judges: ScopedDict[AnyVulnerability, list[str]] = ScopedDict('custom_judges')


def register_judge(vuln: AnyVulnerability, judge: LlmEvaluatorEntity | EvaluatorGetter) -> None:
    """Register the judge for a vulnerability under the vulnerability and its category
    codes (RES-1922). *judge* is an entity or a ``getter(model_id)``; a built-in
    vulnerability keeps its shipped judge, so registering one for it raises.
    """
    from evaluatorq.redteam.vulnerability_registry import get_framework_categories, get_primary_category

    vuln = resolve_open_value(Vulnerability, vuln)
    if isinstance(vuln, Vulnerability):
        msg = f'{vuln.value!r} is a built-in vulnerability and keeps its shipped judge.'
        raise ValueError(msg)  # noqa: TRY004 - a wrong value, not a wrong type
    if isinstance(judge, LlmEvaluatorEntity):
        entity = judge

        def getter(model_id: str | None = None) -> LlmEvaluatorEntity:
            if not model_id:
                return entity
            return entity.model_copy(update={'model': entity.model.model_copy(update={'id': model_id})})

    else:
        getter = judge
    codes = [code for codes in get_framework_categories(vuln).values() for code in codes] or [
        get_primary_category(vuln)
    ]
    keys = [k for code in codes for k in (code, f'OWASP-{code}')]
    VULNERABILITY_EVALUATOR_REGISTRY[vuln] = getter
    for key in keys:
        OWASP_EVALUATOR_REGISTRY[key] = getter
    _custom_judges[vuln] = keys


def clear_custom_judges() -> None:
    """Drop every registered custom judge (test isolation)."""
    for vuln, keys in _custom_judges.items():
        VULNERABILITY_EVALUATOR_REGISTRY.pop(vuln, None)
        for key in keys:
            OWASP_EVALUATOR_REGISTRY.pop(key, None)
    _custom_judges.clear()


class EvaluatorMetadata(TypedDict):
    """Canonical evaluator metadata associated with an OWASP category."""

    evaluator_id: str
    evaluator_name: str


def _normalize_category_code(category: str) -> str:
    """Normalize category to short-form code (e.g., OWASP-LLM01 -> LLM01)."""
    return category.removeprefix('OWASP-')


def get_evaluator_metadata_for_category(category: str) -> EvaluatorMetadata | None:
    """Return evaluator metadata for a category, with fallback to local evaluator identifiers.

    Delegates to get_evaluator_metadata_for_vulnerability after resolving the category
    to a Vulnerability via the vulnerability registry (single source of truth).

    Preference order:
    1. Canonical ORQ evaluator metadata mapping (when known, from vulnerability_registry)
    2. Local evaluator entity id/display_name
    """
    normalized = _normalize_category_code(category)
    vuln = resolve_category_safe(normalized)
    if vuln is not None:
        return get_evaluator_metadata_for_vulnerability(vuln)

    evaluator = get_evaluator_for_category(normalized)
    if evaluator is None:
        return None

    evaluator_id = str(getattr(evaluator, 'id', '') or '').strip()
    if not evaluator_id:
        return None

    evaluator_name = evaluator_id
    display_name = str(getattr(evaluator, 'display_name', '') or '').strip()
    if display_name:
        evaluator_name = display_name

    return {'evaluator_id': evaluator_id, 'evaluator_name': evaluator_name}


def get_evaluator_for_category(
    category: str,
    model_id: str | None = None,
) -> LlmEvaluatorEntity | None:
    """Get the appropriate evaluator for a given OWASP category.

    Args:
        category: OWASP category code (e.g., "ASI01", "LLM07")
        model_id: Optional model ID override

    Returns:
        LlmEvaluatorEntity for the category, or None if not found
    """
    getter = OWASP_EVALUATOR_REGISTRY.get(category)
    if getter is None:
        return None
    return getter(model_id)


def get_evaluator_for_vulnerability(
    vuln: AnyVulnerability,
    model_id: str | None = None,
) -> LlmEvaluatorEntity | None:
    """Get the evaluator for a vulnerability.

    Args:
        vuln: Vulnerability enum value
        model_id: Optional model ID override

    Returns:
        LlmEvaluatorEntity or None if no evaluator registered
    """
    getter = VULNERABILITY_EVALUATOR_REGISTRY.get(vuln)
    if getter is None:
        return None
    return getter(model_id)


def get_evaluator_metadata_for_vulnerability(vuln: AnyVulnerability) -> EvaluatorMetadata | None:
    """Return evaluator metadata for a vulnerability."""
    mapped = CANONICAL_EVALUATOR_METADATA.get(vuln)
    if mapped:
        return {'evaluator_id': mapped['evaluator_id'], 'evaluator_name': mapped['evaluator_name']}

    evaluator = get_evaluator_for_vulnerability(vuln)
    if evaluator is None:
        return None

    evaluator_id = str(getattr(evaluator, 'id', '') or '').strip()
    if not evaluator_id:
        return None

    evaluator_name = evaluator_id
    display_name = str(getattr(evaluator, 'display_name', '') or '').strip()
    if display_name:
        evaluator_name = display_name

    return {'evaluator_id': evaluator_id, 'evaluator_name': evaluator_name}


def list_available_categories() -> list[str]:
    """List all available OWASP category codes.

    Returns:
        List of category codes (ASI01-ASI10, LLM01-LLM09)
    """
    return list(OWASP_EVALUATOR_REGISTRY.keys())
