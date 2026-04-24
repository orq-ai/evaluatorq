"""
OWASP evaluator registry.

Maps OWASP category codes to their corresponding evaluator getter functions.
This registry enables dynamic evaluator selection based on sample category.

Supports two OWASP frameworks:
1. OWASP Agentic Security Initiative (ASI) - Agent-specific vulnerabilities
2. OWASP LLM Top 10 - LLM-specific vulnerabilities

Note: Only categories suitable for static dataset testing are included.

Excluded ASI categories (require live system testing):
- ASI03: Identity/Privilege Abuse (requires actual permission systems)
- ASI04: Supply Chain (requires actual component integration)
- ASI07: Inter-Agent Communication (requires multi-agent architectures)
- ASI08: Cascading Failures (requires runtime observation)
- ASI10: Rogue Agents (requires runtime behavior observation)

Excluded LLM categories:
- LLM03: Supply Chain — excluded (infrastructure-level, no dataset samples)
- LLM10: Unbounded Consumption — excluded (resource exhaustion, not testable via prompt-based red teaming)
"""

from collections.abc import Callable
from typing import TypedDict

from evaluatorq.redteam.contracts import Vulnerability
from evaluatorq.redteam.frameworks.owasp.agent_evaluators import (
    get_asi01_goal_hijacking_evaluator,
    get_asi02_tool_misuse_evaluator,
    get_asi05_code_execution_evaluator,
    get_asi06_memory_poisoning_evaluator,
    get_asi09_trust_exploitation_evaluator,
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
    'ASI05': get_asi05_code_execution_evaluator,
    'ASI06': get_asi06_memory_poisoning_evaluator,
    'ASI09': get_asi09_trust_exploitation_evaluator,
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
OWASP_EVALUATOR_REGISTRY: dict[str, EvaluatorGetter] = {
    **_ASI_REGISTRY,
    **{f'OWASP-{k}': v for k, v in _ASI_REGISTRY.items()},
    **_LLM_REGISTRY,
    **{f'OWASP-{k}': v for k, v in _LLM_REGISTRY.items()},
}

# Backwards compatibility alias
OWASP_EVALUATOR_REGISTRY_FULL: dict[str, EvaluatorGetter] = OWASP_EVALUATOR_REGISTRY.copy()

# Primary registry keyed by vulnerability
VULNERABILITY_EVALUATOR_REGISTRY: dict[Vulnerability, EvaluatorGetter] = {}
for _cat, _getter in _ASI_REGISTRY.items():
    _vuln = CATEGORY_TO_VULNERABILITY.get(_cat)
    if _vuln is not None:
        VULNERABILITY_EVALUATOR_REGISTRY[_vuln] = _getter
for _cat, _getter in _LLM_REGISTRY.items():
    _vuln = CATEGORY_TO_VULNERABILITY.get(_cat)
    if _vuln is not None:
        VULNERABILITY_EVALUATOR_REGISTRY[_vuln] = _getter


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
    vuln: Vulnerability,
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


def get_evaluator_metadata_for_vulnerability(vuln: Vulnerability) -> EvaluatorMetadata | None:
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
