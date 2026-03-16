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
"""

from collections.abc import Callable
from typing import TypedDict

from evaluatorq.redteam.frameworks.owasp.models import LlmEvaluatorEntity

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
    get_llm10_unbounded_consumption_evaluator,
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
    'LLM10': get_llm10_unbounded_consumption_evaluator,
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

# Category code to display name mappings
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

# Unified category names (supports ASI01, OWASP-ASI01, LLM01, OWASP-LLM01 formats)
OWASP_CATEGORY_NAMES: dict[str, str] = {
    **_ASI_CATEGORY_NAMES,
    **{f'OWASP-{k}': v for k, v in _ASI_CATEGORY_NAMES.items()},
    **_LLM_CATEGORY_NAMES,
    **{f'OWASP-{k}': v for k, v in _LLM_CATEGORY_NAMES.items()},
}


class EvaluatorMetadata(TypedDict):
    """Canonical evaluator metadata associated with an OWASP category."""

    evaluator_id: str
    evaluator_name: str


def _normalize_category_code(category: str) -> str:
    """Normalize category to short-form code (e.g., OWASP-LLM01 -> LLM01)."""
    return category.removeprefix('OWASP-')


# Canonical evaluator metadata for categories with known evaluator entities in ORQ.
_CANONICAL_EVALUATOR_METADATA_BY_CATEGORY: dict[str, EvaluatorMetadata] = {
    'ASI01': {'evaluator_id': '01KFTGH3A3ZEQBY4JH0RHTAQYN', 'evaluator_name': 'owasp_asi01_goal_hijacking'},
    'ASI05': {'evaluator_id': '01KFTGH3TW3SHBNT0FF8ZC53KA', 'evaluator_name': 'owasp_asi05_code_execution'},
    'ASI06': {'evaluator_id': '01KFTGH4AHJD0DGMVXEV0QPM21', 'evaluator_name': 'owasp_asi06_memory_poisoning'},
    'LLM01': {'evaluator_id': '01KFX0DEY1S57B1Q19AK10RAYF', 'evaluator_name': 'owasp_llm01_prompt_injection'},
    'LLM02': {
        'evaluator_id': '01KFX0DEY1RWTR466RMS2FT3P7',
        'evaluator_name': 'owasp_llm02_sensitive_info_disclosure',
    },
    'LLM07': {'evaluator_id': '01KFX0DEY1DBEW2ZAQ0M3DN12T', 'evaluator_name': 'owasp_llm07_system_prompt_leakage'},
}


def get_evaluator_metadata_for_category(category: str) -> EvaluatorMetadata | None:
    """Return evaluator metadata for a category, with fallback to local evaluator identifiers.

    Preference order:
    1. Canonical ORQ evaluator metadata mapping (when known)
    2. Local evaluator entity id/display_name
    """
    normalized = _normalize_category_code(category)
    mapped = _CANONICAL_EVALUATOR_METADATA_BY_CATEGORY.get(normalized)
    if mapped:
        return mapped

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


def list_available_categories() -> list[str]:
    """List all available OWASP category codes (short-form only).

    Returns:
        List of category codes (ASI01-ASI10, LLM01-LLM10)
    """
    return [k for k in OWASP_EVALUATOR_REGISTRY if not k.startswith('OWASP-')]
