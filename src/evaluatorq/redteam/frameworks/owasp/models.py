"""OWASP-specific domain models."""

from __future__ import annotations

import sys
from typing import Any

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """Polyfill for Python <3.11."""

from pydantic import BaseModel, Field


class OWASPCategory(StrEnum):
    """OWASP Agentic Security Initiative categories."""

    ASI01 = 'ASI01'  # Agent Goal Hijacking
    ASI02 = 'ASI02'  # Tool Misuse and Exploitation
    ASI03 = 'ASI03'  # Identity and Privilege Abuse
    ASI04 = 'ASI04'  # Agentic Supply Chain Vulnerabilities
    ASI05 = 'ASI05'  # Unexpected Code Execution (RCE)
    ASI06 = 'ASI06'  # Memory and Context Poisoning
    ASI07 = 'ASI07'  # Insecure Inter-Agent Communication
    ASI08 = 'ASI08'  # Cascading Failures
    ASI09 = 'ASI09'  # Human-Agent Trust Exploitation
    ASI10 = 'ASI10'  # Rogue Agents


class LlmEvaluatorOutputFormat(StrEnum):
    """Output format for LLM evaluators."""

    BOOLEAN = 'boolean'
    NUMBER = 'number'
    STRING = 'string'
    CATEGORICAL = 'categorical'


class EvaluatorModelConfig(BaseModel):
    """Model configuration for an LLM evaluator."""

    id: str
    integration_id: str | None = None


class LlmEvaluatorEntity(BaseModel):
    """Lightweight LLM evaluator entity.

    Replaces ``evals_python_runner.models.evaluator.LlmEvaluatorEntity``
    with a minimal representation that holds the evaluation prompt template,
    model config, and output format.
    """

    id: str = ''
    display_name: str | None = None
    description: str | None = None
    model: EvaluatorModelConfig
    prompt: str
    output_type: LlmEvaluatorOutputFormat
    temperature: float = 0.1
    requires_reference: bool = False
    guardrail_config: dict[str, Any] | None = None
    categorical_labels: list[dict[str, Any]] | None = None

    @property
    def model_id(self) -> str:
        """Convenience accessor for the model ID."""
        return self.model.id
