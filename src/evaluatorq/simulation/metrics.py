"""Canonical metadata for per-turn simulation quality metrics."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TurnMetric:
    key: str
    label: str
    high_is_risky: bool = False


TURN_METRICS = (
    TurnMetric('response_quality', 'response quality'),
    TurnMetric('hallucination_risk', 'hallucination risk', high_is_risky=True),
    TurnMetric('tone_appropriateness', 'tone appropriateness'),
    TurnMetric('factual_accuracy', 'factual accuracy'),
)


def is_errored_result(result: object) -> bool:
    """Classify a typed simulation result or its saved dictionary form."""
    if isinstance(result, Mapping):
        metadata = result.get('metadata')
        terminated_by = result.get('terminated_by')
    else:
        metadata = getattr(result, 'metadata', None)
        terminated_by = getattr(result, 'terminated_by', None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    criteria_errors = metadata.get('criteria_errors')
    outcome = getattr(terminated_by, 'value', terminated_by)
    return (
        bool(metadata.get('error'))
        or (isinstance(criteria_errors, list) and bool(criteria_errors))
        or outcome == 'error'
    )
