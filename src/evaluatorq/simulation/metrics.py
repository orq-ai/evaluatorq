"""Canonical metadata for per-turn simulation quality metrics."""

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
