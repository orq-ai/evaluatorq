from __future__ import annotations

import typing

from pydantic import BaseModel, Field

DEFAULT_JUDGE_MODEL = "openai/gpt-5.5"


def _build_verdict_model(
    verdict_kind: str, labels: list[str] | None, score_range: tuple[float, float]
) -> type[BaseModel]:
    """Build a dynamic Pydantic verdict model based on verdict spec.

    Args:
        verdict_kind: The kind of verdict ("categorical" or "numeric")
        labels: List of allowed categorical labels, or None for boolean/numeric
        score_range: The score range (min, max) for numeric verdicts

    Returns:
        A Pydantic BaseModel class with 'value' and 'explanation' fields
    """
    if verdict_kind == "categorical":
        if labels is None:
            # No labels = boolean
            value_annotation = bool
        else:
            # Labels provided = Literal
            value_annotation = typing.Literal[tuple(labels)]  # type: ignore
    else:  # numeric
        value_annotation = float

    # Create model dynamically
    class VerdictModel(BaseModel):
        value: value_annotation  # type: ignore
        explanation: str = Field(default="", description="Explanation for the verdict")

    return VerdictModel
