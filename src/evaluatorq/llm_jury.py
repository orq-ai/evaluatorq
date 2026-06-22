from __future__ import annotations

import json
import typing
from typing import Any

from pydantic import BaseModel, Field

from evaluatorq.common.judge import JudgeOutcome
from evaluatorq.common.jury import JuryDeliberation, Prediction, append_jury_summary
from evaluatorq.contracts import TokenUsage
from evaluatorq.types import DataPoint, EvaluationResult, Output

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


# ---------------------------------------------------------------------------
# Template / prompt helpers
# ---------------------------------------------------------------------------


def _build_replacements(data: DataPoint, output: Output, criteria: str) -> dict[str, Any]:
    """Build the template variable substitution dict for an LLM jury prompt.

    Handles three common ``inputs`` shapes:
    - ``{"messages": [...]}`` — serialised as a JSON message list.
    - ``{"input": ...}`` — single string input.
    - anything else — JSON-dumped as-is.
    """
    inputs = data.inputs
    if isinstance(inputs, dict) and "messages" in inputs:
        input_str = json.dumps(inputs["messages"], indent=2, default=str)
    elif isinstance(inputs, dict) and "input" in inputs:
        input_str = str(inputs["input"])
    else:
        input_str = json.dumps(inputs, indent=2, default=str)

    out_str = output if isinstance(output, str) else json.dumps(output, default=str)
    return {
        "input": input_str,
        "output": out_str,
        "expected_output": "" if data.expected_output is None else str(data.expected_output),
        "criteria": criteria,
    }


def _default_system_prompt(
    verdict_kind: str, labels: list[str] | None, score_range: tuple[float, float]
) -> str:
    """Return a sensible system prompt for the jury judge.

    The prompt instructs the model to return a structured verdict and a 2–3
    sentence explanation.  The ``value`` constraint is tailored to the
    ``verdict_kind``:

    - ``"numeric"`` — a float in ``[lo, hi]``.
    - ``"categorical"`` with labels — one of the given label strings.
    - ``"categorical"`` without labels — a boolean.
    """
    base = (
        "You are a strict evaluator. Read the input, the model's output, any "
        "expected output, and judge against the stated criterion. "
        "Return a structured verdict with a 2-3 sentence explanation."
    )
    if verdict_kind == "numeric":
        lo, hi = score_range
        return f"{base} `value` must be a number between {lo} and {hi} (higher is better)."
    if labels:
        return f"{base} `value` must be exactly one of: {', '.join(labels)}."
    return f"{base} `value` must be a boolean: true if the criterion is met, false otherwise."


def _default_template(criteria: str) -> str:
    """Return a default Mustache-style evaluation prompt template.

    Placeholder tokens use the ``{{name}}`` convention expected by the
    template engine (double-braces).
    """
    return (
        f"# Criterion\n{criteria}\n\n"
        "# Input\n{{input}}\n\n"
        "# Output\n{{output}}\n\n"
        "# Expected output\n{{expected_output}}\n"
    )


# ---------------------------------------------------------------------------
# Jury → domain-type converters
# ---------------------------------------------------------------------------


def _outcome_to_prediction(outcome: JudgeOutcome) -> Prediction:
    """Convert a raw :class:`JudgeOutcome` into a :class:`Prediction`.

    Maps error / abstain states to the matching Prediction fields so the jury
    runner can aggregate them without knowing about judge internals.
    """
    if outcome.error_kind is not None:
        return Prediction(
            error=outcome.error_message or str(outcome.error_kind),
            token_usage=outcome.token_usage,
        )
    payload = outcome.payload
    if payload is None:
        return Prediction(error="judge returned no payload", token_usage=outcome.token_usage)
    if payload.abstain or payload.value is None:
        return Prediction(
            abstained=True,
            explanation=payload.explanation,
            token_usage=outcome.token_usage,
        )
    return Prediction(
        value=payload.value,
        explanation=payload.explanation,
        token_usage=outcome.token_usage,
    )


def _to_evaluation_result(
    deliberation: JuryDeliberation,
    *,
    verdict_kind: str,
    passing_labels: list[str] | None,
    threshold: float,
    score_range: tuple[float, float],
) -> EvaluationResult:
    """Map a :class:`JuryDeliberation` to an :class:`EvaluationResult`.

    Passing logic:

    - ``"numeric"`` — ``passed`` when ``verdict >= threshold``.
    - Boolean verdicts — ``passed`` equals the verdict directly.
    - String label verdicts — ``passed`` when the label is in ``passing_labels``
      (or always if ``passing_labels`` is ``None``).
    """
    verdict = deliberation.verdict
    explanation = append_jury_summary(deliberation.explanation, deliberation.jury)

    if verdict is None:
        # Inconclusive jury — mark as failed.
        return EvaluationResult.model_validate({
            "value": False,
            "explanation": explanation,
            "pass": False,
            "token_usage": deliberation.token_usage,
        })

    if verdict_kind == "numeric":
        value = float(verdict)
        passed = value >= threshold
    elif isinstance(verdict, bool):
        value = verdict
        passed = verdict
    else:
        # String label
        value = verdict
        passed = (verdict in passing_labels) if passing_labels is not None else True

    return EvaluationResult.model_validate({
        "value": value,
        "explanation": explanation,
        "pass": passed,
        "token_usage": deliberation.token_usage,
    })
