from __future__ import annotations

import json
import typing
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from evaluatorq.common.judge import JudgeOutcome, run_judge
from evaluatorq.common.jury import JuryDeliberation, Prediction, VerdictKind, append_jury_summary, run_jury
from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.types import DataPoint, EvaluationResult, Evaluator, Output, ScorerParameter

try:
    from evaluatorq.common.jury import TieBreak
except ImportError:  # pragma: no cover - TieBreak is a type alias
    TieBreak = Any  # type: ignore

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
        value: Any = "inconclusive"
        passed = None
    elif verdict_kind == "numeric":
        lo, hi = score_range
        score = min(max(float(verdict), lo), hi)
        value = score
        passed = score >= threshold
    elif isinstance(verdict, bool):
        value = verdict
        passed = verdict
    else:  # string label
        value = verdict
        passed = (verdict in passing_labels) if passing_labels else None

    return EvaluationResult.model_validate({
        "value": value,
        "explanation": explanation,
        "pass": passed,
        "token_usage": deliberation.token_usage,
    })


# ---------------------------------------------------------------------------
# Panel resolution helper
# ---------------------------------------------------------------------------


def _resolve_panel(judges: list[str] | None, model: str | None) -> list[str]:
    """Resolve a judge panel from either a list of judges or a single model shorthand.

    Raises:
        ValueError: If both ``judges`` and ``model`` are set simultaneously.
    """
    if judges and model:
        raise ValueError("Pass either `judges` or `model`, not both.")
    if model:
        return [model]
    if judges:
        return list(judges)
    return [DEFAULT_JUDGE_MODEL]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


# ponytail: strict_panel reserved for parity; generic jury has no self-judge guard yet
def llm_jury(
    *,
    name: str,
    criteria: str | None = None,
    prompt: str | None = None,
    system_prompt: str | None = None,
    judges: list[str] | None = None,
    model: str | None = None,
    repetitions: int = 1,
    replacement_judges: list[str] | None = None,
    min_successful_judges: int = 1,
    strict_panel: bool = False,
    verdict_kind: Literal["categorical", "numeric"] = "categorical",
    labels: list[str] | None = None,
    passing_labels: list[str] | None = None,
    numeric_aggregation: Literal["mean", "median"] = "mean",
    threshold: float = 0.5,
    score_range: tuple[float, float] = (0.0, 1.0),
    tie_break: Any = None,
    structured_output: bool = True,
    temperature: float | None = None,
    max_tokens: int = 8000,
    timeout_ms: int = 90000,
    extra_kwargs: dict[str, Any] | None = None,
    client: Any = None,
) -> Evaluator:
    """Build a jury (or single-judge) LLM evaluator for ``evaluators=[...]``."""
    # --- validation (fail fast) ---
    if bool(criteria) == bool(prompt):
        raise ValueError("Pass exactly one of `criteria` or `prompt`.")
    if verdict_kind != "categorical" and (labels or passing_labels):
        raise ValueError("`labels`/`passing_labels` are only valid for verdict_kind='categorical'.")
    if labels and passing_labels and not set(passing_labels) <= set(labels):
        raise ValueError("`passing_labels` must be a subset of `labels`.")
    if verdict_kind == "numeric" and not (score_range[0] <= threshold <= score_range[1]):
        raise ValueError(
            f"threshold ({threshold}) must lie within score_range {score_range}."
        )
    panel = _resolve_panel(judges, model)
    deduped = list(dict.fromkeys([*panel]))
    if not (1 <= min_successful_judges <= len(deduped)):
        raise ValueError(
            f"min_successful_judges ({min_successful_judges}) must be between 1 and "
            f"the deduplicated panel size ({len(deduped)})."
        )
    if temperature == 0.0:
        logger.warning(
            "temperature=0.0: reasoning models (o-series, gpt-5, …) often score worse "
            "at temp 0. Leave it unset unless you know your model benefits."
        )

    resolved_client = client if client is not None else resolve_llm_client(config_client=None).client
    verdict_model = _build_verdict_model(verdict_kind, labels, score_range)
    template = prompt if prompt is not None else _default_template(criteria or "")
    sys_prompt = system_prompt if system_prompt is not None else _default_system_prompt(
        verdict_kind, labels, score_range
    )
    vkind = VerdictKind.NUMERIC if verdict_kind == "numeric" else VerdictKind.CATEGORICAL

    async def scorer(params: ScorerParameter) -> EvaluationResult:
        data = params["data"]
        output = params["output"]
        replacements = _build_replacements(data, output, criteria or "")

        async def judge_fn(judge_model: str) -> Prediction:
            cfg = LLMCallConfig(
                model=judge_model, max_tokens=max_tokens, timeout_ms=timeout_ms,
                extra_kwargs=extra_kwargs or {},
            )
            outcome = await run_judge(
                client=resolved_client, model=judge_model, cfg=cfg,
                prompt_template=template, replacements=replacements,
                system_prompt=sys_prompt, response_model=verdict_model,
                structured_output=structured_output, temperature=temperature,
            )
            return _outcome_to_prediction(outcome)

        deliberation = await run_jury(
            judge_fn=judge_fn,
            panel=panel,
            repetitions=repetitions,
            replacement_judges=replacement_judges or [],
            min_successful_judges=min_successful_judges,
            verdict_kind=vkind,
            numeric_aggregation=numeric_aggregation,
            tie_break=tie_break,
        )
        return _to_evaluation_result(
            deliberation, verdict_kind=verdict_kind, passing_labels=passing_labels,
            threshold=threshold, score_range=score_range,
        )

    return {"name": name, "scorer": scorer}
