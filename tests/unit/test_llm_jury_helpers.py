from __future__ import annotations

import pytest

from evaluatorq.common.judge import JudgeOutcome, EvaluatorResponsePayload, JudgeError
from evaluatorq.common.template_engine import render_template
from evaluatorq.llm_jury import (
    DEFAULT_PAIRWISE_TEMPLATE,
    DEFAULT_TEMPLATE,
    _build_replacements,
    _default_system_prompt,
    _outcome_to_prediction,
    _to_evaluation_result,
)
from evaluatorq.types import DataPoint
from evaluatorq.common.jury import JuryDeliberation, JuryResult
from evaluatorq.contracts import JuryVote


# ---------------------------------------------------------------------------
# _build_replacements
# ---------------------------------------------------------------------------

def test_build_replacements_stringifies():
    dp = DataPoint(inputs={"question": "2+2?"}, expected_output="4")
    r = _build_replacements(dp, output="4", criteria="correct?")
    assert "2+2" in render_template("{{input.all_messages}}", r)
    assert render_template("{{output.response}}", r) == "4"
    assert render_template("{{input.expected_output}}", r) == "4"
    assert r["criteria"] == "correct?"


def test_build_replacements_messages_key():
    dp = DataPoint(inputs={"messages": [{"role": "user", "content": "hello"}]}, expected_output=None)
    r = _build_replacements(dp, output="hi", criteria="polite?")
    assert "hello" in render_template("{{input.all_messages}}", r)
    assert render_template("{{input.expected_output}}", r) == ""


def test_build_replacements_input_key():
    dp = DataPoint(inputs={"input": "plain text"}, expected_output=None)
    r = _build_replacements(dp, output="reply", criteria="any")
    assert "plain text" in render_template("{{input.all_messages}}", r)


# ---------------------------------------------------------------------------
# _default_system_prompt
# ---------------------------------------------------------------------------

def test_default_system_prompt_boolean():
    prompt = _default_system_prompt("categorical", None, (0.0, 1.0))
    assert "boolean" in prompt.lower() or "true" in prompt.lower()


def test_default_system_prompt_numeric():
    prompt = _default_system_prompt("numeric", None, (0.0, 10.0))
    assert "0" in prompt and "10" in prompt


def test_default_system_prompt_categorical():
    prompt = _default_system_prompt("categorical", ["good", "bad", "neutral"], (0.0, 1.0))
    assert "good" in prompt
    assert "bad" in prompt
    assert "neutral" in prompt


# ---------------------------------------------------------------------------
# default templates
# ---------------------------------------------------------------------------

def test_default_template_contains_placeholders():
    assert "{{criteria}}" in DEFAULT_TEMPLATE
    assert "{{input.all_messages}}" in DEFAULT_TEMPLATE
    assert "{{output.response}}" in DEFAULT_TEMPLATE
    assert "{{input.expected_output}}" in DEFAULT_TEMPLATE


def test_pairwise_template_substitutes_criteria_like_pointwise():
    # Both juries reach `criteria` the same way, so a `prompt=` override behaves
    # identically on either side.
    assert "{{criteria}}" in DEFAULT_PAIRWISE_TEMPLATE
    assert "{{question}}" in DEFAULT_PAIRWISE_TEMPLATE
    assert "{{response_a.output.response}}" in DEFAULT_PAIRWISE_TEMPLATE
    assert "{{response_b.output.response}}" in DEFAULT_PAIRWISE_TEMPLATE


# ---------------------------------------------------------------------------
# _outcome_to_prediction
# ---------------------------------------------------------------------------

def test_outcome_to_prediction_success():
    payload = EvaluatorResponsePayload(value=True, explanation="looks good")
    outcome = JudgeOutcome(payload=payload, raw_content='{"value": true, "explanation": "looks good"}')
    pred = _outcome_to_prediction(outcome)
    assert pred.value is True
    assert pred.explanation == "looks good"
    assert pred.error is None
    assert not pred.abstained


def test_outcome_to_prediction_error():
    outcome = JudgeOutcome(error_kind=JudgeError.TIMEOUT, error_message="timed out after 5000ms")
    pred = _outcome_to_prediction(outcome)
    assert pred.error is not None and "timed" in pred.error.lower()
    assert pred.value is None


def test_outcome_to_prediction_abstain():
    payload = EvaluatorResponsePayload(value=None, explanation="cannot determine", abstain=True)
    outcome = JudgeOutcome(payload=payload, raw_content='{}')
    pred = _outcome_to_prediction(outcome)
    assert pred.abstained is True


# ---------------------------------------------------------------------------
# _to_evaluation_result
# ---------------------------------------------------------------------------

def _make_deliberation(verdict, explanation="explanation text"):
    vote = JuryVote(
        model="test-model",
        success=True,
        value=verdict,
        explanation=explanation,
    )
    jury = JuryResult(
        judges_configured=1,
        judges_succeeded=1,
        judges_failed=0,
        replacements_used=0,
        tie=False,
        inconclusive=False,
        votes=[vote],
        raw_agreement=1.0,
    )
    return JuryDeliberation(verdict=verdict, explanation=explanation, jury=jury, token_usage=None)


def test_to_evaluation_result_bool_pass():
    delib = _make_deliberation(True)
    result = _to_evaluation_result(
        delib,
        verdict_kind="categorical",
        passing_labels=None,
        threshold=0.5,
        score_range=(0.0, 1.0),
    )
    assert result.pass_ is True
    assert result.value is True


def test_to_evaluation_result_bool_fail():
    delib = _make_deliberation(False)
    result = _to_evaluation_result(
        delib,
        verdict_kind="categorical",
        passing_labels=None,
        threshold=0.5,
        score_range=(0.0, 1.0),
    )
    assert result.pass_ is False
    assert result.value is False


def test_to_evaluation_result_numeric_pass():
    delib = _make_deliberation(0.8)
    result = _to_evaluation_result(
        delib,
        verdict_kind="numeric",
        passing_labels=None,
        threshold=0.5,
        score_range=(0.0, 1.0),
    )
    assert result.pass_ is True
    assert result.value == 0.8


def test_to_evaluation_result_numeric_fail():
    delib = _make_deliberation(0.3)
    result = _to_evaluation_result(
        delib,
        verdict_kind="numeric",
        passing_labels=None,
        threshold=0.5,
        score_range=(0.0, 1.0),
    )
    assert result.pass_ is False


def test_to_evaluation_result_categorical_label():
    delib = _make_deliberation("good")
    result = _to_evaluation_result(
        delib,
        verdict_kind="categorical",
        passing_labels=["good", "excellent"],
        threshold=0.5,
        score_range=(0.0, 1.0),
    )
    assert result.pass_ is True
    assert result.value == "good"


def test_to_evaluation_result_categorical_label_fail():
    delib = _make_deliberation("bad")
    result = _to_evaluation_result(
        delib,
        verdict_kind="categorical",
        passing_labels=["good", "excellent"],
        threshold=0.5,
        score_range=(0.0, 1.0),
    )
    assert result.pass_ is False


def _make_inconclusive_deliberation(explanation="explanation text"):
    """Build an inconclusive JuryDeliberation (verdict=None).

    The JuryResult validator requires len(votes) == judges_configured + replacements_used,
    so we supply one failed vote to satisfy that constraint.
    """
    failed_vote = JuryVote(
        model="test-model",
        success=False,
        value=None,
        error="judge failed",
    )
    jury = JuryResult(
        judges_configured=1,
        judges_succeeded=0,
        judges_failed=1,
        replacements_used=0,
        tie=False,
        inconclusive=True,
        votes=[failed_vote],
        stats=None,
        raw_agreement=None,
    )
    return JuryDeliberation(verdict=None, explanation=explanation, jury=jury, token_usage=None)


def test_to_evaluation_result_inconclusive():
    """verdict=None deliberation -> value='inconclusive' and pass_ is None."""
    delib = _make_inconclusive_deliberation()
    result = _to_evaluation_result(
        delib,
        verdict_kind="categorical",
        passing_labels=None,
        threshold=0.5,
        score_range=(0.0, 1.0),
    )
    assert result.value == "inconclusive"
    assert result.pass_ is None


def test_to_evaluation_result_numeric_clamped():
    """numeric verdict=1.5 with score_range=(0.0, 1.0) -> value clamped to 1.0 and pass_ is True."""
    delib = _make_deliberation(1.5)
    result = _to_evaluation_result(
        delib,
        verdict_kind="numeric",
        passing_labels=None,
        threshold=0.5,
        score_range=(0.0, 1.0),
    )
    assert result.value == 1.0
    assert result.pass_ is True


def test_to_evaluation_result_string_label_no_passing_labels():
    """string label verdict with passing_labels=None -> pass_ is None."""
    delib = _make_deliberation("good")
    result = _to_evaluation_result(
        delib,
        verdict_kind="categorical",
        passing_labels=None,
        threshold=0.5,
        score_range=(0.0, 1.0),
    )
    assert result.pass_ is None
