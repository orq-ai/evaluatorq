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


def test_outcome_to_prediction_retains_classify_raw_output_only():
    classify = JudgeOutcome(
        payload=EvaluatorResponsePayload(value='neutral', explanation="choice='neutral' (confidence 0.96)"),
        endpoint='classify',
        raw_output={'type': 'choice', 'choice': 'neutral', 'confidence': 0.96},
    )
    assert _outcome_to_prediction(classify).raw_output == classify.raw_output

    chat = JudgeOutcome(
        payload=EvaluatorResponsePayload(value=True, explanation='ok'),
        endpoint='chat',
        raw_content='{"value": true, "explanation": "ok"}',
    )
    assert _outcome_to_prediction(chat).raw_output is None


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


def test_outcome_to_prediction_bare_none_is_failure_not_abstention():
    # value=None with abstain=False is a non-usable verdict, not a clean abstention: the
    # Prediction is non-decisive and non-abstained, so the jury counts it as failed.
    payload = EvaluatorResponsePayload(value=None, explanation="no verdict", abstain=False)
    pred = _outcome_to_prediction(JudgeOutcome(payload=payload, raw_content='{}'))
    assert pred.abstained is False
    assert pred.value is None
    assert pred.error is None
    assert pred.decisive is False


def test_outcome_to_prediction_flags_coerced_abstention():
    # abstain=True WITH a value is coerced to an abstention by EvaluatorResponsePayload;
    # the Prediction must carry verdict_coerced so the artifact can tell it apart from a
    # clean abstention.
    payload = EvaluatorResponsePayload(value=True, explanation="contradiction", abstain=True)
    assert payload.coerced_abstain is True  # precondition: the coercion fired
    pred = _outcome_to_prediction(JudgeOutcome(payload=payload, raw_content='{}'))
    assert pred.abstained is True
    assert pred.verdict_coerced is True


def test_jury_vote_still_rejects_the_contradiction_the_payload_coerces():
    # EvaluatorResponsePayload coerces abstain=True + a value; JuryVote raises on it.
    # The two are deliberately different policies, so the backstop needs a test of its
    # own — otherwise the coercion silently becomes the only enforcement and a vote
    # built without a judge payload reintroduces the contradiction unnoticed.
    with pytest.raises(ValueError, match='abstained JuryVote must have a null value'):
        JuryVote(model='m', success=True, abstained=True, value=True)


@pytest.mark.asyncio
async def test_jury_vote_carries_the_coerced_count():
    # A coerced abstention must reach the vote and its repetitions, so the run artifact
    # can tell it from a clean abstention.
    from evaluatorq.common.jury import Prediction, VerdictKind, run_jury

    async def judge(model: str) -> Prediction:
        return Prediction(abstained=True, verdict_coerced=True, explanation='coerced')

    delib = await run_jury(judge_fn=judge, panel=['j'], repetitions=1, verdict_kind=VerdictKind.CATEGORICAL, tie_break=None)
    vote = delib.jury.votes[0]
    assert vote.abstained is True
    assert vote.repetitions_coerced == 1
    assert vote.repetitions[0].verdict_coerced is True


@pytest.mark.asyncio
async def test_bare_none_outcome_counts_as_a_failed_repetition():
    # Driven through _outcome_to_prediction, so it fails if llm_jury goes back to reading
    # value=None as a clean abstention.
    from evaluatorq.common.jury import Prediction, VerdictKind, run_jury

    payload = EvaluatorResponsePayload(value=None, explanation="no verdict", abstain=False)

    async def judge(model: str) -> Prediction:
        return _outcome_to_prediction(JudgeOutcome(payload=payload, raw_content='{}'))

    delib = await run_jury(judge_fn=judge, panel=['j'], repetitions=1, verdict_kind=VerdictKind.CATEGORICAL, tie_break=None)
    vote = delib.jury.votes[0]
    assert vote.repetitions_failed == 1
    assert vote.abstained is False
    assert vote.success is False


@pytest.mark.asyncio
async def test_bare_none_outcome_starts_a_replacement_judge():
    # A failed judge is replaced and a clean abstention is not, so reading value=None as
    # a failure means an extra, billed judge. The changelog states this.
    from evaluatorq.common.jury import Prediction, VerdictKind, run_jury

    calls: list[str] = []

    async def judge(model: str) -> Prediction:
        calls.append(model)
        value = True if model == 'spare' else None
        payload = EvaluatorResponsePayload(value=value, explanation="x", abstain=False)
        return _outcome_to_prediction(JudgeOutcome(payload=payload, raw_content='{}'))

    await run_jury(
        judge_fn=judge,
        panel=['j'],
        repetitions=1,
        replacement_judges=['spare'],
        verdict_kind=VerdictKind.CATEGORICAL,
        tie_break=None,
    )
    assert calls == ['j', 'spare']


def test_classifier_result_carries_a_coerced_verdict():
    from evaluatorq.trace_finder.classifier import _outcome_result

    payload = EvaluatorResponsePayload(value=True, explanation="contradiction", abstain=True)
    result = _outcome_result(JudgeOutcome(payload=payload, raw_content='{}'), model='m')
    assert result.raw_output is not None
    vote = result.raw_output['jury']['votes'][0]
    assert vote['repetitions_coerced'] == 1
    assert vote['repetitions'][0]['verdict_coerced'] is True


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


@pytest.mark.asyncio
async def test_coerced_count_matches_the_flagged_repetitions():
    from evaluatorq.common.jury import Prediction, VerdictKind, run_jury

    passes = iter([
        Prediction(abstained=True, verdict_coerced=True),
        Prediction(abstained=True, verdict_coerced=True),
        Prediction(abstained=True),
    ])

    async def judge(model: str) -> Prediction:
        return next(passes)

    delib = await run_jury(judge_fn=judge, panel=['m'], repetitions=3, verdict_kind=VerdictKind.CATEGORICAL, tie_break=None)
    assert delib.jury.votes[0].repetitions_coerced == 2


def test_jury_vote_rejects_a_coerced_count_the_repetitions_do_not_back():
    from evaluatorq.contracts import JuryRepetition

    with pytest.raises(ValueError, match='repetitions_coerced'):
        JuryVote(model='m', success=True, abstained=True, repetitions=[JuryRepetition()], repetitions_coerced=3)


@pytest.mark.asyncio
async def test_bare_none_passes_name_the_cause_on_the_vote():
    from evaluatorq.common.jury import Prediction, VerdictKind, run_jury

    payload = EvaluatorResponsePayload(value=None, explanation='transcript is empty', abstain=False)

    async def judge(model: str) -> Prediction:
        return _outcome_to_prediction(JudgeOutcome(payload=payload, raw_content='{}'))

    delib = await run_jury(judge_fn=judge, panel=['m'], repetitions=1, verdict_kind=VerdictKind.CATEGORICAL, tie_break=None)
    assert delib.jury.votes[0].error == 'judge returned no verdict: transcript is empty'


@pytest.mark.asyncio
async def test_bare_none_with_no_explanation_names_the_bare_cause():
    from evaluatorq.common.jury import VerdictKind, run_jury

    payload = EvaluatorResponsePayload(value=None, explanation='', abstain=False)

    async def judge(model: str):
        return _outcome_to_prediction(JudgeOutcome(payload=payload, raw_content='{}'))

    delib = await run_jury(judge_fn=judge, panel=['m'], repetitions=1, verdict_kind=VerdictKind.CATEGORICAL, tie_break=None)
    assert delib.jury.votes[0].error == 'judge returned no verdict'


def test_no_verdict_message_caps_a_long_reason():
    from evaluatorq.common.jury import NO_VERDICT_REASON_LIMIT, no_verdict_message

    prefix = 'judge returned no verdict: '
    assert no_verdict_message('x' * 5000) == prefix + 'x' * (NO_VERDICT_REASON_LIMIT - 1) + '…'
    # Whitespace at the cut is dropped, so the result never ends in a space before the ellipsis.
    # The cut keeps NO_VERDICT_REASON_LIMIT - 1 characters; make the last one a space.
    spaced = no_verdict_message('x' * (NO_VERDICT_REASON_LIMIT - 2) + ' ' + 'y' * 100)
    assert spaced == prefix + 'x' * (NO_VERDICT_REASON_LIMIT - 2) + '…'
    # A reason exactly at the limit is kept whole.
    exact = 'y' * NO_VERDICT_REASON_LIMIT
    assert no_verdict_message(exact) == prefix + exact
