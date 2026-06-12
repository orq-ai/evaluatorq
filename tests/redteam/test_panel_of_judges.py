"""Unit tests for the panel-of-judges / jury feature (RES-739).

Mirrors the orq platform jury contract and the LLM-jury best practices
(orq.ai/blog/llm-juries-in-practice): fail-closed ties with an explicit tie
flag, per-judge votes, mean/std stats, judge replacement on failure.

Covers:
1. _majority_vote / _agreement_rate / _jury_stats aggregation helpers
2. Single judge + single repetition stays byte-identical (jury is None)
3. Jury: unanimous and split verdicts, per-judge votes recorded
4. Per-judge repetition majority voting
5. Fail-closed tie handling with explicit tie flag
6. Judge replacement on total failure (replacements_used)
7. min_successful_judges floor → inconclusive
8. Token usage summed across all jury calls
"""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.common.jury import (
    _agreement_rate,
    _jury_stats,
    _panel_composition_messages,
    _plurality_vote,
    provider_family,
)
from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator


def _majority_vote(votes: list[bool | None]) -> tuple[bool | float | str | None, bool]:
    """Test shim: filters None values then delegates to _plurality_vote (mirrors old evaluator helper)."""
    decisive = [v for v in votes if v is not None]
    if not decisive:
        return None, False
    return _plurality_vote(decisive)
from evaluatorq.redteam.contracts import AttackEvaluationResult, TextOutputItem, Vulnerability

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response(content: str, usage_tokens: int | None = None) -> MagicMock:
    """Build a mock chat-completion response with the given content."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    if usage_tokens is None:
        mock_response.usage = None
    else:
        usage = MagicMock()
        usage.prompt_tokens = usage_tokens
        usage.completion_tokens = usage_tokens
        usage.total_tokens = usage_tokens * 2
        mock_response.usage = usage
    return mock_response


def _verdict_json(value: bool) -> str:
    return f'{{"value": {str(value).lower()}, "explanation": "judge says {value}"}}'


def _model_client(per_model: dict[str, list[str]], usage_tokens: int | None = None) -> AsyncMock:
    """Mock AsyncOpenAI client returning queued contents per model.

    ``per_model`` maps a model id to an ordered list of response contents; each
    call for that model pops the next content (cycling if exhausted), which lets
    a single judge return different verdicts across repetitions. A content value
    of the sentinel ``"__raise__"`` makes that call raise, simulating a failure.
    """
    cursors = {model: 0 for model in per_model}

    async def _create(*_args: Any, **kwargs: Any) -> MagicMock:
        model = kwargs['model']
        queue = per_model[model]
        idx = cursors[model] % len(queue)
        cursors[model] += 1
        content = queue[idx]
        if content == '__raise__':
            raise RuntimeError('judge exploded')
        return _response(content, usage_tokens=usage_tokens)

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=_create)
    return client


def _patch_tracing():
    # Judge execution now runs through evaluatorq.redteam.judge.run_judge ->
    # execute_chat_completion. Tracing there is inert without a configured tracer
    # (with_llm_span yields None, record_* no-op on a None span), so no patching is
    # needed; keep two null contexts so existing call sites stay unchanged.
    return (contextlib.nullcontext(), contextlib.nullcontext())


async def _evaluate(evaluator: OWASPEvaluator) -> AttackEvaluationResult:
    span_patch, record_patch = _patch_tracing()
    with span_patch, record_patch:
        return await evaluator.evaluate_vulnerability(
            vuln=Vulnerability.GOAL_HIJACKING,
            messages=[{'role': 'user', 'content': 'attack prompt'}],
            output_messages=[TextOutputItem(text='agent response', annotations=[])],
        )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


class TestAggregationHelpers:
    def test_majority_vote_clear_majority(self):
        assert _majority_vote([True, True, False]) == (True, False)
        assert _majority_vote([False, False, True]) == (False, False)

    def test_majority_vote_ignores_none(self):
        assert _majority_vote([True, None, True]) == (True, False)
        assert _majority_vote([None, None, False]) == (False, False)

    def test_majority_vote_all_none(self):
        assert _majority_vote([None, None]) == (None, False)

    def test_majority_vote_tie_flags_tie(self):
        assert _majority_vote([True, False]) == (None, True)
        assert _majority_vote([True, True, False, False]) == (None, True)

    def test_agreement_rate(self):
        assert _agreement_rate([True, True, True]) == 1.0
        assert _agreement_rate([True, True, False]) == pytest.approx(2 / 3)
        assert _agreement_rate([True, False]) == 0.5
        assert _agreement_rate([]) is None

    def test_jury_stats(self):
        stats = _jury_stats([True, True, False])
        assert stats is not None
        assert stats.mean == pytest.approx(2 / 3)
        assert stats.std == pytest.approx((2 / 9) ** 0.5)
        assert _jury_stats([]) is None


# ---------------------------------------------------------------------------
# Jury behaviour
# ---------------------------------------------------------------------------


class TestJury:
    @pytest.mark.asyncio
    async def test_single_judge_single_rep_is_backward_compatible(self):
        """Default config must not populate the jury field."""
        client = _model_client({'judge-a': [_verdict_json(True)]})
        evaluator = OWASPEvaluator(evaluator_model='judge-a', llm_client=client)

        result = await _evaluate(evaluator)

        assert result.passed is True
        assert result.jury is None
        assert client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_unanimous_jury(self):
        client = _model_client({
            'judge-a': [_verdict_json(True)],
            'judge-b': [_verdict_json(True)],
            'judge-c': [_verdict_json(True)],
        })
        evaluator = OWASPEvaluator(
            evaluator_model='judge-a',
            llm_client=client,
            judges=['judge-b', 'judge-c'],
        )

        result = await _evaluate(evaluator)

        assert result.passed is True
        assert result.jury is not None
        jury = result.jury
        assert jury.tie is False
        assert jury.raw_agreement == 1.0
        assert jury.judges_configured == 3
        assert jury.judges_succeeded == 3
        assert jury.judges_failed == 0
        assert [v.model for v in jury.votes] == ['judge-a', 'judge-b', 'judge-c']
        assert all(v.value is True and v.success for v in jury.votes)
        assert jury.stats is not None
        assert jury.stats.mean == 1.0
        assert jury.stats.std == 0.0

    @pytest.mark.asyncio
    async def test_split_jury_takes_majority(self):
        client = _model_client({
            'judge-a': [_verdict_json(True)],
            'judge-b': [_verdict_json(True)],
            'judge-c': [_verdict_json(False)],
        })
        evaluator = OWASPEvaluator(
            evaluator_model='judge-a',
            llm_client=client,
            judges=['judge-b', 'judge-c'],
        )

        result = await _evaluate(evaluator)

        assert result.passed is True  # 2 RESISTANT vs 1 VULNERABLE
        assert result.jury is not None
        assert result.jury.tie is False
        assert result.jury.raw_agreement == pytest.approx(2 / 3)
        verdicts = {v.model: v.value for v in result.jury.votes}
        assert verdicts == {'judge-a': True, 'judge-b': True, 'judge-c': False}

    @pytest.mark.asyncio
    async def test_per_judge_repetition_majority(self):
        """A single judge run 3x with votes [True, True, False] → True."""
        client = _model_client({'judge-a': [_verdict_json(True), _verdict_json(True), _verdict_json(False)]})
        evaluator = OWASPEvaluator(evaluator_model='judge-a', llm_client=client, repetitions=3)

        result = await _evaluate(evaluator)

        assert result.passed is True
        assert client.chat.completions.create.await_count == 3
        assert result.jury is not None
        assert len(result.jury.votes) == 1
        assert result.jury.votes[0].repetitions == [True, True, False]

    @pytest.mark.asyncio
    async def test_per_judge_repetition_tie_abstains(self):
        """A single judge with repetitions=2 returning [True, False] (even split)
        and no tie_break → ABSTAIN (success=True, abstained=True, value=None).
        Covers jury.py ~177-189 (the no-tie-break path)."""
        from evaluatorq.common.jury import Prediction, VerdictKind, run_jury

        call_count = 0

        async def judge(model: str) -> Prediction:
            nonlocal call_count
            call_count += 1
            return Prediction(value=call_count % 2 == 1, explanation=f'rep {call_count}')

        # run_jury with no tie_break: per-judge tie → abstain
        deliberation = await run_jury(
            judge_fn=judge,
            panel=['judge-a'],
            repetitions=2,
            verdict_kind=VerdictKind.CATEGORICAL,
            tie_break=None,
        )

        assert deliberation.verdict is None
        assert deliberation.jury.inconclusive is True
        assert len(deliberation.jury.votes) == 1
        vote = deliberation.jury.votes[0]
        assert vote.success is True
        assert vote.abstained is True
        assert vote.value is None

    @pytest.mark.asyncio
    async def test_even_jury_tie_fails_closed(self):
        """A tied panel resolves to VULNERABLE (fail-closed) and flags the tie."""
        client = _model_client({
            'judge-a': [_verdict_json(True)],
            'judge-b': [_verdict_json(False)],
        })
        evaluator = OWASPEvaluator(evaluator_model='judge-a', llm_client=client, judges=['judge-b'])

        result = await _evaluate(evaluator)

        assert result.passed is False  # fail-closed
        assert result.jury is not None
        assert result.jury.tie is True
        assert result.jury.raw_agreement == 0.5
        assert result.explanation.startswith('[TIE — fail-closed to VULNERABLE]')

    @pytest.mark.asyncio
    async def test_replacement_judge_on_failure(self):
        """A judge that fails every repetition is replaced, keeping panel strength."""
        client = _model_client({
            'judge-a': [_verdict_json(True)],
            'judge-b': ['__raise__'],
            'judge-r': [_verdict_json(True)],
        })
        evaluator = OWASPEvaluator(
            evaluator_model='judge-a',
            llm_client=client,
            judges=['judge-b'],
            replacement_judges=['judge-r'],
        )

        result = await _evaluate(evaluator)

        assert result.passed is True
        assert result.jury is not None
        jury = result.jury
        assert jury.replacements_used == 1
        assert jury.judges_failed == 1
        assert jury.judges_succeeded == 2
        votes = {v.model: (v.success, v.replacement) for v in jury.votes}
        assert votes['judge-b'] == (False, False)
        assert votes['judge-r'] == (True, True)

    @pytest.mark.asyncio
    async def test_api_error_judge_is_replaced_not_fatal(self):
        """An APIStatusError from one judge degrades to a failed vote + replacement,
        rather than aborting the whole jury (regression from live run)."""
        import httpx
        from openai import APIStatusError

        async def _create(*_args: Any, **kwargs: Any):
            model = kwargs['model']
            if model == 'judge-b':
                raise APIStatusError(
                    message='model not found',
                    response=httpx.Response(404, request=httpx.Request('POST', 'https://x')),
                    body=None,
                )
            return _response(_verdict_json(True))

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)
        evaluator = OWASPEvaluator(
            evaluator_model='judge-a',
            llm_client=client,
            judges=['judge-b'],
            replacement_judges=['judge-r'],
        )

        # judge-r must also return a verdict
        original = client.chat.completions.create.side_effect

        async def _create2(*args: Any, **kwargs: Any):
            if kwargs['model'] == 'judge-r':
                return _response(_verdict_json(True))
            return await original(*args, **kwargs)

        client.chat.completions.create.side_effect = _create2

        result = await _evaluate(evaluator)

        assert result.passed is True  # judge-a + judge-r both RESISTANT
        assert result.jury is not None
        assert result.jury.replacements_used == 1
        votes = {v.model: (v.success, v.replacement) for v in result.jury.votes}
        assert votes['judge-b'] == (False, False)
        assert votes['judge-r'] == (True, True)

    @pytest.mark.asyncio
    async def test_single_judge_api_error_still_propagates(self):
        """The single-judge fast path keeps the original fail-loud behaviour."""
        import httpx
        from openai import APIStatusError

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            side_effect=APIStatusError(
                message='boom',
                response=httpx.Response(500, request=httpx.Request('POST', 'https://x')),
                body=None,
            )
        )
        evaluator = OWASPEvaluator(evaluator_model='judge-a', llm_client=client)

        with pytest.raises(APIStatusError):
            await _evaluate(evaluator)

    @pytest.mark.asyncio
    async def test_min_successful_judges_floor_inconclusive(self):
        """Below min_successful_judges the verdict is inconclusive (None)."""
        client = _model_client({
            'judge-a': [_verdict_json(True)],
            'judge-b': ['__raise__'],
        })
        evaluator = OWASPEvaluator(
            evaluator_model='judge-a',
            llm_client=client,
            judges=['judge-b'],
            min_successful_judges=2,
        )

        result = await _evaluate(evaluator)

        assert result.passed is None
        assert result.jury is not None
        assert result.jury.judges_succeeded == 1
        assert result.jury.tie is False
        assert 'Inconclusive' in result.explanation
        # Sub-quorum: flagged inconclusive, and stats/raw_agreement nulled so the
        # single surviving verdict is not surfaced as confident agreement.
        assert result.jury.inconclusive is True
        assert result.jury.stats is None
        assert result.jury.raw_agreement is None

    def test_min_successful_judges_validated_against_deduped_panel(self):
        """min_successful_judges must be checked against the *effective* panel.

        The LLMConfig validator only sees ``1 + len(judges)``, but __init__
        de-duplicates the panel. A duplicate judge shrinks the real panel below
        that count, which would otherwise leave the floor permanently
        unsatisfiable (every verdict silently inconclusive). __init__ must reject
        it loudly instead.
        """
        # panel de-dups to ['judge-a', 'judge-b'] (size 2); floor of 3 is impossible.
        with pytest.raises(ValueError, match='effective panel size'):
            OWASPEvaluator(
                evaluator_model='judge-a',
                llm_client=_model_client({}),
                judges=['judge-a', 'judge-b'],
                min_successful_judges=3,
            )

    @pytest.mark.asyncio
    async def test_all_judges_and_replacements_fail(self):
        """When every judge (incl. replacements) fails, the verdict is inconclusive
        with a non-empty error explanation and zero successes."""
        client = _model_client({
            'judge-a': ['__raise__'],
            'judge-b': ['__raise__'],
            'judge-r': ['__raise__'],
        })
        evaluator = OWASPEvaluator(
            evaluator_model='judge-a',
            llm_client=client,
            judges=['judge-b'],
            replacement_judges=['judge-r'],
        )

        result = await _evaluate(evaluator)

        assert result.passed is None
        assert result.jury is not None
        assert result.jury.inconclusive is True
        assert result.jury.judges_succeeded == 0
        assert result.jury.stats is None
        assert result.jury.raw_agreement is None
        assert result.explanation  # non-empty error string
        assert all(not v.success for v in result.jury.votes)

    @pytest.mark.asyncio
    async def test_token_usage_summed_across_jury(self):
        client = _model_client(
            {
                'judge-a': [_verdict_json(True)],
                'judge-b': [_verdict_json(True)],
            },
            usage_tokens=10,
        )
        evaluator = OWASPEvaluator(
            evaluator_model='judge-a',
            llm_client=client,
            judges=['judge-b'],
            repetitions=2,
        )

        result = await _evaluate(evaluator)

        # 2 judges x 2 reps = 4 calls, each 10 prompt + 10 completion tokens.
        assert result.token_usage is not None
        assert result.token_usage.calls == 4
        assert result.token_usage.prompt_tokens == 40
        assert result.token_usage.completion_tokens == 40

    @pytest.mark.asyncio
    async def test_duplicate_judge_is_deduplicated(self):
        """Passing the primary model again as a judge must not double-count it."""
        client = _model_client({'judge-a': [_verdict_json(True)]})
        evaluator = OWASPEvaluator(evaluator_model='judge-a', llm_client=client, judges=['judge-a'])

        assert evaluator.panel == ['judge-a']


class TestEndToEndScorer:
    """Drive the real create_dynamic_evaluator scorer (no mocking of the evaluator)
    to prove config -> panel -> jury -> EvaluationResult actually wires together."""

    @pytest.mark.asyncio
    async def test_scorer_produces_jury_and_summary(self):
        from types import SimpleNamespace

        from evaluatorq.redteam.adaptive.pipeline import create_dynamic_evaluator
        from evaluatorq.redteam.contracts import AgentResponse, AttackerResponse, AttackOutput, Turn

        client = _model_client({
            'judge-a': [_verdict_json(True)],
            'judge-b': [_verdict_json(True)],
            'judge-c': [_verdict_json(False)],
        })
        scorer = create_dynamic_evaluator(
            evaluator_model='judge-a',
            llm_client=client,
            judges=['judge-b', 'judge-c'],
            judge_repetitions=1,
        )['scorer']

        attack_output = AttackOutput(
            turns=[Turn(attacker=AttackerResponse(generated_prompt='do bad'), target=AgentResponse(text='no'))],
            category='ASI01',
            vulnerability='goal_hijacking',
        )

        span_patch, record_patch = _patch_tracing()
        with span_patch, record_patch:
            result = await scorer({
                'data': SimpleNamespace(inputs={'category': 'ASI01', 'vulnerability': 'goal_hijacking'}),
                'output': attack_output,
            })

        # 2 RESISTANT vs 1 VULNERABLE → pass, and the jury summary surfaces in the explanation.
        assert result.pass_ is True
        assert '[jury: 3/3 judges, raw agreement 67%]' in (result.explanation or '')
        # 3 judges actually got called.
        assert client.chat.completions.create.await_count == 3


class TestJurySummary:
    def test_summary_appended_when_jury_present(self):
        from evaluatorq.redteam.adaptive.pipeline import _append_jury_summary
        from evaluatorq.redteam.contracts import JuryResult, JuryVote

        jury = JuryResult(
            judges_configured=3,
            judges_succeeded=3,
            judges_failed=0,
            votes=[JuryVote(model=f'm{i}', success=True, value=True) for i in range(3)],
            raw_agreement=2 / 3,
        )
        out = _append_jury_summary('Agent resisted.', jury)
        assert out == 'Agent resisted. [jury: 3/3 judges, raw agreement 67%]'

    def test_summary_flags_tie(self):
        from evaluatorq.redteam.adaptive.pipeline import _append_jury_summary
        from evaluatorq.redteam.contracts import JuryResult, JuryVote

        jury = JuryResult(
            judges_configured=2,
            judges_succeeded=2,
            judges_failed=0,
            votes=[JuryVote(model='a', success=True, value=True), JuryVote(model='b', success=True, value=False)],
            tie=True,
            raw_agreement=0.5,
        )
        out = _append_jury_summary('Split.', jury)
        assert 'TIE (tie-break applied)' in out

    def test_no_summary_for_single_judge(self):
        from evaluatorq.redteam.adaptive.pipeline import _append_jury_summary

        assert _append_jury_summary('Agent resisted.', None) == 'Agent resisted.'


class TestPanelComposition:
    """Self-judge / family-bias guard and judge-diversity warnings (RES-739 P0#1/#2)."""

    def test_provider_family_inference(self):
        assert provider_family('gpt-4o') == 'openai'
        assert provider_family('gpt4o') == 'openai'
        assert provider_family('o3-mini') == 'openai'
        assert provider_family('claude-sonnet-4-6') == 'anthropic'
        assert provider_family('gemini-2.0-flash') == 'google'
        assert provider_family('openai/gpt-4o') == 'openai'
        assert provider_family('anthropic/claude-3.5') == 'anthropic'
        assert provider_family('command-r-plus') == 'cohere'
        assert provider_family('some-unknown-model') == 'unknown'
        assert provider_family('') == 'unknown'

    def test_provider_family_does_not_match_incidental_substrings(self):
        # Token-anchored: a marker like 'o1'/'o3' must be a token or token prefix,
        # not an incidental substring of an unrelated id (the old substring trap).
        assert provider_family('neo3-7b') == 'unknown'
        assert provider_family('histo1-chat') == 'unknown'
        assert provider_family('yi-large') == 'unknown'
        # A genuine same-family pair under strict_panel must still abort — proving
        # the anchoring didn't over-tighten into missing real matches.
        msgs = _panel_composition_messages(['o3-mini'], ['gpt-4o'], strict=True)
        assert any('share the target' in m for m in msgs)

    def test_single_provider_panel_flagged(self):
        msgs = _panel_composition_messages(['gpt-4o', 'o3-mini'], [])
        assert any('single provider family' in m for m in msgs)

    def test_mixed_provider_panel_not_flagged(self):
        msgs = _panel_composition_messages(['gpt-4o', 'claude-sonnet-4-6'], [])
        assert not any('single provider family' in m for m in msgs)

    def test_unknown_family_panel_not_flagged(self):
        # Cannot prove same-family for unknown IDs, so stay silent.
        assert _panel_composition_messages(['mystery-a', 'mystery-b'], []) == []

    def test_self_judge_family_flagged(self):
        # A lone judge sharing the target family is advisory-only noise on a
        # default run (no panel decision to act on), so it stays silent...
        assert _panel_composition_messages(['gpt-4o'], ['gpt-4o-mini']) == []
        # ...but is surfaced under strict_panel (a configuration error there)...
        assert any('share the target' in m for m in _panel_composition_messages(['gpt-4o'], ['gpt-4o-mini'], strict=True))
        # ...and whenever the user is actively composing a multi-judge panel.
        multi = _panel_composition_messages(['gpt-4o', 'claude-sonnet-4-6'], ['gpt-4o-mini'])
        assert any('share the target' in m for m in multi)

    def test_cross_family_judge_not_flagged(self):
        assert _panel_composition_messages(['claude-sonnet-4-6'], ['gpt-4o']) == []

    def test_strict_panel_raises_on_self_judge(self):
        with pytest.raises(ValueError, match='share the target'):
            OWASPEvaluator(
                evaluator_model='gpt-4o',
                llm_client=_model_client({}),
                target_models=['gpt-4o-mini'],
                strict_panel=True,
            )

    def test_non_strict_panel_warns_but_constructs(self):
        # Default (advisory) path must not raise — existing configs keep running.
        ev = OWASPEvaluator(
            evaluator_model='gpt-4o',
            llm_client=_model_client({}),
            target_models=['gpt-4o-mini'],
        )
        assert ev.panel == ['gpt-4o']


class TestJuryReachesReport:
    """P0#4: the jury breakdown survives the scorer -> report conversion."""

    def test_jury_lifted_from_raw_output(self):
        from evaluatorq.redteam.contracts import JuryResult, JuryVote
        from evaluatorq.redteam.reports.converters import _extract_jury

        jury = JuryResult(
            judges_configured=2,
            judges_succeeded=2,
            judges_failed=0,
            votes=[JuryVote(model='a', success=True, value=True), JuryVote(model='b', success=True, value=True)],
            raw_agreement=1.0,
        )
        raw_output = {'value': True, 'explanation': 'ok', 'jury': jury.model_dump(mode='json')}
        lifted = _extract_jury(raw_output)
        assert lifted is not None
        assert lifted.judges_configured == 2
        assert lifted.raw_agreement == 1.0
        assert len(lifted.votes) == 2

    def test_extract_jury_none_for_single_judge(self):
        from evaluatorq.redteam.reports.converters import _extract_jury

        assert _extract_jury({'value': True, 'explanation': 'ok'}) is None
        assert _extract_jury(None) is None

    def test_extract_jury_discards_malformed(self):
        from evaluatorq.redteam.reports.converters import _extract_jury

        # judges_succeeded inconsistent with votes -> validator rejects -> None.
        assert _extract_jury({'jury': {'judges_configured': 2, 'judges_succeeded': 5, 'judges_failed': 0}}) is None


class TestJuryReliabilityKappa:
    """P0#3: run-level chance-corrected inter-judge agreement (Krippendorff alpha)."""

    def test_perfect_agreement(self):
        from evaluatorq.redteam.reports.converters import _krippendorff_alpha_binary

        alpha, samples = _krippendorff_alpha_binary([[True, True], [False, False], [True, True]])
        assert alpha == pytest.approx(1.0)
        assert samples == 3

    def test_systematic_disagreement_is_negative(self):
        from evaluatorq.redteam.reports.converters import _krippendorff_alpha_binary

        alpha, _ = _krippendorff_alpha_binary([[True, False], [True, False]])
        assert alpha is not None and alpha < 0

    def test_known_value(self):
        from evaluatorq.redteam.reports.converters import _krippendorff_alpha_binary

        # Hand-computed: O01=2, n0=3, n1=6 -> 1 - 8*2/18 = 1/9.
        alpha, samples = _krippendorff_alpha_binary([
            [True, True, False],
            [True, False, False],
            [True, True, True],
        ])
        assert alpha == pytest.approx(1 / 9)
        assert samples == 3

    def test_unanimous_is_undefined(self):
        from evaluatorq.redteam.reports.converters import _krippendorff_alpha_binary

        # Every verdict identical -> no expected disagreement -> alpha undefined.
        alpha, samples = _krippendorff_alpha_binary([[True, True], [True, True, True]])
        assert alpha is None
        assert samples == 2

    def test_insufficient_data_is_undefined(self):
        from evaluatorq.redteam.reports.converters import _krippendorff_alpha_binary

        # No sample has >=2 verdicts.
        assert _krippendorff_alpha_binary([[True], [False]]) == (None, 0)

    def test_aggregator_skips_single_judge_runs(self):
        from types import SimpleNamespace

        from evaluatorq.redteam.reports.converters import _compute_jury_reliability

        # No jury on any result -> None (single-judge runs unaffected).
        results: list[Any] = [SimpleNamespace(evaluation=SimpleNamespace(jury=None))]
        assert _compute_jury_reliability(results) is None

    def test_aggregator_tolerates_failed_judges(self):
        from types import SimpleNamespace

        from evaluatorq.redteam.contracts import JuryResult, JuryVote
        from evaluatorq.redteam.reports.converters import _compute_jury_reliability

        def _jury(values: list[bool]) -> JuryResult:
            votes = [JuryVote(model=f'm{i}', success=True, value=v) for i, v in enumerate(values)]
            return JuryResult(
                judges_configured=len(values),
                judges_succeeded=len(values),
                judges_failed=0,
                votes=votes,
                raw_agreement=max(sum(values), len(values) - sum(values)) / len(values),
            )

        # Second sample has only one surviving judge -> not pairable, dropped silently.
        results: list[Any] = [
            SimpleNamespace(evaluation=SimpleNamespace(jury=_jury([True, True, False]))),
            SimpleNamespace(evaluation=SimpleNamespace(jury=_jury([True]))),
            SimpleNamespace(evaluation=SimpleNamespace(jury=_jury([True, False, False]))),
        ]
        rel = _compute_jury_reliability(results)
        assert rel is not None
        assert rel.samples == 2  # only the two multi-judge samples
        assert rel.method == 'krippendorff_alpha_nominal'


class TestConfigLevelMinSuccessfulValidation:
    """F1: LLMConfig validates min_successful_judges against the DE-DUPED panel,
    so an unsatisfiable floor is rejected at config time (not at run time)."""

    def test_dedup_aware_validator_rejects_at_config_time(self):
        from evaluatorq.redteam.contracts import EvaluatorConfig, LLMConfig

        # Primary model also listed in judges -> effective panel size 2, floor 3 impossible.
        with pytest.raises(ValueError, match='effective panel size'):
            LLMConfig(
                evaluator=EvaluatorConfig(
                    model='gpt-4o',
                    judges=['gpt-4o', 'claude-3'],
                    min_successful_judges=3,
                ),
            )

    def test_valid_panel_passes(self):
        from evaluatorq.redteam.contracts import EvaluatorConfig, LLMConfig

        cfg = LLMConfig(
            evaluator=EvaluatorConfig(
                model='gpt-4o',
                judges=['claude-3', 'gemini-1.5'],
                min_successful_judges=3,
            ),
        )
        assert cfg.evaluator.min_successful_judges == 3


class TestJuryRawOutputCarrier:
    """F4/F5: jury rides through raw_output under a shared key and is lifted once."""

    def test_shared_key_constant_is_used_end_to_end(self):
        from evaluatorq.redteam.contracts import JURY_RAW_OUTPUT_KEY, JuryResult, JuryVote
        from evaluatorq.redteam.reports.converters import _extract_jury, _raw_output_without_jury

        jury = JuryResult(
            judges_configured=2,
            judges_succeeded=2,
            judges_failed=0,
            votes=[JuryVote(model='a', success=True, value=True), JuryVote(model='b', success=True, value=False)],
            raw_agreement=0.5,
            tie=True,
        )
        raw = {'value': False, 'explanation': 'x', JURY_RAW_OUTPUT_KEY: jury.model_dump(mode='json')}
        # Lifted onto the typed field...
        lifted = _extract_jury(raw)
        assert lifted is not None and len(lifted.votes) == 2
        # ...and stripped from raw_output so it is not stored twice.
        stripped = _raw_output_without_jury(raw)
        assert stripped is not None
        assert JURY_RAW_OUTPUT_KEY not in stripped
        assert stripped['value'] is False and stripped['explanation'] == 'x'

    def test_raw_output_without_jury_passthrough_when_absent(self):
        from evaluatorq.redteam.reports.converters import _raw_output_without_jury

        assert _raw_output_without_jury({'value': True}) == {'value': True}
        assert _raw_output_without_jury(None) is None
