"""TDD tests for SimulationEntry / CriteriaRow / TranscriptMessage + individual_entries().

T-1: byte-identity — key order from model_dump(mode='json') must match the
     hand-built dict that _build_individual_results_section used to emit.
T-2: fallback — when criteria_meta is absent the fallback path still populates
     the same key structure.
"""

from __future__ import annotations

import pytest

from evaluatorq.contracts import Message, TokenUsage
from evaluatorq.simulation.reports.sections import individual_entries
from evaluatorq.simulation.types import SimulationResult, TerminatedBy, TurnMetrics


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

EXPECTED_KEY_ORDER = [
    'index',
    'persona',
    'scenario',
    'model',
    'target_model',
    'terminated_by',
    'goal_achieved',
    'goal_completion_score',
    'rules_broken',
    'criteria',
    'turn_count',
    'total_tokens',
    'judge_reason',
    'error',
    'evaluator_scores',
    'transcript',
]

EXPECTED_CRITERIA_KEY_ORDER = ['id', 'description', 'type', 'passed', 'safety']
EXPECTED_TRANSCRIPT_KEY_ORDER = ['role', 'content']


def _make_result(
    *,
    persona: str = 'Alice',
    scenario: str = 'Billing',
    model: str = 'gpt-4o',
    target_model: str | None = 'gpt-4o-mini',
    goal_achieved: bool = True,
    goal_completion_score: float = 0.9,
    terminated_by: TerminatedBy = TerminatedBy.judge,
    rules_broken: list[str] | None = None,
    criteria_meta: list[dict] | None = None,
    turn_count: int = 3,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    reason: str = 'goal achieved',
    error: str | None = None,
    evaluator_scores: dict[str, float] | None = None,
    messages: list[Message] | None = None,
) -> SimulationResult:
    metadata: dict[str, object] = {
        'persona': persona,
        'scenario': scenario,
        'model': model,
    }
    if target_model is not None:
        metadata['target_model'] = target_model
    if criteria_meta is not None:
        metadata['criteria_meta'] = criteria_meta
    if error is not None:
        metadata['error'] = error
    if evaluator_scores is not None:
        metadata['evaluator_scores'] = evaluator_scores
    if messages is None:
        messages = [
            Message(role='user', content='hello'),
            Message(role='assistant', content='hi there'),
        ]
    return SimulationResult(
        messages=messages,
        terminated_by=terminated_by,
        reason=reason,
        goal_achieved=goal_achieved,
        goal_completion_score=goal_completion_score,
        rules_broken=rules_broken or [],
        turn_count=turn_count,
        token_usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        turn_metrics=[TurnMetrics(turn_number=1, token_usage=TokenUsage(), judge_reason='ok')],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# T-1: byte-identity — key order matches the hand-built dict exactly
# ---------------------------------------------------------------------------


def test_individual_entries_top_level_key_order():
    """model_dump keys must be in the exact same order as the hand-built dict."""
    result = _make_result(
        criteria_meta=[
            {'id': 'c0', 'description': 'must greet', 'type': 'must_happen', 'passed': True},
        ]
    )
    entries = individual_entries([result])
    dumped = entries[0].model_dump(mode='json')
    assert list(dumped.keys()) == EXPECTED_KEY_ORDER


def test_individual_entries_criteria_key_order():
    """CriteriaRow.model_dump keys must match _criteria_rows() dict keys."""
    result = _make_result(
        criteria_meta=[
            {'id': 'c0', 'description': 'greet user', 'type': 'must_happen', 'passed': True},
            {'id': 'c1', 'description': 'no insults', 'type': 'must_not_happen', 'passed': True},
        ]
    )
    entries = individual_entries([result])
    dumped = entries[0].model_dump(mode='json')
    for row in dumped['criteria']:
        assert list(row.keys()) == EXPECTED_CRITERIA_KEY_ORDER


def test_individual_entries_transcript_key_order():
    """TranscriptMessage.model_dump keys must match the hand-built dict keys."""
    result = _make_result()
    entries = individual_entries([result])
    dumped = entries[0].model_dump(mode='json')
    for msg in dumped['transcript']:
        assert list(msg.keys()) == EXPECTED_TRANSCRIPT_KEY_ORDER


def test_individual_entries_field_values():
    """Spot-check that values match what the old hand-built dict would produce."""
    result = _make_result(
        persona='Alice',
        scenario='Billing',
        model='gpt-4o',
        target_model='gpt-4o-mini',
        terminated_by=TerminatedBy.max_turns,
        goal_achieved=False,
        goal_completion_score=0.3,
        rules_broken=['rule-a'],
        criteria_meta=[
            {
                'id': 'c0',
                'description': 'no insults',
                'type': 'must_not_happen',
                'passed': False,  # safety violation
            }
        ],
        turn_count=5,
        total_tokens=42,
        reason='max turns reached',
        error=None,
        evaluator_scores={'goal_achieved': 0.0},
    )
    entries = individual_entries([result])
    d = entries[0].model_dump(mode='json')

    assert d['index'] == 0
    assert d['persona'] == 'Alice'
    assert d['scenario'] == 'Billing'
    assert d['model'] == 'gpt-4o'
    assert d['target_model'] == 'gpt-4o-mini'
    # terminated_by must be a plain str, not an enum repr
    assert d['terminated_by'] == 'max_turns'
    assert isinstance(d['terminated_by'], str)
    assert d['goal_achieved'] is False
    assert d['goal_completion_score'] == 0.3
    assert d['rules_broken'] == ['rule-a']
    assert d['turn_count'] == 5
    assert d['total_tokens'] == 42
    assert d['judge_reason'] == 'max turns reached'
    assert d['error'] is None
    assert d['evaluator_scores'] == {'goal_achieved': 0.0}

    crit = d['criteria'][0]
    assert crit['id'] == 'c0'
    assert crit['description'] == 'no insults'
    assert crit['type'] == 'must_not_happen'
    assert crit['passed'] is False
    assert crit['safety'] is True  # must_not_happen + failed = safety violation

    assert d['transcript'][0] == {'role': 'user', 'content': 'hello'}
    assert d['transcript'][1] == {'role': 'assistant', 'content': 'hi there'}


def test_individual_entries_index_is_zero_based():
    """index is 0-based as in the original hand-built dict."""
    results = [_make_result(), _make_result(persona='Bob')]
    entries = individual_entries(results)
    assert entries[0].model_dump(mode='json')['index'] == 0
    assert entries[1].model_dump(mode='json')['index'] == 1


def test_individual_entries_target_model_none_when_absent():
    """target_model serialises as None when not in metadata."""
    result = _make_result(target_model=None)
    entries = individual_entries([result])
    d = entries[0].model_dump(mode='json')
    assert d['target_model'] is None


def test_individual_entries_terminated_by_is_str_not_enum():
    """terminated_by must be a JSON-scalar str, matching .value behaviour."""
    for tb in TerminatedBy:
        result = _make_result(terminated_by=tb)
        entries = individual_entries([result])
        d = entries[0].model_dump(mode='json')
        assert d['terminated_by'] == tb.value
        assert isinstance(d['terminated_by'], str)


# ---------------------------------------------------------------------------
# T-2: fallback — criteria_meta absent → criteria_results fallback path
# ---------------------------------------------------------------------------


def test_individual_entries_fallback_without_criteria_meta():
    """When criteria_meta is absent, fallback to criteria_results.

    The fallback still populates the same 5-key structure per row;
    'type' is None and 'safety' is False (classification unavailable).
    """
    result = SimulationResult(
        messages=[Message(role='user', content='hi')],
        terminated_by=TerminatedBy.judge,
        reason='done',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        turn_metrics=[],
        metadata={'persona': 'X', 'scenario': 'Y'},
        criteria_results={'must greet user': True, 'no profanity': False},
    )
    entries = individual_entries([result])
    d = entries[0].model_dump(mode='json')

    # key order still holds
    assert list(d.keys()) == EXPECTED_KEY_ORDER
    # criteria present and each row has the right keys
    assert len(d['criteria']) == 2
    for row in d['criteria']:
        assert list(row.keys()) == EXPECTED_CRITERIA_KEY_ORDER
    # type is None (not available in fallback), safety is False
    for row in d['criteria']:
        assert row['type'] is None
        assert row['safety'] is False


# ---------------------------------------------------------------------------
# Wiring test: _build_individual_results_section uses model_dump output
# ---------------------------------------------------------------------------


def test_section_data_entries_matches_model_dump():
    """After rewiring, section.data['entries'][i] == individual_entries()[i].model_dump(mode='json')."""
    from evaluatorq.simulation.reports.sections import build_report_sections

    result = _make_result(
        criteria_meta=[
            {'id': 'c0', 'description': 'greet', 'type': 'must_happen', 'passed': True}
        ]
    )
    sections = build_report_sections([result])
    ind = next(s for s in sections if s.kind == 'individual_results')
    expected = individual_entries([result])[0].model_dump(mode='json')
    assert ind.data['entries'][0] == expected
