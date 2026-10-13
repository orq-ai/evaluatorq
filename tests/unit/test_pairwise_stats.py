from __future__ import annotations

import pytest

from evaluatorq.contracts import Usage
from evaluatorq.pairwise import PairwiseComparison, PairwiseVote
from evaluatorq.pairwise_reports.stats import (
    comparison_flags,
    fleiss_kappa,
    judge_usage,
    kappa_label,
    length_lean,
    outcome_counts,
    trust_signals,
    verdict,
    wilson_interval,
)
from evaluatorq.pairwise_run import new_run

J = ['j1', 'j2', 'j3']


def _cmp(values, winner, *, flipped=(), failures=0, usage=None):
    votes = [
        PairwiseVote(model=m, vote=None if m in flipped else v, flipped=m in flipped, repetition_failures=failures)
        for m, v in zip(J, values)
    ]
    return PairwiseComparison(winner=winner, votes=votes, token_usage=usage)


def _run(rows, *, judges=J):
    r = new_run(run_name='r', label_a='left', label_b='right', judges=judges)
    for i, (values, winner, kw) in enumerate(rows):
        kw = dict(kw)
        ra, rb = kw.pop('ra', 'a'), kw.pop('rb', 'bb')
        r.add(_cmp(values, winner, **kw), question=f'q{i}', response_a=ra, response_b=rb)
    return r


def test_wilson_interval_with_no_trials_is_the_whole_range():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(18, 30)
    assert lo < 0.6 < hi
    assert lo == pytest.approx(0.4232, abs=1e-3)
    assert hi == pytest.approx(0.7541, abs=1e-3)


def test_verdict_is_too_close_when_the_interval_crosses_half():
    r = _run([(['B'] * 3, 'B', {})] * 18 + [(['A'] * 3, 'A', {})] * 12)
    v = verdict(r)
    assert (v.state, v.leader, v.leader_label, v.decided) == ('too_close', 'B', 'right', 30)
    assert v.win_rate == pytest.approx(0.6)


def test_verdict_is_clear_when_the_lower_bound_is_above_half():
    r = _run([(['B'] * 3, 'B', {})] * 40 + [(['A'] * 3, 'A', {})] * 5)
    assert verdict(r).state == 'clear'


def test_verdict_with_equal_wins_names_no_leader():
    r = _run([(['A'] * 3, 'A', {}), (['B'] * 3, 'B', {})])
    v = verdict(r)
    assert (v.state, v.leader, v.win_rate) == ('too_close', None, 0.5)


def test_verdict_without_decided_comparisons():
    r = _run([(['tie'] * 3, 'tie', {}), ([None] * 3, 'inconclusive', {})])
    v = verdict(r)
    assert (v.state, v.leader, v.win_rate, v.low, v.high) == ('no_decided', None, None, None, None)
    assert (v.ties, v.inconclusive, v.total) == (1, 1, 2)


def test_verdict_of_an_empty_run():
    assert verdict(_run([])).state == 'no_decided'


def test_fleiss_kappa_is_one_for_perfect_agreement_across_categories():
    r = _run([(['A'] * 3, 'A', {}), (['B'] * 3, 'B', {}), (['tie'] * 3, 'tie', {})])
    assert fleiss_kappa(r) == (pytest.approx(1.0), 3)


def test_fleiss_kappa_skips_rows_without_a_full_panel():
    r = _run([(['A', 'A', 'B'], 'A', {'flipped': ('j3',)})])
    assert fleiss_kappa(r) == (None, 0)


def test_fleiss_kappa_needs_two_judges():
    r = new_run(run_name='r', judges=['solo'])
    r.add(
        PairwiseComparison(winner='A', votes=[PairwiseVote(model='solo', vote='A')]),
        question='q',
        response_a='a',
        response_b='b',
    )
    assert fleiss_kappa(r) == (None, 0)


@pytest.mark.parametrize(
    ('k', 'label'),
    [(-0.1, 'poor'), (0.1, 'slight'), (0.3, 'fair'), (0.5, 'moderate'), (0.7, 'substantial'), (0.85, 'almost perfect')],
)
def test_kappa_label_bands(k, label):
    assert kappa_label(k) == label


def test_length_lean_counts_the_longer_answer_winning():
    r = _run([
        (['B'] * 3, 'B', {'ra': 'a', 'rb': 'bbbb'}),
        (['A'] * 3, 'A', {'ra': 'aaaa', 'rb': 'b'}),
        (['A'] * 3, 'A', {'ra': 'a', 'rb': 'bbbb'}),
    ])
    lean = length_lean(r)
    assert lean is not None
    assert (lean.longer_wins, lean.compared) == (2, 3)


def test_length_lean_is_none_when_lengths_never_differ():
    r = _run([(['A'] * 3, 'A', {'ra': 'xx', 'rb': 'yy'})])
    assert length_lean(r) is None


def test_comparison_flags():
    r = _run([(['A', 'B', 'A'], 'A', {'flipped': ('j3',), 'failures': 1})])
    f = comparison_flags(r.entries[0], r.judges)
    assert (f.split, f.thrown_out, f.judge_failed) == (True, True, True)


def test_comparison_flags_marks_a_missing_configured_judge_as_failed():
    r = new_run(run_name='r', judges=J)
    r.add(
        PairwiseComparison(winner='A', votes=[PairwiseVote(model='j1', vote='A')]),
        question='q',
        response_a='a',
        response_b='b',
    )
    assert comparison_flags(r.entries[0], r.judges).judge_failed is True


def test_outcome_counts():
    r = _run([(['A'] * 3, 'A', {}), (['tie'] * 3, 'tie', {}), ([None] * 3, 'inconclusive', {})])
    assert outcome_counts(r) == {'A': 1, 'B': 0, 'tie': 1, 'inconclusive': 1}


def test_judge_usage_sums_tokens_and_keeps_unknown_cost_unknown():
    u1 = Usage(input_tokens=10, output_tokens=2, total_tokens=12, total_cost=0.5)
    u2 = Usage(input_tokens=5, output_tokens=1, total_tokens=6)
    r = _run([(['A'] * 3, 'A', {'usage': u1}), (['B'] * 3, 'B', {'usage': u2}), (['A'] * 3, 'A', {})])
    t = judge_usage(r)
    assert (t.input_tokens, t.output_tokens, t.total_tokens, t.with_usage) == (15, 3, 18, 2)
    assert t.cost == pytest.approx(0.5)
    assert judge_usage(_run([(['A'] * 3, 'A', {})])).cost is None


def test_trust_signals_order_and_tones():
    rows = [(['A'] * 3, 'A', {})] * 8 + [([None] * 3, 'inconclusive', {'flipped': tuple(J)})] * 2
    sig = trust_signals(_run(rows))
    assert [s.label for s in sig] == [
        'Decided',
        'Votes thrown out',
        'Judge agreement',
        'Position bias',
        'Length',
        'Judge failures',
    ]
    by = {s.label: s for s in sig}
    assert by['Decided'].tone == 'good'
    assert by['Votes thrown out'].tone == 'warn'
    assert by['Judge failures'].tone == 'good'


def test_trust_signals_omit_position_bias_when_swap_was_never_observed():
    r = new_run(run_name='r', judges=J, swap=False)
    r.add(
        PairwiseComparison(winner='A', votes=[PairwiseVote(model=m, vote='A', completed=False) for m in J]),
        question='q',
        response_a='a',
        response_b='b',
    )
    assert 'Position bias' not in [s.label for s in trust_signals(r)]
