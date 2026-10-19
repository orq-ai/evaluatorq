from __future__ import annotations

from typing import Literal

import pytest

from evaluatorq.contracts import Usage
from evaluatorq.pairwise import PairwiseComparison, PairwiseVote
from evaluatorq.pairwise_reports.stats import (
    comparison_flags,
    fleiss_kappa,
    generation_metrics,
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


def test_fleiss_kappa_is_measured_over_the_panel_despite_a_stand_in():
    r = _run([(['A'] * 3, 'A', {}), (['B'] * 3, 'B', {}), (['tie'] * 3, 'tie', {})])
    # j3 failed here and the stand-in s1 voted instead, as run_pairwise records it.
    r.add(
        PairwiseComparison(
            winner='A',
            votes=[
                PairwiseVote(model='j1', vote='A'),
                PairwiseVote(model='j2', vote='A'),
                PairwiseVote(model='j3', vote=None),
                PairwiseVote(model='s1', vote='A'),
            ],
        ),
        question='q-standin',
        response_a='a',
        response_b='bb',
    )
    assert fleiss_kappa(r) == (pytest.approx(1.0), 3)


@pytest.mark.parametrize(
    'duplicate_votes',
    [
        [('j1', 'A'), ('j1', None)],
        [('j1', None), ('j1', 'A')],
        [('j1', 'A'), ('j1', 'B')],
    ],
)
def test_fleiss_kappa_excludes_rows_with_duplicate_panel_votes(
    duplicate_votes: list[tuple[str, Literal['A', 'B', 'tie'] | None]],
):
    r = new_run(run_name='r', judges=J)
    rows: list[list[tuple[str, Literal['A', 'B', 'tie'] | None]]] = [
        [('j1', 'A'), ('j2', 'A'), ('j3', 'A')],
        [('j1', 'B'), ('j2', 'B'), ('j3', 'B')],
        [*duplicate_votes, ('j2', 'A'), ('j3', 'A')],
    ]
    for i, votes in enumerate(rows):
        r.add(
            PairwiseComparison(
                winner='A',
                votes=[PairwiseVote(model=model, vote=vote) for model, vote in votes],
            ),
            question=f'q{i}',
            response_a='a',
            response_b='b',
        )

    assert fleiss_kappa(r) == (pytest.approx(1.0), 2)


def test_fleiss_kappa_is_undefined_for_unanimous_single_category_panels():
    r = _run([(['A'] * 3, 'A', {}), (['A'] * 3, 'A', {})])

    assert fleiss_kappa(r) == (None, 2)


def test_fleiss_kappa_matches_known_mixed_agreement_example():
    r = new_run(run_name='r', judges=['j1', 'j2'])
    ballots: list[list[Literal['A', 'B', 'tie']]] = [['A', 'A'], ['A', 'B']]
    for i, votes in enumerate(ballots):
        r.add(
            PairwiseComparison(
                winner='A',
                votes=[PairwiseVote(model=f'j{j + 1}', vote=vote) for j, vote in enumerate(votes)],
            ),
            question=f'q{i}',
            response_a='a',
            response_b='b',
        )

    assert fleiss_kappa(r) == (pytest.approx(-1 / 3), 2)


def test_fleiss_kappa_needs_two_judges():
    r = new_run(run_name='r', judges=['solo'])
    r.add(
        PairwiseComparison(winner='A', votes=[PairwiseVote(model='solo', vote='A')]),
        question='q',
        response_a='a',
        response_b='b',
    )
    assert fleiss_kappa(r) == (None, 0)


def test_fleiss_kappa_does_not_infer_a_panel_from_replacement_votes():
    r = new_run(run_name='legacy', judges=[])
    r.add(
        PairwiseComparison(
            winner='A',
            votes=[
                PairwiseVote(model='stand-in-1', vote='A', replacement=True),
                PairwiseVote(model='stand-in-2', vote='A', replacement=True),
            ],
        ),
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
    assert t.cost_partial is True
    assert judge_usage(_run([(['A'] * 3, 'A', {})])).cost is None
    assert judge_usage(_run([(['A'] * 3, 'A', {'usage': u1})])).cost == pytest.approx(0.5)
    partial = judge_usage(_run([(['A'] * 3, 'A', {'usage': u1}), (['B'] * 3, 'B', {})]))
    assert partial.cost == pytest.approx(0.5) and partial.cost_partial


def test_generation_metrics_keep_metric_coverage_and_same_entry_pairs():
    from evaluatorq.pairwise_run import PairwiseResponseMetrics

    run = _run([(['A'] * 3, 'A', {}), (['B'] * 3, 'B', {})])
    run.entries[0].metrics_a = PairwiseResponseMetrics(
        usage=Usage(total_cost=0.0004, calls=1, priced_calls=1), latency_ms=0
    )
    run.entries[0].metrics_b = PairwiseResponseMetrics(
        usage=Usage(total_cost=0.0008, calls=1, priced_calls=1), latency_ms=20
    )
    run.entries[1].metrics_a = PairwiseResponseMetrics(
        usage=Usage(total_cost=0.0005, calls=1, priced_calls=1), latency_ms=30
    )
    run.entries[1].metrics_b = None

    totals = generation_metrics(run)
    assert (totals.a_cost, totals.a_cost_recorded, totals.a_cost_partial) == (0.0009, 2, False)
    assert (totals.b_cost, totals.b_cost_recorded, totals.b_cost_partial) == (0.0008, 1, True)
    assert (totals.a_latency_ms, totals.a_latency_recorded) == (30, 2)
    assert (totals.b_latency_ms, totals.b_latency_recorded) == (20, 1)
    assert totals.paired_cost_entries == totals.paired_latency_entries == 1


def test_generation_cost_coverage_honors_unknown_and_partial_prices():
    from evaluatorq.pairwise_run import PairwiseResponseMetrics

    run = _run([(['A'] * 3, 'A', {}), (['B'] * 3, 'B', {}), (['A'] * 3, 'A', {})])
    run.entries[0].metrics_a = PairwiseResponseMetrics(usage=Usage(total_cost=0.2))
    run.entries[1].metrics_a = PairwiseResponseMetrics(usage=Usage(total_cost=0.3, calls=2, priced_calls=1))
    run.entries[2].metrics_a = PairwiseResponseMetrics(usage=Usage(total_cost=0.0))
    totals = generation_metrics(run)
    assert totals.a_cost == pytest.approx(0.5)
    assert totals.a_cost_recorded == 3
    assert totals.a_cost_partial is True
    run.entries[0].metrics_a = PairwiseResponseMetrics(usage=Usage(total_cost=0.2, calls=2, priced_calls=1))
    assert generation_metrics(run).a_cost_partial is True

def test_generation_metrics_reaggregate_only_filtered_entries():
    from evaluatorq.pairwise_run import PairwiseResponseMetrics

    run = _run([(['A'] * 3, 'A', {}), (['B'] * 3, 'B', {})])
    run.entries[0].metrics_a = PairwiseResponseMetrics(usage=Usage(total_cost=0.1), latency_ms=10)
    run.entries[1].metrics_a = PairwiseResponseMetrics(usage=Usage(total_cost=0.9), latency_ms=90)
    filtered = generation_metrics(run.model_copy(update={'entries': [run.entries[1]]}))
    assert (filtered.a_cost, filtered.a_cost_recorded, filtered.a_latency_ms, filtered.a_latency_recorded) == (
        0.9, 1, 90, 1
    )


def test_judge_usage_exposes_recorded_subtotal_but_not_invents_missing_price():
    usage = Usage(input_tokens=10, output_tokens=2, total_cost=0.001)
    run = _run([(['A'] * 3, 'A', {'usage': usage}), (['B'] * 3, 'B', {})])
    totals = judge_usage(run)
    assert totals.cost == pytest.approx(0.001)
    assert totals.cost_recorded == 1
    assert totals.cost_partial is True


@pytest.mark.parametrize('calls, priced_calls, partial', [(0, 0, True), (2, 0, True), (2, 1, True), (2, 2, False)])
def test_cost_totals_require_known_complete_call_pricing(calls, priced_calls, partial):
    from evaluatorq.pairwise_run import PairwiseResponseMetrics

    usage = Usage(total_cost=0.3, calls=calls, priced_calls=priced_calls)
    run = _run([(['A'] * 3, 'A', {'usage': usage})])
    run.entries[0].metrics_a = PairwiseResponseMetrics(usage=usage)
    run.entries[0].metrics_b = PairwiseResponseMetrics(usage=usage)
    generation = generation_metrics(run)
    judging = judge_usage(run)
    assert generation.a_cost == generation.b_cost == pytest.approx(0.3)
    assert generation.a_cost_partial is generation.b_cost_partial is partial
    assert generation.paired_cost_entries == (0 if partial else 1)
    assert judging.cost == pytest.approx(0.3)
    assert judging.cost_partial is partial
    assert judging.cost_complete is not partial


def test_equal_side_counts_do_not_make_disjoint_measurements_paired():
    from evaluatorq.pairwise_run import PairwiseResponseMetrics

    run = _run([(['A'] * 3, 'A', {}), (['B'] * 3, 'B', {})])
    run.entries[0].metrics_a = PairwiseResponseMetrics(
        usage=Usage(total_cost=0.2, calls=1, priced_calls=1), latency_ms=10
    )
    run.entries[1].metrics_b = PairwiseResponseMetrics(
        usage=Usage(total_cost=0.3, calls=1, priced_calls=1), latency_ms=20
    )
    totals = generation_metrics(run)
    assert (totals.a_cost_recorded, totals.b_cost_recorded) == (1, 1)
    assert (totals.a_latency_recorded, totals.b_latency_recorded) == (1, 1)
    assert totals.paired_cost_entries == totals.paired_latency_entries == 0
    assert totals.a_cost_partial is totals.b_cost_partial is True


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


def test_judge_agreement_signal_explains_unanimous_single_outcome_panel():
    sig = {s.label: s for s in trust_signals(_run([(['A'] * 3, 'A', {}), (['A'] * 3, 'A', {})]))}
    row = sig['Judge agreement']
    assert 'two or more judges' not in row.detail
    assert '2' in row.detail
    assert 'same outcome' in row.detail
    assert row.tone == ''


def test_position_bias_row_warns_and_counts_only_measured_judges():
    def entry(j1_flipped: bool):
        votes = [
            PairwiseVote(model='j1', vote=None if j1_flipped else 'A', flipped=j1_flipped, completed=True),
            PairwiseVote(model='j2', vote='A', completed=True),
            # Never completed both orderings, so a flip was never observable for j3.
            PairwiseVote(model='j3', vote='A', completed=False),
        ]
        return PairwiseComparison(winner='A', votes=votes)

    r = new_run(run_name='r', judges=J)
    for i, flipped in enumerate([True, False, False, False]):
        r.add(entry(flipped), question=f'q{i}', response_a='a', response_b='bb')
    row = next(s for s in trust_signals(r) if s.label == 'Position bias')
    assert row.tone == 'warn'
    # j1 flips on 1 of 4 completed pairs (0.25 >= the cut-off); j3 is unmeasured and not counted.
    assert row.value == '1 of 2 judges'


def test_length_row_reads_as_one_sentence():
    r = _run([(['B'] * 3, 'B', {'ra': 'a', 'rb': 'bbbb'}), (['A'] * 3, 'A', {'ra': 'aaaa', 'rb': 'b'})])
    row = next(s for s in trust_signals(r) if s.label == 'Length')
    assert row.value == 'longer won 100%'
    assert row.tone == ''
    assert row.detail.startswith('2 of 2 decided comparisons with answers of different length')
    assert 'includes 50%' in row.detail


def test_length_row_warns_when_the_lean_is_beyond_chance():
    r = _run([(['B'] * 3, 'B', {'ra': 'a', 'rb': 'bbbb'})] * 12)
    row = next(s for s in trust_signals(r) if s.label == 'Length')
    assert row.value == 'longer won 100%'
    assert row.tone == 'warn'
    assert 'judges may favour longer answers' in row.detail
