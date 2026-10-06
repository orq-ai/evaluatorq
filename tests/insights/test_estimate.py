"""Unit tests for `evaluatorq.insights.estimate`: trace bounds, token constants, cost and time."""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from evaluatorq.common.model_catalogue import ModelInfo
from evaluatorq.common.run_manifest import list_manifests, start_manifest
from evaluatorq.contracts import ManifestStatus, StageRecord
from evaluatorq.insights import estimate, summarize, transcript
from evaluatorq.insights.estimate import (
    TOKENS_PER_TRACE,
    RunEstimate,
    StageEstimate,
    StageModels,
    TraceBound,
    estimate_run,
    stage_seconds,
    StageTiming,
    trace_bound,
)
from evaluatorq.trace_finder.models import FacetCatalogue, FacetSelection

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from evaluatorq.insights.models import DimensionName

CATALOGUE = FacetCatalogue(
    status=('completed', 'error'),
    model=('gpt-5', 'claude'),
    value_counts={
        'status': {'completed': 90, 'error': 30},
        'model': {'gpt-5': 70, 'claude': 40},
    },
)
MODELS = StageModels(summary='summary-m', classifier='classifier-m', embedding='embedding-m')


def _bound(selection: FacetSelection | None = None, *, catalogue: FacetCatalogue = CATALOGUE, limit: int = 500) -> TraceBound:
    return trace_bound(
        'recent',
        catalogue=catalogue,
        facets=selection or FacetSelection(),
        limit=limit,
        finder_count=None,
        snapshot_count=None,
    )


def _prices() -> dict[str, ModelInfo | None]:
    return {
        'summary-m': ModelInfo(0.01, 0.03, 'p', supports_responses=True),
        'classifier-m': ModelInfo(0.002, 0.004, 'p', supports_responses=False),
        'embedding-m': ModelInfo(0.0001, 0.0, 'p', supports_responses=False),
    }


def _estimate(
    *,
    bound: TraceBound | None = None,
    dimensions: tuple[DimensionName, ...] = ('intent',),
    question_count: int = 1,
    coding: bool = False,
    prices: dict[str, ModelInfo | None] | None = None,
    seconds: Mapping[str, StageTiming] | None = None,
    parallelism: int = 10,
    query: bool = False,
    sentiment_selected: bool = False,
) -> RunEstimate:
    return estimate_run(
        bound=bound or TraceBound(n=100, exact=True, basis='x'),
        dimensions=dimensions,
        question_count=question_count,
        coding=coding,
        models=MODELS,
        prices=_prices() if prices is None else prices,
        seconds=seconds or {},
        parallelism=parallelism,
        query=query,
        sentiment_selected=sentiment_selected,
    )


def _f(value: float | None) -> float:
    assert value is not None
    return value


def _row(result: RunEstimate, name: str) -> StageEstimate:
    return next(row for row in result.rows if row.name == name)


def test_no_filter_bound_sums_status_counts() -> None:
    bound = _bound()
    assert (bound.n, bound.exact) == (120, False)
    assert 'status' in bound.basis


def test_one_facet_bound_sums_its_selected_values() -> None:
    assert _bound(FacetSelection(model=frozenset({'gpt-5'}))).n == 70
    assert _bound(FacetSelection(model=frozenset({'gpt-5', 'claude'}))).n == 110


def test_two_facets_take_the_smaller_sum() -> None:
    bound = _bound(FacetSelection(model=frozenset({'gpt-5'}), status=frozenset({'error'})))
    assert bound.n == 30


def test_truncated_facet_falls_back_to_the_limit() -> None:
    truncated = CATALOGUE.model_copy(update={'truncated_facets': frozenset({'model'})})
    bound = _bound(FacetSelection(model=frozenset({'gpt-5'})), catalogue=truncated, limit=200)
    assert bound.n == 200
    assert 'limit' in bound.basis


def test_facet_without_counts_falls_back_to_the_limit() -> None:
    bound = _bound(FacetSelection(provider=frozenset({'openai'})), limit=200)
    assert bound.n == 200


def test_every_bound_is_capped_at_the_limit() -> None:
    assert _bound(limit=50).n == 50
    assert _bound(FacetSelection(model=frozenset({'gpt-5'})), limit=10).n == 10


def test_finder_and_snapshot_counts_are_exact() -> None:
    finder = trace_bound(
        'finder', catalogue=CATALOGUE, facets=FacetSelection(), limit=5, finder_count=321, snapshot_count=None
    )
    snapshot = trace_bound(
        'snapshot', catalogue=CATALOGUE, facets=FacetSelection(), limit=5, finder_count=None, snapshot_count=44
    )
    assert (finder.n, finder.exact) == (321, True)
    assert (snapshot.n, snapshot.exact) == (44, True)


def test_token_constants_follow_the_caps_they_derive_from() -> None:
    assert estimate.LABEL_INPUT_TOKENS == transcript.VIEW_BUDGET // estimate.CHARS_PER_TOKEN
    assert estimate.SUMMARY_INPUT_TOKENS == transcript.VIEW_BUDGET // estimate.CHARS_PER_TOKEN
    assert estimate.SUMMARY_OUTPUT_TOKENS == summarize.SUMMARY_MAX_TOKENS
    assert estimate.EMBEDDING_INPUT_TOKENS == summarize.SUMMARY_MAX_TOKENS
    assert TOKENS_PER_TRACE['label'] == (estimate.LABEL_INPUT_TOKENS, estimate.LABEL_ANSWER_TOKENS)
    assert TOKENS_PER_TRACE['summary'] == (estimate.SUMMARY_INPUT_TOKENS, estimate.SUMMARY_OUTPUT_TOKENS)
    assert TOKENS_PER_TRACE['embedding'] == (estimate.EMBEDDING_INPUT_TOKENS, 0)


def test_a_changed_cap_moves_the_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _row(_estimate(), 'summary').cost_high
    monkeypatch.setattr(summarize, 'SUMMARY_MAX_TOKENS', summarize.SUMMARY_MAX_TOKENS * 2)
    try:
        reloaded = importlib.reload(estimate)
        after = reloaded.estimate_run(
            bound=TraceBound(n=100, exact=True, basis='x'),
            dimensions=('intent',),
            question_count=1,
            coding=False,
            models=MODELS,
            prices=_prices(),
            seconds={},
            parallelism=10,
            query=False,
        )
    finally:
        monkeypatch.undo()
        importlib.reload(estimate)
    assert _row(after, 'summary').cost_high != before


def test_the_sentiment_dimension_adds_one_label_answer_unless_already_selected() -> None:
    _, answer = TOKENS_PER_TRACE['label']
    plain = _row(_estimate(dimensions=('intent',)), 'label')
    added = _row(_estimate(dimensions=('sentiment',)), 'label')
    selected = _row(_estimate(dimensions=('sentiment',), sentiment_selected=True), 'label')
    assert added.tokens_out == plain.tokens_out + answer
    assert selected.tokens_out == plain.tokens_out


def test_a_question_source_counts_the_population_match_answer() -> None:
    _, answer = TOKENS_PER_TRACE['label']
    assert _row(_estimate(query=True), 'label').tokens_out == _row(_estimate(), 'label').tokens_out + answer
    only_match = _row(_estimate(query=True, question_count=0), 'label')
    assert only_match.tokens_out == answer


def test_the_coding_label_row_says_it_is_an_upper_bound() -> None:
    row = _row(_estimate(coding=True), 'label_coding')
    assert 'upper bound' in row.basis


def test_no_earlier_manifests_leaves_time_unknown_but_cost_given(tmp_path: Path) -> None:
    result = _estimate(seconds=stage_seconds(tmp_path))
    assert result.seconds_high is None
    assert result.seconds_low is None
    assert result.cost_high is not None
    assert result.time_note == 'estimated after your first run'


def test_time_is_seconds_times_traces_scaled_by_the_parallelism_ratio() -> None:
    seconds = {'label': StageTiming(5.0, 1, 3), 'summary': StageTiming(20.0, 1, 3)}
    base = _estimate(seconds=seconds, parallelism=10, dimensions=())
    assert base.seconds_high == pytest.approx(100 * (5.0 + 20.0) / 10)
    faster = _estimate(seconds=seconds, parallelism=20, dimensions=())
    assert faster.seconds_high == pytest.approx(_f(base.seconds_high) / 2)
    assert base.time_note is None


def test_a_stage_without_timing_makes_time_partial_and_is_named() -> None:
    result = _estimate(seconds={'summary': StageTiming(1.0, 1, 1)}, dimensions=('intent',))
    assert result.time_partial is True
    assert any('dimension:intent' in unknown for unknown in result.unknowns)


def test_time_from_runs_without_recorded_parallelism_is_unscaled_and_says_so() -> None:
    seconds = {'summary': StageTiming(2.0, None, 1)}
    slow = _estimate(seconds=seconds, parallelism=10, dimensions=(), question_count=0)
    fast = _estimate(seconds=seconds, parallelism=100, dimensions=(), question_count=0)
    assert slow.seconds_high == fast.seconds_high == pytest.approx(200.0)
    assert slow.time_note is not None
    assert 'parallelism' in slow.time_note


def _manifest_with_parallelism(directory: Path, run_id: str, parallelism: int | None, seconds: float) -> None:
    writer = start_manifest(
        run_id=run_id, surface='insights', run_name=run_id, runs_dir=directory, parallelism=parallelism
    )
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    writer.manifest.stages.append(
        StageRecord(
            name='summary',
            status=ManifestStatus.COMPLETED,
            started_at=start,
            ended_at=start + timedelta(seconds=seconds),
            completed=100,
            total=100,
        )
    )
    writer.flush()


def test_runs_at_different_parallelism_normalise_to_one_figure(tmp_path: Path) -> None:
    _manifest_with_parallelism(tmp_path, 'a', 10, 100)
    _manifest_with_parallelism(tmp_path, 'b', 20, 50)
    timing = stage_seconds(tmp_path)['summary']
    assert timing.parallelism == 1
    assert timing.seconds == pytest.approx(10.0)
    assert timing.runs == 2


def test_runs_without_recorded_parallelism_give_an_unscaled_median(tmp_path: Path) -> None:
    _manifest_with_parallelism(tmp_path, 'a', None, 100)
    timing = stage_seconds(tmp_path)['summary']
    assert timing.parallelism is None
    assert timing.seconds == pytest.approx(1.0)


def test_a_recorded_parallelism_survives_the_manifest_round_trip(tmp_path: Path) -> None:
    _manifest_with_parallelism(tmp_path, 'a', 7, 10)
    assert list_manifests(tmp_path)[0].parallelism == 7


def test_cost_is_traces_times_tokens_times_each_stages_own_price() -> None:
    result = _estimate(dimensions=('intent',), question_count=2)
    label_in, label_out = TOKENS_PER_TRACE['label']
    summary_in, summary_out = TOKENS_PER_TRACE['summary']
    embed_in, _ = TOKENS_PER_TRACE['embedding']
    assert _row(result, 'label').cost_high == pytest.approx(100 * (label_in * 0.002 + 2 * label_out * 0.004) / 1000)
    assert _row(result, 'summary').cost_high == pytest.approx(
        100 * (summary_in * 0.01 + summary_out * 0.03) / 1000
    )
    assert _row(result, 'dimension:intent').cost_high == pytest.approx(100 * embed_in * 0.0001 / 1000)
    assert result.cost_high == pytest.approx(sum(_f(row.cost_high) for row in result.rows))
    assert result.unknowns == ()


def test_embedding_cost_scales_with_dimension_count() -> None:
    one = _estimate(dimensions=('intent',))
    three = _estimate(dimensions=('intent', 'failure', 'sentiment'))
    embed = lambda r: sum(_f(row.cost_high) for row in r.rows if row.name.startswith('dimension:'))  # noqa: E731
    assert embed(three) == pytest.approx(3 * embed(one))


def test_a_coding_question_adds_one_label_call_per_trace() -> None:
    plain = _estimate(coding=False)
    coding = _estimate(coding=True)
    label_in, label_out = TOKENS_PER_TRACE['label']
    extra = _row(coding, 'label_coding')
    assert extra.cost_high == pytest.approx(100 * (label_in * 0.002 + label_out * 0.004) / 1000)
    assert _f(coding.cost_high) == pytest.approx(_f(plain.cost_high) + _f(extra.cost_high))


def test_one_unpriced_model_makes_the_total_partial_and_names_its_stage() -> None:
    prices = _prices() | {'summary-m': None}
    result = _estimate(prices=prices)
    assert _row(result, 'summary').cost_high is None
    assert result.cost_partial is True
    assert any('summary' in unknown for unknown in result.unknowns)
    assert result.cost_high == pytest.approx(
        sum(row.cost_high for row in result.rows if row.cost_high is not None)
    )


def test_a_price_missing_from_the_mapping_is_unpriced() -> None:
    prices = {key: value for key, value in _prices().items() if key != 'classifier-m'}
    result = _estimate(prices=prices)
    assert _row(result, 'label').cost_high is None
    assert result.cost_partial is True


def test_every_stage_unpriced_leaves_no_total() -> None:
    result = _estimate(prices={})
    assert result.cost_high is None
    assert result.cost_low is None


def test_a_question_source_returns_a_range_from_label_only_to_every_match() -> None:
    result = _estimate(query=True, bound=TraceBound(n=100, exact=False, basis='x'))
    label = _row(result, 'label')
    summary = _row(result, 'summary')
    assert label.cost_low == label.cost_high
    assert summary.cost_low == 0
    assert _f(summary.cost_high) > 0
    assert result.cost_low == pytest.approx(label.cost_high)
    assert result.cost_high == pytest.approx(sum(_f(row.cost_high) for row in result.rows))


def test_other_sources_give_one_figure() -> None:
    result = _estimate(query=False)
    assert result.cost_low == result.cost_high


def _manifest_with_stages(directory: Path, run_id: str, stages: dict[str, tuple[float, int]]) -> None:
    writer = start_manifest(run_id=run_id, surface='insights', run_name=run_id, runs_dir=directory)
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    for name, (seconds, total) in stages.items():
        writer.manifest.stages.append(
            StageRecord(
                name=name,
                status=ManifestStatus.COMPLETED,
                started_at=start,
                ended_at=start + timedelta(seconds=seconds),
                completed=total,
                total=total,
            )
        )
    writer.flush()


def test_stage_seconds_is_the_median_per_trace_across_manifests(tmp_path: Path) -> None:
    _manifest_with_stages(tmp_path, 'a', {'summary': (100, 100), 'label': (50, 100)})
    _manifest_with_stages(tmp_path, 'b', {'summary': (300, 100)})
    _manifest_with_stages(tmp_path, 'c', {'summary': (200, 100)})
    seconds = stage_seconds(tmp_path)
    assert seconds['summary'].seconds == pytest.approx(2.0)
    assert seconds['label'].seconds == pytest.approx(0.5)
    assert seconds['summary'].parallelism is None


def test_stage_seconds_ignores_stages_without_a_recorded_total(tmp_path: Path) -> None:
    _manifest_with_stages(tmp_path, 'a', {'dimension:intent': (40, 0), 'summary': (10, 10)})
    assert set(stage_seconds(tmp_path)) == {'summary'}

