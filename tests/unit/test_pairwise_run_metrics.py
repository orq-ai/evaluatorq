from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from evaluatorq.contracts import Usage
from evaluatorq.pairwise import PairwiseComparison
from evaluatorq.pairwise_run import PairwiseResponseMetrics, PairwiseRun, new_run


def test_response_metrics_reject_negative_or_nonfinite_latency() -> None:
    for latency in (-0.001, float('nan'), float('inf'), float('-inf')):
        with pytest.raises(ValidationError):
            PairwiseResponseMetrics(latency_ms=latency)


def test_response_metrics_preserve_zero_latency_and_usage() -> None:
    usage = Usage(total_cost=0.000001, calls=1, priced_calls=1)

    metrics = PairwiseResponseMetrics(usage=usage, latency_ms=0.0)

    assert metrics.latency_ms == 0.0
    assert metrics.usage == usage


def test_comparison_rejects_negative_or_nonfinite_judge_latency() -> None:
    for latency in (-0.001, float('nan'), float('inf'), float('-inf')):
        with pytest.raises(ValidationError):
            PairwiseComparison(winner='A', judge_latency_ms=latency)


def test_run_add_and_save_load_preserve_response_metrics(tmp_path) -> None:
    run = new_run(run_name='metrics')
    metrics_a = PairwiseResponseMetrics(usage=Usage(total_cost=0.000001, calls=1, priced_calls=1), latency_ms=0.0)
    metrics_b = PairwiseResponseMetrics(usage=Usage(calls=1), latency_ms=14.25)
    run.add(
        PairwiseComparison(winner='A', judge_latency_ms=3.5),
        question='q',
        response_a='a',
        response_b='b',
        metrics_a=metrics_a,
        metrics_b=metrics_b,
    )

    loaded = PairwiseRun.load(run.save(tmp_path / 'metrics.json'))

    assert loaded.entries[0].metrics_a == metrics_a
    assert loaded.entries[0].metrics_b == metrics_b
    assert loaded.entries[0].comparison.judge_latency_ms == 3.5


def test_legacy_run_without_measurement_fields_remains_loadable(tmp_path) -> None:
    run = new_run(run_name='legacy')
    run.add(PairwiseComparison(winner='inconclusive'), question='q', response_a='a', response_b='b')
    artifact = run.model_dump(mode='json')
    artifact['entries'][0].pop('metrics_a', None)
    artifact['entries'][0].pop('metrics_b', None)
    artifact['entries'][0]['comparison'].pop('judge_latency_ms', None)
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(artifact), encoding='utf-8')

    loaded = PairwiseRun.load(path)

    assert loaded.entries[0].metrics_a is None
    assert loaded.entries[0].metrics_b is None
    assert loaded.entries[0].comparison.judge_latency_ms is None
