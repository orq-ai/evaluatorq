"""Data contract tests for the Insights review payload."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import cast

from evaluatorq.dashboard.insights_review_data import build_review_payload, review_trace_key
from evaluatorq.insights.models import (
    ClusterAssignment,
    InsightsConfig,
    InsightsRun,
    LabelAnswer,
    LabelResult,
    LabelSpec,
    TraceInsight,
    TraceSummary,
)
from evaluatorq.insights.transcript import SHELL_TOOLS, ToolStats


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _trace_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    traces = payload['traces']
    assert isinstance(traces, list)
    return [_mapping(trace) for trace in traces]


def _run(traces: list[TraceInsight]) -> InsightsRun:
    spec = LabelSpec(
        name='user_frustration',
        kind='score',
        instructions='Rate frustration.',
        criteria=['1 calm', '2 mild', '3 irritated', '4 frustrated', '5 angry'],
    )
    failed_spec = LabelSpec(name='failed_question', kind='noul', instructions='Was the question answered?')
    return InsightsRun(
        run_id='review-test',
        run_name='Review test',
        created_at=datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
        status='completed',
        stage_failures=[],
        population={'mode': 'snapshot', 'limit': len(traces)},
        config=InsightsConfig(labels=[spec, failed_spec], dimensions=['intent']),
        traces=traces,
        dimensions={},
        labels={
            spec.name: LabelResult(
                spec=spec, counts={'1': 1, '5': 1}, mean_confidence=None, n_low_confidence=0, n_failed=1
            ),
            failed_spec.name: LabelResult(
                spec=failed_spec, counts={}, mean_confidence=None, n_low_confidence=0, n_failed=1
            ),
        },
        priority=None,
        priority_reason=None,
        counts={'n_traces': len(traces)},
        warnings=[],
    )


def _trace(span_id: str, *, tool_stats: ToolStats | None = None, value: float = 0.0) -> TraceInsight:
    return TraceInsight(
        trace_id='same-trace',
        span_id=span_id,
        timestamp=datetime(2026, 9, 1, 9, 30),
        labels={
            'user_frustration': LabelAnswer(value=value, confidence=None, probabilities=None, error=None),
            'failed_question': LabelAnswer(value=None, confidence=None, probabilities=None, error='classifier failed'),
        },
        summary=TraceSummary(
            summary='Short summary',
            request=None,
            task=None,
            topic='topic',
            assistant_errors=['None', 'A real assistant error.'],
            sentiment_explanation=None,
        ),
        tool_stats=tool_stats,
        assignments={'intent': ClusterAssignment(top='intent-t0', base='intent-b0')},
    )


def test_review_payload_provides_safe_server_built_trace_links(monkeypatch):
    from urllib.parse import urlencode

    monkeypatch.setenv('ORQ_WORKSPACE', 'workspace')
    monkeypatch.setenv('ORQ_UI_BASE_URL', 'https://orq.example')
    trace = _trace('span ?&').model_copy(update={'trace_id': 'trace /?&'})
    run = _run([trace]).model_copy(update={'population': {'mode': 'query'}})

    row = _trace_rows(build_review_payload(run))[0]

    assert row['trace_url'] == (
        '/insights/review-test/trace?'
        + urlencode({'trace_id': 'trace /?&', 'span_id': 'span ?&'})
    )
    assert row['orq_url'] is None
    safe_trace = _trace('span_a-~').model_copy(update={'trace_id': 'trace.a-~'})
    safe_run = _run([safe_trace]).model_copy(update={'population': {'mode': 'query'}})
    safe_row = _trace_rows(build_review_payload(safe_run))[0]
    assert safe_row['orq_url'] == (
        'https://orq.example/workspace/traces?query=%28trace%3Atrace.a-~%2F%2Fspan%3Aspan_a-~%29'
    )
    snapshot_row = _trace_rows(build_review_payload(_run([trace])))[0]
    assert snapshot_row['orq_url'] is None


def test_review_payload_preserves_ui_trace_identity_scores_errors_and_tool_stats():
    populated = ToolStats(tools={'Bash': 2, 'Read': 1}, skills={'testing': 3}, commands={'pytest': 2})
    run = _run([_trace('span-a', tool_stats=populated), _trace('span-b', tool_stats=ToolStats())])
    payload = build_review_payload(run)
    first, second = _trace_rows(payload)

    assert first['id'] != second['id']
    assert first['id'] == review_trace_key(run.traces[0])
    assert first['ts'] == '2026-09-01T09:30:00Z'
    assert [
        cast(
            dict[str, list[str]],
            _trace_rows(build_review_payload(_run([_trace(f'score-{index}', value=index / 4)])))[0]['l'],
        )['user_frustration'][0]
        for index in range(5)
    ] == ['1', '2', '3', '4', '5']
    assert first['label_errors'] == {'failed_question': 'classifier failed'}
    assert first['errors'] == ['A real assistant error.']
    assert _mapping(first['a'])['intent'] == 'intent-b0'
    assert first['has_tool_stats'] is True
    assert first['tools'] == {'Bash': 2, 'Read': 1}
    assert 'Skill' not in _mapping(first['tools'])
    assert first['skills'] == {'testing': 3}
    assert first['commands'] == {'pytest': 2}
    assert second['has_tool_stats'] is True
    assert second['tools'] == second['skills'] == second['commands'] == {}
    assert _mapping(payload['activity'])['shell_tools'] == sorted(SHELL_TOOLS)
    assert _mapping(payload['run'])['cost'] is None
    rerun = _mapping(payload['rerun'])
    assert rerun['name'] == 'Review test'
    assert rerun['source'] == 'snapshot'
    assert rerun['limit'] == 2
    assert rerun['dimensions'] == ['intent']
    assert rerun['labels'] == ['user_frustration']
    assert [spec['name'] for spec in rerun['custom_labels']] == ['failed_question']
    assert rerun['source_path'] is None
    assert rerun['fresh_selection_required'] is True
    json.dumps(payload)


def test_rerun_payload_retains_available_file_source_path(tmp_path):
    source = tmp_path / 'snapshot.json'
    source.write_text('{}', encoding='utf-8')
    run = _run([]).model_copy(update={'population': {'mode': 'snapshot', 'snapshot_path': str(source)}})

    rerun = _mapping(build_review_payload(run)['rerun'])

    assert rerun['source'] == 'snapshot'
    assert rerun['source_path'] == str(source)
    assert rerun['fresh_selection_required'] is False


def test_legacy_run_without_tool_stats_is_explicitly_unmeasured():
    run = _run([_trace('span-old')])
    saved = json.loads(run.model_dump_json())
    del saved['traces'][0]['tool_stats']
    legacy = InsightsRun.model_validate(saved)

    trace = _trace_rows(build_review_payload(legacy))[0]
    assert trace['has_tool_stats'] is False
    assert trace['tools'] == trace['skills'] == trace['commands'] == {}
    assert trace['summary'] == 'Short summary'
    assert trace['request'] is None
    assert trace['has_summary'] is True

    missing_summary = TraceInsight(
        trace_id='missing-summary', span_id='span-summary', timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc)
    )
    no_summary = _trace_rows(build_review_payload(_run([missing_summary])))[0]
    assert no_summary['has_summary'] is False
    assert no_summary['summary'] is None
    assert no_summary['errors'] == []


def test_review_payload_keeps_all_5000_traces_and_json_serializes():
    traces = [
        TraceInsight(
            trace_id=f'trace-{index}',
            span_id=f'span-{index}',
            timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
        for index in range(5000)
    ]
    payload = build_review_payload(_run(traces))

    assert len(payload['traces']) == 5000
    last_trace = _trace_rows(payload)[-1]
    assert last_trace['trace_id'] == 'trace-4999'
    assert last_trace['has_summary'] is False
    json.dumps(payload)
