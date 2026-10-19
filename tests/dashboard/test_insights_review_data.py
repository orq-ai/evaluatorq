"""Data contract tests for the Insights review payload."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import cast

from evaluatorq.dashboard.insights_review_data import (
    build_review_payload,
    build_signal_detail_payload,
    review_trace_key,
)
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
from evaluatorq.signals.models import Evidence, Precondition, SignalReport, SignalResult


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
        + urlencode({'trace_id': 'trace /?&', 'span_id': 'span ?&', 'view': 'trace'})
    )
    assert row['orq_url'] is None
    safe_trace = _trace('span_a-~').model_copy(update={'trace_id': 'trace.a-~'})
    safe_run = _run([safe_trace]).model_copy(update={'population': {'mode': 'query'}})
    safe_row = _trace_rows(build_review_payload(safe_run))[0]
    assert safe_row['orq_url'] == (
        'https://orq.example/workspace/traces/(trace:trace.a-~//span:span_a-~)'
    )
    snapshot_row = _trace_rows(build_review_payload(_run([trace])))[0]
    assert snapshot_row['orq_url'] is None


def test_review_payload_lands_on_analysis_when_the_conversation_cannot_be_re_read(tmp_path):
    from evaluatorq.dashboard.insights_uploads import uploads_dir
    from evaluatorq.insights.store import get_insights_runs_dir

    kept = tmp_path / 'kept.json'
    kept.write_text('{}', encoding='utf-8')
    upload = uploads_dir(get_insights_runs_dir()) / f'snapshot-{"0" * 32}.json'
    populations = {
        'filter': {'mode': 'filter'},
        'kept snapshot': {'mode': 'snapshot', 'snapshot_path': str(kept)},
        'deleted snapshot': {'mode': 'snapshot', 'snapshot_path': str(tmp_path / 'gone.json')},
        'deleted upload': {'mode': 'snapshot', 'snapshot_path': str(upload)},
        'dataset': {'mode': 'dataset'},
    }
    landings = {
        name: _trace_rows(build_review_payload(_run([_trace('s')]).model_copy(update={'population': population})))[0][
            'trace_url'
        ]
        for name, population in populations.items()
    }

    trace_tab = '/insights/review-test/trace?trace_id=same-trace&span_id=s&view=trace'
    analysis_tab = '/insights/review-test/trace?trace_id=same-trace&span_id=s'
    assert landings == {
        'filter': trace_tab,
        'kept snapshot': trace_tab,
        'deleted snapshot': analysis_tab,
        'deleted upload': analysis_tab,
        'dataset': analysis_tab,
    }


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
    assert 'rerun' not in payload
    json.dumps(payload)


def test_review_payload_normalizes_coding_scores_from_legacy_full_bundle():
    run = _run([_trace('coding-span')])
    trace = run.traces[0].model_copy(
        update={
            'labels': {
                'user_corrections': LabelAnswer(value=0.5, confidence=None, probabilities=None, error=None),
            }
        }
    )
    run = run.model_copy(
        update={
            'traces': [trace],
            'config': run.config.model_copy(update={'coding_analysis': True}),
        }
    )

    row = _trace_rows(build_review_payload(run))[0]

    assert row['l'] == {'user_corrections': ['2']}


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


def test_review_payload_includes_compact_signal_summaries_without_evidence_or_preconditions():
    report = SignalReport(
        trajectory_id='trajectory-1',
        config_version='signals-v3',
        results={
            'tool_retries': SignalResult(
                name='tool_retries',
                group='B',
                value=1,
                evidence=[Evidence(step_id=4, agent_path=('agent-1',), call_id='call-2', reason='retry')],
                approximate=True,
                preconditions=[Precondition(name='tool_results_present', met='partial', detail='One result omitted')],
            ),
            'tool_success': SignalResult(
                name='tool_success',
                group='B',
                no_basis='tool results were not captured',
                preconditions=[Precondition(name='tool_results_present', met=False, detail='No tool results')],
            ),
            'trajectory_tags': SignalResult(name='trajectory_tags', group='D', value={'coding': True}),
        },
    )
    trace = _trace('signal-span').model_copy(
        update={'signals': report, 'source_coverage': {'source': 'otel', 'partial': True}}
    )

    row = _trace_rows(build_review_payload(_run([trace])))[0]

    assert row['has_signals'] is True
    assert row['has_signal_details'] is True
    assert row['trace_id'] == 'same-trace'
    assert row['span_id'] == 'signal-span'
    assert row['signals'] == {
        'config_version': 'signals-v3',
        'results': {
            'tool_retries': {
                'name': 'tool_retries', 'group': 'B', 'value': 1, 'approximate': True,
                'no_basis': None, 'reason': None, 'rule_version': None,
            },
            'tool_success': {
                'name': 'tool_success', 'group': 'B', 'value': None, 'approximate': False,
                'no_basis': 'tool results were not captured', 'reason': None, 'rule_version': None,
            },
            'trajectory_tags': {
                'name': 'trajectory_tags', 'group': 'D', 'value': {'coding': True}, 'approximate': False,
                'no_basis': None, 'reason': None, 'rule_version': None,
            },
        },
    }
    assert row['signal_detail_url'] == '/insights/review-test/trace-signals.json?trace_id=same-trace&span_id=signal-span'
    serialized = json.dumps(row)
    assert 'conversation' not in serialized
    assert 'tool results were not captured' in serialized
    assert 'trajectory_tags' in serialized
    assert 'call-2' not in serialized
    assert 'tool_results_present' not in serialized

    detail = build_signal_detail_payload(_run([trace]), 'same-trace', 'signal-span')
    assert detail is not None
    assert detail['signals'] == report.model_dump(mode='json')
    assert detail['source_coverage'] == {'source': 'otel', 'partial': True}


def test_review_payload_marks_missing_legacy_signals_as_unmeasured():
    trace = _trace_rows(build_review_payload(_run([_trace('legacy-signal-span')])))[0]

    assert trace['has_signals'] is False
    assert trace['signals'] is None
    assert trace['has_signal_details'] is False
    assert trace['signal_detail_url'] is None


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


def test_review_payload_names_the_source_like_the_run_form_tabs() -> None:
    run = _run([])
    for mode, label in (
        ('filter', 'Orq traces'),
        ('query', 'Orq traces matching a question'),
        ('export', 'Trace file (Finder export)'),
        ('snapshot', 'Trace file'),
    ):
        payload = build_review_payload(run.model_copy(update={'population': {'mode': mode}}))
        assert _mapping(payload['run'])['source_label'] == label
