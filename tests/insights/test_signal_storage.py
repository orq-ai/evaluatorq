from __future__ import annotations

import json

import pytest

from evaluatorq.insights.models import InsightsRun
from evaluatorq.insights.store import load_run, save_run
from evaluatorq.signals.config import SignalsConfig
from evaluatorq.signals.models import Evidence, Precondition, SignalReport, SignalResult


def test_signal_report_and_effective_config_survive_run_storage(minimal_run, tmp_path) -> None:
    config = SignalsConfig(retry_window=7, error_detection='status_and_content')
    config = config.model_copy(update={'tag_thresholds': config.resolved_tag_thresholds()})
    report = SignalReport(
        trajectory_id='trajectory-1',
        config_version=config.version,
        results={
            'tool_call_count': SignalResult(
                name='tool_call_count', group='A', value=1,
                evidence=[Evidence(step_id=2, call_id='call-1', agent_path=('subagent-1',),
                                   related=[((), 1)], related_call_ids=['spawn-1'])],
            ),
            'tool_time_ms': SignalResult(
                name='tool_time_ms', group='C', no_basis='missing tool timing',
                preconditions=[Precondition(name='tool timing', met=False, detail='not captured')],
            ),
            'wall_time_ms': SignalResult(name='wall_time_ms', group='C', value=1000, approximate=True),
        },
    )
    minimal_run.config = minimal_run.config.model_copy(update={'signals': config})
    minimal_run.traces[0].signals = report
    minimal_run.traces[0].source_coverage = {'source': 'snapshot', 'timing': 'missing'}
    path = save_run(minimal_run, tmp_path)
    restored = load_run(path)
    assert restored.schema_version == 2
    assert restored.traces[0].signals == report
    assert restored.config.signals == config
    assert restored.config.signals.retry_window == 7
    assert restored.config.signals.tag_thresholds is not None
    assert restored.traces[1].signals is None
    assert restored.traces[0].source_coverage == minimal_run.traces[0].source_coverage
    raw = json.loads(path.read_text())
    assert 'trajectory' not in raw['traces'][0]
    assert 'messages' not in raw['traces'][0]


def test_old_run_has_unmeasured_signals_and_no_invented_configuration(minimal_run, tmp_path) -> None:
    raw = minimal_run.model_dump(mode='json')
    raw['schema_version'] = 1
    raw['config'].pop('signals')
    for trace in raw['traces']:
        trace.pop('signals')
        trace.pop('source_coverage')
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(raw))
    restored = load_run(path)
    assert restored.schema_version == 1
    assert restored.config.signals is None
    assert all(trace.signals is None for trace in restored.traces)
    assert all(trace.source_coverage == {} for trace in restored.traces)


def test_unknown_run_schema_is_rejected(minimal_run) -> None:
    raw = minimal_run.model_dump(mode='json')
    raw['schema_version'] = 99
    with pytest.raises(ValueError, match='schema_version'):
        InsightsRun.model_validate(raw)
