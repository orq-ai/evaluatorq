from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import typer
from typer.testing import CliRunner

from evaluatorq import cli as cli_module
from evaluatorq.common.judge import ClassifyQuestion
from evaluatorq.trace_finder import (
    CompiledQuery,
    RunSnapshot,
    TraceClassification,
    TraceRecord,
    ValueSelection,
)


def _app() -> typer.Typer:
    app = typer.Typer()
    cli_module._register_subapps(app)
    return app


def _trace() -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id='trace-cli-1',
        span_id='span-cli-1',
        timestamp=datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': 'I need a refund.'},),
        project='support-agent',
        model='gpt-5.6-luna',
        provider='openai',
        status='ok',
        product='production',
        trace_type='llm',
    )


class FakeStore:
    def __init__(self) -> None:
        self.request: Any | None = None

    async def compile(self, request: Any) -> RunSnapshot:
        self.request = request
        trace = _trace()
        compiled = CompiledQuery(
            task=ClassifyQuestion(
                kind='choice',
                instructions='Does the trace mention a refund?',
                criteria={'yes': 'It mentions a refund.', 'no': 'It does not.'},
                state={},
            ),
            selection=ValueSelection(kind='values', values=('yes',)),
        )
        result = TraceClassification(
            trace_id=trace.trace_id,
            span_id=trace.span_id,
            value='yes',
            confidence=0.94,
            matched=True,
            raw_result={'value': 'yes'},
        )
        return RunSnapshot(
            state='completed',
            request=request,
            compiled=compiled,
            trace_ids=(trace.trace_id,),
            traces=(trace,),
            results={trace.trace_id: result},
            total=1,
            completed=1,
            matched=1,
        )


def test_find_missing_orq_key_exits_two(monkeypatch: Any) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    def missing_key() -> Any:
        raise ValueError('ORQ_API_KEY environment variable must be set to reach the Orq API.')

    monkeypatch.setattr(find_cli, 'resolve_orq_client', missing_key)
    result = CliRunner().invoke(_app(), ['find', 'refund requests'])

    assert result.exit_code == 2
    assert 'ORQ_API_KEY' in result.output


def test_find_writes_json_and_prints_fake_trace(monkeypatch: Any, tmp_path: Path) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    store = FakeStore()
    monkeypatch.setattr(find_cli, 'resolve_orq_client', lambda: object())
    monkeypatch.setattr(find_cli, 'resolve_llm_client', lambda **_: SimpleNamespace(client=object()))
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: store)
    output = tmp_path / 'finder.json'

    result = CliRunner().invoke(
        _app(),
        [
            'find',
            'refund requests',
            '--json',
            str(output),
            '--project',
            'support-agent',
            '--tokens-min',
            '100',
        ],
    )

    assert result.exit_code == 0, result.output
    assert 'trace-cli-1' in result.output
    assert output.exists()
    assert 'trace-cli-1' in output.read_text()
    assert store.request is not None
    assert store.request.population.facets.project == frozenset({'support-agent'})
    assert store.request.population.numeric.tokens_min == 100


def test_dashboard_flags_are_handed_to_reload_worker_environment(monkeypatch: Any, tmp_path: Path) -> None:
    from evaluatorq.dashboard import launch

    names = (
        'EVALUATORQ_COMPILER_MODEL',
        'EVALUATORQ_JEV_MODEL',
        'EVALUATORQ_FINDER_WINDOW_DAYS',
        'EVALUATORQ_FINDER_LIMIT',
        'EVALUATORQ_FINDER_PARALLELISM',
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launch, 'serve', lambda *args, **kwargs: None)

    try:
        result = CliRunner().invoke(
            cli_module.app,
            [
                'dashboard',
                str(tmp_path),
                '--compiler-model',
                'compiler/model',
                '--jev-model',
                'jev/model',
                '--window-days',
                '14',
                '--limit',
                '120',
                '--parallelism',
                '12',
            ],
        )

        assert result.exit_code == 0, result.output
        assert os.environ['EVALUATORQ_COMPILER_MODEL'] == 'compiler/model'
        assert os.environ['EVALUATORQ_JEV_MODEL'] == 'jev/model'
        assert os.environ['EVALUATORQ_FINDER_WINDOW_DAYS'] == '14'
        assert os.environ['EVALUATORQ_FINDER_LIMIT'] == '120'
        assert os.environ['EVALUATORQ_FINDER_PARALLELISM'] == '12'
    finally:
        for name in names:
            os.environ.pop(name, None)
