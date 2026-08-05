"""Simulation entrypoints bind a run id onto the invocation-metadata rail.

Every sim entrypoint mints its own ``run_id`` and binds it via
``_sim_run_scope``, so the LLM calls it issues carry ``evaluatorq_run_id`` in
the request ``metadata`` (the key Orq's trace UI filters on).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from evaluatorq.simulation.api import _sim_run_scope, generate_personas, generate_scenarios


class _FakeCompletions:
    """Captures the kwargs of the single ``parse`` call generate_structured makes."""

    def __init__(self, sink: dict[str, Any], build_parsed) -> None:
        self._sink = sink
        self._build_parsed = build_parsed

    async def parse(self, **kwargs: Any) -> Any:
        self._sink.update(kwargs)
        parsed = self._build_parsed(kwargs['response_format'])
        message = SimpleNamespace(parsed=parsed, refusal=None, content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='stop')], usage=None)


class _FakeClient:
    def __init__(self, sink: dict[str, Any], build_parsed) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(sink, build_parsed))
        self.base_url = 'https://my.orq.ai/v3/router'

    async def close(self) -> None:  # pragma: no cover - generator doesn't own us
        pass


@pytest.fixture
def force_orq_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tagging is gated on the client routing through Orq; force it for the fake."""
    monkeypatch.setattr('evaluatorq.common.llm_call.client_routes_through_orq', lambda _client: True)


def _one_persona(response_format: type) -> Any:
    return response_format(
        personas=[
            {
                'name': 'Angry Customer',
                'patience': 0.2,
                'assertiveness': 0.9,
                'politeness': 0.3,
                'technical_level': 0.4,
                'communication_style': 'terse',
                'background': 'Frustrated with a refund denial',
            }
        ]
    )


def _one_scenario(response_format: type) -> Any:
    return response_format(
        scenarios=[
            {
                'name': 'Refund dispute',
                'goal': 'Obtain a refund for a late order',
                'context': 'Order placed 40 days ago',
            }
        ]
    )


@pytest.mark.asyncio
async def test_generate_personas_binds_a_run_id(force_orq_routing: None) -> None:
    """generate_personas mints its own run_id; the parse call carries it."""
    sink: dict[str, Any] = {}
    personas = await generate_personas(
        ['angry customer'],
        agent_description='support bot',
        generation_client=_FakeClient(sink, _one_persona),  # type: ignore[arg-type]
    )

    assert len(personas) == 1
    assert sink['metadata']['evaluatorq_run_id']


@pytest.mark.asyncio
async def test_generate_scenarios_binds_a_run_id(force_orq_routing: None) -> None:
    """generate_scenarios mints its own run_id; the parse call carries it."""
    sink: dict[str, Any] = {}
    scenarios = await generate_scenarios(
        ['disputes a refund denial'],
        agent_description='support bot',
        generation_client=_FakeClient(sink, _one_scenario),  # type: ignore[arg-type]
    )

    assert len(scenarios) == 1
    assert sink['metadata']['evaluatorq_run_id']


@pytest.mark.asyncio
async def test_each_entrypoint_call_gets_a_distinct_run_id(force_orq_routing: None) -> None:
    """Two separate calls are two separate runs — the ids must not match."""
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    await generate_personas(['a'], generation_client=_FakeClient(first, _one_persona))  # type: ignore[arg-type]
    await generate_personas(['b'], generation_client=_FakeClient(second, _one_persona))  # type: ignore[arg-type]

    assert first['metadata']['evaluatorq_run_id'] != second['metadata']['evaluatorq_run_id']


def test_sim_run_scope_stamps_the_root_span() -> None:
    """The scope stamps the span attribute and binds the ContextVar."""
    from evaluatorq.common.thread_context import pipeline_metadata

    attrs: dict[str, Any] = {}
    span = SimpleNamespace(set_attribute=lambda k, v: attrs.__setitem__(k, v))

    with _sim_run_scope('r1', span):
        assert pipeline_metadata()['evaluatorq_run_id'] == 'r1'
    assert attrs == {'orq.evaluatorq_run_id': 'r1'}
    assert 'evaluatorq_run_id' not in pipeline_metadata()


def test_sim_run_scope_without_a_span_still_binds() -> None:
    """Span-less entrypoints (generate_personas/_scenarios) still bind the run id."""
    from evaluatorq.common.thread_context import pipeline_metadata

    with _sim_run_scope('r2', None):
        assert pipeline_metadata()['evaluatorq_run_id'] == 'r2'


# ---------------------------------------------------------------------------
# Gap 1: prove a REAL root span (not a hand-built SimpleNamespace) actually
# receives the ``orq.evaluatorq_run_id`` stamp when driving the real
# entrypoints. `_sim_run_scope` alone is tested above, but nothing upstream
# proves `simulate()`/`generate()` hand it a live span.
# ---------------------------------------------------------------------------


class _CollectingExporter(SpanExporter):
    """Minimal in-memory exporter that collects finished spans."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Any) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


@pytest.fixture
def span_collector():
    """Set up an in-memory OTel TracerProvider; patch the simulation tracer."""
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('evaluatorq-simulation-test')

    with patch('evaluatorq.simulation.tracing.get_tracer', return_value=tracer):
        yield exporter

    provider.shutdown()


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _find(exporter: _CollectingExporter, name: str) -> ReadableSpan:
    for s in exporter.spans:
        if s.name == name:
            return s
    raise AssertionError(f'span {name!r} not found; got {[s.name for s in exporter.spans]}')


@pytest.mark.asyncio
async def test_simulate_stamps_run_id_on_root_span(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the real ``simulate()`` end to end (inner core mocked out so no
    actual simulation runs) and confirm the ``orq.simulation.pipeline`` root
    span comes back carrying a non-empty ``orq.evaluatorq_run_id``."""
    from evaluatorq.simulation import api
    from evaluatorq.simulation.api import simulate

    monkeypatch.setattr(
        api,
        '_simulate_core',
        AsyncMock(return_value=MagicMock(results=[])),
    )

    results = await simulate(
        target=lambda messages: 'ok',
        datapoints=[],
        sim_model='test',
        upload_results=False,
        executive_summary=False,
    )

    assert results == []
    span = _find(span_collector, 'orq.simulation.pipeline')
    run_id = _attrs(span).get('orq.evaluatorq_run_id')
    assert run_id, f'expected a non-empty run id on the root span, got {run_id!r}'


@pytest.mark.asyncio
async def test_generate_stamps_run_id_on_root_span(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the real ``generate()`` end to end (inner generation mocked out)
    and confirm the ``orq.simulation.generate`` root span — wholly untested
    before this — comes back carrying a non-empty ``orq.evaluatorq_run_id``."""
    from evaluatorq.simulation import api
    from evaluatorq.simulation.api import generate

    async def _fake_generate(**_kwargs: Any) -> tuple[list[Any], None, bool]:
        return [], None, False

    monkeypatch.setattr(api, '_generate_datapoints_inner', _fake_generate)

    datapoints = await generate(
        agent_description='a helpful assistant',
        num_personas=1,
        num_scenarios=1,
    )

    assert datapoints == []
    span = _find(span_collector, 'orq.simulation.generate')
    run_id = _attrs(span).get('orq.evaluatorq_run_id')
    assert run_id, f'expected a non-empty run id on the root span, got {run_id!r}'
