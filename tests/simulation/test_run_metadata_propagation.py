"""Simulation entrypoints bind a run id onto the invocation-metadata rail.

The three entrypoints that own a root span (``simulate``, ``generate_and_simulate``,
``generate``) mint a ``run_id``, stamp it on that span as ``orq.evaluatorq_run_id``,
and bind it so every LLM call they issue carries ``evaluatorq_run_id`` in the
request ``metadata`` (the key Orq's trace UI filters on).

``generate_personas`` / ``generate_scenarios`` deliberately do NOT: they own no
root span, so an id bound there would be undiscoverable. Their calls are still
traced via ``orq.simulation.persona_generation`` and its child LLM span.
"""

from __future__ import annotations

# ruff: noqa: S101
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from evaluatorq.common.thread_context import evaluatorq_pipeline, evaluatorq_run_id, pipeline_metadata
from evaluatorq.simulation.api import generate_personas, generate_scenarios
from openai import AsyncOpenAI


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
async def test_generate_personas_binds_no_run_id() -> None:
    """No root span, so no run id — an id here would be undiscoverable."""
    sink: dict[str, Any] = {}
    personas = await generate_personas(
        ['angry customer'],
        agent_description='support bot',
        generation_client=cast(AsyncOpenAI, cast(object, _FakeClient(sink, _one_persona))),
    )

    assert len(personas) == 1
    assert 'evaluatorq_run_id' not in (sink.get('metadata') or {})


@pytest.mark.asyncio
async def test_generate_scenarios_binds_no_run_id() -> None:
    """Mirror of the personas case — see the module docstring."""
    sink: dict[str, Any] = {}
    scenarios = await generate_scenarios(
        ['disputes a refund denial'],
        agent_description='support bot',
        generation_client=cast(AsyncOpenAI, cast(object, _FakeClient(sink, _one_scenario))),
    )

    assert len(scenarios) == 1
    assert 'evaluatorq_run_id' not in (sink.get('metadata') or {})


@pytest.mark.asyncio
async def test_generate_personas_inherits_an_outer_run_id() -> None:
    """Called from inside a run, the outer id still reaches the call.

    This is why dropping the mint is safe: the sim paths that matter bind a run
    id upstream, and the ContextVar rail carries it down here.
    """
    sink: dict[str, Any] = {}
    with evaluatorq_pipeline('agent_simulation'), evaluatorq_run_id('outer-run'):
        await generate_personas(
            ['angry customer'],
            generation_client=cast(AsyncOpenAI, cast(object, _FakeClient(sink, _one_persona))),
        )

    assert sink['metadata'] == {
        'evaluatorq_pipeline': 'agent_simulation',
        'evaluatorq_run_id': 'outer-run',
    }


@pytest.mark.asyncio
async def test_generate_scenarios_inherits_an_outer_run_id() -> None:
    """The span-less scenario helper preserves, rather than replaces, an outer id."""
    sink: dict[str, Any] = {}
    with evaluatorq_pipeline('agent_simulation'), evaluatorq_run_id('outer-run'):
        await generate_scenarios(
            ['disputes a refund denial'],
            generation_client=cast(AsyncOpenAI, cast(object, _FakeClient(sink, _one_scenario))),
        )

    assert sink['metadata'] == {
        'evaluatorq_pipeline': 'agent_simulation',
        'evaluatorq_run_id': 'outer-run',
    }


# ---------------------------------------------------------------------------
# Prove a REAL root span (not a hand-built SimpleNamespace) actually receives
# the ``orq.evaluatorq_run_id`` stamp when driving the real entrypoints.
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
async def test_post_run_processor_stays_under_pipeline_root(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-run LLM work inherits both the root span and its run metadata."""
    from evaluatorq.simulation import api
    from evaluatorq.simulation.api import _simulate_run
    from evaluatorq.simulation.tracing import with_llm_span

    async def fake_core(**kwargs: Any) -> Any:
        run = MagicMock(results=[])
        await kwargs['post_run'](run)
        return run

    monkeypatch.setattr(api, '_simulate_core', fake_core)

    async def post_run(_run: Any) -> None:
        assert pipeline_metadata() == {
            'evaluatorq_pipeline': 'agent_simulation',
            'evaluatorq_run_id': pipeline_metadata()['evaluatorq_run_id'],
        }
        async with with_llm_span(model='recommendation-model', purpose='recommendations'):
            pass

    await _simulate_run(
        target=lambda messages: 'ok',
        datapoints=[],
        sim_model='test',
        upload_results=False,
        executive_summary=False,
        post_run=post_run,
    )

    root = _find(span_collector, 'orq.simulation.pipeline')
    recommendation = _find(span_collector, 'chat recommendation-model')
    assert recommendation.parent is not None
    assert recommendation.parent.span_id == root.context.span_id


@pytest.mark.asyncio
async def test_generate_and_simulate_stamps_run_id_on_root_span(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The combined root owns one visible run id before generation begins."""
    from evaluatorq.simulation import api
    from evaluatorq.simulation.api import generate_and_simulate

    async def _fake_generate(**_kwargs: Any) -> tuple[list[Any], None, bool]:  # noqa: RUF029
        return [], None, False

    monkeypatch.setattr(api, '_generate_datapoints_inner', _fake_generate)
    monkeypatch.setattr(api, '_simulate_core', AsyncMock(return_value=MagicMock(results=[])))

    results = await generate_and_simulate(
        agent_description='a helpful assistant',
        target=lambda _messages: 'ok',
        num_personas=1,
        num_scenarios=1,
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
    and confirm the ``orq.simulation.generate`` root span comes back carrying a
    non-empty ``orq.evaluatorq_run_id``."""
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


@pytest.mark.asyncio
async def test_generate_binds_pipeline_and_run_id_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generation calls inherit both simulation metadata keys from the root."""
    from evaluatorq.simulation import api
    from evaluatorq.simulation.api import generate

    seen_metadata: dict[str, str] = {}

    async def _fake_generate(**_kwargs: Any) -> tuple[list[Any], None, bool]:
        seen_metadata.update(pipeline_metadata())
        return [], None, False

    monkeypatch.setattr(api, '_generate_datapoints_inner', _fake_generate)

    await generate(agent_description='a helpful assistant', num_personas=1, num_scenarios=1)

    assert seen_metadata['evaluatorq_pipeline'] == 'agent_simulation'
    assert seen_metadata['evaluatorq_run_id']


@pytest.mark.asyncio
async def test_two_generate_calls_get_distinct_run_ids(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate spans, separate ids — two calls are two runs."""
    from evaluatorq.simulation import api
    from evaluatorq.simulation.api import generate

    async def _fake_generate(**_kwargs: Any) -> tuple[list[Any], None, bool]:
        return [], None, False

    monkeypatch.setattr(api, '_generate_datapoints_inner', _fake_generate)

    await generate(agent_description='a', num_personas=1, num_scenarios=1)
    await generate(agent_description='b', num_personas=1, num_scenarios=1)

    ids = [_attrs(s).get('orq.evaluatorq_run_id') for s in span_collector.spans if s.name == 'orq.simulation.generate']
    assert len(ids) == 2
    assert all(ids)
    assert ids[0] != ids[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('helper', 'seeds', 'build_parsed', 'span_name'),
    [
        (generate_personas, ['angry customer'], _one_persona, 'orq.simulation.persona_generation'),
        (generate_scenarios, ['refund dispute'], _one_scenario, 'orq.simulation.scenario_generation'),
    ],
)
async def test_seeded_generation_emits_its_generation_span(
    span_collector: _CollectingExporter,
    helper: Any,
    seeds: list[str],
    build_parsed: Any,
    span_name: str,
) -> None:
    """Both public seeded helpers retain their dedicated generation span."""
    await helper(
        seeds,
        agent_description='support bot',
        generation_client=_FakeClient({}, build_parsed),  # type: ignore[arg-type]
    )

    assert _find(span_collector, span_name).name == span_name
