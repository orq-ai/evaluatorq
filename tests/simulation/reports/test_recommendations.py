"""Tests for simulation/reports/recommendations.py.

Covers the selective trigger filter (benign failures produce no LLM calls), the
generation path with a mocked LLM client, and rendering of the recommendations
section in the Markdown / HTML exports.

RES-822: generation now routes through ``common.structured_output.generate_structured``
— the raw Responses ``create()`` leg first, then ``json_object`` fallback with
fence-tolerant parsing — so the generation tests mock the Responses response and
assert on its kwargs, plus the fenced and malformed fallback paths.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIStatusError
from pydantic import ValidationError

from evaluatorq.common.thread_context import evaluatorq_pipeline, evaluatorq_run_id
from evaluatorq.contracts import Message, TokenUsage
from evaluatorq.simulation.reports import export_html, export_markdown
from evaluatorq.simulation.reports.recommendations import (
    _SuggestionsLLMResponse,
    SimulationRecommendationConfig,
    find_triggers,
    generate_recommendations,
)
from evaluatorq.simulation.reports.sections import build_report_sections
from evaluatorq.simulation.types import (
    SimulationRecommendation,
    SimulationResult,
    TerminatedBy,
    TurnMetrics,
)


def _make_result(
    *,
    goal_achieved: bool = True,
    rules_broken: list[str] | None = None,
    terminated_by: TerminatedBy = TerminatedBy.judge,
    metadata: dict[str, Any] | None = None,
    factual_accuracy: float | None = None,
) -> SimulationResult:
    meta = {'persona': 'Alice', 'scenario': 'Billing'}
    meta.update(metadata or {})
    return SimulationResult(
        messages=[
            Message(role='user', content='hi'),
            Message(role='assistant', content='response'),
        ],
        terminated_by=terminated_by,
        reason='judge decided',
        goal_achieved=goal_achieved,
        goal_completion_score=1.0 if goal_achieved else 0.2,
        rules_broken=rules_broken or [],
        turn_count=2,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        turn_metrics=[
            TurnMetrics(
                turn_number=1,
                token_usage=TokenUsage(),
                judge_reason='ok',
                factual_accuracy=factual_accuracy,
            )
        ],
        metadata=meta,
    )


def _parsed_response(suggestions: list[str]) -> Any:
    """A raw Responses response carrying valid structured JSON (happy path)."""
    output_text = _SuggestionsLLMResponse(suggestions=suggestions).model_dump_json()
    content = SimpleNamespace(type='output_text', text=output_text, annotations=[])
    content.to_dict = lambda: {'type': content.type, 'text': content.text, 'annotations': content.annotations}
    output = SimpleNamespace(type='message', role='assistant', content=[content], status='completed')
    output.to_dict = lambda: {
        'type': output.type,
        'role': output.role,
        'content': [content.to_dict()],
        'status': output.status,
    }
    response = SimpleNamespace(
        output=[output],
        incomplete_details=None,
        stop_reason='stop',
        output_text=output_text,
    )
    response.to_dict = lambda: {
        'output': [output.to_dict()],
        'incomplete_details': response.incomplete_details,
        'stop_reason': response.stop_reason,
        'output_text': response.output_text,
    }
    return response


def _fallback_response(content: str) -> Any:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _schema_400() -> APIStatusError:
    request = httpx.Request('POST', 'https://router.example/v3/router')
    response = httpx.Response(400, request=request)
    return APIStatusError('response_format not supported', response=response, body=None)


def _mock_client(payload: list[str] | Exception) -> MagicMock:
    """Client whose raw Responses call succeeds with the given suggestions, or raises.

    Generation asks for ``api='responses'``, so the Responses leg is the one
    exercised; the chat legs only run when this leg declines (see the fallback
    tests, which make ``responses.create`` raise a schema rejection).
    """
    client = MagicMock()
    if isinstance(payload, Exception):
        client.responses.create = AsyncMock(side_effect=payload)
    else:
        client.responses.create = AsyncMock(return_value=_parsed_response(payload))
    client.responses.parse = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Trigger filter
# ---------------------------------------------------------------------------


def test_clean_run_has_no_triggers():
    assert find_triggers(_make_result(goal_achieved=True)) == []


def test_benign_max_turns_has_no_triggers():
    result = _make_result(goal_achieved=False, terminated_by=TerminatedBy.max_turns)
    assert find_triggers(result) == []


def test_errored_run_has_no_triggers():
    result = _make_result(
        goal_achieved=False,
        rules_broken=['leaked internal data'],
        metadata={'error': 'boom'},
    )
    assert find_triggers(result) == []


def test_rules_broken_triggers():
    result = _make_result(goal_achieved=False, rules_broken=['leaked internal data'])
    assert ('rule_broken', 'leaked internal data') in find_triggers(result)


def test_failed_criterion_triggers():
    result = _make_result(
        goal_achieved=False,
        metadata={
            'criteria_meta': [
                {'id': 'c0', 'description': 'explain the charge', 'type': 'must_happen', 'passed': False},
            ]
        },
    )
    assert ('criterion_failed', 'explain the charge') in find_triggers(result)


def test_low_factual_accuracy_triggers():
    result = _make_result(goal_achieved=True, factual_accuracy=0.2)
    triggers = find_triggers(result)
    assert any(kind == 'low_factual_accuracy' for kind, _ in triggers)


def test_config_threshold_overrides_default():
    result = _make_result(goal_achieved=True, factual_accuracy=0.2)
    config = SimulationRecommendationConfig(factual_accuracy_below=0.1)
    assert find_triggers(result, config) == []


def test_threshold_endpoints_disable_their_check():
    """0.0 / 1.0 are documented as "never trigger on this metric" — assert they do that."""
    result = _make_result(goal_achieved=True, factual_accuracy=0.0)
    off = SimulationRecommendationConfig(factual_accuracy_below=0.0, hallucination_risk_above=1.0)
    assert not [kind for kind, _ in find_triggers(result, off) if kind == 'low_factual_accuracy']


def test_config_rejects_meaningless_values():
    with pytest.raises(ValidationError):
        SimulationRecommendationConfig(max_suggestions=0)
    with pytest.raises(ValidationError):
        SimulationRecommendationConfig(hallucination_risk_above=1.5)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_triggered_results_makes_no_llm_calls():
    client = _mock_client(['unused'])
    recs = await generate_recommendations([_make_result()], client, 'test-model')
    assert recs == []
    client.responses.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_broken_rule_produces_targeted_recommendation():
    client = _mock_client(['Add a system prompt rule forbidding internal data'])
    result = _make_result(
        goal_achieved=False,
        rules_broken=['leaked internal data'],
        metadata={'datapoint_id': 'dp-7'},
    )
    recs = await generate_recommendations([_make_result(), result], client, 'test-model')

    assert len(recs) == 1
    rec = recs[0]
    assert rec.result_index == 1
    assert rec.datapoint_id == 'dp-7'
    assert rec.persona == 'Alice'
    assert rec.scenario == 'Billing'
    assert rec.triggers == ['rule_broken: leaked internal data']
    assert rec.suggestions == ['Add a system prompt rule forbidding internal data']

    call_kwargs = client.responses.create.await_args.kwargs
    user_prompt = call_kwargs['input'][1]['content']
    assert 'leaked internal data' in user_prompt
    # Reasoning models reject non-default temperatures; omit unless asked for.
    assert 'temperature' not in call_kwargs


@pytest.mark.asyncio
async def test_recommendation_call_inherits_run_metadata():
    client = _mock_client(['Fix it'])
    result = _make_result(goal_achieved=False, rules_broken=['leaked data'])

    with evaluatorq_pipeline('agent_simulation'), evaluatorq_run_id('sim-run-1'):
        await generate_recommendations([result], client, 'test-model')

    assert client.responses.create.await_args.kwargs['metadata'] == {
        'evaluatorq_pipeline': 'agent_simulation',
        'evaluatorq_run_id': 'sim-run-1',
    }


@pytest.mark.asyncio
async def test_recommendation_call_injects_active_trace_headers(monkeypatch: pytest.MonkeyPatch):
    client = _mock_client(['Fix it'])
    result = _make_result(goal_achieved=False, rules_broken=['leaked data'])
    # The Responses leg injects headers through llm_call.apply_trace_headers,
    # not through structured_output's own import — patch the transport.
    monkeypatch.setattr(
        'evaluatorq.common.llm_call.get_trace_context_headers',
        AsyncMock(return_value={'traceparent': 'active-trace', 'tracestate': 'active-state'}),
    )

    await generate_recommendations(
        [result],
        client,
        'test-model',
        llm_kwargs={'extra_headers': {'x-caller': 'preserved', 'traceparent': 'caller-trace'}},
    )

    # The active trace wins over a stale caller traceparent, but other caller
    # headers survive.
    assert client.responses.create.await_args.kwargs['extra_headers'] == {
        'x-caller': 'preserved',
        'traceparent': 'active-trace',
        'tracestate': 'active-state',
    }


@pytest.mark.asyncio
async def test_llm_failure_skips_result_without_raising():
    client = _mock_client(RuntimeError('boom'))
    result = _make_result(goal_achieved=False, rules_broken=['rude tone'])
    recs = await generate_recommendations([result], client, 'test-model')
    assert recs == []


@pytest.mark.asyncio
async def test_fenced_json_object_fallback_parses():
    """parse() rejected (400) -> json_object fallback, and a fenced ```json
    payload still yields suggestions instead of dropping the section."""
    client = MagicMock()
    client.responses.create = AsyncMock(side_effect=_schema_400())
    client.responses.parse = AsyncMock()
    client.chat.completions.parse = AsyncMock(side_effect=_schema_400())
    fenced = '```json\n{"suggestions": ["Add a guardrail on ticket IDs"]}\n```'
    client.chat.completions.create = AsyncMock(return_value=_fallback_response(fenced))
    result = _make_result(goal_achieved=False, rules_broken=['leaked data'])

    recs = await generate_recommendations([result], client, 'some/legacy-model')

    assert len(recs) == 1
    assert recs[0].suggestions == ['Add a guardrail on ticket IDs']


@pytest.mark.asyncio
async def test_malformed_fallback_is_swallowed():
    """A malformed fallback payload skips the result with a warning, no crash."""
    client = MagicMock()
    client.responses.create = AsyncMock(side_effect=_schema_400())
    client.responses.parse = AsyncMock()
    client.chat.completions.parse = AsyncMock(side_effect=_schema_400())
    client.chat.completions.create = AsyncMock(return_value=_fallback_response('not json at all'))
    result = _make_result(goal_achieved=False, rules_broken=['leaked data'])

    recs = await generate_recommendations([result], client, 'some/legacy-model')

    assert recs == []


@pytest.mark.asyncio
async def test_max_results_caps_llm_calls():
    client = _mock_client(['fix it'])
    results = [_make_result(goal_achieved=False, rules_broken=['r']) for _ in range(5)]
    recs = await generate_recommendations(results, client, 'test-model', max_results=2)
    assert len(recs) == 2
    assert client.responses.create.await_count == 2


@pytest.mark.asyncio
async def test_config_max_results_caps_llm_calls():
    """The cap is a config field now; the keyword is an override on top of it."""
    client = _mock_client(['fix it'])
    results = [_make_result(goal_achieved=False, rules_broken=['r']) for _ in range(5)]

    recs = await generate_recommendations(
        results, client, 'test-model', config=SimulationRecommendationConfig(max_results=2)
    )

    assert len(recs) == 2
    assert client.responses.create.await_count == 2


@pytest.mark.asyncio
async def test_config_max_results_is_overridden_by_the_keyword():
    client = _mock_client(['fix it'])
    results = [_make_result(goal_achieved=False, rules_broken=['r']) for _ in range(5)]

    recs = await generate_recommendations(
        results, client, 'test-model', max_results=1, config=SimulationRecommendationConfig(max_results=4)
    )

    assert len(recs) == 1


@pytest.mark.asyncio
async def test_config_drives_token_budget_and_prompt_budgets():
    client = _mock_client(['fix it'])
    result = _make_result(goal_achieved=False, rules_broken=['leaked data'])
    result.messages = [Message(role='user', content='x' * 5000)]

    await generate_recommendations(
        [result],
        client,
        'test-model',
        config=SimulationRecommendationConfig(
            max_tokens=123, max_suggestions=9, max_message_chars=60, max_transcript_chars=200
        ),
    )

    kwargs = client.responses.create.await_args.kwargs
    assert kwargs['max_output_tokens'] == 123
    system, user = kwargs['input']
    assert 'a list of 1-9 concise' in system['content']
    # Per-message budget bites first (60 + the ellipsis), then the transcript cap.
    assert 'x' * 61 not in user['content']


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _sample_recommendation() -> SimulationRecommendation:
    return SimulationRecommendation(
        result_index=0,
        datapoint_id='dp-1',
        persona='Alice',
        scenario='Billing',
        triggers=['rule_broken: leaked internal data'],
        suggestions=['Add a guardrail blocking internal identifiers'],
    )


def test_sections_include_recommendations_only_when_provided():
    results = [_make_result()]
    kinds = [s.kind for s in build_report_sections(results)]
    assert 'recommendations' not in kinds

    kinds = [s.kind for s in build_report_sections(results, recommendations=[_sample_recommendation()])]
    assert 'recommendations' in kinds


def test_markdown_renders_recommendations():
    md = export_markdown([_make_result()], recommendations=[_sample_recommendation()])
    assert 'Remediation Suggestions' in md
    assert 'Add a guardrail blocking internal identifiers' in md
    assert 'Rule broken: leaked internal data' in md


def test_html_renders_recommendations():
    html = export_html([_make_result()], recommendations=[_sample_recommendation()])
    assert 'Remediation Suggestions' in html
    assert 'Add a guardrail blocking internal identifiers' in html


def test_html_omits_recommendations_section_when_absent():
    html = export_html([_make_result()])
    assert 'Remediation Suggestions' not in html


def test_find_triggers_resolves_rule_ids_and_dedupes():
    """Broken-rule ids resolve to the criterion description, and a rule that is
    also a failed criterion is reported once, not twice."""
    r = _make_result()
    r.rules_broken = ['criteria_0']
    r.metadata['criteria_meta'] = [
        {
            'id': 'criteria_0',
            'description': 'No internal identifiers leak.',
            'type': 'must_not_happen',
            'passed': False,
        },
    ]
    triggers = find_triggers(r)
    descs = [evidence for _, evidence in triggers]
    assert descs.count('No internal identifiers leak.') == 1
    assert 'criteria_0' not in descs


@pytest.mark.asyncio
async def test_fallback_tolerates_non_string_suggestions():
    """A stray number/null in the fallback payload is coerced or dropped instead
    of dropping the whole suggestion (review fix: keep the old tolerance)."""
    client = MagicMock()
    client.responses.create = AsyncMock(side_effect=_schema_400())
    client.responses.parse = AsyncMock()
    client.chat.completions.parse = AsyncMock(side_effect=_schema_400())
    sloppy = '{"suggestions": [1, "Add a guardrail", null]}'
    client.chat.completions.create = AsyncMock(return_value=_fallback_response(sloppy))
    result = _make_result(goal_achieved=False, rules_broken=['leaked data'])

    recs = await generate_recommendations([result], client, 'some/legacy-model')

    assert len(recs) == 1
    assert recs[0].suggestions == ['1', 'Add a guardrail']


@pytest.mark.asyncio
async def test_batch_is_wrapped_in_one_span(monkeypatch: pytest.MonkeyPatch):
    """Without the wrapper each per-result call attaches to the trace root, so a
    batch renders as N loose top-level LLM spans after the run span has closed."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('test')
    monkeypatch.setattr('evaluatorq.simulation.tracing.get_tracer', lambda: tracer)
    monkeypatch.setattr('evaluatorq.common.tracing.get_tracer', lambda: tracer)

    client = _mock_client(['fix it'])
    results = [_make_result(goal_achieved=False, rules_broken=['r']) for _ in range(3)]
    await generate_recommendations(results, client, 'test-model')

    provider.shutdown()
    finished = exporter.get_finished_spans()
    batch = [s for s in finished if s.name == 'orq.simulation.recommendation_generation']
    assert len(batch) == 1
    batch_attrs = batch[0].attributes or {}
    batch_context = batch[0].context
    assert batch_context is not None
    assert batch_attrs['orq.simulation.recommendation_count'] == 3
    llm_spans = [s for s in finished if s.name == 'responses test-model']
    assert len(llm_spans) == 3
    assert all(s.parent is not None and s.parent.span_id == batch_context.span_id for s in llm_spans)


@pytest.mark.asyncio
async def test_no_triggers_opens_no_batch_span(monkeypatch: pytest.MonkeyPatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr('evaluatorq.simulation.tracing.get_tracer', lambda: provider.get_tracer('test'))

    assert await generate_recommendations([_make_result()], _mock_client(['unused']), 'test-model') == []
    provider.shutdown()
    assert exporter.get_finished_spans() == ()


@pytest.mark.asyncio
async def test_batch_runs_concurrently():
    """The per-result calls are independent; running them serially made a
    10-result batch a 20s staircase at the end of every run."""
    import asyncio

    in_flight = 0
    peak = 0

    async def _slow_create(**_kwargs: Any) -> Any:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.01)
            return _parsed_response(['fix it'])
        finally:
            in_flight -= 1

    client = MagicMock()
    client.responses.create = _slow_create
    client.responses.parse = AsyncMock()
    results = [_make_result(goal_achieved=False, rules_broken=['r']) for _ in range(5)]

    recs = await generate_recommendations(results, client, 'test-model')

    assert len(recs) == 5
    assert peak == 5
    # gather preserves order, so recommendations still map back to result order.
    assert [r.result_index for r in recs] == [0, 1, 2, 3, 4]
