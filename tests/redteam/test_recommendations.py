"""Tests for focus-area recommendation generation.

RES-822: the call now goes through ``common.structured_output.generate_structured``
— ``chat.completions.parse()`` first, ``json_object`` (``create()``) fallback for
models that reject structured output, with fence-tolerant parsing of the
fallback payload. These tests pin that path plus the RES-817 regression that the
temperature and other call kwargs come from ``cfg.evaluator`` rather than a
hardcoded value.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIStatusError

from evaluatorq.common.thread_context import evaluatorq_pipeline, evaluatorq_run_id
from evaluatorq.redteam.contracts import (
    AgentInfo,
    AttackInfo,
    AttackTechnique,
    DeliveryMethod,
    EvaluatorConfig,
    Framework,
    LLMConfig,
    Message,
    Pipeline,
    RedTeamReport,
    RedTeamResult,
    ReportSummary,
    Severity,
    TurnType,
    UnifiedEvaluationResult,
)
from evaluatorq.redteam.reports import recommendations as rec_mod
from evaluatorq.redteam.reports.recommendations import _FocusAreaLLMResponse, generate_focus_area_recommendations


def _empty_report() -> RedTeamReport:
    return RedTeamReport(
        created_at=datetime.now(timezone.utc),
        pipeline=Pipeline.DYNAMIC,
        categories_tested=['ASI01'],
        total_results=0,
        results=[],
        summary=ReportSummary(),
    )


def _vulnerable_result() -> RedTeamResult:
    return RedTeamResult(
        attack=AttackInfo(
            id='LLM06-test-001',
            category='LLM06',
            framework=Framework.OWASP_LLM,
            attack_technique=AttackTechnique.INDIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
            turn_type=TurnType.SINGLE,
            severity=Severity.MEDIUM,
            source='template_dynamic',
        ),
        agent=AgentInfo(key='test_agent', model='openai/gpt-5-mini'),
        messages=cast(list[Message], [{'role': 'user', 'content': 'do bad thing'}]),
        response='Sure, here is the bad thing.',
        evaluation=UnifiedEvaluationResult(
            passed=False,
            explanation='Agent complied with adversarial request.',
            evaluator_id='LLM06',
        ),
        vulnerable=True,
    )


def _fake_area() -> dict[str, Any]:
    return {
        'category': 'LLM06',
        'category_name': 'Excessive Agency',
        'vulnerability': 'LLM06',
        'vulnerability_name': 'Excessive Agency',
        'vulnerability_rate': 0.5,
        'risk_score': 0.5,
        'vulnerable_results': [_vulnerable_result()],
    }


def _parsed_response() -> Any:
    """A parse() response whose message carries a validated model (happy path)."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.refusal = None
    response.choices[0].message.parsed = _FocusAreaLLMResponse(
        recommendations=['Reduce agent permissions'],
        patterns_observed='Agent acted beyond scope',
    )
    return response


def _fallback_response(content: str) -> Any:
    """A create() response for the json_object fallback path."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _schema_400() -> APIStatusError:
    """A 400 whose body reads as a structured-output-unsupported error, which is
    what trips generate_structured's json_object fallback."""
    request = httpx.Request('POST', 'https://router.example/v3/router')
    response = httpx.Response(400, request=request)
    return APIStatusError('response_format not supported', response=response, body=None)


@pytest.fixture
def mock_client_and_capture(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    """AsyncOpenAI mock whose ``parse()`` succeeds and captures its kwargs.

    Stubs ``_compute_top_risk_areas`` so no full report fixture is needed.
    """

    def _fake_compute(_r: RedTeamReport, _n: int) -> list[dict[str, Any]]:
        return [_fake_area()]

    monkeypatch.setattr(rec_mod, '_compute_top_risk_areas', _fake_compute)

    captured: dict[str, Any] = {}

    async def fake_parse(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _parsed_response()

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=fake_parse)
    return client, captured


@pytest.mark.asyncio
async def test_default_cfg_uses_reasoning_safe_temperature(
    mock_client_and_capture: tuple[Any, dict[str, Any]],
) -> None:
    """Without an explicit cfg, the call uses ``LLMConfig`` defaults
    (``temperature=1.0``) — which reasoning models accept."""
    client, captured = mock_client_and_capture

    recs = await generate_focus_area_recommendations(_empty_report(), client, model='openai/gpt-5-mini')

    assert recs, 'Expected at least one recommendation'
    assert recs[0].recommendations == ['Reduce agent permissions']
    assert recs[0].patterns_observed == 'Agent acted beyond scope'
    assert captured['temperature'] == 1.0


@pytest.mark.asyncio
async def test_explicit_evaluator_temperature_is_forwarded(
    mock_client_and_capture: tuple[Any, dict[str, Any]],
) -> None:
    """Caller-supplied ``cfg.evaluator.temperature`` is passed through verbatim."""
    client, captured = mock_client_and_capture
    cfg = LLMConfig(evaluator=EvaluatorConfig(model='openai/gpt-4o-mini', temperature=0.0))

    await generate_focus_area_recommendations(_empty_report(), client, model='openai/gpt-4o-mini', cfg=cfg)

    assert captured['temperature'] == 0.0


@pytest.mark.asyncio
async def test_evaluator_extra_kwargs_merged(
    mock_client_and_capture: tuple[Any, dict[str, Any]],
) -> None:
    """``cfg.evaluator.extra_kwargs`` are merged into the call."""
    client, captured = mock_client_and_capture
    cfg = LLMConfig(evaluator=EvaluatorConfig(temperature=1.0, extra_kwargs={'seed': 42}))

    await generate_focus_area_recommendations(_empty_report(), client, model='openai/gpt-5-mini', cfg=cfg)

    assert captured['temperature'] == 1.0
    assert captured.get('seed') == 42


@pytest.mark.asyncio
async def test_extra_body_carries_retry_hints_for_router_client(
    mock_client_and_capture: tuple[Any, dict[str, Any]],
) -> None:
    """``extra_body`` carries router retry hints when the client routes through the Orq router."""
    client, captured = mock_client_and_capture
    client.base_url = 'https://my.orq.ai/v3/router'
    cfg = LLMConfig()

    await generate_focus_area_recommendations(_empty_report(), client, model='openai/gpt-5-mini', cfg=cfg)

    assert captured['extra_body'] == {'retry': {'count': cfg.retry_count, 'on_codes': cfg.retry_on_codes}}


@pytest.mark.asyncio
async def test_extra_body_empty_for_non_router_client(
    mock_client_and_capture: tuple[Any, dict[str, Any]],
) -> None:
    """No ORQ-specific ``retry`` is sent to a plain OpenAI client (decision: gate on base_url)."""
    client, captured = mock_client_and_capture
    client.base_url = 'https://api.openai.com/v1'
    cfg = LLMConfig()

    await generate_focus_area_recommendations(_empty_report(), client, model='gpt-5-mini', cfg=cfg)

    assert captured['extra_body'] == {}


@pytest.mark.asyncio
async def test_run_metadata_is_top_level_and_preserves_router_extra_body(
    mock_client_and_capture: tuple[Any, dict[str, Any]],
) -> None:
    client, captured = mock_client_and_capture
    client.base_url = 'https://my.orq.ai/v3/router'

    with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('recommendation-run'):
        await generate_focus_area_recommendations(_empty_report(), client, model='gpt-5-mini')

    assert captured['metadata'] == {
        'evaluatorq_pipeline': 'red_teaming',
        'evaluatorq_run_id': 'recommendation-run',
    }
    assert captured['extra_body'] == {'retry': {'count': 3, 'on_codes': [429, 500, 502, 503, 504]}}


@pytest.mark.asyncio
async def test_llm_kwargs_override_evaluator_extra_kwargs(
    mock_client_and_capture: tuple[Any, dict[str, Any]],
) -> None:
    """Caller-supplied ``llm_kwargs`` win over ``cfg.evaluator.extra_kwargs``."""
    client, captured = mock_client_and_capture
    cfg = LLMConfig(evaluator=EvaluatorConfig(extra_kwargs={'seed': 1}))

    await generate_focus_area_recommendations(
        _empty_report(),
        client,
        model='openai/gpt-5-mini',
        cfg=cfg,
        llm_kwargs={'seed': 99},
    )

    assert captured['seed'] == 99


@pytest.mark.asyncio
async def test_fenced_json_object_fallback_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """When parse() is rejected (400) the call falls back to json_object, and a
    fenced ```json payload still parses instead of dropping the section — the
    exact provider path RES-822 exists to fix."""

    def _fake_compute(_r: RedTeamReport, _n: int) -> list[dict[str, Any]]:
        return [_fake_area()]

    monkeypatch.setattr(rec_mod, '_compute_top_risk_areas', _fake_compute)

    async def fake_parse(**_kwargs: Any) -> Any:
        raise _schema_400()

    fenced = '```json\n{"recommendations": ["Add an allowlist"], "patterns_observed": "Broad tool scope"}\n```'

    async def fake_create(**_kwargs: Any) -> Any:
        return _fallback_response(fenced)

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=fake_parse)
    client.chat.completions.create = AsyncMock(side_effect=fake_create)

    recs = await generate_focus_area_recommendations(_empty_report(), client, model='some/legacy-model')

    assert recs, 'fenced fallback payload should still produce a recommendation'
    assert recs[0].recommendations == ['Add an allowlist']
    assert recs[0].patterns_observed == 'Broad tool scope'


@pytest.mark.asyncio
async def test_malformed_fallback_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed fallback payload degrades to no-recommendations with a
    warning, never crashing the run."""

    def _fake_compute(_r: RedTeamReport, _n: int) -> list[dict[str, Any]]:
        return [_fake_area()]

    monkeypatch.setattr(rec_mod, '_compute_top_risk_areas', _fake_compute)

    async def fake_parse(**_kwargs: Any) -> Any:
        raise _schema_400()

    async def fake_create(**_kwargs: Any) -> Any:
        return _fallback_response('not json at all')

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=fake_parse)
    client.chat.completions.create = AsyncMock(side_effect=fake_create)

    recs = await generate_focus_area_recommendations(_empty_report(), client, model='some/legacy-model')

    assert recs == []


@pytest.mark.asyncio
async def test_chat_completion_failure_returns_empty_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hard LLM failure is swallowed: the function warns and returns the
    successful subset, never propagating the exception (RES-817)."""

    def _fake_compute(_r: RedTeamReport, _n: int) -> list[dict[str, Any]]:
        return [_fake_area()]

    monkeypatch.setattr(rec_mod, '_compute_top_risk_areas', _fake_compute)

    async def fake_parse(**_kwargs: Any) -> Any:
        raise RuntimeError('400 Bad Request: temperature must be 1.0 for this model')

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=fake_parse)

    recs = await generate_focus_area_recommendations(_empty_report(), client, model='openai/gpt-5-mini')

    assert recs == []
