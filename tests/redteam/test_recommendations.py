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


@pytest.mark.asyncio
async def test_wire_param_is_max_completion_tokens(
    mock_client_and_capture: tuple[Any, dict[str, Any]],
) -> None:
    """The token budget goes out as ``max_completion_tokens`` — OpenAI rejects
    ``max_tokens`` outright for the o-series and gpt-5 families, and every other
    chat call in the repo sends this key (review fix)."""
    client, captured = mock_client_and_capture

    await generate_focus_area_recommendations(_empty_report(), client, model='openai/gpt-5-mini')

    assert captured['max_completion_tokens'] == 1500
    assert 'max_tokens' not in captured


@pytest.mark.asyncio
async def test_user_extra_body_merges_with_router_retry(
    mock_client_and_capture: tuple[Any, dict[str, Any]],
) -> None:
    """A caller-supplied ``extra_body`` merges INTO the router retry body instead
    of silently replacing it (review fix: retry hints must not vanish)."""
    client, captured = mock_client_and_capture
    client.base_url = 'https://my.orq.ai/v3/router'
    cfg = LLMConfig()

    await generate_focus_area_recommendations(
        _empty_report(),
        client,
        model='openai/gpt-5-mini',
        cfg=cfg,
        llm_kwargs={'extra_body': {'provider_hint': 'x'}},
    )

    assert captured['extra_body']['retry'] == {'count': cfg.retry_count, 'on_codes': cfg.retry_on_codes}
    assert captured['extra_body']['provider_hint'] == 'x'


@pytest.mark.asyncio
async def test_fallback_tolerates_non_string_items(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stray number/null in the fallback payload is coerced or dropped instead
    of dropping the whole focus area — the fallback runs on exactly the models
    most likely to emit one (review fix: keep the pre-RES-822 tolerance)."""

    def _fake_compute(_r: RedTeamReport, _n: int) -> list[dict[str, Any]]:
        return [_fake_area()]

    monkeypatch.setattr(rec_mod, '_compute_top_risk_areas', _fake_compute)

    async def fake_parse(**_kwargs: Any) -> Any:
        raise _schema_400()

    sloppy = '{"recommendations": [1, "Add an allowlist", null], "patterns_observed": 5}'

    async def fake_create(**_kwargs: Any) -> Any:
        return _fallback_response(sloppy)

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=fake_parse)
    client.chat.completions.create = AsyncMock(side_effect=fake_create)

    recs = await generate_focus_area_recommendations(_empty_report(), client, model='some/legacy-model')

    assert recs, 'one bad item must not drop the section'
    assert recs[0].recommendations == ['1', 'Add an allowlist']
    assert recs[0].patterns_observed == '5'


# ---------------------------------------------------------------------------
# RedTeamRecommendationConfig knobs (RES-1286)
# ---------------------------------------------------------------------------


def _capturing_client(monkeypatch: pytest.MonkeyPatch, recommendations: list[str]) -> tuple[Any, dict[str, Any]]:
    """Like ``mock_client_and_capture`` but with a caller-chosen reply."""
    monkeypatch.setattr(rec_mod, '_compute_top_risk_areas', lambda _r, _n: [_fake_area()])
    captured: dict[str, Any] = {}

    async def fake_parse(**kwargs: Any) -> Any:
        captured.update(kwargs)
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.refusal = None
        response.choices[0].message.parsed = _FocusAreaLLMResponse(
            recommendations=recommendations,
            patterns_observed='Agent acted beyond scope',
        )
        return response

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=fake_parse)
    return client, captured


@pytest.mark.asyncio
async def test_config_drives_token_budget_and_suggestion_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_tokens and max_suggestions reach the call, and the cap is enforced on
    the reply rather than only requested in the prompt."""
    from evaluatorq.redteam.contracts import RedTeamRecommendationConfig

    client, captured = _capturing_client(monkeypatch, ['one', 'two', 'three'])

    recs = await generate_focus_area_recommendations(
        _empty_report(),
        client,
        model='openai/gpt-5-mini',
        recommendations=RedTeamRecommendationConfig(max_tokens=222, max_suggestions=2),
    )

    assert captured['max_completion_tokens'] == 222
    assert 'a list of at most 2 concise' in captured['messages'][0]['content']
    assert recs[0].recommendations == ['one', 'two']


@pytest.mark.asyncio
async def test_config_attack_budget_truncates_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-attack budget is a config field, not the old 500-char constant."""
    from evaluatorq.redteam.contracts import RedTeamRecommendationConfig

    long_result = _vulnerable_result()
    long_result.response = 'z' * 400
    monkeypatch.setattr(
        rec_mod, '_compute_top_risk_areas', lambda _r, _n: [{**_fake_area(), 'vulnerable_results': [long_result]}]
    )

    captured: dict[str, Any] = {}

    async def fake_parse(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _parsed_response()

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=fake_parse)

    await generate_focus_area_recommendations(
        _empty_report(),
        client,
        model='openai/gpt-5-mini',
        recommendations=RedTeamRecommendationConfig(max_attack_chars=100),
    )

    user_prompt = captured['messages'][1]['content']
    assert 'z' * 100 + '...' in user_prompt
    assert 'z' * 101 not in user_prompt


# ---------------------------------------------------------------------------
# Map-then-reduce: oversized attacks are condensed before the focus-area call
# ---------------------------------------------------------------------------


def _long_attack(chars: int) -> RedTeamResult:
    result = _vulnerable_result()
    result.response = 'z' * chars
    return result


def _routing_client(monkeypatch: pytest.MonkeyPatch, results: list[RedTeamResult]) -> tuple[Any, list[dict[str, Any]]]:
    """Client whose parse() answers both the condense and focus-area schemas.

    Records every call so a test can assert which ones actually happened.
    """
    monkeypatch.setattr(
        rec_mod, '_compute_top_risk_areas', lambda _r, _n: [{**_fake_area(), 'vulnerable_results': results}]
    )
    calls: list[dict[str, Any]] = []

    async def fake_parse(**kwargs: Any) -> Any:
        calls.append(kwargs)
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.refusal = None
        if kwargs['response_format'] is rec_mod._CondensedAttackLLMResponse:  # noqa: SLF001
            response.choices[0].message.parsed = rec_mod._CondensedAttackLLMResponse(  # noqa: SLF001
                analysis='Agent ran the tool without confirming.'
            )
        else:
            response.choices[0].message.parsed = _FocusAreaLLMResponse(
                recommendations=['Gate the tool'], patterns_observed='No confirmation step'
            )
        return response

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=fake_parse)
    return client, calls


@pytest.mark.asyncio
async def test_short_attacks_cost_no_condense_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The map step is conditional: a normal attack goes into the aggregate verbatim."""
    from evaluatorq.redteam.contracts import RedTeamRecommendationConfig

    client, calls = _routing_client(monkeypatch, [_vulnerable_result()])

    await generate_focus_area_recommendations(
        _empty_report(),
        client,
        model='openai/gpt-5-mini',
        recommendations=RedTeamRecommendationConfig(condense_above_chars=1000),
    )

    assert len(calls) == 1  # the focus-area call only
    assert '<prompt>' in calls[0]['messages'][1]['content']


@pytest.mark.asyncio
async def test_oversized_attack_is_condensed_before_the_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long attack becomes an <analysis> block, so the aggregate never sees the bulk."""
    from evaluatorq.redteam.contracts import RedTeamRecommendationConfig

    client, calls = _routing_client(monkeypatch, [_long_attack(5000)])

    await generate_focus_area_recommendations(
        _empty_report(),
        client,
        model='openai/gpt-5-mini',
        recommendations=RedTeamRecommendationConfig(condense_above_chars=1000, condense_max_tokens=42),
    )

    condense, aggregate = calls
    assert condense['max_completion_tokens'] == 42
    aggregate_prompt = aggregate['messages'][1]['content']
    assert '<analysis>Agent ran the tool without confirming.</analysis>' in aggregate_prompt
    assert 'z' * 1000 not in aggregate_prompt


@pytest.mark.asyncio
async def test_failed_condense_truncates_instead_of_losing_the_area(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented degradation: the condense call dies, the area still gets advice."""
    from evaluatorq.redteam.contracts import RedTeamRecommendationConfig

    monkeypatch.setattr(
        rec_mod,
        '_compute_top_risk_areas',
        lambda _r, _n: [{**_fake_area(), 'vulnerable_results': [_long_attack(5000)]}],
    )
    calls: list[dict[str, Any]] = []

    async def fake_parse(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if kwargs['response_format'] is rec_mod._CondensedAttackLLMResponse:  # noqa: SLF001
            raise RuntimeError('condense exploded')
        return _parsed_response()

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=fake_parse)

    recs = await generate_focus_area_recommendations(
        _empty_report(),
        client,
        model='openai/gpt-5-mini',
        recommendations=RedTeamRecommendationConfig(condense_above_chars=1000),
    )

    assert recs[0].recommendations == ['Reduce agent permissions']
    aggregate_prompt = calls[-1]['messages'][1]['content']
    assert 'z' * 1001 not in aggregate_prompt  # hard-truncated to the same budget


@pytest.mark.asyncio
async def test_prompt_ceiling_truncates_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backstop: condensing that did not shrink enough truncates loudly, never silently."""
    from evaluatorq.redteam.contracts import RedTeamRecommendationConfig

    client, calls = _routing_client(monkeypatch, [_long_attack(5000)])
    warnings_seen: list[str] = []
    monkeypatch.setattr(rec_mod.logger, 'warning', lambda msg, *a, **k: warnings_seen.append(str(msg)))

    await generate_focus_area_recommendations(
        _empty_report(),
        client,
        model='openai/gpt-5-mini',
        # Nothing is condensed (budget above the block), so the ceiling is what bites.
        recommendations=RedTeamRecommendationConfig(condense_above_chars=99_000, max_area_prompt_chars=1_000),
    )

    assert len(calls[-1]['messages'][1]['content']) <= 1_003  # budget + the '...' marker
    assert any('max_area_prompt_chars' in w for w in warnings_seen)
