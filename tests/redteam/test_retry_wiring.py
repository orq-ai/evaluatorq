"""Retry is wired at the client-construction seam, not per call site.

Pipeline LLM calls rely on the OpenAI SDK's built-in retry (429/5xx/network,
Retry-After honored), driven by ``LLMConfig.retry_count`` through
``resolve_llm_client(max_retries=...)``. These tests pin that plumbing so a
refactor cannot silently revert a client to a different retry budget, and that
the Orq-SDK backend derives an equivalent client-level ``RetryConfig``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.retry import without_client_retries
from evaluatorq.common.target_call import call_target_with_retry
from evaluatorq.contracts import LLMCallConfig, Message
from evaluatorq.openresponses.client import build_simulation_client
from evaluatorq.redteam.backends.registry import create_async_llm_client
from evaluatorq.redteam.contracts import LLMConfig


@pytest.fixture(autouse=True)
def _orq_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORQ_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_resolve_llm_client_passes_max_retries() -> None:
    client = resolve_llm_client(max_retries=4).client
    assert client.max_retries == 4


def test_resolve_llm_client_default_keeps_sdk_retries() -> None:
    """Unset means the SDK default (2 retries) — never zero, never a third layer."""
    client = resolve_llm_client().client
    assert client.max_retries == 2


def test_resolve_llm_client_openai_branch_passes_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = resolve_llm_client(max_retries=5).client
    assert client.max_retries == 5


def test_without_client_retries_clones_and_preserves_injected_client_settings() -> None:
    http_client = httpx.AsyncClient(base_url="https://example.test/v1")
    injected = AsyncOpenAI(
        api_key="user-key",
        base_url="https://example.test/v1",
        timeout=17.5,
        max_retries=5,
        http_client=http_client,
    )

    disarmed = without_client_retries(injected)

    assert disarmed is not injected
    assert disarmed.max_retries == 0
    assert disarmed.api_key == injected.api_key
    assert disarmed.base_url == injected.base_url
    assert disarmed.timeout == injected.timeout
    assert disarmed._client is injected._client


def test_build_simulation_client_disarms_injected_client_at_with_retry_boundary() -> None:
    injected = AsyncOpenAI(api_key="user-key", max_retries=5)

    client, owned = build_simulation_client(injected, max_retries=0)

    assert client is not injected
    assert client.max_retries == 0
    assert injected.max_retries == 5
    assert owned is False


def test_create_async_llm_client_plumbs_retry_count() -> None:
    cfg = LLMConfig(retry_count=6)
    client = create_async_llm_client(max_retries=cfg.retry_count)
    assert client.max_retries == 6


def test_retry_count_bounds_validated() -> None:
    with pytest.raises(ValueError):
        LLMConfig(retry_count=-1)
    with pytest.raises(ValueError):
        LLMConfig(retry_count=11)


def test_retry_attempts_property() -> None:
    assert LLMConfig(retry_count=2).retry_attempts == 3
    assert LLMConfig(retry_count=0).retry_attempts == 1


def test_orq_sdk_retry_config_derived_from_pipeline_config() -> None:
    pytest.importorskip("orq_ai_sdk")
    from evaluatorq.redteam.backends.orq import _orq_retry_config

    cfg = _orq_retry_config(3, [429, 500, 502, 503, 504], timeout_ms=240_000)
    assert cfg is not None
    assert cfg.strategy == "backoff"
    assert cfg.retry_connection_errors is True
    assert cfg.status_codes_override == ["429", "500", "502", "503", "504"]
    # Backoff rounds (0.5/1/2s) plus the attempt budget: Speakeasy's
    # max_elapsed_time is total wall clock from before the first attempt, so
    # the window must cover retry_count failed attempts of up to timeout_ms.
    assert cfg.backoff.max_elapsed_time == (500 + 1000 + 2000) + 3 * 240_000


def test_orq_retry_window_survives_a_slow_first_failure() -> None:
    """A single failed attempt taking the full per-call timeout must leave the
    elapsed-time budget with room to retry, or the config is inert for exactly
    the slow-failure cases retry exists for."""
    pytest.importorskip("orq_ai_sdk")
    from evaluatorq.redteam.backends.orq import _orq_retry_config

    timeout_ms = 240_000
    cfg = _orq_retry_config(3, [429], timeout_ms=timeout_ms)
    assert cfg.backoff.max_elapsed_time > timeout_ms


def test_orq_sdk_retry_config_disabled_at_zero() -> None:
    pytest.importorskip("orq_ai_sdk")
    from evaluatorq.redteam.backends.orq import _orq_retry_config

    assert _orq_retry_config(0, [429]) is None


def test_orq_responses_target_client_carries_no_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The target owns retry via with_retry, so its self-built client must have
    max_retries=0 — otherwise the two layers stack to retry_attempts x SDK
    attempts HTTP requests per call (the defect RES-832 removed)."""
    monkeypatch.setenv("ORQ_API_KEY", "test-key")
    from evaluatorq.openresponses.target import LLMCallConfig, OrqResponsesTarget

    target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"))
    assert target._client.max_retries == 0

    # An injected client is cloned so with_retry remains the sole owner.
    from openai import AsyncOpenAI

    injected = AsyncOpenAI(api_key="user-key", max_retries=5)
    target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), client=injected)
    assert target._client is not injected
    assert target._client.max_retries == 0
    assert injected.max_retries == 5


@pytest.mark.asyncio
async def test_orchestrator_target_call_has_exactly_outer_retry_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The orchestrator's two retries must produce three HTTP calls, not 12."""
    from evaluatorq.redteam.backends.registry import _create_openresponses_backend

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            429,
            request=request,
            headers={"content-type": "application/json"},
            json={"error": {"message": "rate limited", "type": "rate_limit_error"}},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://example.test/v1")
    sdk_client = AsyncOpenAI(api_key="test-key", max_retries=0, http_client=http_client)
    monkeypatch.setattr(
        "evaluatorq.openresponses.target.build_simulation_client",
        lambda *_args, **_kwargs: (sdk_client, False),
    )
    monkeypatch.setattr("evaluatorq.common.retry.asyncio.sleep", AsyncMock())

    backend = _create_openresponses_backend(pipeline_config=LLMConfig())
    target = backend.create_target("gpt-4o")
    result = await call_target_with_retry(
        target,
        [Message(role="user", content="hello")],
        target_agent_timeout_ms=10_000,
        max_target_retries=2,
    )
    await http_client.aclose()

    assert result.succeeded is False
    assert result.attempts == 3
    assert attempts == 3


def test_openai_model_target_disables_sdk_retries_when_auto_built() -> None:
    from evaluatorq.redteam.backends.openai import OpenAIModelTarget

    with patch("evaluatorq.redteam.backends.openai.create_async_llm_client", return_value=MagicMock()) as create:
        OpenAIModelTarget("gpt-4o")

    create.assert_called_once_with(max_retries=0)


def test_openai_backend_disables_sdk_retries_when_auto_built() -> None:
    from evaluatorq.redteam.backends.openai import OpenAIBackend

    with patch("evaluatorq.redteam.backends.openai.create_async_llm_client", return_value=MagicMock()) as create:
        OpenAIBackend()

    create.assert_called_once_with(max_retries=0)


def test_openai_backend_warns_when_target_retry_settings_are_ignored() -> None:
    """OpenAI target retries belong to call_target_with_retry, so ignored settings must be visible."""
    from evaluatorq.redteam.backends.registry import _create_openai_backend

    with patch('evaluatorq.redteam.backends._retry.logger.warning') as warning:
        _create_openai_backend(llm_client=MagicMock(), pipeline_config=LLMConfig(retry_count=2))

    warning.assert_called_once()
    message = warning.call_args.args[0]
    assert 'retry_count' in message
    assert 'retry_on_codes' in message


def test_openresponses_backend_uses_the_same_ignored_retry_warning() -> None:
    from evaluatorq.redteam.backends.registry import _create_openresponses_backend

    with patch('evaluatorq.redteam.backends._retry.logger.warning') as warning:
        _create_openresponses_backend(pipeline_config=LLMConfig(retry_count=2))

    warning.assert_called_once()
    message = warning.call_args.args[0]
    assert 'retry_count' in message
    assert 'retry_on_codes' in message


def test_openai_backend_does_not_warn_for_default_pipeline_retry_settings() -> None:
    from evaluatorq.redteam.backends.registry import _create_openai_backend

    with patch('evaluatorq.redteam.backends._retry.logger.warning') as warning:
        _create_openai_backend(llm_client=MagicMock(), pipeline_config=LLMConfig())

    warning.assert_not_called()


def test_openai_backend_warns_when_direct_retry_settings_are_ignored() -> None:
    """Direct factory callers must not lose the warning for ignored retry kwargs."""
    from evaluatorq.redteam.backends.registry import _create_openai_backend

    with patch('evaluatorq.redteam.backends._retry.logger.warning') as warning:
        _create_openai_backend(llm_client=MagicMock(), retry_count=2, retry_on_codes=[429, 503])

    warning.assert_called_once()
    message = warning.call_args.args[0]
    assert 'retry_count' in message
    assert 'retry_on_codes' in message


def test_orq_backend_preserves_shared_sdk_retries_when_auto_built(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluatorq.redteam.backends.orq as orq_backend

    calls: list[dict[str, object]] = []

    class FakeOrq:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(orq_backend, "_orq_cls", FakeOrq)
    monkeypatch.setenv("ORQ_API_KEY", "test-key")

    backend = orq_backend.ORQBackend(retry_count=3, retry_on_codes=[429, 503], timeout_ms=240_000)

    assert len(calls) == 1
    assert calls[0]["retry_config"] is not None
    assert backend.create_target("agent").orq_client is backend._orq_client


def test_orq_backend_warns_when_injected_retry_config_cannot_be_reconfigured() -> None:
    import evaluatorq.redteam.backends.orq as orq_backend

    with patch.object(orq_backend.logger, "warning") as warning:
        orq_backend.ORQBackend(orq_client=MagicMock(), retry_count=2, retry_on_codes=[429])

    warning.assert_called_once()
    assert warning.call_args.args[0] == (
        'Ignoring retry_count and retry_on_codes for ORQ target calls; '
        'call_target_with_retry owns target retries and the SDK retry budget is '
        'disabled at the target-call boundary'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'path',
    [
        'openai-auto',
        'openai-injected',
        'openresponses-auto',
        'openresponses-injected',
        'orq-auto',
        'orq-injected',
        'simulation-client',
        'judge',
    ],
)
async def test_retry_paths_make_exactly_three_http_attempts(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every retry owner gets three transport attempts at a two-retry budget.

    The clients deliberately start with ``max_retries=2``. A path that forgets
    to disarm the SDK before adding its own retry wrapper therefore produces
    nine requests, making this a transport-level matrix rather than a test of
    mocked method call counts.
    """
    max_target_retries = 2
    expected_attempts = max_target_retries + 1
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            429,
            request=request,
            headers={'content-type': 'application/json'},
            json={'error': {'message': 'rate limited', 'type': 'rate_limit_error'}},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url='https://example.test/v1')
    sdk_client = AsyncOpenAI(api_key='test-key', max_retries=2, http_client=http_client)
    monkeypatch.setattr('evaluatorq.common.retry.asyncio.sleep', AsyncMock())

    target: Any = None
    sync_client: httpx.Client | None = None

    if path == 'openai-auto':
        from evaluatorq.redteam.backends.openai import OpenAIBackend

        def create_openai(*_args: object, **kwargs: object) -> AsyncOpenAI:
            assert kwargs['max_retries'] == 0
            return sdk_client

        with patch('evaluatorq.redteam.backends.openai.create_async_llm_client', side_effect=create_openai):
            target = OpenAIBackend().create_target('gpt-4o')
    elif path == 'openai-injected':
        from evaluatorq.redteam.backends.openai import OpenAIBackend

        target = OpenAIBackend(client=sdk_client).create_target('gpt-4o')
    elif path.startswith('openresponses'):
        from evaluatorq.redteam.backends.registry import _create_openresponses_backend

        if path == 'openresponses-auto':
            def build(*_args: object, **kwargs: object) -> tuple[AsyncOpenAI, bool]:
                assert kwargs['max_retries'] == 0
                return build_simulation_client(sdk_client, max_retries=0)

            monkeypatch.setattr('evaluatorq.openresponses.target.build_simulation_client', build)
            backend = _create_openresponses_backend(pipeline_config=LLMConfig())
        else:
            backend = _create_openresponses_backend(llm_client=sdk_client, pipeline_config=LLMConfig())
        target = backend.create_target('gpt-4o')
    elif path.startswith('orq'):
        pytest.importorskip('orq_ai_sdk')
        from orq_ai_sdk import Orq

        import evaluatorq.redteam.backends.orq as orq_backend
        from evaluatorq.redteam.backends.orq import ORQBackend

        sync_client = httpx.Client(transport=httpx.MockTransport(handler), base_url='https://example.test/v1')
        if path == 'orq-auto':
            def create_orq(**kwargs: Any) -> Orq:
                return Orq(client=sync_client, **kwargs)

            monkeypatch.setattr(orq_backend, '_orq_cls', create_orq)
            backend = ORQBackend()
        else:
            orq_client = Orq(
                api_key='test-key',
                server_url='https://example.test',
                client=sync_client,
                retry_config=None,
            )
            backend = ORQBackend(orq_client=orq_client)
        target = backend.create_target('agent')
    elif path == 'simulation-client':
        from evaluatorq.simulation.agents.user_simulator import UserSimulatorAgent

        # No monkeypatch on with_retry: the budget must come from the agent's own
        # config, or this leg proves nothing about the production call path.
        agent = UserSimulatorAgent(LLMCallConfig(model='gpt-4o', client=sdk_client, retry_count=max_target_retries))
        with pytest.raises(APIStatusError):
            await agent.respond_async([Message(role='user', content='hello')])
    else:
        from evaluatorq.common.judge import run_judge

        outcome = await run_judge(
            client=sdk_client,
            model='gpt-4o',
            cfg=LLMCallConfig(model='gpt-4o', api='chat_completions', retry_count=2),
            prompt_template='judge this',
            replacements={},
            structured_output=False,
        )
        assert outcome.error_kind is not None

    if path in {'simulation-client', 'judge'}:
        await http_client.aclose()
    else:
        result = await call_target_with_retry(
            target,
            [Message(role='user', content='hello')],
            target_agent_timeout_ms=10_000,
            max_target_retries=max_target_retries,
        )
        assert result.succeeded is False
        assert result.attempts == expected_attempts

    if sync_client is not None:
        sync_client.close()
    assert attempts == expected_attempts
