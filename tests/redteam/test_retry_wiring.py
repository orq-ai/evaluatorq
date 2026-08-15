"""Retry is wired at the client-construction seam, not per call site.

Pipeline LLM calls rely on the OpenAI SDK's built-in retry (429/5xx/network,
Retry-After honored), driven by ``LLMConfig.retry_count`` through
``resolve_llm_client(max_retries=...)``. These tests pin that plumbing so a
refactor cannot silently revert a client to a different retry budget, and that
the Orq-SDK backend derives an equivalent client-level ``RetryConfig``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AsyncOpenAI

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.target_call import call_target_with_retry
from evaluatorq.contracts import LLMCallConfig, Message
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


def test_injected_client_is_never_reconfigured() -> None:
    from openai import AsyncOpenAI

    injected = AsyncOpenAI(api_key="user-key", max_retries=0)
    resolved = resolve_llm_client(injected, max_retries=7)
    assert resolved.client is injected
    assert resolved.client.max_retries == 0


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

    # An injected client stays exactly as the caller built it.
    from openai import AsyncOpenAI

    injected = AsyncOpenAI(api_key="user-key", max_retries=5)
    assert OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), client=injected)._client.max_retries == 5


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
    assert "injected client" in warning.call_args.args[0]
