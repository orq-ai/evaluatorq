"""Tests for the openresponses backend registration.

Verifies the registry can resolve ``backend="openresponses"`` and that the
backend wires target construction, context resolution, and error mapping.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.contracts import AgentResponse
from evaluatorq.redteam.backends.registry import resolve_backend
from evaluatorq.redteam.contracts import AgentContext, LLMConfig, TargetConfig
from evaluatorq.simulation.target import OrqResponsesTarget


class TestResolveBackendOpenResponses:
    def test_resolves_to_backend_with_correct_name(self):
        client = MagicMock()
        backend = resolve_backend("openresponses", llm_client=client)
        assert backend.name == "openresponses"

    def test_create_target_returns_orq_responses_target_with_correct_agent_id(self):
        client = MagicMock()
        backend = resolve_backend("openresponses", llm_client=client)
        target = backend.create_target("my-agent")
        assert isinstance(target, OrqResponsesTarget)
        assert target.config.model == "my-agent"

    def test_instructions_are_threaded_from_target_config(self):
        client = MagicMock()
        backend = resolve_backend(
            "openresponses",
            llm_client=client,
            target_config=TargetConfig(system_prompt="be safe"),
        )
        target = backend.create_target("agent-id")
        assert isinstance(target, OrqResponsesTarget)
        assert target.instructions == "be safe"

    def test_retry_settings_thread_from_pipeline_config(self):
        client = MagicMock()
        backend = resolve_backend(
            "openresponses",
            llm_client=client,
            pipeline_config=LLMConfig(retry_count=2, retry_on_codes=[429, 503]),
        )
        target = backend.create_target("agent-id")

        assert isinstance(target, OrqResponsesTarget)
        assert target.retry_attempts == 3
        assert target.retry_statuses == {429, 503}

    def test_retry_count_none_uses_default(self):
        client = MagicMock()
        backend = resolve_backend("openresponses", llm_client=client)
        target = backend.create_target("agent-id")
        assert isinstance(target, OrqResponsesTarget)
        assert target.retry_attempts is None

    @pytest.mark.asyncio
    async def test_resolve_context_returns_minimal_agent_context(self):
        client = MagicMock()
        backend = resolve_backend(
            "openresponses",
            llm_client=client,
            target_config=TargetConfig(system_prompt="hi"),
        )
        ctx = await backend.resolve_context("agent-id")
        assert ctx.key == "agent-id"
        assert ctx.system_prompt == "hi"

    @pytest.mark.asyncio
    async def test_resolve_context_cache_hit_returns_same_object(self):
        backend = resolve_backend("openresponses", llm_client=MagicMock())
        ctx1 = await backend.resolve_context("agent-id")
        ctx2 = await backend.resolve_context("agent-id")
        assert ctx1 is ctx2

    def test_lookup_is_case_insensitive(self):
        client = MagicMock()
        backend = resolve_backend("OpenResponses", llm_client=client)
        assert backend.name == "openresponses"


class TestCleanupMemory:
    @pytest.mark.asyncio
    async def test_cleanup_memory_is_noop_and_does_not_raise(self):
        backend = resolve_backend("openresponses", llm_client=MagicMock())
        ctx = AgentContext(key="k", display_name="k", description="d")
        # Must not raise for any entity_ids input
        await backend.cleanup_memory(ctx, [])
        await backend.cleanup_memory(ctx, ["id1", "id2", "id3"])


class TestErrorMapper:
    def test_maps_http_status_codes(self):
        backend = resolve_backend("openresponses", llm_client=MagicMock())
        exc = type("HTTPErr", (Exception,), {"status_code": 429})()
        code, _ = backend.map_error(exc)
        assert code == "openresponses.http.429"

    def test_maps_provider_error_code(self):
        backend = resolve_backend("openresponses", llm_client=MagicMock())
        exc = type("ProviderErr", (Exception,), {"code": "content_filter"})()
        code, _ = backend.map_error(exc)
        assert code == "openresponses.code.content_filter"

    def test_maps_rate_limit_by_class_name(self):
        backend = resolve_backend("openresponses", llm_client=MagicMock())
        exc = type("RateLimitError", (Exception,), {})()
        code, _ = backend.map_error(exc)
        assert code == "openresponses.rate_limit"

    def test_maps_timeout_by_class_name(self):
        backend = resolve_backend("openresponses", llm_client=MagicMock())
        exc = type("TimeoutError", (Exception,), {})()
        code, _ = backend.map_error(exc)
        assert code == "openresponses.timeout"

    def test_maps_authentication_error_by_class_name(self):
        backend = resolve_backend("openresponses", llm_client=MagicMock())
        exc = type("AuthenticationError", (Exception,), {})()
        code, _ = backend.map_error(exc)
        assert code == "openresponses.auth"

    def test_maps_unknown_to_unknown(self):
        backend = resolve_backend("openresponses", llm_client=MagicMock())
        with patch("evaluatorq.redteam.backends.openresponses.logger") as mock_logger:
            code, _ = backend.map_error(RuntimeError("boom"))
        assert code == "openresponses.unknown"
        mock_logger.opt.assert_called_once_with(exception=mock_logger.opt.call_args[1]["exception"])
        mock_logger.opt.return_value.error.assert_called_once()

    def test_message_includes_exception_type_and_text(self):
        backend = resolve_backend("openresponses", llm_client=MagicMock())
        exc = type("HTTPErr", (Exception,), {"status_code": 500})("internal error")
        _, msg = backend.map_error(exc)
        assert "HTTPErr" in msg
        assert "internal error" in msg


class TestAgentContextProvider:
    @pytest.mark.asyncio
    async def test_returns_basic_context(self):
        backend = resolve_backend(
            "openresponses",
            llm_client=MagicMock(),
            target_config=TargetConfig(system_prompt="be safe"),
        )
        ctx = await backend.resolve_context("agent-id")
        assert ctx.key == "agent-id"
        assert ctx.system_prompt == "be safe"
        assert ctx.tools == []
        assert ctx.memory_stores == []


class TestCallResponsesApiTokenUsage:
    """Verify _call_responses_api returns AgentResponse with correct TokenUsage."""

    @pytest.mark.asyncio
    async def test_token_usage_is_populated_from_response(self):
        from evaluatorq.simulation.types import TokenUsage

        mock_response = MagicMock()
        mock_response.id = "resp-123"
        mock_response.model = "gpt-4o"
        mock_response.status = "completed"
        mock_response.output = [
            MagicMock(type="message", content=[MagicMock(type="output_text", text="hello")])
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        client = MagicMock()
        client.responses.create = AsyncMock(return_value=mock_response)

        target = OrqResponsesTarget(
            MagicMock(model="gpt-4o", api="responses", timeout_ms=None, max_tokens=None),
            client=client,
        )

        result = await target._call_responses_api(responses_input="hello")
        assert isinstance(result, AgentResponse)
        assert result.usage is not None
        assert isinstance(result.usage, TokenUsage)
        assert result.usage.calls == 1
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_token_usage_is_none_when_response_has_no_usage(self):
        mock_response = MagicMock()
        mock_response.id = "resp-456"
        mock_response.model = "gpt-4o"
        mock_response.status = "completed"
        mock_response.output = [
            MagicMock(type="message", content=[MagicMock(type="output_text", text="hi")])
        ]
        mock_response.usage = None

        client = MagicMock()
        client.responses.create = AsyncMock(return_value=mock_response)

        target = OrqResponsesTarget(
            MagicMock(model="gpt-4o", api="responses", timeout_ms=None, max_tokens=None),
            client=client,
        )

        result = await target._call_responses_api(responses_input="hi")
        assert isinstance(result, AgentResponse)
        assert result.usage is None
