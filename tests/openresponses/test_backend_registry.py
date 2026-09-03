"""Tests for the openresponses backend registration.

Verifies the registry can resolve ``backend="openresponses"`` and that the
backend wires target construction, context resolution, and error mapping.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.contracts import AgentResponse, LLMCallConfig
from evaluatorq.redteam.backends.registry import resolve_backend
from evaluatorq.redteam.contracts import AgentContext, LLMConfig, TargetConfig
from evaluatorq.openresponses.target import OrqResponsesTarget
from evaluatorq.common.thread_context import conversation_thread, evaluatorq_pipeline, evaluatorq_run_id


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

    def test_reasoning_effort_is_threaded_from_pipeline_config(self):
        client = MagicMock()
        backend = resolve_backend(
            "openresponses",
            llm_client=client,
            pipeline_config=LLMConfig(target_reasoning_effort="high"),
        )
        target = backend.create_target("agent-id")
        assert isinstance(target, OrqResponsesTarget)
        assert target.config.reasoning_effort == "high"

    def test_reasoning_effort_defaults_to_none(self):
        client = MagicMock()
        backend = resolve_backend("openresponses", llm_client=client)
        target = backend.create_target("agent-id")
        assert isinstance(target, OrqResponsesTarget)
        assert target.config.reasoning_effort is None

    def test_pipeline_retry_settings_do_not_stack_on_target_path(self):
        client = MagicMock()
        with patch("evaluatorq.redteam.backends._retry.logger.warning") as warning:
            backend = resolve_backend(
                "openresponses",
                llm_client=client,
                pipeline_config=LLMConfig(retry_count=2, retry_on_codes=[429, 503]),
            )
        target = backend.create_target("agent-id")

        assert isinstance(target, OrqResponsesTarget)
        assert target.retry_attempts == 1
        assert target.retry_statuses is None
        warning.assert_called_once()
        assert "retry_count and retry_on_codes" in warning.call_args.args[0]

    def test_retry_count_none_uses_default(self):
        client = MagicMock()
        backend = resolve_backend("openresponses", llm_client=client)
        target = backend.create_target("agent-id")
        assert isinstance(target, OrqResponsesTarget)
        assert target.retry_attempts == 1

    def test_default_pipeline_retry_settings_do_not_warn(self):
        client = MagicMock()
        with patch("evaluatorq.redteam.backends._retry.logger.warning") as warning:
            resolve_backend("openresponses", llm_client=client, pipeline_config=LLMConfig())

        warning.assert_not_called()

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
        from evaluatorq.contracts import TokenUsage

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
            LLMCallConfig(model="gpt-4o", api="responses"),
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
            LLMCallConfig(model="gpt-4o", api="responses"),
            client=client,
        )

        result = await target._call_responses_api(responses_input="hi")
        assert isinstance(result, AgentResponse)
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_direct_openai_uses_native_metadata_without_thread(self):
        mock_response = MagicMock(
            model='gpt-4o',
            output=[MagicMock(type='message', content=[MagicMock(type='output_text', text='hello')])],
            usage=None,
            telemetry=None,
        )
        client = MagicMock()
        client.base_url = 'https://api.openai.com/v1'
        client.responses.create = AsyncMock(return_value=mock_response)
        target = OrqResponsesTarget(
            LLMCallConfig(model='gpt-4o', api='responses'),
            client=client,
        )

        with evaluatorq_pipeline('agent_simulation'), evaluatorq_run_id('run-1'), conversation_thread('thread-1'):
            await target._call_responses_api(responses_input='hello')

        _, kwargs = client.responses.create.call_args
        assert kwargs['metadata'] == {'evaluatorq_pipeline': 'agent_simulation', 'evaluatorq_run_id': 'run-1'}
        assert 'thread' not in kwargs.get('extra_body', {})

    @pytest.mark.asyncio
    async def test_orq_router_uses_native_metadata_and_thread_extra_body(self):
        mock_response = MagicMock(
            model='gpt-4o',
            output=[MagicMock(type='message', content=[MagicMock(type='output_text', text='hello')])],
            usage=None,
            telemetry=None,
        )
        client = MagicMock()
        client.base_url = 'https://my.orq.ai/v3/router'
        client.responses.create = AsyncMock(return_value=mock_response)
        target = OrqResponsesTarget(
            LLMCallConfig(model='gpt-4o', api='responses'),
            client=client,
        )

        with evaluatorq_pipeline('agent_simulation'), evaluatorq_run_id('run-1'), conversation_thread('thread-1'):
            await target._call_responses_api(responses_input='hello')

        _, kwargs = client.responses.create.call_args
        assert kwargs['metadata'] == {'evaluatorq_pipeline': 'agent_simulation', 'evaluatorq_run_id': 'run-1'}
        assert kwargs['extra_body']['thread'] == {'id': 'thread-1'}
