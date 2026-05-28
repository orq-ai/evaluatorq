"""Tests for the openresponses backend registration.

Verifies the registry can resolve ``backend="openresponses"`` and that the
backend wires target construction, context resolution, and error mapping.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evaluatorq.redteam.backends.registry import resolve_backend
from evaluatorq.redteam.contracts import LLMConfig, TargetConfig
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

    def test_lookup_is_case_insensitive(self):
        client = MagicMock()
        backend = resolve_backend("OpenResponses", llm_client=client)
        assert backend.name == "openresponses"


class TestErrorMapper:
    def test_maps_http_status_codes(self):
        backend = resolve_backend("openresponses", llm_client=MagicMock())
        exc = type("HTTPErr", (Exception,), {"status_code": 429})()
        code, _ = backend.map_error(exc)
        assert code == "openresponses.http.429"

    def test_maps_unknown_to_unknown(self):
        backend = resolve_backend("openresponses", llm_client=MagicMock())
        code, _ = backend.map_error(RuntimeError("boom"))
        assert code == "openresponses.unknown"


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
