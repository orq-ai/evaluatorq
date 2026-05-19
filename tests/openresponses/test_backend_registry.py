"""Tests for the openresponses backend bundle registration.

Verifies the registry can resolve ``backend="openresponses"`` and that the
bundle wires the right factory / context provider / error mapper.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evaluatorq.redteam.backends.registry import resolve_backend
from evaluatorq.redteam.contracts import LLMConfig, TargetConfig
from evaluatorq.simulation.target import OrqResponsesTarget


class TestResolveBackendOpenResponses:
    def test_resolves_to_bundle_with_correct_components(self):
        client = MagicMock()
        bundle = resolve_backend("openresponses", llm_client=client)
        assert bundle.name == "openresponses"
        assert bundle.target_factory is not None
        assert bundle.context_provider is not None
        assert bundle.error_mapper is not None

    def test_factory_creates_targets_with_correct_agent_id(self):
        client = MagicMock()
        bundle = resolve_backend("openresponses", llm_client=client)
        target = bundle.target_factory.create_target("my-agent")
        assert isinstance(target, OrqResponsesTarget)
        assert target.config.model == "my-agent"

    def test_instructions_are_threaded_from_target_config(self):
        client = MagicMock()
        bundle = resolve_backend(
            "openresponses",
            llm_client=client,
            target_config=TargetConfig(system_prompt="be safe"),
        )
        target = bundle.target_factory.create_target("agent-id")
        assert isinstance(target, OrqResponsesTarget)
        assert target.instructions == "be safe"

    def test_retry_settings_thread_from_pipeline_config(self):
        client = MagicMock()
        bundle = resolve_backend(
            "openresponses",
            llm_client=client,
            pipeline_config=LLMConfig(retry_count=2, retry_on_codes=[429, 503]),
        )
        target = bundle.target_factory.create_target("agent-id")

        assert isinstance(target, OrqResponsesTarget)
        assert target.retry_attempts == 3
        assert target.retry_statuses == {429, 503}

    @pytest.mark.asyncio
    async def test_context_provider_returns_minimal_agent_context(self):
        client = MagicMock()
        bundle = resolve_backend(
            "openresponses",
            llm_client=client,
            target_config=TargetConfig(system_prompt="hi"),
        )
        provider = bundle.context_provider
        ctx = await provider.get_agent_context("agent-id")
        assert ctx.key == "agent-id"
        assert ctx.system_prompt == "hi"

    def test_lookup_is_case_insensitive(self):
        client = MagicMock()
        bundle = resolve_backend("OpenResponses", llm_client=client)
        assert bundle.name == "openresponses"


class TestErrorMapper:
    def test_maps_http_status_codes(self):
        mapper = resolve_backend("openresponses", llm_client=MagicMock()).error_mapper
        # Build an exception with a status_code attribute at construction time
        # so the type checker doesn't complain about dynamic attribute assignment.
        exc = type("HTTPErr", (Exception,), {"status_code": 429})()
        code, _ = mapper.map_error(exc)
        assert code == "openresponses.http.429"

    def test_maps_unknown_to_unknown(self):
        mapper = resolve_backend("openresponses", llm_client=MagicMock()).error_mapper
        code, _ = mapper.map_error(RuntimeError("boom"))
        assert code == "openresponses.unknown"


class TestAgentContextProvider:
    @pytest.mark.asyncio
    async def test_returns_basic_context(self):
        provider = resolve_backend(
            "openresponses",
            llm_client=MagicMock(),
            target_config=TargetConfig(system_prompt="be safe"),
        ).context_provider
        ctx = await provider.get_agent_context("agent-id")
        assert ctx.key == "agent-id"
        assert ctx.system_prompt == "be safe"
        assert ctx.tools == []
        assert ctx.memory_stores == []
