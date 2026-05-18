"""Tests for the openresponses backend bundle registration.

Verifies the registry can resolve ``backend="openresponses"`` and that the
bundle wires the right factory / context provider / error mapper.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from evaluatorq.redteam.backends.openresponses import (
    OpenResponsesAgentTarget,
    OpenResponsesContextProvider,
    OpenResponsesErrorMapper,
    OpenResponsesTargetFactory,
)
from evaluatorq.redteam.backends.registry import resolve_backend
from evaluatorq.redteam.contracts import TargetConfig


class TestResolveBackendOpenResponses:
    def test_resolves_to_bundle_with_correct_components(self):
        client = MagicMock()
        bundle = resolve_backend("openresponses", llm_client=client)
        assert bundle.name == "openresponses"
        assert isinstance(bundle.target_factory, OpenResponsesTargetFactory)
        assert isinstance(bundle.context_provider, OpenResponsesContextProvider)
        assert isinstance(bundle.error_mapper, OpenResponsesErrorMapper)

    def test_factory_creates_targets_with_correct_agent_id(self):
        client = MagicMock()
        bundle = resolve_backend("openresponses", llm_client=client)
        target = bundle.target_factory.create_target("my-agent")
        assert isinstance(target, OpenResponsesAgentTarget)
        assert target.agent_id == "my-agent"

    def test_instructions_are_threaded_from_target_config(self):
        client = MagicMock()
        bundle = resolve_backend(
            "openresponses",
            llm_client=client,
            target_config=TargetConfig(system_prompt="be safe"),
        )
        target = bundle.target_factory.create_target("agent-id")
        assert isinstance(target, OpenResponsesAgentTarget)
        assert target.instructions == "be safe"

    def test_context_provider_returns_minimal_agent_context(self):
        client = MagicMock()
        bundle = resolve_backend(
            "openresponses",
            llm_client=client,
            target_config=TargetConfig(system_prompt="hi"),
        )
        provider = bundle.context_provider
        assert isinstance(provider, OpenResponsesContextProvider)

    def test_lookup_is_case_insensitive(self):
        client = MagicMock()
        bundle = resolve_backend("OpenResponses", llm_client=client)
        assert bundle.name == "openresponses"

    def test_resolution_without_explicit_client_uses_env(self):
        """When no client is passed, the registry calls the factory which builds one
        from env vars. Make sure that path doesn't crash when env is configured."""
        with patch(
            "evaluatorq.redteam.backends.openresponses.create_openresponses_client",
            return_value=MagicMock(),
        ):
            bundle = resolve_backend("openresponses")
            assert bundle.name == "openresponses"

    def test_private_alias_still_works_for_backwards_compatibility(self):
        """The pre-public name was ``_create_openresponses_client``; keep the alias
        for any in-flight callers that already imported it."""
        from evaluatorq.redteam.backends.openresponses import (
            _create_openresponses_client,
            create_openresponses_client,
        )
        assert _create_openresponses_client is create_openresponses_client


class TestErrorMapper:
    def test_maps_http_status_codes(self):
        mapper = OpenResponsesErrorMapper()
        # Build an exception with a status_code attribute at construction time
        # so the type checker doesn't complain about dynamic attribute assignment.
        exc = type("HTTPErr", (Exception,), {"status_code": 429})()
        code, _ = mapper.map_error(exc)
        assert code == "openresponses.http.429"

    def test_maps_unknown_to_unknown(self):
        mapper = OpenResponsesErrorMapper()
        code, _ = mapper.map_error(RuntimeError("boom"))
        assert code == "openresponses.unknown"


class TestAgentContextProvider:
    @pytest.mark.asyncio
    async def test_returns_basic_context(self):
        provider = OpenResponsesContextProvider(instructions="be safe")
        ctx = await provider.get_agent_context("agent-id")
        assert ctx.key == "agent-id"
        assert ctx.system_prompt == "be safe"
        assert ctx.tools == []
        assert ctx.memory_stores == []
