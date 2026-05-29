"""Tests for the openresponses backend registration.

Verifies the registry can resolve ``backend="openresponses"`` and that the
backend wires target construction, context resolution, and error mapping.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from evaluatorq.redteam.backends.base import Backend
from evaluatorq.redteam.backends.openresponses import (
    OpenResponsesAgentTarget,
    OpenResponsesBackend,
    OpenResponsesContextProvider,
    OpenResponsesErrorMapper,
)
from evaluatorq.redteam.backends.registry import resolve_backend
from evaluatorq.redteam.contracts import TargetConfig


class TestResolveBackendOpenResponses:
    def test_resolves_to_backend_with_correct_name(self):
        client = MagicMock()
        backend = resolve_backend("openresponses", llm_client=client)
        assert backend.name == "openresponses"
        assert isinstance(backend, OpenResponsesBackend)
        assert isinstance(backend, Backend)

    def test_create_target_returns_openresponses_target_with_correct_agent_id(self):
        client = MagicMock()
        backend = resolve_backend("openresponses", llm_client=client)
        target = backend.create_target("my-agent")
        assert isinstance(target, OpenResponsesAgentTarget)
        assert target.agent_id == "my-agent"

    def test_instructions_are_threaded_from_target_config(self):
        client = MagicMock()
        backend = resolve_backend(
            "openresponses",
            llm_client=client,
            target_config=TargetConfig(system_prompt="be safe"),
        )
        target = backend.create_target("agent-id")
        assert isinstance(target, OpenResponsesAgentTarget)
        assert target.instructions == "be safe"

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

    def test_resolution_without_explicit_client_uses_env(self):
        """When no client is passed, the registry builds one from env vars."""
        with patch(
            "evaluatorq.redteam.backends.openresponses.create_openresponses_client",
            return_value=MagicMock(),
        ):
            backend = resolve_backend("openresponses")
            assert backend.name == "openresponses"

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
