"""Tests for the _make_agent_backend helper and its wiring inside _prepare_target.

Test approach
-------------
We test ``_make_agent_backend`` in isolation rather than calling ``_prepare_target``
directly.  ``_prepare_target`` is a deeply async function that requires a dozen
mock parameters (LLM clients, hooks, dataset handles, etc.) and fires network I/O
via ``resolve_context`` before it ever touches the backend.  Testing it end-to-end
would require a second layer of async machinery and would make this a partial
integration test, not a unit test.

``_make_agent_backend`` is the unit of composition: it calls ``resolve_backend``
twice (with different names) and wraps the results in a ``HybridAgentBackend``.
Two targeted assertions cover the routing contract completely:

1. ``create_target("k")`` on the returned composite delegates to the *exec* backend
   with key ``"agent/k"`` → the produced target is an ``OrqResponsesTarget`` whose
   ``config.model == "agent/k"``.

2. ``resolve_context("k")`` delegates to the *context* backend (the ORQ backend)
   with the bare key ``"k"``.

The ORQ backend (``_create_orq_backend``) requires the ``orq`` SDK, so we patch
``resolve_backend`` at the runner module level to inject lightweight fakes,
keeping the test hermetic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.redteam.backends.base import HybridAgentBackend
from evaluatorq.redteam.backends.openresponses import OpenResponsesBackend
from evaluatorq.redteam.runner import _make_agent_backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orq_backend_mock() -> MagicMock:
    """Lightweight mock that quacks like an ORQBackend."""
    backend = MagicMock(name="orq_backend")
    backend.resolve_context = AsyncMock()
    backend.cleanup_memory = AsyncMock()
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMakeAgentBackend:
    """_make_agent_backend constructs a HybridAgentBackend with correct delegation."""

    def test_returns_hybrid_agent_backend(self):
        """Return type is always HybridAgentBackend."""
        orq_mock = _make_orq_backend_mock()
        openresponses_mock = OpenResponsesBackend(client=None)

        def _fake_resolve_backend(name, *, llm_client=None, target_config=None, pipeline_config=None):
            if name == "orq":
                return orq_mock
            if name == "openresponses":
                return openresponses_mock
            raise ValueError(f"Unexpected backend name: {name!r}")

        with patch("evaluatorq.redteam.runner.resolve_backend", side_effect=_fake_resolve_backend):
            result = _make_agent_backend(target_config=None, pipeline_config=None)

        assert isinstance(result, HybridAgentBackend)

    def test_exec_backend_receives_agent_prefixed_key(self):
        """create_target('k') on the composite calls the exec backend with 'agent/k'.

        The OrqResponsesTarget produced by OpenResponsesBackend stores the model
        name on config.model, so we verify that attribute equals 'agent/k'.
        """
        orq_mock = _make_orq_backend_mock()
        # Inject a client so create_target builds the target without resolving real
        # ORQ credentials (CI has no ORQ_API_KEY); we only assert the prefixed model.
        openresponses_backend = OpenResponsesBackend(client=MagicMock())

        def _fake_resolve_backend(name, *, llm_client=None, target_config=None, pipeline_config=None):
            if name == "orq":
                return orq_mock
            if name == "openresponses":
                return openresponses_backend
            raise ValueError(f"Unexpected backend name: {name!r}")

        with patch("evaluatorq.redteam.runner.resolve_backend", side_effect=_fake_resolve_backend):
            hybrid = _make_agent_backend(target_config=None, pipeline_config=None)

        target = hybrid.create_target("my-agent")
        # OrqResponsesTarget stores the model in its config
        assert target.config.model == "agent/my-agent"  # pyright: ignore[reportAttributeAccessIssue]

    @pytest.mark.asyncio
    async def test_resolve_context_delegates_to_orq_backend_with_bare_key(self):
        """resolve_context('k') routes to the ORQ context backend with the bare key 'k',
        NOT the prefixed form 'agent/k'.
        """
        from evaluatorq.redteam.contracts import AgentContext

        orq_mock = _make_orq_backend_mock()
        sentinel_ctx = AgentContext(key="my-agent")
        orq_mock.resolve_context.return_value = sentinel_ctx

        openresponses_mock = OpenResponsesBackend(client=None)

        def _fake_resolve_backend(name, *, llm_client=None, target_config=None, pipeline_config=None):
            if name == "orq":
                return orq_mock
            if name == "openresponses":
                return openresponses_mock
            raise ValueError(f"Unexpected backend name: {name!r}")

        with patch("evaluatorq.redteam.runner.resolve_backend", side_effect=_fake_resolve_backend):
            hybrid = _make_agent_backend(target_config=None, pipeline_config=None)
            # The ORQ context backend is built lazily on first context use, so resolve
            # inside the patch to capture the mocked backend.
            result = await hybrid.resolve_context("my-agent")

        orq_mock.resolve_context.assert_called_once_with("my-agent")
        assert result is sentinel_ctx

    def test_exec_backend_built_with_llm_client_none(self):
        """The openresponses exec backend must receive llm_client=None.

        This ensures the attacker LLM client is never forwarded to the target.
        """
        received_kwargs: dict[str, object] = {}
        orq_mock = _make_orq_backend_mock()
        openresponses_mock = MagicMock(name="openresponses_backend")

        def _fake_resolve_backend(name, *, llm_client=None, target_config=None, pipeline_config=None):
            if name == "orq":
                return orq_mock
            if name == "openresponses":
                received_kwargs["llm_client"] = llm_client
                return openresponses_mock
            raise ValueError(f"Unexpected backend name: {name!r}")

        with patch("evaluatorq.redteam.runner.resolve_backend", side_effect=_fake_resolve_backend):
            _make_agent_backend(target_config=None, pipeline_config=None)

        assert received_kwargs.get("llm_client") is None


class TestExecBackendRequiresOrq:
    """The agent exec backend must never resolve a self-built client to OpenAI."""

    def test_default_require_orq_is_true(self):
        """OpenResponsesBackend exclusively serves agent/<key> → require_orq defaults True."""
        assert OpenResponsesBackend(client=None)._require_orq is True

    def test_create_target_refuses_openai_fallback(self, monkeypatch):
        """With only OPENAI_API_KEY set, building an agent target must raise, not
        silently route the hosted agent to OpenAI."""
        from evaluatorq.common.llm_client import MissingLLMCredentialsError

        monkeypatch.delenv("ORQ_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

        backend = OpenResponsesBackend(client=None)
        with pytest.raises(MissingLLMCredentialsError):
            backend.create_target("agent/my-agent")

    def test_create_target_uses_orq_router_when_orq_key_present(self, monkeypatch):
        """ORQ_API_KEY present → target's self-built client routes through the Orq router."""
        from evaluatorq.common.llm_client import client_routes_through_orq

        monkeypatch.setenv("ORQ_API_KEY", "sk-orq")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.delenv("ORQ_BASE_URL", raising=False)

        backend = OpenResponsesBackend(client=None)
        target = backend.create_target("agent/my-agent")

        assert client_routes_through_orq(target._client) is True  # pyright: ignore[reportAttributeAccessIssue]
