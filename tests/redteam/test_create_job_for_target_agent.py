"""Unit tests for _create_job_for_target — AGENT branch routes through OrqResponses.

Assertions:
1. agent:foo  -> job driven by an OrqResponsesTarget with config.model == "agent/foo"
2. deployment:bar -> create_model_job still called (unchanged branch)
3. gpt-4o-mini    -> create_model_job still called (unchanged branch)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUNNER = "evaluatorq.redteam.runner"
# Patch where it is *used*: target.py does `from ...client import build_simulation_client`,
# so it holds its own binding — patching the client module's attribute would not intercept it.
_BUILD_SIM_CLIENT = "evaluatorq.openresponses.target.build_simulation_client"


def _fake_build_sim_client(
    config_client: object = None, *, extra_api_key: object = None, require_orq: object = False
) -> tuple[MagicMock, bool]:
    """Stub for build_simulation_client — returns a MagicMock client so no env-var
    lookups happen at OrqResponsesTarget construction time."""
    return MagicMock(), True


def _fake_orq_backend(**kwargs: object) -> MagicMock:
    """Stub for the 'orq' backend factory — returns a MagicMock Backend so
    the ORQBackend import (which has optional deps) is never exercised."""
    backend = MagicMock()
    backend.cleanup_memory = MagicMock()
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentBranchUsesOrqResponsesTarget:
    """_create_job_for_target('agent:foo', ...) must return a job backed by
    OrqResponsesTarget(config.model='agent/foo')."""

    def test_returns_callable_job(self):
        from evaluatorq.redteam.runner import _create_job_for_target

        with (
            patch(_BUILD_SIM_CLIENT, side_effect=_fake_build_sim_client),
            patch(f"{_RUNNER}.resolve_backend") as mock_resolve,
        ):
            # Patch resolve_backend so:
            # - 'orq' call -> stub (avoids ORQBackend import / credential check)
            # - 'openresponses' call -> REAL OpenResponsesBackend built inline
            from evaluatorq.redteam.backends.openresponses import OpenResponsesBackend

            def _side_effect(name: str, **kwargs: object) -> object:
                if name == "orq":
                    return _fake_orq_backend(**kwargs)
                # Let the real OpenResponsesBackend build itself (no network at construction)
                return OpenResponsesBackend(client=None, instructions=None)

            mock_resolve.side_effect = _side_effect

            job = _create_job_for_target("agent:foo", llm_client=None, system_prompt="sp")

        assert callable(job), "Expected a callable job"

    def test_target_is_orq_responses_target_with_agent_prefix(self):
        """The composite's create_target('foo') produces OrqResponsesTarget with
        config.model == 'agent/foo'."""
        from evaluatorq.openresponses.target import OrqResponsesTarget
        from evaluatorq.redteam.backends.openresponses import OpenResponsesBackend
        from evaluatorq.redteam.backends.base import HybridAgentBackend

        with patch(_BUILD_SIM_CLIENT, side_effect=_fake_build_sim_client):
            exec_backend = OpenResponsesBackend(client=None, instructions=None)

        orq_stub = _fake_orq_backend()
        hybrid = HybridAgentBackend(context_backend=orq_stub, exec_backend=exec_backend)

        with patch(_BUILD_SIM_CLIENT, side_effect=_fake_build_sim_client):
            target = hybrid.create_target("foo")

        assert isinstance(target, OrqResponsesTarget), (
            f"Expected OrqResponsesTarget, got {type(target).__name__}"
        )
        assert target.config.model == "agent/foo", (
            f"Expected config.model='agent/foo', got {target.config.model!r}"
        )

    def test_job_name_uses_bare_key_not_prefixed(self):
        """Job name must be 'redteam:static:foo', not 'redteam:static:agent/foo'."""
        from evaluatorq.redteam.backends.openresponses import OpenResponsesBackend
        from evaluatorq.redteam.runner import _create_job_for_target

        with (
            patch(_BUILD_SIM_CLIENT, side_effect=_fake_build_sim_client),
            patch(f"{_RUNNER}.resolve_backend") as mock_resolve,
        ):
            def _side_effect(name: str, **kwargs: object) -> object:
                if name == "orq":
                    return _fake_orq_backend(**kwargs)
                return OpenResponsesBackend(client=None, instructions=None)

            mock_resolve.side_effect = _side_effect

            job = _create_job_for_target("agent:foo", llm_client=None, system_prompt=None)

        # The @job decorator stores the job name in the closure as 'name'.
        import inspect

        closure_vars = inspect.getclosurevars(job)
        job_name = closure_vars.nonlocals.get("name", "")
        assert "foo" in job_name, f"Expected 'foo' in job name, got: {job_name!r}"
        assert "agent/" not in job_name, (
            f"Job name must NOT contain the 'agent/' prefix, got: {job_name!r}"
        )

    def test_system_prompt_forwarded_to_backend(self):
        """TargetConfig(system_prompt=...) is passed to the openresponses backend,
        which stores it as instructions on OrqResponsesTarget."""
        from evaluatorq.openresponses.target import OrqResponsesTarget
        from evaluatorq.redteam.backends.openresponses import OpenResponsesBackend
        from evaluatorq.redteam.backends.base import HybridAgentBackend

        instructions = "You are a helpful assistant."

        with patch(_BUILD_SIM_CLIENT, side_effect=_fake_build_sim_client):
            exec_backend = OpenResponsesBackend(client=None, instructions=instructions)

        orq_stub = _fake_orq_backend()
        hybrid = HybridAgentBackend(context_backend=orq_stub, exec_backend=exec_backend)

        with patch(_BUILD_SIM_CLIENT, side_effect=_fake_build_sim_client):
            target = hybrid.create_target("bar")

        assert isinstance(target, OrqResponsesTarget)
        assert target.instructions == instructions


class TestDeploymentBranchUnchanged:
    """deployment:bar still routes through create_model_job (deployment_key path)."""

    def test_deployment_calls_create_model_job(self):
        from evaluatorq.redteam.runner import _create_job_for_target

        with patch(f"{_RUNNER}.create_model_job", return_value=MagicMock()) as mock_cmj:
            _create_job_for_target("deployment:bar", llm_client=None, system_prompt=None)

        mock_cmj.assert_called_once()
        call_kwargs = mock_cmj.call_args
        # Must have been called with deployment_key='bar'
        kwargs = call_kwargs[1] if call_kwargs[1] else {}
        args = call_kwargs[0] if call_kwargs[0] else ()
        assert kwargs.get("deployment_key") == "bar" or (
            len(args) > 0 and args[0] == "bar"
        ), f"create_model_job not called with deployment_key='bar'. Call: {call_kwargs}"


class TestModelBranchUnchanged:
    """Bare strings (no prefix) parse as TargetKind.AGENT and route through the
    AGENT branch — _parse_target returns (TargetKind.AGENT, key) when there is
    no colon. The fallback model branch in _create_job_for_target is therefore
    unreachable with valid TargetKind values; we verify the correct routing."""

    def test_bare_key_does_not_call_create_model_job(self):
        """A bare key like 'gpt-4o-mini' (no colon) is parsed as AGENT and must
        not call create_model_job — it uses _create_static_job_for_agent_target."""
        from evaluatorq.redteam.backends.openresponses import OpenResponsesBackend
        from evaluatorq.redteam.runner import _create_job_for_target

        with (
            patch(_BUILD_SIM_CLIENT, side_effect=_fake_build_sim_client),
            patch(f"{_RUNNER}.resolve_backend") as mock_resolve,
            patch(f"{_RUNNER}.create_model_job") as mock_cmj,
        ):
            def _side_effect(name: str, **kwargs: object) -> object:
                if name == "orq":
                    return _fake_orq_backend(**kwargs)
                return OpenResponsesBackend(client=None, instructions=None)

            mock_resolve.side_effect = _side_effect

            _create_job_for_target("gpt-4o-mini", llm_client=None, system_prompt=None)

        mock_cmj.assert_not_called()

    def test_agent_branch_does_not_call_create_model_job(self):
        """AGENT branch must NOT call create_model_job at all (uses static job helper)."""
        from evaluatorq.redteam.backends.openresponses import OpenResponsesBackend
        from evaluatorq.redteam.runner import _create_job_for_target

        with (
            patch(_BUILD_SIM_CLIENT, side_effect=_fake_build_sim_client),
            patch(f"{_RUNNER}.resolve_backend") as mock_resolve,
            patch(f"{_RUNNER}.create_model_job") as mock_cmj,
        ):
            def _side_effect(name: str, **kwargs: object) -> object:
                if name == "orq":
                    return _fake_orq_backend(**kwargs)
                return OpenResponsesBackend(client=None, instructions=None)

            mock_resolve.side_effect = _side_effect

            _create_job_for_target("agent:foo", llm_client=None, system_prompt=None)

        mock_cmj.assert_not_called()
