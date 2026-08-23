"""Pre-flight of the target reasoning effort happens before any LLM spend.

Only ``agent:`` string targets route through a backend
(``make_agent_backend``'s OpenResponses exec leg) that actually forwards
``target_reasoning_effort`` — so the pre-flight below is exercised against an
``agent:`` string target, with the ORQ SDK context lookup stubbed out. A bare
``AgentTarget`` (e.g. a user's own implementation) is a separate case, covered
by ``test_bare_agent_target_is_warned_not_raised`` below: its ability to
forward the setting is unknown, so it must warn and proceed rather than
raise (see the module docstring on ``runner._run_dynamic_or_hybrid``'s
pre-flight block for the full rationale).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evaluatorq.common import model_catalogue
from evaluatorq.common.model_catalogue import ModelInfo
from evaluatorq.contracts import AgentContext, AgentResponse, AgentTarget, Message
from evaluatorq.redteam import red_team
from evaluatorq.redteam.contracts import LLMConfig, SaveMode


_DATASET = Path(__file__).parent / "fixtures" / "static_e2e_dataset.json"

_AGENT_TARGET = "agent:thinky-agent"


class _StubOrqBackend:
    """Stand-in for ``resolve_backend('orq', ...)``: resolves a fixed context
    without touching the network, matching the shape ``Backend.resolve_context``
    returns.
    """

    async def resolve_context(self, agent_key: str) -> AgentContext:
        return AgentContext(key=agent_key, model="openai/thinky")


class _Target(AgentTarget):
    async def respond(self, messages: list[Message]) -> AgentResponse:  # pragma: no cover - never reached
        raise AssertionError("target must not be called")

    def new(self) -> "_Target":
        return _Target()

    async def get_agent_context(self) -> AgentContext:
        return AgentContext(key="thinky-agent", model="openai/thinky")


@pytest.fixture(autouse=True)
def _catalogue(monkeypatch: pytest.MonkeyPatch):
    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {
            "thinky": ModelInfo(
                0.1, 0.2, "openai", supports_responses=True, reasoning_efforts=frozenset({"low", "high"})
            )
        }

    monkeypatch.setattr(model_catalogue, "_load_catalogue", fake_load)
    monkeypatch.setenv("ORQ_API_KEY", "test-key")


def _stub_orq_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the ``resolve_backend`` binding ``runner.py`` imported, so both
    legs' context-retrieval step resolves ``_StubOrqBackend`` instead of
    building a real ORQ SDK client.
    """
    from evaluatorq.redteam import runner as runner_module

    def fake_resolve_backend(name: str, **kwargs: Any) -> Any:  # noqa: ARG001
        return _StubOrqBackend()

    monkeypatch.setattr(runner_module, "resolve_backend", fake_resolve_backend)


@pytest.mark.asyncio
async def test_static_mode_fails_before_any_target_call(monkeypatch: pytest.MonkeyPatch):
    """The static leg has no generation step, so the first thing an invalid
    effort would waste is the target calls themselves — and every judge call
    scoring their errors."""
    import evaluatorq as evaluatorq_pkg

    _stub_orq_backend(monkeypatch)

    async def no_run(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise AssertionError("evaluatorq() ran despite an invalid reasoning effort")

    # _run_static imports the entry point from the package at call time.
    monkeypatch.setattr(evaluatorq_pkg, "evaluatorq", no_run)

    with pytest.raises(ValueError, match="not accepted by openai/thinky"):
        await red_team(
            target=_AGENT_TARGET,
            mode="static",
            dataset=_DATASET,
            llm_config=LLMConfig(target_reasoning_effort="xhigh"),
            categories=["ASI01"],
            save=SaveMode.NONE,
        )


@pytest.mark.asyncio
async def test_unsupported_effort_fails_before_generation(monkeypatch: pytest.MonkeyPatch):
    from evaluatorq.redteam.adaptive import capability_classifier

    _stub_orq_backend(monkeypatch)

    async def no_calls(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise AssertionError("classification ran despite an invalid reasoning effort")

    monkeypatch.setattr(capability_classifier, "classify_agent_capabilities", no_calls)

    with pytest.raises(ValueError, match="not accepted by openai/thinky"):
        await red_team(
            target=_AGENT_TARGET,
            mode="dynamic",
            llm_config=LLMConfig(target_reasoning_effort="xhigh"),
            categories=["LLM01"],
            save=SaveMode.NONE,
        )


@pytest.mark.asyncio
async def test_bare_agent_target_is_warned_not_raised(monkeypatch: pytest.MonkeyPatch, caplog):
    """A bare user ``AgentTarget`` cannot be known to forward reasoning effort,
    so an unsupported value must warn-and-skip rather than kill the run —
    unlike the ``agent:`` string-target cases above, which do raise because
    that path genuinely forwards it.
    """
    import logging
    from unittest.mock import AsyncMock

    # The run proceeds past the pre-flight (it warns, not raises), so
    # everything downstream — attack generation, target calls — would try to
    # reach a real LLM host without this. Fail fast instead of hitting the
    # network: the `_block_outbound_network` fixture in conftest treats any
    # real connection attempt as a test failure.
    no_network_client = AsyncMock()
    no_network_client.chat.completions.create = AsyncMock(side_effect=RuntimeError('network disabled in this test'))

    with caplog.at_level(logging.WARNING):
        # No ValueError from the pre-flight itself: it warns and lets the run
        # proceed past context retrieval instead of validating (and rejecting)
        # an effort the bare target may never have received in the first
        # place. Whatever happens afterwards (attack generation, target calls,
        # report assembly) is unrelated to the pre-flight, so any exception
        # raised there is tolerated here as long as it is not the pre-flight's
        # own "not accepted by" ValueError.
        try:
            await red_team(
                target=_Target(),
                mode="dynamic",
                llm_config=LLMConfig(target_reasoning_effort="xhigh"),
                categories=["LLM01"],
                save=SaveMode.NONE,
                max_turns=1,
                generate_strategies=False,
                llm_client=no_network_client,
            )
        except ValueError as exc:
            assert "not accepted by" not in str(exc), (
                f"pre-flight raised for a bare AgentTarget instead of warning-and-skipping: {exc}"
            )

    assert any(
        "target_reasoning_effort" in record.message and "_Target" in record.message for record in caplog.records
    )
