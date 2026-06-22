"""RES-912: agent sim forwards the inference base_url to the results upload."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, patch

import pytest

import evaluatorq.evaluatorq  # noqa: F401  populate sys.modules; the package attr is shadowed by the function

# evaluatorq.evaluatorq resolves to the re-exported function, not the submodule,
# so patch the module object directly (3.10's mock can't resolve the shadowed path).
_EQ_MOD = sys.modules["evaluatorq.evaluatorq"]

from evaluatorq.simulation.types import (
    CommunicationStyle,
    Datapoint,
    Message,
    Persona,
    Scenario,
)


def _make_datapoint() -> Datapoint:
    return Datapoint(
        id="dp-001",
        persona=Persona(
            name="P",
            patience=0.5,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.5,
            communication_style=CommunicationStyle.casual,
            background="b",
        ),
        scenario=Scenario(name="S", goal="g"),
        user_system_prompt="system",
        first_message="hi",
    )


async def _target(messages: list[Message]) -> str:
    return "ok"


@pytest.mark.asyncio
async def test_sim_forwards_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORQ_API_KEY", "test")
    monkeypatch.setenv("ORQ_BASE_URL", "https://my.staging.orq.ai")

    from evaluatorq.simulation.api import simulate

    spy = AsyncMock(return_value=[])
    with patch.object(_EQ_MOD, "evaluatorq", spy):
        await simulate(
            target=_target,
            datapoints=[_make_datapoint()],
            sim_model="test",
            max_turns=1,
            exit_on_failure=False,
        )

    assert spy.await_args is not None
    assert spy.await_args.kwargs["_base_url"] == "https://my.staging.orq.ai"
