"""RES-912: agent sim forwards the inference base_url to the results upload."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

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
    with patch("evaluatorq.evaluatorq.evaluatorq", spy):
        await simulate(
            target=_target,
            datapoints=[_make_datapoint()],
            sim_model="test",
            max_turns=1,
            exit_on_failure=False,
        )

    assert spy.await_args is not None
    assert spy.await_args.kwargs["_base_url"] == "https://my.staging.orq.ai"
