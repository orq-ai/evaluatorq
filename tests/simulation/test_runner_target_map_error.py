"""SimulationRunner passes the target's own map_error, so cli.* codes reach simulation results."""

from __future__ import annotations

import pytest

from evaluatorq.contracts import AgentResponse, AgentTarget, Message
from evaluatorq.simulation.runner.simulation import SimulationRunner


class _Boom(Exception):
    pass


class _MappingTarget(AgentTarget):
    def __init__(self) -> None:
        super().__init__()

    async def respond(self, messages: list[Message]) -> AgentResponse:
        raise _Boom('exit 2')

    def new(self) -> _MappingTarget:
        return type(self)()

    def map_error(self, exc: Exception) -> tuple[str, str] | None:
        if isinstance(exc, _Boom):
            return 'cli.exit.2', str(exc)
        return None


class _NoOpinionTarget(_MappingTarget):
    def map_error(self, exc: Exception) -> tuple[str, str] | None:
        return None


@pytest.mark.asyncio
async def test_runner_uses_target_map_error() -> None:
    runner = SimulationRunner(target_agent=_MappingTarget(), max_target_retries=0)
    result = await runner._get_target_response([Message(role='user', content='hi')], target=_MappingTarget())
    assert result.error is not None
    assert result.error.code == 'cli.exit.2'


@pytest.mark.asyncio
async def test_runner_falls_back_to_default_mapping_when_target_returns_none() -> None:
    runner = SimulationRunner(target_agent=_NoOpinionTarget(), max_target_retries=0)
    result = await runner._get_target_response([Message(role='user', content='hi')], target=_NoOpinionTarget())
    assert result.error is not None
    assert result.error.code != 'cli.exit.2'
