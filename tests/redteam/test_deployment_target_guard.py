"""Deployment targets are refused by the adaptive pipelines (RES-1493).

Only the static pipeline drives a deployment end to end. Dynamic and hybrid would
send the key to the agents API and fail as a missing agent, so they refuse it
first, before any client, credential or paid call.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest

from evaluatorq.redteam.contracts import Pipeline, SaveMode
from evaluatorq.redteam.runner import _reject_deployment_targets, red_team

STATIC = f'mode="{Pipeline.STATIC.value}"'


@pytest.mark.parametrize('mode', ['dynamic', 'hybrid'])
@pytest.mark.asyncio
async def test_red_team_refuses_deployment_target(mode: str) -> None:
    with pytest.raises(ValueError) as exc:
        await red_team(target='deployment:my-key', mode=mode)

    message = str(exc.value)
    assert 'deployment:my-key' in message
    assert STATIC in message
    assert 'simulate()' in message
    assert mode in message


@pytest.mark.asyncio
async def test_red_team_refuses_deployment_target_by_default() -> None:
    """No explicit mode means dynamic, which is the combination users hit first."""
    with pytest.raises(ValueError, match='deployment:my-key'):
        await red_team(target='deployment:my-key')


@pytest.mark.asyncio
async def test_red_team_refuses_a_deployment_among_agents() -> None:
    with pytest.raises(ValueError, match='deployment:second'):
        await red_team(target=['agent:first', 'deployment:second'], mode='dynamic')


@pytest.mark.asyncio
async def test_refusal_precedes_the_credential_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """A keyless user gets the deployment message, not 'missing LLM credentials'.

    The wrong-diagnosis error is the whole failure this guard exists to prevent,
    so its position relative to the credential check is part of the contract.
    """
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('ORQ_API_KEY', raising=False)

    with pytest.raises(ValueError) as exc:
        await red_team(target='deployment:my-key', mode='dynamic')

    assert STATIC in str(exc.value)


@pytest.mark.asyncio
async def test_static_pipeline_still_accepts_a_deployment() -> None:
    """The mode gate lives at the call site, so nothing else pins it.

    Without this, hoisting the gate into ``_reject_deployment_targets`` would kill
    the only supported way to red-team a deployment with every test still green.
    The static leg is replaced by a sentinel: reaching it is the assertion, and
    running it would need a live dataset and a live deployment.
    """

    class _Reached(Exception):
        pass

    with patch('evaluatorq.redteam.runner._run_static', new_callable=AsyncMock, side_effect=_Reached):
        with pytest.raises(_Reached):
            await red_team(target='deployment:my-key', mode='static', save=SaveMode.NONE)


@pytest.mark.parametrize('mode', [Pipeline.DYNAMIC, Pipeline.HYBRID])
def test_guard_allows_agent_targets(mode: Pipeline) -> None:
    _reject_deployment_targets(['agent:my-agent'], mode)


def test_guard_names_the_offending_target_among_several() -> None:
    with pytest.raises(ValueError, match='deployment:second'):
        _reject_deployment_targets(['agent:first', 'deployment:second'], Pipeline.DYNAMIC)


def test_remedy_names_a_parameter_red_team_actually_takes() -> None:
    """The message is the deliverable, so the keyword it quotes must exist."""
    with pytest.raises(ValueError) as exc:
        _reject_deployment_targets(['deployment:x'], Pipeline.DYNAMIC)

    assert 'mode=' in str(exc.value)
    assert 'mode' in inspect.signature(red_team).parameters
