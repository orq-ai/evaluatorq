"""Deployment targets are refused by the adaptive pipelines (RES-1493).

Only the static pipeline builds a deployment job. Dynamic and hybrid would send
the key to the agents API and fail as a missing agent, so they refuse it first.
"""

from __future__ import annotations

import pytest

from evaluatorq.redteam.contracts import Pipeline
from evaluatorq.redteam.runner import _reject_deployment_targets, red_team


@pytest.mark.parametrize('mode', ['dynamic', 'hybrid'])
@pytest.mark.asyncio
async def test_red_team_refuses_deployment_target(mode: str) -> None:
    with pytest.raises(ValueError) as exc:
        await red_team(target='deployment:my-key', mode=mode)

    message = str(exc.value)
    assert 'deployment:my-key' in message
    assert 'pipeline="static"' in message
    assert mode in message


@pytest.mark.parametrize('mode', [Pipeline.DYNAMIC, Pipeline.HYBRID])
def test_guard_allows_agent_targets(mode: Pipeline) -> None:
    _reject_deployment_targets(['agent:my-agent'], mode)


def test_guard_names_the_offending_target_among_several() -> None:
    with pytest.raises(ValueError, match='deployment:second'):
        _reject_deployment_targets(['agent:first', 'deployment:second'], Pipeline.DYNAMIC)
