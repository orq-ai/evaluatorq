"""E2E: a saved run can be replayed verbatim against a different target."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import cast
from unittest.mock import patch

import pytest
from openai import AsyncOpenAI

from evaluatorq.redteam import red_team
from evaluatorq.redteam.replay import load_redteam_replay
from evaluatorq.redteam.runner import get_runs_dir
from evaluatorq.tracing import TracingContext

from .conftest import (
    DeterministicAsyncOpenAI,
    MockBackend,
    validate_report_structure,
)


@asynccontextmanager
async def _noop_tracing_session(*args, **kwargs):
    yield TracingContext(run_id='test', run_name='test', enabled=False, parent_context=None, trace_type='redteam')


@contextmanager
def _patches(mock_backend_bundle: MockBackend):
    with (
        patch('evaluatorq.redteam.runner.resolve_backend', return_value=mock_backend_bundle),
        patch('evaluatorq.redteam.backends.registry.resolve_backend', return_value=mock_backend_bundle),
        patch('evaluatorq.redteam.runner.tracing_session', _noop_tracing_session),
    ):
        yield


def _attack_ids(report) -> list[str]:
    return sorted(r.attack.id for r in report.results)


@pytest.mark.asyncio
async def test_dynamic_run_replays_the_same_attacks(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
) -> None:
    """The replay re-runs the stored cases and generates nothing new."""
    client = cast(AsyncOpenAI, cast(object, mock_llm_client))

    with _patches(mock_backend_bundle):
        original = await red_team(
            'agent:e2e-test-agent',
            mode='dynamic',
            categories=['ASI01'],
            generate_strategies=False,
            parallelism=2,
            llm_client=client,
            name='replay-source',
        )

    stored = load_redteam_replay('latest', get_runs_dir())
    assert stored.datapoints, 'the saved run should carry its datapoints'

    # A replay against a *different* target: same cases, new subject.
    with _patches(mock_backend_bundle):
        replayed = await red_team(
            'agent:other-agent',
            previous_run='latest',
            parallelism=2,
            llm_client=client,
            name='replay-run',
        )

    errors = validate_report_structure(replayed, expected_pipeline='dynamic', min_results=1)
    assert not errors, f'Report validation errors: {errors}'
    assert _attack_ids(replayed) == _attack_ids(original)
    assert replayed.tested_agents == ['other-agent']


@pytest.mark.asyncio
async def test_replay_skips_strategy_generation(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
) -> None:
    """No planner call happens on replay even with generation left enabled."""
    client = cast(AsyncOpenAI, cast(object, mock_llm_client))

    with _patches(mock_backend_bundle):
        await red_team(
            'agent:e2e-test-agent',
            mode='dynamic',
            categories=['ASI01'],
            generate_strategies=False,
            parallelism=2,
            llm_client=client,
            name='replay-source',
        )

    with (
        _patches(mock_backend_bundle),
        patch('evaluatorq.redteam.runner.generate_dynamic_datapoints_for_vulnerabilities') as gen_vulns,
        patch('evaluatorq.redteam.runner.generate_dynamic_datapoints') as gen_cats,
        patch('evaluatorq.redteam.runner.classify_agent_capabilities') as classify,
    ):
        replayed = await red_team(
            'agent:e2e-test-agent',
            previous_run='latest',
            parallelism=2,
            llm_client=client,
            name='replay-run',
        )

    assert gen_vulns.call_count == 0
    assert gen_cats.call_count == 0
    assert classify.call_count == 0
    assert replayed.total_results > 0


@pytest.mark.asyncio
async def test_replayed_run_is_itself_replayable(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
) -> None:
    """Replays are saved with their datapoints too, so the chain doesn't break."""
    client = cast(AsyncOpenAI, cast(object, mock_llm_client))

    with _patches(mock_backend_bundle):
        await red_team(
            'agent:e2e-test-agent',
            mode='dynamic',
            categories=['ASI01'],
            generate_strategies=False,
            parallelism=2,
            llm_client=client,
            name='replay-source',
        )
    first = load_redteam_replay('latest', get_runs_dir())

    with _patches(mock_backend_bundle):
        await red_team(
            'agent:e2e-test-agent',
            previous_run='replay-source',
            parallelism=2,
            llm_client=client,
            name='replay-run',
        )

    second = load_redteam_replay('replay-run', get_runs_dir())
    assert [dp.inputs for dp in second.datapoints] == [dp.inputs for dp in first.datapoints]
