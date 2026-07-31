"""E2E: a saved run can be replayed verbatim against a different target."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from pathlib import Path  # noqa: TC003 — pytest fixture annotation
from typing import cast
from unittest.mock import patch

import pytest
from openai import AsyncOpenAI

from evaluatorq.redteam import red_team
from evaluatorq.redteam import runner as runner_mod
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


@pytest.mark.asyncio
async def test_hybrid_and_static_replays_keep_their_pipeline_and_split(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
    static_dataset_path: Path,
) -> None:
    """A hybrid replay restores both legs; a static replay stays static."""
    client = cast(AsyncOpenAI, cast(object, mock_llm_client))

    with _patches(mock_backend_bundle):
        hybrid = await red_team(
            'agent:e2e-test-agent',
            mode='hybrid',
            categories=['ASI01'],
            generate_strategies=False,
            parallelism=2,
            llm_client=client,
            dataset=str(static_dataset_path),
            name='hybrid-source',
        )
    with _patches(mock_backend_bundle):
        hybrid_replay = await red_team(
            'agent:e2e-test-agent', previous_run='hybrid-source', parallelism=2, llm_client=client, name='hybrid-replay'
        )

    assert hybrid_replay.pipeline == hybrid.pipeline
    assert _attack_ids(hybrid_replay) == _attack_ids(hybrid)
    assert hybrid_replay.summary.datapoint_breakdown == hybrid.summary.datapoint_breakdown

    with _patches(mock_backend_bundle):
        static = await red_team(
            'agent:e2e-test-agent',
            mode='static',
            categories=['ASI01'],
            parallelism=2,
            llm_client=client,
            dataset=str(static_dataset_path),
            name='static-source',
        )
    with _patches(mock_backend_bundle):
        static_replay = await red_team(
            'agent:e2e-test-agent', previous_run='static-source', parallelism=2, llm_client=client, name='static-replay'
        )

    assert static_replay.pipeline.value == 'static'
    assert _attack_ids(static_replay) == _attack_ids(static)


@pytest.mark.asyncio
async def test_replay_restores_the_original_turn_budget(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
) -> None:
    """max_turns isn't in the datapoints, so it must be carried by the run record."""
    client = cast(AsyncOpenAI, cast(object, mock_llm_client))

    with _patches(mock_backend_bundle):
        await red_team(
            'agent:e2e-test-agent',
            mode='dynamic',
            categories=['ASI01'],
            generate_strategies=False,
            max_turns=9,
            attacker_instructions='handles refunds',
            parallelism=2,
            llm_client=client,
            name='turns-source',
        )

    replay = load_redteam_replay('turns-source', get_runs_dir())
    assert replay.max_turns == 9
    assert replay.attacker_instructions == 'handles refunds'

    seen: list[int] = []
    real_job = runner_mod.create_dynamic_redteam_job

    def _record(*args, **kwargs):
        seen.append(kwargs['max_turns'])
        return real_job(*args, **kwargs)

    with _patches(mock_backend_bundle), patch.object(runner_mod, 'create_dynamic_redteam_job', _record):
        await red_team(
            'agent:e2e-test-agent', previous_run='turns-source', parallelism=2, llm_client=client, name='turns-replay'
        )
    assert seen == [9], f'replay should run at the original turn budget, got {seen}'

    # An explicit value still wins over the restored one.
    seen.clear()
    with _patches(mock_backend_bundle), patch.object(runner_mod, 'create_dynamic_redteam_job', _record):
        await red_team(
            'agent:e2e-test-agent',
            previous_run='turns-source',
            max_turns=3,
            parallelism=2,
            llm_client=client,
            name='turns-override',
        )
    assert seen == [3]


@pytest.mark.asyncio
async def test_multi_target_replay_runs_every_target_on_the_same_cases(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
) -> None:
    client = cast(AsyncOpenAI, cast(object, mock_llm_client))

    with _patches(mock_backend_bundle):
        original = await red_team(
            'agent:e2e-test-agent',
            mode='dynamic',
            categories=['ASI01'],
            generate_strategies=False,
            parallelism=2,
            llm_client=client,
            name='multi-source',
        )

    with _patches(mock_backend_bundle):
        replayed = await red_team(
            ['agent:v1', 'agent:v2'], previous_run='multi-source', parallelism=2, llm_client=client, name='multi-replay'
        )

    assert sorted(replayed.tested_agents) == ['v1', 'v2']
    assert replayed.total_results == original.total_results * 2
    assert set(_attack_ids(replayed)) == set(_attack_ids(original))


@pytest.mark.asyncio
async def test_hybrid_replay_against_a_bare_agent_target_keeps_the_static_leg(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
    static_dataset_path: Path,
) -> None:
    """The AgentTarget leg must split replayed datapoints by their hybrid tag.

    Before the split fix this branch only recognised a static leg when a dataset
    had been loaded in-process, so on a replay (where none is) every static row
    was relabelled dynamic and sent to the wrong inner job. With no string target
    present the AgentTarget is the first prepared target, which is what makes the
    difference reach the report.
    """
    from .conftest import MockAgentTarget

    client = cast(AsyncOpenAI, cast(object, mock_llm_client))

    with _patches(mock_backend_bundle):
        original = await red_team(
            'agent:e2e-test-agent',
            mode='hybrid',
            categories=['ASI01'],
            generate_strategies=False,
            parallelism=2,
            llm_client=client,
            dataset=str(static_dataset_path),
            name='at-hybrid-source',
        )
    source_static = (original.summary.datapoint_breakdown or {}).get('static', 0)
    assert source_static > 0, 'fixture should produce a static leg'

    with _patches(mock_backend_bundle):
        replayed = await red_team(
            MockAgentTarget('direct-agent'),
            previous_run='at-hybrid-source',
            parallelism=2,
            llm_client=client,
            name='at-hybrid-replay',
        )

    assert (replayed.summary.datapoint_breakdown or {}).get('static') == source_static
    assert _attack_ids(replayed) == _attack_ids(original)
