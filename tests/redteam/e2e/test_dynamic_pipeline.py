"""E2E tests for the dynamic red teaming pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import cast
from unittest.mock import patch

import pytest
from openai import AsyncOpenAI

from evaluatorq.redteam import red_team
from evaluatorq.tracing import TracingContext
from .conftest import (
    DeterministicAsyncOpenAI,
    MockBackend,
    validate_report_structure,
)


@contextmanager
def _dynamic_patches(mock_backend_bundle: MockBackend):
    """Patch lazy imports used by _run_dynamic and _run_hybrid."""
    with (
        patch('evaluatorq.redteam.runner.resolve_backend', return_value=mock_backend_bundle),
        patch('evaluatorq.redteam.backends.registry.resolve_backend', return_value=mock_backend_bundle),
        patch('evaluatorq.redteam.runner.tracing_session', _noop_tracing_session),
    ):
        yield


@asynccontextmanager
async def _noop_tracing_session(*args, **kwargs):
    yield TracingContext(run_id='test', run_name='test', enabled=False, parent_context=None, trace_type='redteam')


@pytest.mark.asyncio
async def test_full_dynamic_run(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
) -> None:
    """Full dynamic pipeline run with a single category, no strategy generation."""
    with _dynamic_patches(mock_backend_bundle):
        report = await red_team(
            'agent:e2e-test-agent',
            mode='dynamic',
            categories=['ASI01'],
            generate_strategies=False,
            parallelism=2,
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
            description='E2E dynamic test',
        )

    errors = validate_report_structure(report, expected_pipeline='dynamic', min_results=1)
    assert not errors, f'Report validation errors: {errors}'
    assert 'ASI01' in report.categories_tested


@pytest.mark.asyncio
async def test_dynamic_with_strategy_generation(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
) -> None:
    """Dynamic run with strategy generation enabled."""
    with _dynamic_patches(mock_backend_bundle):
        report = await red_team(
            'agent:e2e-test-agent',
            mode='dynamic',
            categories=['ASI01'],
            generate_strategies=True,
            generated_strategy_count=1,
            parallelism=2,
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
            description='E2E dynamic with generation',
        )

    errors = validate_report_structure(report, expected_pipeline='dynamic', min_results=1)
    assert not errors, f'Report validation errors: {errors}'


@pytest.mark.asyncio
async def test_dynamic_datapoint_capping(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
) -> None:
    """max_dynamic_datapoints=2 should cap results."""
    with _dynamic_patches(mock_backend_bundle):
        report = await red_team(
            'agent:e2e-test-agent',
            mode='dynamic',
            categories=['ASI01'],
            generate_strategies=False,
            max_dynamic_datapoints=2,
            parallelism=2,
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
        )

    assert report.total_results <= 2


@pytest.mark.asyncio
async def test_dynamic_memory_cleanup(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
) -> None:
    """Memory cleanup should be invoked when agent has memory stores."""
    with _dynamic_patches(mock_backend_bundle):
        await red_team(
            'agent:e2e-test-agent',
            mode='dynamic',
            categories=['ASI01'],
            generate_strategies=False,
            cleanup_memory=True,
            parallelism=2,
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
        )

    assert len(mock_backend_bundle.cleaned_entity_ids) > 0, 'Expected memory cleanup to be called'


# Covers the re-widening fix in `_run_dynamic_or_hybrid` — the expensive path, where
# reading `categories=[]` as "no filter" meant a full category sweep against a live
# attacker model. The static equivalent lives in test_static_pipeline.py.
@pytest.mark.parametrize(
    ('categories', 'vulnerabilities'),
    [([], None), (None, [])],
    ids=['categories', 'vulnerabilities'],
)
@pytest.mark.asyncio
async def test_dynamic_empty_filter_hard_fails_instead_of_sweeping_everything(
    categories: list[str] | None,
    vulnerabilities: list[str] | None,
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: MockBackend,
) -> None:
    """An empty selection must not re-widen to every category on the dynamic path."""
    from evaluatorq.redteam.exceptions import RedTeamError

    with _dynamic_patches(mock_backend_bundle), pytest.raises(RedTeamError, match='zero datapoints') as exc:
        await red_team(
            'agent:e2e-test-agent',
            mode='dynamic',
            categories=categories,
            vulnerabilities=vulnerabilities,
            generate_strategies=False,
            parallelism=2,
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
        )

    # The error names the argument the caller actually passed, not a merged
    # `categories=` list carrying vocabulary they never typed.
    expected = 'categories=[]' if categories is not None else 'vulnerabilities=[]'
    assert expected in str(exc.value)
