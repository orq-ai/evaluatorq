"""E2E tests for the dynamic red teaming pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from typing import cast
from unittest.mock import patch

import pytest
from openai import AsyncOpenAI

from evaluatorq.redteam import red_team
from evaluatorq.redteam.backends.base import BackendBundle

from .conftest import (
    DeterministicAsyncOpenAI,
    MockMemoryCleanup,
    validate_report_structure,
)


@contextmanager
def _dynamic_patches(mock_backend_bundle: BackendBundle):
    """Patch lazy imports used by _run_dynamic and _run_hybrid."""
    with (
        patch("evaluatorq.redteam.backends.registry.resolve_backend", return_value=mock_backend_bundle),
        patch("evaluatorq.tracing.setup.init_tracing_if_needed", return_value=None),
        patch("evaluatorq.tracing.context.capture_parent_context", return_value=None),
    ):
        yield


@pytest.mark.asyncio
async def test_full_dynamic_run(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: BackendBundle,
) -> None:
    """Full dynamic pipeline run with a single category, no strategy generation."""
    with _dynamic_patches(mock_backend_bundle):
        report = await red_team(
            "agent:e2e-test-agent",
            mode="dynamic",
            categories=["ASI01"],
            generate_strategies=False,
            attack_model="e2e-attack-model",
            evaluator_model="e2e-evaluator",
            parallelism=2,
            backend="openai",
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
            description="E2E dynamic test",
        )

    errors = validate_report_structure(report, expected_pipeline="dynamic", min_results=1)
    assert not errors, f"Report validation errors: {errors}"
    assert "ASI01" in report.categories_tested


@pytest.mark.asyncio
async def test_dynamic_with_strategy_generation(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: BackendBundle,
) -> None:
    """Dynamic run with strategy generation enabled."""
    with _dynamic_patches(mock_backend_bundle):
        report = await red_team(
            "agent:e2e-test-agent",
            mode="dynamic",
            categories=["ASI01"],
            generate_strategies=True,
            generated_strategy_count=1,
            attack_model="e2e-attack-model",
            evaluator_model="e2e-evaluator",
            parallelism=2,
            backend="openai",
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
            description="E2E dynamic with generation",
        )

    errors = validate_report_structure(report, expected_pipeline="dynamic", min_results=1)
    assert not errors, f"Report validation errors: {errors}"


@pytest.mark.asyncio
async def test_dynamic_datapoint_capping(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: BackendBundle,
) -> None:
    """max_dynamic_datapoints=2 should cap results."""
    with _dynamic_patches(mock_backend_bundle):
        report = await red_team(
            "agent:e2e-test-agent",
            mode="dynamic",
            categories=["ASI01"],
            generate_strategies=False,
            max_dynamic_datapoints=2,
            attack_model="e2e-attack-model",
            evaluator_model="e2e-evaluator",
            parallelism=2,
            backend="openai",
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
        )

    assert report.total_results <= 2


@pytest.mark.asyncio
async def test_dynamic_memory_cleanup(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: BackendBundle,
) -> None:
    """Memory cleanup should be invoked when agent has memory stores."""
    with _dynamic_patches(mock_backend_bundle):
        await red_team(
            "agent:e2e-test-agent",
            mode="dynamic",
            categories=["ASI01"],
            generate_strategies=False,
            cleanup_memory=True,
            attack_model="e2e-attack-model",
            evaluator_model="e2e-evaluator",
            parallelism=2,
            backend="openai",
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
            memory_cleanup=mock_backend_bundle.memory_cleanup,
        )

    cleanup: MockMemoryCleanup = cast(MockMemoryCleanup, mock_backend_bundle.memory_cleanup)
    assert len(cleanup.cleaned_entity_ids) > 0, "Expected memory cleanup to be called"
