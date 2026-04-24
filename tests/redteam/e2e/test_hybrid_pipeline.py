"""E2E tests for the hybrid red teaming pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from openai import AsyncOpenAI

from evaluatorq.redteam import red_team
from evaluatorq.redteam.backends.base import BackendBundle

from .conftest import DeterministicAsyncOpenAI, validate_report_structure


@contextmanager
def _hybrid_patches(mock_backend_bundle: BackendBundle):
    """Patch lazy imports used by _run_hybrid."""
    with (
        patch("evaluatorq.redteam.runner.resolve_backend", return_value=mock_backend_bundle),
        patch("evaluatorq.redteam.runner.init_tracing_if_needed", return_value=None),
        patch("evaluatorq.redteam.runner.capture_parent_context", return_value=None),
    ):
        yield


@pytest.mark.asyncio
async def test_full_hybrid_run(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: BackendBundle,
    static_dataset_path: Path,
) -> None:
    """Full hybrid pipeline: dynamic + static merged."""
    with _hybrid_patches(mock_backend_bundle):
        report = await red_team(
            "agent:e2e-test-agent",
            mode="hybrid",
            categories=["ASI01"],
            generate_strategies=False,
            parallelism=2,
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
            dataset=str(static_dataset_path),
            description="E2E hybrid test",
        )

    errors = validate_report_structure(report, expected_pipeline="hybrid", min_results=1)
    assert not errors, f"Report validation errors: {errors}"

    # Hybrid should have a datapoint_breakdown
    breakdown = report.summary.datapoint_breakdown
    assert breakdown is not None, "Hybrid report should have datapoint_breakdown"
    assert "static" in breakdown or "template_dynamic" in breakdown, (
        f"Expected breakdown keys, got {breakdown}"
    )


@pytest.mark.asyncio
async def test_hybrid_independent_caps(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: BackendBundle,
    static_dataset_path: Path,
) -> None:
    """Independent capping: max_dynamic_datapoints=1 + max_static_datapoints=1."""
    with _hybrid_patches(mock_backend_bundle):
        report = await red_team(
            "agent:e2e-test-agent",
            mode="hybrid",
            categories=["ASI01"],
            generate_strategies=False,
            max_dynamic_datapoints=1,
            max_static_datapoints=1,
            parallelism=2,
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
            dataset=str(static_dataset_path),
        )

    assert report.total_results <= 2, f"Expected at most 2 results, got {report.total_results}"
