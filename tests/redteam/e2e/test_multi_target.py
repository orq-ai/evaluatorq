"""E2E tests for multi-target red teaming."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from openai import AsyncOpenAI

from evaluatorq.redteam import red_team

from .conftest import DeterministicAsyncOpenAI, validate_report_structure


@pytest.mark.asyncio
async def test_multi_target_static_merge(
    mock_llm_client: DeterministicAsyncOpenAI,
    static_dataset_path: Path,
) -> None:
    """Two static targets should produce a merged report with results from both."""
    report = await red_team(
        ["openai:model-a", "openai:model-b"],
        mode="static",
        evaluator_model="e2e-evaluator",
        parallelism=2,
        backend="openai",
        dataset=str(static_dataset_path),
        llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
        description="E2E multi-target test",
    )

    errors = validate_report_structure(report, expected_pipeline="static", min_results=2)
    assert not errors, f"Report validation errors: {errors}"

    # 3 samples x 2 targets = 6 results
    assert report.total_results == 6

    # Each result should have an agent model set
    agent_models = {r.agent.model for r in report.results}
    assert "model-a" in agent_models, f"Expected model-a in results, got {agent_models}"
    assert "model-b" in agent_models, f"Expected model-b in results, got {agent_models}"
