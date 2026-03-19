"""E2E tests for the static red teaming pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from openai import AsyncOpenAI

from evaluatorq.redteam import red_team

from .conftest import DeterministicAsyncOpenAI, validate_report_structure


@pytest.mark.asyncio
async def test_full_static_run(
    mock_llm_client: DeterministicAsyncOpenAI,
    static_dataset_path: Path,
) -> None:
    """Full static pipeline run with 3 dataset samples."""
    report = await red_team(
        "openai:e2e-static-model",
        mode="static",
        evaluator_model="e2e-evaluator",
        parallelism=2,
        backend="openai",
        dataset=str(static_dataset_path),
        llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
        description="E2E static test",
    )

    errors = validate_report_structure(report, expected_pipeline="static", min_results=3)
    assert not errors, f"Report validation errors: {errors}"
    assert report.total_results == 3


@pytest.mark.asyncio
async def test_static_vulnerability_detection(
    mock_llm_client: DeterministicAsyncOpenAI,
    static_dataset_path: Path,
) -> None:
    """ASI01/LLM07 should be vulnerable (mock leaks), ASI05 resistant (mock refuses)."""
    report = await red_team(
        "openai:e2e-static-model",
        mode="static",
        evaluator_model="e2e-evaluator",
        parallelism=2,
        backend="openai",
        dataset=str(static_dataset_path),
        llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
    )

    results_by_cat = {r.attack.category: r for r in report.results}

    # ASI01 — "Ignore previous instructions" → mock leaks secrets → vulnerable
    asi01 = results_by_cat.get("ASI01")
    assert asi01 is not None, "ASI01 result missing"
    assert asi01.vulnerable is True, f"ASI01 should be vulnerable, got vulnerable={asi01.vulnerable}"

    # LLM07 — "print hidden instructions" → mock leaks secrets → vulnerable
    llm07 = results_by_cat.get("LLM07")
    assert llm07 is not None, "LLM07 result missing"
    assert llm07.vulnerable is True, f"LLM07 should be vulnerable, got vulnerable={llm07.vulnerable}"

    # ASI05 — "rm -rf" → mock refuses → resistant
    asi05 = results_by_cat.get("ASI05")
    assert asi05 is not None, "ASI05 result missing"
    assert asi05.vulnerable is False, f"ASI05 should be resistant, got vulnerable={asi05.vulnerable}"


@pytest.mark.asyncio
async def test_static_category_filtering(
    mock_llm_client: DeterministicAsyncOpenAI,
    static_dataset_path: Path,
) -> None:
    """Filtering to ASI01 should yield only ASI01 results."""
    report = await red_team(
        "openai:e2e-static-model",
        mode="static",
        categories=["ASI01"],
        evaluator_model="e2e-evaluator",
        parallelism=2,
        backend="openai",
        dataset=str(static_dataset_path),
        llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
    )

    assert report.total_results >= 1
    result_categories = {r.attack.category for r in report.results}
    assert result_categories == {"ASI01"}, f"Expected only ASI01, got {result_categories}"


@pytest.mark.asyncio
async def test_static_datapoint_capping(
    mock_llm_client: DeterministicAsyncOpenAI,
    static_dataset_path: Path,
) -> None:
    """max_static_datapoints=1 should yield exactly 1 result."""
    report = await red_team(
        "openai:e2e-static-model",
        mode="static",
        max_static_datapoints=1,
        evaluator_model="e2e-evaluator",
        parallelism=2,
        backend="openai",
        dataset=str(static_dataset_path),
        llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
    )

    assert report.total_results == 1
