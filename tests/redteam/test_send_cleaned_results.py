"""Tests for _send_cleaned_results URL persistence on the report."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from evaluatorq.redteam.contracts import Pipeline, RedTeamReport, ReportSummary
from evaluatorq.redteam.runner import _send_cleaned_results
from evaluatorq.types import DataPoint, DataPointResult, JobResult


def _make_report() -> RedTeamReport:
    return RedTeamReport(  # pyright: ignore[reportArgumentType]
        created_at=datetime.now(tz=timezone.utc),
        description="test",
        pipeline=Pipeline.DYNAMIC,
        framework=None,
        categories_tested=["ASI01"],
        tested_agents=["agent:test"],
        total_results=0,
        results=[],
        summary=ReportSummary(),
    )


def _make_result() -> DataPointResult:
    return DataPointResult(
        data_point=DataPoint(inputs={"x": 1}),
        job_results=[JobResult(job_name="j", output="ok")],  # pyright: ignore[reportArgumentType]
    )


@pytest.mark.asyncio
async def test_sets_experiment_url_on_success() -> None:
    """Successful upload populates report.experiment_url — guards the order
    of upload-before-save so URL lands in the persisted summary report."""
    report = _make_report()
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
        patch(
            "evaluatorq.redteam.runner.send_results_to_orq",
            new_callable=AsyncMock,
            return_value="https://orq.example/experiments/abc",
        ),
    ):
        await _send_cleaned_results(
            results=[_make_result()],
            name="n",
            description="d",
            start_time=datetime.now(tz=timezone.utc),
            report=report,
        )
    assert report.experiment_url == "https://orq.example/experiments/abc"


@pytest.mark.asyncio
async def test_no_url_when_api_key_missing() -> None:
    report = _make_report()
    with patch.dict(os.environ, {}, clear=True):
        await _send_cleaned_results(
            results=[_make_result()],
            name="n",
            description="d",
            start_time=datetime.now(tz=timezone.utc),
            report=report,
        )
    assert report.experiment_url is None


@pytest.mark.asyncio
async def test_no_url_when_upload_returns_none() -> None:
    report = _make_report()
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
        patch(
            "evaluatorq.redteam.runner.send_results_to_orq",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        await _send_cleaned_results(
            results=[_make_result()],
            name="n",
            description="d",
            start_time=datetime.now(tz=timezone.utc),
            report=report,
        )
    assert report.experiment_url is None


@pytest.mark.asyncio
async def test_upload_exception_does_not_break_report() -> None:
    """Upload failures are swallowed; report.experiment_url stays None."""
    report = _make_report()
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
        patch(
            "evaluatorq.redteam.runner.send_results_to_orq",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ),
    ):
        await _send_cleaned_results(
            results=[_make_result()],
            name="n",
            description="d",
            start_time=datetime.now(tz=timezone.utc),
            report=report,
        )
    assert report.experiment_url is None


@pytest.mark.asyncio
async def test_forwards_inference_client_base_url() -> None:
    """RES-912: the Orq host used for inference is forwarded to the upload so
    results land on the same server (e.g. staging)."""
    from types import SimpleNamespace

    report = _make_report()
    spy = AsyncMock(return_value=None)
    client = SimpleNamespace(base_url="https://my.staging.orq.ai/v3/router")
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
        patch("evaluatorq.redteam.runner.send_results_to_orq", spy),
    ):
        await _send_cleaned_results(
            results=[_make_result()],
            name="n",
            description="d",
            start_time=datetime.now(tz=timezone.utc),
            report=report,
            inference_client=client,  # type: ignore[arg-type]
        )
    assert spy.await_args.kwargs["base_url"] == "https://my.staging.orq.ai"


@pytest.mark.asyncio
async def test_base_url_falls_back_to_env_without_client() -> None:
    report = _make_report()
    spy = AsyncMock(return_value=None)
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test", "ORQ_BASE_URL": "https://my.staging.orq.ai"}),
        patch("evaluatorq.redteam.runner.send_results_to_orq", spy),
    ):
        await _send_cleaned_results(
            results=[_make_result()],
            name="n",
            description="d",
            start_time=datetime.now(tz=timezone.utc),
            report=report,
        )
    assert spy.await_args.kwargs["base_url"] == "https://my.staging.orq.ai"
