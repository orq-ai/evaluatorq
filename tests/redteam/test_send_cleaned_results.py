"""Tests for _send_cleaned_results URL persistence on the report."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from typing import TYPE_CHECKING

import pytest

from evaluatorq.redteam.contracts import Pipeline, RedTeamReport, ReportSummary
from evaluatorq.redteam.runner import _send_cleaned_results
from evaluatorq.send_results import OrqResponse
from evaluatorq.types import DataPoint, DataPointResult, JobResult

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def _make_response(rows_created: int = 1, url: str | None = "https://orq.example/experiments/abc") -> OrqResponse:
    return OrqResponse(
        sheet_id="sheet-1",
        manifest_id="manifest-1",
        experiment_name="n",
        rows_created=rows_created,
        experiment_url=url,
    )


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
            return_value=_make_response(),
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
    from typing import cast

    report = _make_report()
    spy = AsyncMock(return_value=None)
    client = cast("AsyncOpenAI", cast(object, SimpleNamespace(base_url="https://my.staging.orq.ai/v3/router")))
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
            inference_client=client,
        )
    assert spy.await_args is not None
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
    assert spy.await_args is not None
    assert spy.await_args.kwargs["base_url"] == "https://my.staging.orq.ai"


@pytest.mark.asyncio
async def test_persists_upload_diagnostics_on_report() -> None:
    """uploaded_count + rows_created land on the report alongside the URL, so a
    local JSON is enough to diagnose an Explorer sample-count mismatch."""
    report = _make_report()
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
        patch(
            "evaluatorq.redteam.runner.send_results_to_orq",
            new_callable=AsyncMock,
            return_value=_make_response(rows_created=1),
        ),
    ):
        await _send_cleaned_results(
            results=[_make_result()],
            name="n",
            description="d",
            start_time=datetime.now(tz=timezone.utc),
            report=report,
        )
    assert report.uploaded_count == 1
    assert report.rows_created == 1
    assert report.experiment_url == "https://orq.example/experiments/abc"


@pytest.mark.asyncio
async def test_diagnostics_none_when_upload_returns_none() -> None:
    """A failed upload leaves rows_created unset but records the attempt count."""
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
    assert report.uploaded_count == 1
    assert report.rows_created is None
    assert report.experiment_url is None


@pytest.mark.asyncio
async def test_url_persisted_even_when_rows_created_gap() -> None:
    """A rows_created < uploaded gap still persists all three diagnostics."""
    report = _make_report()
    results = [_make_result(), _make_result()]
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
        patch(
            "evaluatorq.redteam.runner.send_results_to_orq",
            new_callable=AsyncMock,
            return_value=_make_response(rows_created=1),
        ),
    ):
        await _send_cleaned_results(
            results=results,
            name="n",
            description="d",
            start_time=datetime.now(tz=timezone.utc),
            report=report,
        )
    assert report.uploaded_count == 2
    assert report.rows_created == 1
    assert report.experiment_url == "https://orq.example/experiments/abc"


@pytest.mark.asyncio
async def test_report_json_roundtrips_diagnostics() -> None:
    """The new optional fields serialize into the run JSON and old JSONs
    without them still validate."""
    report = _make_report()
    report.uploaded_count = 17
    report.rows_created = 8
    data = report.model_dump(mode="json")
    assert data["uploaded_count"] == 17
    assert data["rows_created"] == 8
    # backward compat: a legacy payload without the new keys still validates
    data.pop("uploaded_count")
    data.pop("rows_created")
    legacy = RedTeamReport.model_validate(data)
    assert legacy.uploaded_count is None
    assert legacy.rows_created is None


@pytest.mark.asyncio
async def test_all_rows_stripped_warns_and_records_zero_uploaded() -> None:
    """Raw rows that all strip to nothing must warn loudly (not DEBUG) and
    persist uploaded_count=0 — the worst-case '0 samples in Explorer' path."""
    from loguru import logger as _logger

    report = _make_report()
    stripped = DataPointResult(
        data_point=DataPoint(inputs={"x": 1}),
        job_results=[JobResult(job_name="j", output=None)],  # pyright: ignore[reportArgumentType]
    )
    lines: list[str] = []
    handler_id = _logger.add(lambda m: lines.append(str(m)), level="WARNING", format="{message}")
    try:
        with (
            patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
            patch(
                "evaluatorq.redteam.runner.send_results_to_orq",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            await _send_cleaned_results(
                results=[stripped],
                name="n",
                description="d",
                start_time=datetime.now(tz=timezone.utc),
                report=report,
            )
    finally:
        _logger.remove(handler_id)
    mock_send.assert_not_awaited()
    assert report.uploaded_count == 0
    assert report.rows_created is None
    warnings = [ln for ln in lines if "nothing uploaded" in ln]
    assert len(warnings) == 1
    assert "1 result row" in warnings[0]


@pytest.mark.asyncio
async def test_empty_results_stays_quiet() -> None:
    """No raw rows at all is not a mismatch — no warning, uploaded_count=0."""
    from loguru import logger as _logger

    report = _make_report()
    lines: list[str] = []
    handler_id = _logger.add(lambda m: lines.append(str(m)), level="WARNING", format="{message}")
    try:
        with patch.dict(os.environ, {"ORQ_API_KEY": "test"}):
            await _send_cleaned_results(
                results=[],
                name="n",
                description="d",
                start_time=datetime.now(tz=timezone.utc),
                report=report,
            )
    finally:
        _logger.remove(handler_id)
    assert report.uploaded_count == 0
    assert not lines


@pytest.mark.asyncio
async def test_uploaded_count_recorded_when_upload_raises() -> None:
    """The attempt count survives an upload exception in the persisted report."""
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
    assert report.uploaded_count == 1
    assert report.rows_created is None


@pytest.mark.asyncio
async def test_auto_saved_run_json_contains_diagnostics(tmp_path) -> None:
    """The runs-index JSON written by _auto_save_run carries the diagnostics
    after _send_cleaned_results mutates the report (guards the mutate-before-
    save ordering the pipelines rely on)."""
    import json as _json

    from evaluatorq.redteam import runner as runner_mod

    report = _make_report()
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
        patch(
            "evaluatorq.redteam.runner.send_results_to_orq",
            new_callable=AsyncMock,
            return_value=_make_response(rows_created=1),
        ),
    ):
        await _send_cleaned_results(
            results=[_make_result()],
            name="n",
            description="d",
            start_time=datetime.now(tz=timezone.utc),
            report=report,
        )
    with patch.object(runner_mod, "get_runs_dir", return_value=tmp_path):
        path = runner_mod._auto_save_run(report, name="diag-test")
    assert path is not None
    data = _json.loads(path.read_text())
    assert data["uploaded_count"] == 1
    assert data["rows_created"] == 1
    assert data["experiment_url"] == "https://orq.example/experiments/abc"


def _make_empty_result() -> DataPointResult:
    """A row whose only job output is None — stripped during cleaning."""
    return DataPointResult(
        data_point=DataPoint(inputs={"x": 2}),
        job_results=[JobResult(job_name="j", output=None)],  # pyright: ignore[reportArgumentType]
    )


def _capture_loguru() -> tuple[list[str], int]:
    from loguru import logger as _logger

    lines: list[str] = []
    handler_id = _logger.add(lambda m: lines.append(str(m)), level="DEBUG")
    return lines, handler_id


@pytest.mark.asyncio
async def test_unconfirmed_upload_warns_instead_of_silent_return() -> None:
    """send_results_to_orq returning None (it swallowed an upload error) must
    leave a WARNING, not exit wordless after the 'Uploading N...' line."""
    from loguru import logger as _logger

    report = _make_report()
    lines, handler_id = _capture_loguru()
    try:
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
    finally:
        _logger.remove(handler_id)
    warnings = [ln for ln in lines if "was not confirmed" in ln]
    assert len(warnings) == 1
    assert report.uploaded_count == 1
    assert report.rows_created is None


@pytest.mark.asyncio
async def test_partial_cleaning_drop_warns() -> None:
    """A partial drop (some rows stripped, some uploaded) warns with both counts."""
    from loguru import logger as _logger

    lines, handler_id = _capture_loguru()
    try:
        with (
            patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
            patch(
                "evaluatorq.redteam.runner.send_results_to_orq",
                new_callable=AsyncMock,
                return_value=_make_response(rows_created=1),
            ),
        ):
            await _send_cleaned_results(
                results=[_make_result(), _make_empty_result(), _make_empty_result()],
                name="n",
                description="d",
                start_time=datetime.now(tz=timezone.utc),
            )
    finally:
        _logger.remove(handler_id)
    drop_warnings = [ln for ln in lines if "2 of 3 result row(s) had no real" in ln]
    assert len(drop_warnings) == 1


@pytest.mark.asyncio
async def test_no_cleaning_drop_warning_when_all_rows_survive() -> None:
    from loguru import logger as _logger

    lines, handler_id = _capture_loguru()
    try:
        with (
            patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
            patch(
                "evaluatorq.redteam.runner.send_results_to_orq",
                new_callable=AsyncMock,
                return_value=_make_response(rows_created=1),
            ),
        ):
            await _send_cleaned_results(
                results=[_make_result()],
                name="n",
                description="d",
                start_time=datetime.now(tz=timezone.utc),
            )
    finally:
        _logger.remove(handler_id)
    assert not [ln for ln in lines if "had no real output and were dropped" in ln]
