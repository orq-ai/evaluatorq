"""evaluatorq() reads experiment_url off the OrqResponse and feeds the sink."""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import evaluatorq.evaluatorq  # noqa: F401  populate sys.modules; the package attr is shadowed by the function
from evaluatorq import DataPoint, evaluatorq
from evaluatorq.send_results import OrqResponse

# evaluatorq.evaluatorq resolves to the re-exported function, not the submodule,
# so patch the module object directly (3.10's mock can't resolve the shadowed path).
_EQ_MOD = sys.modules["evaluatorq.evaluatorq"]


async def _job(data: DataPoint, _row: int) -> dict[str, Any]:
    return {"name": "j", "output": "ok"}


def _response(url: str | None) -> OrqResponse:
    return OrqResponse(
        sheet_id="s1",
        manifest_id="m1",
        experiment_name="run",
        rows_created=1,
        experiment_url=url,
    )


async def _run_with_upload_response(response: OrqResponse | None) -> list[str]:
    sink: list[str] = []
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
        patch.object(_EQ_MOD, "send_results_to_orq", AsyncMock(return_value=response)),
    ):
        await evaluatorq(
            "run",
            data=[DataPoint(inputs={"x": 1})],
            jobs=[_job],
            print_results=False,
            _experiment_url_out=sink,
        )
    return sink


@pytest.mark.asyncio
async def test_sink_receives_experiment_url_from_orq_response() -> None:
    """The happy path: a real OrqResponse lands its URL in the caller's sink."""
    sink = await _run_with_upload_response(_response("https://my.orq.ai/e/abc"))
    assert sink == ["https://my.orq.ai/e/abc"]


@pytest.mark.asyncio
async def test_sink_untouched_when_upload_unconfirmed() -> None:
    sink = await _run_with_upload_response(None)
    assert sink == []


@pytest.mark.asyncio
async def test_sink_untouched_when_response_has_no_url() -> None:
    sink = await _run_with_upload_response(_response(None))
    assert sink == []
