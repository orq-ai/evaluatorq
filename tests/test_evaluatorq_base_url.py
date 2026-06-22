"""RES-912: evaluatorq() forwards _base_url to send_results_to_orq."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from evaluatorq import DataPoint, evaluatorq


async def _job(data: DataPoint, _row: int) -> dict[str, Any]:
    return {"name": "j", "output": "ok"}


@pytest.mark.asyncio
async def test_forwards_base_url_to_upload() -> None:
    spy = AsyncMock(return_value=None)
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
        patch("evaluatorq.evaluatorq.send_results_to_orq", spy),
    ):
        await evaluatorq(
            "run",
            data=[DataPoint(inputs={"x": 1})],
            jobs=[_job],
            print_results=False,
            _base_url="https://my.staging.orq.ai",
        )
    assert spy.await_args is not None
    assert spy.await_args.kwargs["base_url"] == "https://my.staging.orq.ai"


@pytest.mark.asyncio
async def test_base_url_defaults_to_none() -> None:
    """Without _base_url, send_results_to_orq receives None and reads env itself."""
    spy = AsyncMock(return_value=None)
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
        patch("evaluatorq.evaluatorq.send_results_to_orq", spy),
    ):
        await evaluatorq(
            "run",
            data=[DataPoint(inputs={"x": 1})],
            jobs=[_job],
            print_results=False,
        )
    assert spy.await_args is not None
    assert spy.await_args.kwargs.get("base_url") is None
