"""RES-912: evaluatorq() forwards _base_url to send_results_to_orq."""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import evaluatorq.evaluatorq  # noqa: F401  populate sys.modules; the package attr is shadowed by the function
from evaluatorq import DataPoint, evaluatorq

# evaluatorq.evaluatorq resolves to the re-exported function, not the submodule,
# so patch the module object directly (3.10's mock can't resolve the shadowed path).
_EQ_MOD = sys.modules["evaluatorq.evaluatorq"]


async def _job(data: DataPoint, _row: int) -> dict[str, Any]:
    return {"name": "j", "output": "ok"}


@pytest.mark.asyncio
async def test_forwards_base_url_to_upload() -> None:
    spy = AsyncMock(return_value=None)
    with (
        patch.dict(os.environ, {"ORQ_API_KEY": "test"}),
        patch.object(_EQ_MOD, "send_results_to_orq", spy),
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
        patch.object(_EQ_MOD, "send_results_to_orq", spy),
    ):
        await evaluatorq(
            "run",
            data=[DataPoint(inputs={"x": 1})],
            jobs=[_job],
            print_results=False,
        )
    assert spy.await_args is not None
    assert spy.await_args.kwargs.get("base_url") is None
