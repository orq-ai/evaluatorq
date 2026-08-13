from __future__ import annotations

# ruff: noqa: S101
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import LengthFinishReasonError
from pydantic import BaseModel

from evaluatorq.simulation.utils.structured_output import generate_structured


class SampleResponse(BaseModel):
    value: str


@pytest.mark.asyncio
async def test_generate_structured_raises_when_parse_hits_length_limit() -> None:
    # Length-truncated structured output is unusable — fail with a clear,
    # actionable error instead of falling back to a same-budget json_object call
    # that would truncate again.
    parse_error = LengthFinishReasonError(completion=MagicMock())

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=parse_error)
    client.chat.completions.create = AsyncMock()

    with pytest.raises(RuntimeError, match="Raise the max_tokens budget"):
        await generate_structured(
            client,
            model="local-model",
            messages=[{"role": "user", "content": "return json"}],
            response_format=SampleResponse,
            temperature=0.0,
            max_tokens=4000,
            label="Sample.generate",
        )

    # No json_object fallback call — the truncated result is not salvaged.
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_structural_extra_kwargs_are_rejected() -> None:
    # extra_kwargs silently replacing response_format would defeat the schema
    # the helper exists to enforce — reserved keys raise instead (review fix).
    client = MagicMock()

    with pytest.raises(ValueError, match="structural"):
        await generate_structured(
            client,
            model="local-model",
            messages=[{"role": "user", "content": "return json"}],
            response_format=SampleResponse,
            max_tokens=100,
            label="Sample.generate",
            extra_kwargs={"response_format": {"type": "json_object"}},
        )

    client.chat.completions.parse.assert_not_called()
