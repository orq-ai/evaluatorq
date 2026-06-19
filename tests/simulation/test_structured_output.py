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

    with pytest.raises(RuntimeError, match="EVALUATORQ_LLM_MAX_TOKENS"):
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
