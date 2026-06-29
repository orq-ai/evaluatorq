"""Real-server integration tests for OrqResponsesTarget.

These tests require ORQ_API_KEY to be set and make live network calls.
They are excluded from the default test run (skipped unless -m integration).

After RES-808 PR3 the target is stateless: conversation continuity is the
caller's responsibility — the full transcript is passed to ``respond`` each
turn.
"""

from __future__ import annotations

import os

import pytest

from evaluatorq.contracts import AgentResponse, LLMCallConfig, Message
from evaluatorq.openresponses.convert_models import InputImageContent, InputTextContent
from evaluatorq.openresponses.target import OrqResponsesTarget

# 1x1 solid-color PNGs as base64 data URLs. Self-contained so the tests do not
# depend on any live external URL (which could rate-limit, move, or change).
_RED_PIXEL_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
_GREEN_PIXEL_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNg+M8AAAICAQB7CYF4AAAAAElFTkSuQmCC"
)


@pytest.mark.integration
class TestOrqResponsesTargetIntegration:
    @pytest.mark.asyncio
    async def test_responses_v3_real_call_recalls_context_from_transcript(self):
        """Multi-turn recall works when the caller passes the full transcript.

        Turn 1: tell the model a name.
        Turn 2: pass turn-1 user + assistant + new user question; verify recall.
        Statelessness means no server-side threading — the model only knows what
        is in the message list it receives.
        """
        if not os.environ.get("ORQ_API_KEY"):
            pytest.skip("ORQ_API_KEY not set")

        config = LLMCallConfig(model="openai/gpt-4o-mini")
        target = OrqResponsesTarget(config, instructions="Reply tersely.")

        # Turn 1: establish context.
        r1 = await target.respond([Message(role="user", content="My name is Banana.")])
        assert isinstance(r1, AgentResponse)
        assert r1.text

        # Turn 2: caller threads the transcript explicitly.
        r2 = await target.respond(
            [
                Message(role="user", content="My name is Banana."),
                Message(role="assistant", content=r1.text),
                Message(role="user", content="What is my name?"),
            ]
        )
        assert "banana" in r2.text.lower()

        # Usage is reported on the response itself (no instance accumulation).
        assert r2.usage is not None
        assert r2.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_multipart_image_base64_round_trips(self):
        """RES-879: a base64 image part actually reaches the vision model.

        Asserts the model reports the image color, not merely that the HTTP call
        succeeded -- a truthy ``r.text`` would pass even if the image were dropped.
        """
        if not os.environ.get("ORQ_API_KEY"):
            pytest.skip("ORQ_API_KEY not set")

        config = LLMCallConfig(model="openai/gpt-4o-mini")
        target = OrqResponsesTarget(config, instructions="Reply tersely.")

        r = await target.respond(
            [
                Message(
                    role="user",
                    content=[
                        InputTextContent(
                            type="input_text",
                            text="What color is this image? Reply with just the color name.",
                        ),
                        InputImageContent(type="input_image", image_url=_RED_PIXEL_DATA_URL),
                    ],
                )
            ]
        )
        assert isinstance(r, AgentResponse)
        assert "red" in r.text.lower()

    @pytest.mark.asyncio
    async def test_multipart_second_image_round_trips(self):
        """RES-879: a second, distinct image part also reaches the vision model.

        Uses a different color so a stale/cached/dropped image would fail the
        assertion. Base64 data URL keeps the test free of any live dependency.
        """
        if not os.environ.get("ORQ_API_KEY"):
            pytest.skip("ORQ_API_KEY not set")

        config = LLMCallConfig(model="openai/gpt-4o-mini")
        target = OrqResponsesTarget(config, instructions="Reply tersely.")

        r = await target.respond(
            [
                Message(
                    role="user",
                    content=[
                        InputTextContent(
                            type="input_text",
                            text="What color is this image? Reply with just the color name.",
                        ),
                        InputImageContent(type="input_image", image_url=_GREEN_PIXEL_DATA_URL),
                    ],
                )
            ]
        )
        assert isinstance(r, AgentResponse)
        assert "green" in r.text.lower()
