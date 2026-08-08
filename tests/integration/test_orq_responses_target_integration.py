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

from evaluatorq.contracts import AgentResponse, FunctionCall, LLMCallConfig, Message, StrategyToolCall
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

    @pytest.mark.asyncio
    async def test_tool_turn_replay_reaches_the_model(self):
        """RES-1231: a transcript containing a tool turn survives replay.

        This is the exact shape that produced the original
        ``400 Invalid value: 'tool'`` — the transcript was serialized in
        chat-completions form (``role: "tool"`` plus a message-level
        ``tool_calls`` key) and posted as Responses ``input``.

        Asserts the model repeats the fact carried by the tool result, not
        merely that the call returned 200: the second defect was that assistant
        tool calls were silently ignored, which a status check would not catch.
        """
        if not os.environ.get("ORQ_API_KEY"):
            pytest.skip("ORQ_API_KEY not set")

        config = LLMCallConfig(model="openai/gpt-4o-mini")
        target = OrqResponsesTarget(config, instructions="Answer using the tool result. Reply tersely.")

        r = await target.respond(
            [
                Message(role="user", content="What is the price of item X? Use the tool."),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        StrategyToolCall(
                            id="call_price_1",
                            type="function",
                            function=FunctionCall(name="get_price", arguments='{"item": "X"}'),
                        )
                    ],
                ),
                Message(role="tool", tool_call_id="call_price_1", content="1234 euro"),
            ]
        )
        assert isinstance(r, AgentResponse)
        assert "1234" in r.text

    @pytest.mark.asyncio
    async def test_tool_result_multipart_output_reaches_the_model(self):
        """``function_call_output.output`` also accepts a content-parts list.

        Flattening those parts to text would silently drop an image or file a
        tool returned, so the converter passes them through. Verifies the API
        accepts that form and the model reads it.
        """
        if not os.environ.get("ORQ_API_KEY"):
            pytest.skip("ORQ_API_KEY not set")

        config = LLMCallConfig(model="openai/gpt-4o-mini")
        target = OrqResponsesTarget(config, instructions="Answer using the tool result. Reply tersely.")

        r = await target.respond(
            [
                Message(role="user", content="What is the price of item X? Use the tool."),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        StrategyToolCall(
                            id="call_price_2",
                            type="function",
                            function=FunctionCall(name="get_price", arguments='{"item": "X"}'),
                        )
                    ],
                ),
                Message(
                    role="tool",
                    tool_call_id="call_price_2",
                    content=[InputTextContent(type="input_text", text="4321 euro")],
                ),
            ]
        )
        assert isinstance(r, AgentResponse)
        assert "4321" in r.text
