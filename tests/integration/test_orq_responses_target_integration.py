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
from evaluatorq.openresponses.target import OrqResponsesTarget


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
