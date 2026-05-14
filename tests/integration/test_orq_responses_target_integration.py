"""Real-server integration tests for OrqResponsesTarget.

These tests require ORQ_API_KEY to be set and make live network calls.
They are excluded from the default test run (skipped unless -m integration).
"""

from __future__ import annotations

import os

import pytest

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.target import OrqResponsesTarget


@pytest.mark.integration
class TestOrqResponsesTargetIntegration:
    @pytest.mark.asyncio
    async def test_responses_v3_real_call_threads_previous_response_id(self):
        """Multi-turn conversation threads via previous_response_id.

        Turn 1: tell the model a name.
        Turn 2: ask it to recall the name.
        Verify the model remembers AND that previous_response_id was threaded.
        """
        if not os.environ.get("ORQ_API_KEY"):
            pytest.skip("ORQ_API_KEY not set")

        config = LLMCallConfig(model="openai/gpt-4o-mini")
        target = OrqResponsesTarget(
            config,
            instructions="Reply tersely.",
        )

        # Turn 1: establish context
        r1 = await target.send_prompt("My name is Banana.")
        assert r1.text  # some response came back

        # previous_response_id must be set after the first call
        assert target._previous_response_id is not None
        assert isinstance(target._previous_response_id, str)

        # Turn 2: verify memory threading
        r2 = await target.send_prompt("What is my name?")
        assert "banana" in r2.text.lower()

        # Token usage must be non-zero after two calls
        usage = target.get_usage()
        assert usage.total_tokens > 0
