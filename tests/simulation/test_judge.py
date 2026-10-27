"""Tests for JudgeAgent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.simulation.agents.base import LLMResult
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.agents.judge import JudgeAgent, JudgeDecision
from evaluatorq.simulation.contracts import Criterion, Message


@pytest.fixture
def judge():
    config = LLMCallConfig(client=MagicMock())
    return JudgeAgent(
        config,
        goal="Get a refund",
        criteria=[
            Criterion(description="Agent offers refund", type="must_happen"),
            Criterion(description="Agent is rude", type="must_not_happen"),
        ],
        ground_truth="Refund approved",
    )


class TestJudgeAgent:
    def test_name(self, judge):
        assert judge.name == "JudgeAgent"

    @pytest.mark.asyncio
    async def test_evaluate_sends_no_temperature(self):
        """The judge used to pin ``temperature=0.0``, which reasoning models reject.

        Determinism is now the caller's to ask for via ``LLMCallConfig(temperature=...)``;
        unset must reach the wire as an absent parameter, not as a value.
        """
        client = MagicMock()
        client.responses = MagicMock()
        response = MagicMock()
        response.output = []
        response.usage = None
        client.responses.create = AsyncMock(return_value=response)
        judge = JudgeAgent(LLMCallConfig(client=client), goal="Get a refund")

        await judge.evaluate([Message(role="user", content="hello")])

        call = client.responses.create.await_args
        assert call is not None
        assert "temperature" not in call.kwargs

    def test_system_prompt_contains_goal(self, judge):
        assert "Get a refund" in judge.system_prompt

    def test_system_prompt_contains_criteria(self, judge):
        prompt = judge.system_prompt
        assert "MUST HAPPEN" in prompt
        assert "Agent offers refund" in prompt
        assert "MUST NOT HAPPEN" in prompt
        assert "Agent is rude" in prompt

    def test_system_prompt_contains_ground_truth(self, judge):
        assert "GROUND TRUTH" in judge.system_prompt
        assert "Refund approved" in judge.system_prompt

    def test_parse_continue_judgment(self, judge):
        result = LLMResult(
            content=json.dumps({"reason": "Goal not yet achieved", "should_terminate": False, "response_quality": 0.8})
        )
        judgment = judge._parse_judgment(result)

        assert judgment.should_terminate is False
        assert judgment.reason == "Goal not yet achieved"
        assert judgment.goal_achieved is False
        assert judgment.response_quality == 0.8

    def test_continue_judgment_carries_partial_score(self, judge):
        # A continue at the last turn IS the final score on max_turns runs, so it
        # must reflect the judge's partial estimate, not a hardcoded 0.
        content = json.dumps({"reason": "Most of the goal done", "goal_completion_score": 0.75})

        judgment = judge._parse_judgment(LLMResult(content=content))

        assert judgment.goal_completion_score == 0.75

    def test_a_continuing_conversation_has_not_achieved_its_goal(self, judge):
        content = json.dumps({"should_terminate": False, "goal_achieved": True, "goal_completion_score": 0.9})

        judgment = judge._parse_judgment(LLMResult(content=content))

        assert judgment.should_terminate is False
        assert judgment.goal_achieved is False

    def test_parse_finish_judgment(self, judge):
        content = json.dumps(
            {
                "reason": "Goal achieved",
                "should_terminate": True,
                "goal_achieved": True,
                "goal_completion_score": 1.0,
                "tone_appropriateness": 0.9,
            }
        )
        judgment = judge._parse_judgment(LLMResult(content=content))

        assert judgment.should_terminate is True
        assert judgment.goal_achieved is True
        assert judgment.goal_completion_score == 1.0
        assert judgment.tone_appropriateness == 0.9

    def test_a_fenced_reply_is_still_parsed(self, judge):
        content = '```json\n{"reason": "ok", "should_terminate": true, "goal_achieved": true}\n```'

        judgment = judge._parse_judgment(LLMResult(content=content))

        assert judgment.should_terminate is True
        assert judgment.reason == "ok"

    def test_parse_judgment_not_json(self, judge):
        judgment = judge._parse_judgment(LLMResult(content="Some text"))

        assert judgment.should_terminate is True
        assert "Failed to parse" in judgment.reason

    def test_parse_judgment_invalid_json(self, judge):
        judgment = judge._parse_judgment(LLMResult(content="{not json"))

        assert judgment.should_terminate is True
        assert "Failed to parse" in judgment.reason

    def test_parse_judgment_clamps_score(self, judge):
        content = json.dumps(
            {
                "reason": "Done",
                "should_terminate": True,
                "goal_achieved": True,
                "goal_completion_score": 1.5,
                "response_quality": -0.5,
            }
        )
        judgment = judge._parse_judgment(LLMResult(content=content))

        assert judgment.goal_completion_score == 1.0
        assert judgment.response_quality == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_asks_for_the_decision_as_structured_output(self):
        """The judge asks for a schema-shaped reply instead of offering tools."""
        client = MagicMock()
        client.responses = MagicMock()
        response = MagicMock()
        response.output = []
        response.usage = None
        client.responses.create = AsyncMock(return_value=response)
        judge = JudgeAgent(LLMCallConfig(client=client), goal="Get a refund")

        await judge.evaluate([Message(role="user", content="hello")])

        call = client.responses.create.await_args
        assert call is not None
        sent = call.kwargs
        assert "tools" not in sent
        assert sent["text"]["format"]["name"] == "JudgeDecision"
        assert sent["text"]["format"]["strict"] is True


class TestJudgeDecisionSchema:
    @pytest.fixture
    def schema(self):
        from evaluatorq.common.responses import responses_text_config

        return responses_text_config(JudgeDecision)["format"]["schema"]

    def test_every_field_is_required_on_the_wire(self, schema):
        assert set(schema["required"]) == set(JudgeDecision.model_fields)

    def test_does_not_ask_for_a_free_text_rules_broken(self, schema):
        """Violations are derived in code from the occurrence audit + criterion type.

        Asking the judge for them as well gave the two channels something to
        disagree about, and the free-text one is structurally unable to fail a
        must_happen criterion (RES-1308).
        """
        assert "rules_broken" not in schema["properties"]

    def test_decides_to_stop_with_one_flag(self, schema):
        assert schema["properties"]["should_terminate"]["type"] == "boolean"
        assert schema["properties"]["goal_achieved"]["type"] == "boolean"

    def test_quality_scores_may_be_null(self, schema):
        """``null`` is a real answer for these ("no ground truth, not scored"); forcing a
        number would make the model invent one."""
        for name in ("response_quality", "hallucination_risk", "tone_appropriateness", "factual_accuracy"):
            types = [branch["type"] for branch in schema["properties"][name]["anyOf"]]
            assert types == ["number", "null"], name

    def test_maintainer_docstrings_stay_out_of_the_prompt(self, schema):
        assert "description" not in schema
        assert "description" not in schema["$defs"]["CriterionVerdict"]
        assert schema["properties"]["reason"]["description"]
