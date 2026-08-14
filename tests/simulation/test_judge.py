"""Tests for JudgeAgent."""

import json
from unittest.mock import MagicMock

import pytest

from evaluatorq.simulation.agents.base import LLMResult
from evaluatorq.simulation.agents.judge import JUDGE_TOOLS, JudgeAgent, JudgeAgentConfig
from evaluatorq.simulation.types import Criterion


@pytest.fixture
def judge():
    config = JudgeAgentConfig(
        goal="Get a refund",
        criteria=[
            Criterion(description="Agent offers refund", type="must_happen"),
            Criterion(description="Agent is rude", type="must_not_happen"),
        ],
        ground_truth="Refund approved",
        api_key="test-key",
    )
    return JudgeAgent(config)


class TestJudgeAgent:
    def test_name(self, judge):
        assert judge.name == "JudgeAgent"

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
        tool_call = MagicMock()
        tool_call.function.name = "continue_conversation"
        tool_call.function.arguments = json.dumps(
            {
                "reason": "Goal not yet achieved",
                "response_quality": 0.8,
            }
        )

        result = LLMResult(content="", tool_calls=[tool_call])
        judgment = judge._parse_judgment(result)

        assert judgment.should_terminate is False
        assert judgment.reason == "Goal not yet achieved"
        assert judgment.goal_achieved is False
        assert judgment.response_quality == 0.8

    def test_continue_judgment_carries_partial_score(self, judge):
        # A continue at the last turn IS the final score on max_turns runs, so it
        # must reflect the judge's partial estimate, not a hardcoded 0.
        tool_call = MagicMock()
        tool_call.function.name = "continue_conversation"
        tool_call.function.arguments = json.dumps(
            {"reason": "Most of the goal done", "goal_completion_score": 0.75}
        )

        judgment = judge._parse_judgment(LLMResult(content="", tool_calls=[tool_call]))

        assert judgment.goal_completion_score == 0.75

    def test_parse_finish_judgment(self, judge):
        tool_call = MagicMock()
        tool_call.function.name = "finish_conversation"
        tool_call.function.arguments = json.dumps(
            {
                "reason": "Goal achieved",
                "goal_achieved": True,
                "rules_broken": [],
                "goal_completion_score": 1.0,
                "tone_appropriateness": 0.9,
            }
        )

        result = LLMResult(content="", tool_calls=[tool_call])
        judgment = judge._parse_judgment(result)

        assert judgment.should_terminate is True
        assert judgment.goal_achieved is True
        assert judgment.goal_completion_score == 1.0
        assert judgment.tone_appropriateness == 0.9

    def test_parse_judgment_no_tool_calls(self, judge):
        result = LLMResult(content="Some text", tool_calls=None)
        judgment = judge._parse_judgment(result)

        assert judgment.should_terminate is True
        assert "failed to make explicit decision" in judgment.reason

    def test_parse_judgment_invalid_json(self, judge):
        tool_call = MagicMock()
        tool_call.function.name = "finish_conversation"
        tool_call.function.arguments = "not json"

        result = LLMResult(content="", tool_calls=[tool_call])
        judgment = judge._parse_judgment(result)

        assert judgment.should_terminate is True
        assert "Failed to parse" in judgment.reason

    def test_parse_judgment_clamps_score(self, judge):
        tool_call = MagicMock()
        tool_call.function.name = "finish_conversation"
        tool_call.function.arguments = json.dumps(
            {
                "reason": "Done",
                "goal_achieved": True,
                "rules_broken": [],
                "goal_completion_score": 1.5,
                "response_quality": -0.5,
            }
        )

        result = LLMResult(content="", tool_calls=[tool_call])
        judgment = judge._parse_judgment(result)

        assert judgment.goal_completion_score == 1.0
        assert judgment.response_quality == 0.0


class TestJudgeTools:
    def test_has_two_tools(self):
        assert len(JUDGE_TOOLS) == 2

    def test_first_tool_is_continue(self):
        assert JUDGE_TOOLS[0]["function"]["name"] == "continue_conversation"

    def test_second_tool_is_finish(self):
        assert JUDGE_TOOLS[1]["function"]["name"] == "finish_conversation"

    def test_finish_required_fields(self):
        required = JUDGE_TOOLS[1]["function"]["parameters"]["required"]
        assert "goal_achieved" in required
        assert "goal_completion_score" in required
        assert "reason" in required
        assert "criteria_verdicts" in required

    def test_finish_does_not_ask_for_a_free_text_rules_broken(self):
        """Violations are derived in code from the occurrence audit + criterion type.

        Asking the judge for them as well gave the two channels something to
        disagree about, and the free-text one is structurally unable to fail a
        must_happen criterion (RES-1308).
        """
        params = JUDGE_TOOLS[1]["function"]["parameters"]
        assert "rules_broken" not in params["properties"]
        assert "rules_broken" not in params["required"]

    def test_each_tool_describes_its_own_reason_field(self):
        """Field descriptions are prompt text the judge reads, so the two tools must
        not share one generic wording for the decision they just made."""
        continue_props = JUDGE_TOOLS[0]["function"]["parameters"]["properties"]
        finish_props = JUDGE_TOOLS[1]["function"]["parameters"]["properties"]
        assert "continue" in continue_props["reason"]["description"]
        assert "end" in finish_props["reason"]["description"]

    def test_finish_does_not_score_the_goal_SO_FAR(self):
        """"SO FAR ... if the run hits max turns this is the final score" is written
        for a turn that continues; on the call that ends the run there is no so-far."""
        continue_desc = JUDGE_TOOLS[0]["function"]["parameters"]["properties"]["goal_completion_score"]["description"]
        finish_desc = JUDGE_TOOLS[1]["function"]["parameters"]["properties"]["goal_completion_score"]["description"]
        assert "SO FAR" in continue_desc
        assert "SO FAR" not in finish_desc
        assert "partial completion" in finish_desc

    def test_both_tools_have_quality_scores(self):
        quality_fields = {
            "response_quality",
            "hallucination_risk",
            "tone_appropriateness",
            "factual_accuracy",
        }
        for tool in JUDGE_TOOLS:
            props = tool["function"]["parameters"]["properties"]
            assert quality_fields.issubset(props.keys())
