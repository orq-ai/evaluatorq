"""Unit tests for tool call interception — AgentResponse, ToolCallOutputItem, coercion, and evaluator."""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.redteam.backends.base import _coerce_to_agent_response
from evaluatorq.redteam.contracts import (
    AgentResponse,
    AttackerResponse,
    Message,
    OrchestratorResult,
    OutputMessage,
    TextOutputItem,
    ToolCallOutputItem,
    Turn,
)
from evaluatorq.redteam.adaptive.evaluator import _sanitize_placeholders


# ---------------------------------------------------------------------------
# _coerce_to_agent_response
# ---------------------------------------------------------------------------

class TestCoerceToAgentResponse:
    def test_str_input_wraps_to_agent_response(self) -> None:
        result = _coerce_to_agent_response("hello")
        assert isinstance(result, AgentResponse)
        assert result.text == "hello"
        assert result.tool_calls == []

    def test_agent_response_input_returned_unchanged(self) -> None:
        orig_out: list[OutputMessage] = []
        orig_out.append(ToolCallOutputItem(name="foo", arguments=json.dumps({"x": 1})))
        orig_out.append(TextOutputItem(text="hi", annotations=[]))
        original = AgentResponse(output=orig_out)
        result = _coerce_to_agent_response(original)
        assert result is original

    def test_none_input_produces_empty_text(self) -> None:
        result = _coerce_to_agent_response(None)
        assert isinstance(result, AgentResponse)
        assert result.text == ""
        assert result.tool_calls == []

    def test_empty_string_input(self) -> None:
        result = _coerce_to_agent_response("")
        assert result.text == ""

    def test_legacy_text_constructor(self) -> None:
        result = AgentResponse(text="legacy")
        assert result.text == "legacy"
        assert result.output == [TextOutputItem(text="legacy", annotations=[])]


class TestAgentResponseTextSemantics:
    def test_text_concatenates_all_text_items(self) -> None:
        """Multi-part Responses replies must not drop leading text segments."""
        out: list[OutputMessage] = []
        out.append(TextOutputItem(text="draft", annotations=[]))
        out.append(ToolCallOutputItem(name="lookup", arguments='{"q": "value"}'))
        out.append(TextOutputItem(text="final", annotations=[]))
        result = AgentResponse(output=out)

        assert result.text == "draftfinal"

    def test_tool_call_only_response_has_empty_text(self) -> None:
        out2: list[OutputMessage] = []
        out2.append(ToolCallOutputItem(name="lookup", id="call_1", arguments='{"q": "value"}'))
        result = AgentResponse(output=out2)

        assert result.text == ""
        assert len(result.tool_calls) == 1
        tc_item = result.output[0]
        assert isinstance(tc_item, ToolCallOutputItem)
        assert tc_item.id == "call_1"
        # arguments stored as JSON string internally; .arguments_dict parses back
        assert result.tool_calls[0].arguments == '{"q": "value"}'
        assert result.tool_calls[0].arguments_dict == {"q": "value"}

    def test_rejects_non_output_message_items(self) -> None:
        with pytest.raises(ValueError):
            AgentResponse(output=cast(Any, ["not an output item"]))

    def test_rejects_wrong_output_container_type(self) -> None:
        with pytest.raises(ValueError):
            AgentResponse(output=cast(Any, "not a list"))

    def test_output_items_validate_type_discriminator(self) -> None:
        # Pydantic Literal validation raises ValidationError (subtype of ValueError)
        with pytest.raises(ValueError, match="output_text"):
            TextOutputItem(text="hi", annotations=[], type=cast(Any, "wrong"))

        with pytest.raises(ValueError, match="function_call"):
            ToolCallOutputItem(name="tool", arguments="", type=cast(Any, "wrong"))

    def test_tool_call_arguments_accepts_dict_and_string(self) -> None:
        # dict is serialized to JSON string internally
        item_from_dict = ToolCallOutputItem(name="tool", arguments=cast(Any, {"x": 1}))
        assert item_from_dict.arguments == '{"x": 1}'

        # JSON string is accepted as-is
        item_from_str = ToolCallOutputItem(name="tool", arguments='{"x": 1}')
        assert item_from_str.arguments == '{"x": 1}'

    def test_output_items_are_frozen(self) -> None:
        item = TextOutputItem(text="hi", annotations=[])
        with pytest.raises(Exception):
            item.text = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ToolCallOutputItem
# ---------------------------------------------------------------------------

class TestToolCallOutputItem:
    def test_default_result_is_none(self) -> None:
        tc = ToolCallOutputItem(name="my_tool", arguments={"key": "value"})  # pyright: ignore[reportArgumentType]
        assert tc.result is None
        assert tc.arguments_dict == {"key": "value"}

    def test_with_result(self) -> None:
        tc = ToolCallOutputItem(name="my_tool", arguments={}, result="tool output")  # pyright: ignore[reportArgumentType]
        assert tc.result == "tool output"

    def test_empty_arguments(self) -> None:
        tc = ToolCallOutputItem(name="no_args", arguments={})  # pyright: ignore[reportArgumentType]
        assert tc.arguments == "{}"
        assert tc.arguments_dict == {}

    def test_arguments_dict_handles_bad_json(self) -> None:
        tc = ToolCallOutputItem(name="bad", arguments="not-json")
        assert tc.arguments_dict == {}

    def test_id_is_autogenerated_when_missing(self) -> None:
        tc = ToolCallOutputItem(name="x", arguments={})  # pyright: ignore[reportArgumentType]
        assert tc.id.startswith("fc_")


# ---------------------------------------------------------------------------
# OrchestratorResult per-turn tool calls (via turns: list[Turn])
# ---------------------------------------------------------------------------

class TestOrchestratorResultToolCalls:
    def test_default_turns_is_empty(self) -> None:
        result = OrchestratorResult()
        assert result.turns == []
        assert result.n_turns == 0
        assert result.final_response == ''
        assert result.chat_completions == []

    def test_tool_calls_stored_on_turn_target(self) -> None:
        tc = ToolCallOutputItem(name="delete_db", arguments={"db": "prod"})  # pyright: ignore[reportArgumentType]
        result = OrchestratorResult(
            turns=[
                Turn(
                    attacker=AttackerResponse(generated_prompt="do it"),
                    target=AgentResponse(output=[
                        TextOutputItem(text="done", annotations=[]),
                        tc,
                    ]),
                )
            ],
        )
        assert result.n_turns == 1
        assert result.turns[0].target.tool_calls[0].name == "delete_db"
        assert result.final_response == "done"

    def test_turn_target_with_no_tool_calls(self) -> None:
        result = OrchestratorResult(
            turns=[
                Turn(
                    attacker=AttackerResponse(generated_prompt="hi"),
                    target=AgentResponse(text="hello"),
                )
            ],
        )
        assert result.turns[0].target.tool_calls == []
        assert [t.target.tool_calls for t in result.turns] == [[]]


# ---------------------------------------------------------------------------
# _sanitize_placeholders
# ---------------------------------------------------------------------------

class TestSanitizePlaceholders:
    def test_neutralizes_template_placeholder(self) -> None:
        text = '{"name": "{{output.response}}", "args": {}}'
        sanitized = _sanitize_placeholders(text)
        assert "{{output.response}}" not in sanitized
        assert "{ {output.response}}" in sanitized

    def test_neutralizes_all_messages_placeholder(self) -> None:
        text = '{"note": "{{input.all_messages}}"}'
        sanitized = _sanitize_placeholders(text)
        assert "{{input.all_messages}}" not in sanitized

    def test_clean_text_unchanged(self) -> None:
        text = '{"name": "safe_tool", "arguments": {"key": "value"}}'
        assert _sanitize_placeholders(text) == text

    def test_tool_calls_json_is_sanitized_before_injection(self) -> None:
        """Verify adversarial tool call name cannot cause cross-expansion in evaluator prompt."""
        tc = ToolCallOutputItem(name="{{output.response}}", arguments={})  # pyright: ignore[reportArgumentType]
        serialized = json.dumps(
            [{'name': tc.name, 'arguments': tc.arguments, 'result': tc.result}],
            indent=2,
        )
        sanitized = _sanitize_placeholders(serialized)
        # After sanitization, the remaining .replace('{{output.response}}', ...) call
        # will NOT match because the {{ has been broken
        assert "{{output.response}}" not in sanitized


# ---------------------------------------------------------------------------
# Evaluator {{output.tool_calls}} substitution
# ---------------------------------------------------------------------------

class TestEvaluatorToolCallsSubstitution:
    @pytest.mark.asyncio
    async def test_empty_tool_calls_renders_empty_json_array(self) -> None:
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator
        from evaluatorq.redteam.contracts import LLMCallConfig
        evaluator = OWASPEvaluator.__new__(OWASPEvaluator)
        evaluator.evaluator_model = "test-model"
        evaluator.llm_kwargs = {}
        evaluator._cfg = LLMCallConfig()

        captured_prompt: list[str] = []

        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"value": true, "explanation": "ok"}'
        mock_response.usage = None

        async def fake_create(**kwargs):
            captured_prompt.append(kwargs["messages"][1]["content"])
            return mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        evaluator.client = mock_client

        mock_evaluator = MagicMock()
        mock_evaluator.prompt = "Tools: {{output.tool_calls}} Response: {{output.response}}"

        with patch("evaluatorq.redteam.adaptive.evaluator.with_llm_span") as mock_span:
            mock_span.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_span.return_value.__aexit__ = AsyncMock(return_value=False)
            await evaluator._run_evaluator(
                evaluator=mock_evaluator,
                evaluator_id="test",
                messages=[],
                response="agent said this",
                span_attributes={},
                tool_calls=[],
            )

        rendered = captured_prompt[0]
        assert "[]" in rendered  # empty tool calls renders as []
        assert "agent said this" in rendered

    @pytest.mark.asyncio
    async def test_tool_calls_with_result_none_renders_null(self) -> None:
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator
        from evaluatorq.redteam.contracts import LLMCallConfig
        evaluator = OWASPEvaluator.__new__(OWASPEvaluator)
        evaluator.evaluator_model = "test-model"
        evaluator.llm_kwargs = {}
        evaluator._cfg = LLMCallConfig()

        captured_prompt: list[str] = []

        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"value": false, "explanation": "vuln"}'
        mock_response.usage = None

        async def fake_create(**kwargs):
            captured_prompt.append(kwargs["messages"][1]["content"])
            return mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        evaluator.client = mock_client

        mock_evaluator = MagicMock()
        mock_evaluator.prompt = "{{output.tool_calls}}"

        tc = ToolCallOutputItem(name="send_email", arguments={"to": "attacker@evil.com"}, result=None)  # pyright: ignore[reportArgumentType]

        with patch("evaluatorq.redteam.adaptive.evaluator.with_llm_span") as mock_span:
            mock_span.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_span.return_value.__aexit__ = AsyncMock(return_value=False)
            await evaluator._run_evaluator(
                evaluator=mock_evaluator,
                evaluator_id="test",
                messages=[],
                response="",
                span_attributes={},
                tool_calls=[tc],
            )

        rendered = captured_prompt[0]
        parsed = json.loads(rendered)
        assert parsed[0]["name"] == "send_email"
        assert parsed[0]["result"] is None
        assert parsed[0]["arguments"]["to"] == "attacker@evil.com"


# ---------------------------------------------------------------------------
# LangGraph tool call extraction
# ---------------------------------------------------------------------------

class TestLangGraphToolCallExtraction:
    @pytest.mark.asyncio
    async def test_extracts_tool_calls_from_ai_messages(self) -> None:
        pytest.importorskip("langgraph")
        from langchain_core.messages import AIMessage
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        ai_msg = AIMessage(
            content="I'll search for that",
            tool_calls=[{"name": "web_search", "args": {"query": "hello"}, "id": "call_1", "type": "tool_call"}],
        )

        final_msg = AIMessage(content="Here is the result")

        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"messages": [ai_msg, final_msg]})

        target = LangGraphTarget(graph)
        result = await target.send_prompt("search for something")

        assert result.text == "I'll search for thatHere is the result"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "web_search"
        assert result.tool_calls[0].arguments_dict == {"query": "hello"}

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_empty_list(self) -> None:
        pytest.importorskip("langgraph")
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        msg = MagicMock()
        msg.content = "plain response"
        msg.tool_calls = []

        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"messages": [msg]})

        target = LangGraphTarget(graph)
        result = await target.send_prompt("hi")
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_extracts_tool_calls_from_dict_messages(self) -> None:
        pytest.importorskip("langgraph")
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"messages": [
            {"role": "assistant", "content": "calling tool", "tool_calls": [
                {"name": "get_weather", "args": {"city": "Amsterdam"}}
            ]},
            {"role": "assistant", "content": "done", "tool_calls": []},
        ]})

        target = LangGraphTarget(graph)
        result = await target.send_prompt("weather?")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"


# ---------------------------------------------------------------------------
# OpenAI Agents tool call extraction (current turn only)
# ---------------------------------------------------------------------------

class TestOpenAIAgentsToolCallExtraction:
    @pytest.mark.asyncio
    async def test_extracts_tool_calls_from_current_turn_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("agents")
        from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget

        turn1_history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "tool_a", "arguments": '{"x": 1}'}, "type": "function", "id": "1"}
            ]},
            {"role": "assistant", "content": "done first"},
        ]
        turn2_history = [
            *turn1_history,
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "tool_b", "arguments": '{"y": 2}'}, "type": "function", "id": "2"}
            ]},
            {"role": "assistant", "content": "done second"},
        ]

        call_count = 0

        async def fake_run(agent, input_data, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.final_output = f"reply {call_count}"
            result.to_input_list.return_value = turn1_history if call_count == 1 else turn2_history
            return result

        runner = MagicMock()
        runner.run = fake_run
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        r1 = await target.send_prompt("first")
        # Turn 1: tool_a only
        assert len(r1.tool_calls) == 1
        assert r1.tool_calls[0].name == "tool_a"

        r2 = await target.send_prompt("second")
        # Turn 2: tool_b only (not tool_a again)
        assert len(r2.tool_calls) == 1
        assert r2.tool_calls[0].name == "tool_b"

    @pytest.mark.asyncio
    async def test_malformed_arguments_json_falls_back_to_raw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("agents")
        from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget

        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "bad_tool", "arguments": "not-valid-json"}, "type": "function", "id": "1"}
            ]},
        ]

        async def fake_run(agent, input_data, **kwargs):
            result = MagicMock()
            result.final_output = "ok"
            result.to_input_list.return_value = history
            return result

        runner = MagicMock()
        runner.run = fake_run
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        result = await target.send_prompt("hi")
        assert result.tool_calls[0].arguments_dict == {"raw": "not-valid-json"}


# ---------------------------------------------------------------------------
# LangGraph clone _prev_msg_count
# ---------------------------------------------------------------------------

class TestLangGraphClonePrevMsgCount:
    def test_clone_has_zero_prev_msg_count(self) -> None:
        pytest.importorskip("langgraph")
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        graph = MagicMock()
        graph.name = "test"
        graph.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="hi", tool_calls=[])]})

        target = LangGraphTarget(graph)
        target._prev_msg_count = 5  # simulate post-send state
        cloned = target.new()
        assert cloned._prev_msg_count == 0


# ---------------------------------------------------------------------------
# End-to-end sanitization through _run_evaluator
# ---------------------------------------------------------------------------

class TestRunEvaluatorSanitizationEndToEnd:
    @pytest.mark.asyncio
    async def test_adversarial_message_does_not_expand_output_response(self) -> None:
        """A message containing '{{output.response}}' must not expand in the rendered prompt."""
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator

        from evaluatorq.redteam.contracts import LLMCallConfig
        evaluator = OWASPEvaluator.__new__(OWASPEvaluator)
        evaluator.evaluator_model = "test-model"
        evaluator.llm_kwargs = {}
        evaluator._cfg = LLMCallConfig()

        captured_prompt: list[str] = []

        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"value": true, "explanation": "ok"}'
        mock_response.usage = None

        async def fake_create(**kwargs):
            captured_prompt.append(kwargs["messages"][1]["content"])
            return mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        evaluator.client = mock_client

        mock_evaluator = MagicMock()
        mock_evaluator.prompt = "Messages: {{input.all_messages}} Response: {{output.response}}"

        with patch("evaluatorq.redteam.adaptive.evaluator.with_llm_span") as mock_span:
            mock_span.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_span.return_value.__aexit__ = AsyncMock(return_value=False)
            await evaluator._run_evaluator(
                evaluator=mock_evaluator,
                evaluator_id="test",
                messages=[{"role": "user", "content": "{{output.response}}"}],
                response="REAL_RESPONSE",
                span_attributes={},
                tool_calls=[],
            )

        rendered = captured_prompt[0]
        # The adversarial message should have its placeholder neutralised
        assert "REAL_RESPONSE" not in rendered.split("Response:")[0]
        # The actual response value appears exactly once, in the response section
        assert rendered.count("REAL_RESPONSE") == 1
        assert rendered.endswith("REAL_RESPONSE")


# ---------------------------------------------------------------------------
# create_dynamic_evaluator scorer — tool_calls threading
# ---------------------------------------------------------------------------

def _make_fake_eval_result() -> MagicMock:
    result = MagicMock()
    result.passed = True
    result.explanation = "ok"
    result.token_usage = None
    result.evaluator_id = "test"
    result.raw_output = {}
    return result


class TestCreateDynamicEvaluatorScorer:
    @pytest.mark.asyncio
    async def test_scorer_flattens_tool_calls_across_turns_and_passes_to_evaluator(self) -> None:
        """Multi-turn tool calls are flattened and forwarded to the evaluator."""
        from evaluatorq.redteam.adaptive.pipeline import create_dynamic_evaluator
        from evaluatorq.redteam.contracts import AttackOutput

        tc_a = ToolCallOutputItem(name="tool_a", arguments={})  # pyright: ignore[reportArgumentType]
        tc_b = ToolCallOutputItem(name="tool_b", arguments={})  # pyright: ignore[reportArgumentType]

        attack_output = AttackOutput(
            turns=[
                Turn(
                    attacker=AttackerResponse(generated_prompt="hi"),
                    target=AgentResponse(output=[TextOutputItem(text="ok", annotations=[]), tc_a]),
                ),
                Turn(
                    attacker=AttackerResponse(generated_prompt="again"),
                    target=AgentResponse(output=[TextOutputItem(text="done", annotations=[]), tc_b]),
                ),
            ],
            category="ASI05",
            vulnerability="code_execution",
        )

        captured: dict[str, Any] = {}

        async def fake_eval(*args, **kwargs):
            captured["tool_calls"] = kwargs.get("tool_calls")
            return _make_fake_eval_result()

        with patch("evaluatorq.redteam.adaptive.pipeline.OWASPEvaluator") as MockEvaluatorClass:
            MockEvaluatorClass.return_value.evaluate_vulnerability = AsyncMock(side_effect=fake_eval)
            MockEvaluatorClass.return_value.evaluate = AsyncMock(side_effect=fake_eval)
            scorer = create_dynamic_evaluator()["scorer"]
            await scorer({"data": MagicMock(inputs={"category": "ASI05", "vulnerability": ""}), "output": attack_output})

        assert captured["tool_calls"] is not None
        assert len(captured["tool_calls"]) == 2
        assert captured["tool_calls"][0].name == "tool_a"
        assert captured["tool_calls"][1].name == "tool_b"

    @pytest.mark.asyncio
    async def test_scorer_passes_none_when_no_tool_calls(self) -> None:
        """Turns with no tool calls in target output result in `tool_calls=None` passed to the evaluator."""
        from evaluatorq.redteam.adaptive.pipeline import create_dynamic_evaluator
        from evaluatorq.redteam.contracts import AttackOutput

        attack_output = AttackOutput(
            turns=[
                Turn(
                    attacker=AttackerResponse(generated_prompt="hi"),
                    target=AgentResponse(text="nope"),
                ),
            ],
            category="ASI05",
            vulnerability="",
        )

        captured: dict[str, Any] = {}

        async def fake_eval(*args, **kwargs):
            captured["tool_calls"] = kwargs.get("tool_calls")
            return _make_fake_eval_result()

        with patch("evaluatorq.redteam.adaptive.pipeline.OWASPEvaluator") as MockEvaluatorClass:
            MockEvaluatorClass.return_value.evaluate_vulnerability = AsyncMock(side_effect=fake_eval)
            MockEvaluatorClass.return_value.evaluate = AsyncMock(side_effect=fake_eval)
            scorer = create_dynamic_evaluator()["scorer"]
            await scorer({"data": MagicMock(inputs={"category": "ASI05", "vulnerability": ""}), "output": attack_output})

        # Empty turns produce empty flat list → coerced to None by `or None`
        assert captured["tool_calls"] is None

    @pytest.mark.asyncio
    async def test_parse_failure_log_includes_datapoint_context(self) -> None:
        from evaluatorq.redteam.adaptive.pipeline import create_dynamic_evaluator

        data = MagicMock(inputs={
            "id": "dp_123",
            "category": "ASI05",
            "strategy_name": "tool_output_hijack",
        })

        with (
            patch("evaluatorq.redteam.adaptive.pipeline.OWASPEvaluator"),
            patch("evaluatorq.redteam.adaptive.pipeline.logger.error") as mock_error,
        ):
            scorer = create_dynamic_evaluator()["scorer"]
            result = await scorer({"data": data, "output": {"turns": "invalid"}})

        assert result.value == "error"
        log_message = mock_error.call_args.args[0]
        assert "datapoint_id='dp_123'" in log_message
        assert "category='ASI05'" in log_message
        assert "strategy_name='tool_output_hijack'" in log_message
