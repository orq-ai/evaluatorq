"""Unit tests for tool call interception — AgentResponse, ExecutedToolCall, coercion, and evaluator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.redteam.backends.base import _coerce_to_agent_response
from evaluatorq.redteam.contracts import AgentResponse, ExecutedToolCall, OrchestratorResult, Message
from evaluatorq.redteam.adaptive.evaluator import _sanitize_placeholders, _serialize_messages


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
        tc = ExecutedToolCall(name="foo", arguments={"x": 1})
        original = AgentResponse(text="hi", tool_calls=[tc])
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


# ---------------------------------------------------------------------------
# ExecutedToolCall
# ---------------------------------------------------------------------------

class TestExecutedToolCall:
    def test_default_result_is_none(self) -> None:
        tc = ExecutedToolCall(name="my_tool", arguments={"key": "value"})
        assert tc.result is None

    def test_with_result(self) -> None:
        tc = ExecutedToolCall(name="my_tool", arguments={}, result="tool output")
        assert tc.result == "tool output"

    def test_empty_arguments(self) -> None:
        tc = ExecutedToolCall(name="no_args", arguments={})
        assert tc.arguments == {}


# ---------------------------------------------------------------------------
# OrchestratorResult tool_calls_per_turn
# ---------------------------------------------------------------------------

class TestOrchestratorResultToolCalls:
    def test_default_tool_calls_per_turn_is_empty(self) -> None:
        result = OrchestratorResult(
            conversation=[],
            turns=0,
        )
        assert result.tool_calls_per_turn == []

    def test_tool_calls_per_turn_stored(self) -> None:
        tc = ExecutedToolCall(name="delete_db", arguments={"db": "prod"})
        result = OrchestratorResult(
            conversation=[
                Message(role="user", content="do it"),
                Message(role="assistant", content="done"),
            ],
            turns=1,
            tool_calls_per_turn=[[tc]],
        )
        assert len(result.tool_calls_per_turn) == 1
        assert result.tool_calls_per_turn[0][0].name == "delete_db"

    def test_tool_calls_per_turn_empty_turn(self) -> None:
        result = OrchestratorResult(
            conversation=[
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
            ],
            turns=1,
            tool_calls_per_turn=[[]],
        )
        assert result.tool_calls_per_turn == [[]]


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
        tc = ExecutedToolCall(name="{{output.response}}", arguments={})
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
        evaluator = OWASPEvaluator.__new__(OWASPEvaluator)
        evaluator.evaluator_model = "test-model"
        evaluator.llm_kwargs = {}

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
        evaluator = OWASPEvaluator.__new__(OWASPEvaluator)
        evaluator.evaluator_model = "test-model"
        evaluator.llm_kwargs = {}

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

        tc = ExecutedToolCall(name="send_email", arguments={"to": "attacker@evil.com"}, result=None)

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
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        ai_msg = MagicMock()
        ai_msg.content = "I'll search for that"
        ai_msg.tool_calls = [{"name": "web_search", "args": {"query": "hello"}}]

        final_msg = MagicMock()
        final_msg.content = "Here is the result"
        final_msg.tool_calls = []

        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"messages": [ai_msg, final_msg]})

        target = LangGraphTarget(graph)
        result = await target.send_prompt("search for something")

        assert result.text == "Here is the result"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "web_search"
        assert result.tool_calls[0].arguments == {"query": "hello"}

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
        assert result.tool_calls[0].arguments == {"raw": "not-valid-json"}


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
        cloned = target.clone()
        assert cloned._prev_msg_count == 0


# ---------------------------------------------------------------------------
# End-to-end sanitization through _run_evaluator
# ---------------------------------------------------------------------------

class TestRunEvaluatorSanitizationEndToEnd:
    @pytest.mark.asyncio
    async def test_adversarial_message_does_not_expand_output_response(self) -> None:
        """A message containing '{{output.response}}' must not expand in the rendered prompt."""
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator

        evaluator = OWASPEvaluator.__new__(OWASPEvaluator)
        evaluator.evaluator_model = "test-model"
        evaluator.llm_kwargs = {}

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
