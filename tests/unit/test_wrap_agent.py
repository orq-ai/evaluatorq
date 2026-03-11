"""Unit tests for wrap_langchain_agent / wrap_langgraph_agent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evaluatorq.types import DataPoint


# ---------------------------------------------------------------------------
# Helpers — mock agent & convert
# ---------------------------------------------------------------------------

def _make_agent() -> MagicMock:
    """Create a mock CompiledStateGraph with invoke() returning empty messages."""
    agent = MagicMock()
    agent.invoke = MagicMock(return_value={"messages": []})
    agent.nodes = {}
    return agent


def _get_invoke_messages(agent: MagicMock) -> list[dict[str, str]]:
    """Extract the messages list from the first agent.invoke() call."""
    call_args = agent.invoke.call_args
    return call_args[0][0]["messages"]


# ---------------------------------------------------------------------------
# Import with mocked dependencies
# ---------------------------------------------------------------------------

# Mock heavy langchain/langgraph deps before importing the module under test
pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from evaluatorq.integrations.langchain_integration.wrap_agent import (  # noqa: E402
    _extract_messages_from_data,  # pyright: ignore[reportPrivateUsage]
    _normalize_message,  # pyright: ignore[reportPrivateUsage]
    wrap_langchain_agent,
    wrap_langgraph_agent,
)


# ---------------------------------------------------------------------------
# _extract_messages_from_data tests
# ---------------------------------------------------------------------------

class TestExtractMessagesFromData:
    def test_returns_messages_when_present(self) -> None:
        data = DataPoint(inputs={"messages": [{"role": "user", "content": "hi"}]})
        assert _extract_messages_from_data(data) == [{"role": "user", "content": "hi"}]

    def test_returns_none_when_missing(self) -> None:
        data = DataPoint(inputs={})
        assert _extract_messages_from_data(data) is None

    def test_returns_none_when_empty(self) -> None:
        data = DataPoint(inputs={"messages": []})
        assert _extract_messages_from_data(data) is None

    def test_returns_none_when_not_a_list(self) -> None:
        data = DataPoint(inputs={"messages": "not-a-list"})
        assert _extract_messages_from_data(data) is None

    def test_normalizes_pydantic_model_messages(self) -> None:
        """Orq SDK Pydantic message objects are converted to plain dicts."""
        mock_msg = MagicMock()
        mock_msg.model_dump = MagicMock(return_value={"role": "system", "content": "hello"})
        data = DataPoint(inputs={"messages": [mock_msg]})
        result = _extract_messages_from_data(data)
        assert result == [{"role": "system", "content": "hello"}]
        mock_msg.model_dump.assert_called_once_with(exclude_none=True)

    def test_normalizes_mixed_messages(self) -> None:
        """Mix of plain dicts and Pydantic models are all normalized."""
        mock_msg = MagicMock()
        mock_msg.model_dump = MagicMock(return_value={"role": "assistant", "content": "hi"})
        data = DataPoint(inputs={"messages": [
            {"role": "user", "content": "hey"},
            mock_msg,
        ]})
        result = _extract_messages_from_data(data)
        assert result == [
            {"role": "user", "content": "hey"},
            {"role": "assistant", "content": "hi"},
        ]


# ---------------------------------------------------------------------------
# _normalize_message tests
# ---------------------------------------------------------------------------

class TestNormalizeMessage:
    def test_dict_passthrough(self) -> None:
        msg = {"role": "user", "content": "hi"}
        assert _normalize_message(msg) == msg

    def test_pydantic_model_dump(self) -> None:
        mock_msg = MagicMock()
        mock_msg.model_dump = MagicMock(return_value={"role": "system", "content": "ctx"})
        assert _normalize_message(mock_msg) == {"role": "system", "content": "ctx"}
        mock_msg.model_dump.assert_called_once_with(exclude_none=True)

    def test_duck_type_fallback(self) -> None:
        """Object with role/content attrs but no model_dump."""
        msg = MagicMock(spec=["role", "content"])
        msg.role = "assistant"
        msg.content = "reply"
        assert _normalize_message(msg) == {"role": "assistant", "content": "reply"}

    def test_duck_type_defaults(self) -> None:
        """Object with no role/content attrs defaults to user/empty."""
        msg = MagicMock(spec=[])
        assert _normalize_message(msg) == {"role": "user", "content": ""}


# ---------------------------------------------------------------------------
# wrap_langchain_agent tests
# ---------------------------------------------------------------------------

class TestWrapLangChainAgent:
    @pytest.mark.asyncio
    async def test_1_prompt_only(self) -> None:
        agent = _make_agent()
        job = wrap_langchain_agent(agent, name="t")
        data = DataPoint(inputs={"prompt": "hi"})

        await job(data, 0)

        msgs = _get_invoke_messages(agent)
        assert msgs == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_2_messages_only(self) -> None:
        agent = _make_agent()
        job = wrap_langchain_agent(agent, name="t")
        data = DataPoint(inputs={"messages": [{"role": "user", "content": "hi"}]})

        await job(data, 0)

        msgs = _get_invoke_messages(agent)
        assert msgs == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_3_messages_and_prompt(self) -> None:
        agent = _make_agent()
        job = wrap_langchain_agent(agent, name="t")
        data = DataPoint(inputs={
            "prompt": "hi",
            "messages": [{"role": "system", "content": "ctx"}],
        })

        await job(data, 0)

        msgs = _get_invoke_messages(agent)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "ctx"}
        assert msgs[1] == {"role": "user", "content": "hi"}

    @pytest.mark.asyncio
    async def test_4_prompt_with_instructions_string(self) -> None:
        agent = _make_agent()
        job = wrap_langchain_agent(agent, name="t", instructions="be nice")
        data = DataPoint(inputs={"prompt": "hi"})

        await job(data, 0)

        msgs = _get_invoke_messages(agent)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "be nice"}
        assert msgs[1] == {"role": "user", "content": "hi"}

    @pytest.mark.asyncio
    async def test_5_messages_with_instructions(self) -> None:
        agent = _make_agent()
        job = wrap_langchain_agent(agent, name="t", instructions="be nice")
        data = DataPoint(inputs={"messages": [{"role": "user", "content": "hi"}]})

        await job(data, 0)

        msgs = _get_invoke_messages(agent)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "be nice"}
        assert msgs[1] == {"role": "user", "content": "hi"}

    @pytest.mark.asyncio
    async def test_6_messages_prompt_and_instructions(self) -> None:
        agent = _make_agent()
        job = wrap_langchain_agent(agent, name="t", instructions="be nice")
        data = DataPoint(inputs={
            "prompt": "hi",
            "messages": [{"role": "user", "content": "q"}],
        })

        await job(data, 0)

        msgs = _get_invoke_messages(agent)
        assert len(msgs) == 3
        assert msgs[0] == {"role": "system", "content": "be nice"}
        assert msgs[1] == {"role": "user", "content": "q"}
        assert msgs[2] == {"role": "user", "content": "hi"}

    @pytest.mark.asyncio
    async def test_7_instructions_as_callable(self) -> None:
        agent = _make_agent()
        job = wrap_langchain_agent(
            agent,
            name="t",
            instructions=lambda data: str(data.inputs.get("city", "")),
        )
        data = DataPoint(inputs={"prompt": "hi", "city": "SF"})

        await job(data, 0)

        msgs = _get_invoke_messages(agent)
        assert msgs[0] == {"role": "system", "content": "SF"}

    @pytest.mark.asyncio
    async def test_8_no_prompt_no_messages_raises(self) -> None:
        agent = _make_agent()
        job = wrap_langchain_agent(agent, name="t")
        data = DataPoint(inputs={})

        with pytest.raises(ValueError, match="neither was provided"):
            await job(data, 0)

    @pytest.mark.asyncio
    async def test_9_empty_messages_falls_through_to_prompt(self) -> None:
        agent = _make_agent()
        job = wrap_langchain_agent(agent, name="t")
        data = DataPoint(inputs={"prompt": "hi", "messages": []})

        await job(data, 0)

        msgs = _get_invoke_messages(agent)
        assert msgs == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_10_messages_not_a_list_falls_through(self) -> None:
        agent = _make_agent()
        job = wrap_langchain_agent(agent, name="t")
        data = DataPoint(inputs={"prompt": "hi", "messages": "not-a-list"})

        await job(data, 0)

        msgs = _get_invoke_messages(agent)
        assert msgs == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_11_custom_prompt_key(self) -> None:
        agent = _make_agent()
        job = wrap_langchain_agent(agent, name="t", prompt_key="question")
        data = DataPoint(inputs={"question": "hi"})

        await job(data, 0)

        msgs = _get_invoke_messages(agent)
        assert msgs == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_8_no_prompt_no_messages_with_instructions_raises(self) -> None:
        """Instructions present but no prompt/messages still raises."""
        agent = _make_agent()
        job = wrap_langchain_agent(agent, name="t", instructions="be nice")
        data = DataPoint(inputs={})

        with pytest.raises(ValueError, match="neither was provided"):
            await job(data, 0)


class TestWrapLangGraphAgentAlias:
    def test_alias_is_same_function(self) -> None:
        assert wrap_langgraph_agent is wrap_langchain_agent
