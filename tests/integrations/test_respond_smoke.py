"""Smoke tests: every integration target implements respond -> AgentResponse.

RES-808 PR3 unifies targets on ``respond(messages: list[Message])``. The
abstract-method enforcement only guarantees the method exists; these tests
exercise the body of each integration target's ``respond`` against a mocked
underlying SDK so behavior drift is caught.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.contracts import AgentResponse, AgentTarget, Message


def _msgs(content: str = "hi") -> list[Message]:
    return [Message(role="user", content=content)]


@pytest.mark.asyncio
async def test_callable_target_respond_returns_agent_response():
    from evaluatorq.integrations.callable_integration import CallableTarget

    target = CallableTarget(lambda prompt: f"echo: {prompt}")
    assert isinstance(target, AgentTarget)

    result = await target.respond(_msgs("hello"))
    assert isinstance(result, AgentResponse)
    assert "hello" in result.text


@pytest.mark.asyncio
async def test_callable_target_respond_rejects_non_user_last():
    from evaluatorq.integrations.callable_integration import CallableTarget

    target = CallableTarget(lambda prompt: prompt)
    with pytest.raises(ValueError, match="messages\\[-1\\].role"):
        await target.respond([Message(role="assistant", content="x")])


@pytest.mark.asyncio
async def test_vercel_target_respond_returns_agent_response():
    from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget

    target = VercelAISdkTarget("http://example.local/api/chat")
    assert isinstance(target, AgentTarget)

    response = MagicMock()
    response.headers = {"content-type": "text/html"}
    response.text = "plain reply"
    response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=response)

    with patch(
        "evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient",
        return_value=mock_client,
    ):
        result = await target.respond(_msgs())

    assert isinstance(result, AgentResponse)
    assert result.text == "plain reply"
    # Stateless: no _history accumulation — target is fully stateless.
    assert not hasattr(target, "_history")
    # Full transcript is posted as the body's messages.
    body = mock_client.post.call_args.kwargs["json"]
    assert body["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_langgraph_target_respond_returns_agent_response():
    pytest.importorskip("langgraph")
    from evaluatorq.integrations.langgraph_integration import LangGraphTarget

    msg = MagicMock()
    msg.content = "graph reply"
    graph = MagicMock()
    graph.name = "smoke_graph"
    graph.ainvoke = AsyncMock(return_value={"messages": [msg]})

    target = LangGraphTarget(graph)
    assert isinstance(target, AgentTarget)

    result = await target.respond(_msgs())
    assert isinstance(result, AgentResponse)
    # The latest user turn is forwarded to the graph.
    sent = graph.ainvoke.call_args.args[0]
    assert sent == {"messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.asyncio
async def test_openai_agents_target_respond_returns_agent_response():
    pytest.importorskip("agents")
    from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget

    run_result = MagicMock()
    run_result.final_output = "agent reply"
    run_result.to_input_list.return_value = [{"role": "user", "content": "hi"}]
    run_result.context_wrapper = None

    target = OpenAIAgentTarget(MagicMock())
    assert isinstance(target, AgentTarget)

    with patch(
        "evaluatorq.integrations.openai_agents_integration.target.Runner.run",
        new=AsyncMock(return_value=run_result),
    ):
        result = await target.respond(_msgs())

    assert isinstance(result, AgentResponse)
    assert result.text == "agent reply"
    # Stateless: no _history accumulation — target is fully stateless.
    assert not hasattr(target, "_history")
