"""Unit tests for LangGraph red teaming target."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

pytest.importorskip('langgraph')

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from evaluatorq.contracts import Message  # noqa: E402
from evaluatorq.integrations.langgraph_integration import LangGraphTarget  # noqa: E402
from evaluatorq.redteam.contracts import AgentResponse  # noqa: E402


def _make_graph(response_content: str = "I'm fine") -> MagicMock:
    graph = MagicMock()
    graph.name = 'test_graph'
    msg = MagicMock()
    msg.content = response_content
    graph.ainvoke = AsyncMock(return_value={'messages': [msg]})
    return graph


def _graph_returning(*messages: Any) -> MagicMock:
    """A graph whose single ainvoke returns exactly these state messages."""
    graph = MagicMock()
    graph.name = 'test'
    graph.ainvoke = AsyncMock(return_value={'messages': list(messages)})
    return graph


def _tool_turn(name: str = 'order_status', call_id: str | None = 'call_1', **kwargs: Any) -> AIMessage:
    """An assistant turn carrying exactly one tool call."""
    return AIMessage(content='', tool_calls=[{'name': name, 'args': {}, 'id': call_id, 'type': 'tool_call', **kwargs}])


class TestLangGraphTarget:
    @pytest.mark.asyncio
    async def test_structured_fallback_content_is_json_not_repr(self) -> None:
        class Answer(BaseModel):
            field: str

        graph = MagicMock()
        graph.name = 'test_graph'
        graph.ainvoke = AsyncMock(return_value={'messages': [MagicMock(content=Answer(field='x'))]})

        result = await LangGraphTarget(graph).respond([Message(role='user', content='hi')])

        assert result.text == '{"field":"x"}'
        assert 'Answer(' not in result.text

    @pytest.mark.asyncio
    async def test_respond_returns_response(self) -> None:
        graph = _make_graph('hello back')
        target = LangGraphTarget(graph)
        result = await target.respond([Message(role='user', content='hello')])
        assert isinstance(result, AgentResponse)
        assert result.text == 'hello back'

    @pytest.mark.asyncio
    async def test_respond_passes_user_message(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        await target.respond([Message(role='user', content='test prompt')])

        call_args = graph.ainvoke.call_args
        messages = call_args[0][0]['messages']
        assert messages == [{'role': 'user', 'content': 'test prompt'}]

    @pytest.mark.asyncio
    async def test_respond_passes_memory_entity_id(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        await target.respond([Message(role='user', content='hi')])

        config = graph.ainvoke.call_args[1]['config']
        assert 'thread_id' in config['configurable']

    @pytest.mark.asyncio
    async def test_reset_generates_new_memory_entity_id(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        old_thread = target.memory_entity_id
        target.reset_conversation()
        assert target.memory_entity_id != old_thread

    @pytest.mark.asyncio
    async def test_new_does_not_change_original_memory_entity_id(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        old_thread = target.memory_entity_id
        target.new()
        assert target.memory_entity_id == old_thread

    @pytest.mark.asyncio
    async def test_reset_conversation_resets_prev_msg_count(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        await target.respond([Message(role='user', content='hi')])
        assert target._prev_msg_count > 0
        target.reset_conversation()
        assert target._prev_msg_count == 0

    @pytest.mark.asyncio
    async def test_multi_turn_tool_calls_excludes_previous_turns(self) -> None:
        """Turn N must not include tool calls from turns 1..N-1 (checkpointer accumulates state)."""
        from langchain_core.messages import AIMessage

        tool_a = AIMessage(
            content='turn 1 result',
            tool_calls=[{'name': 'tool_A', 'args': {'x': 1}, 'id': 'c1', 'type': 'tool_call'}],
        )
        final_1 = AIMessage(content='done turn 1')

        tool_b = AIMessage(
            content='turn 2 result',
            tool_calls=[{'name': 'tool_B', 'args': {'y': 2}, 'id': 'c2', 'type': 'tool_call'}],
        )
        final_2 = AIMessage(content='done turn 2')

        call_count = 0

        async def fake_ainvoke(state, config):  # noqa: ANN001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {'messages': [tool_a, final_1]}
            # Checkpointer returns full accumulated state
            return {'messages': [tool_a, final_1, tool_b, final_2]}

        graph = MagicMock()
        graph.name = 'test'
        graph.ainvoke = fake_ainvoke

        target = LangGraphTarget(graph)
        r1 = await target.respond([Message(role='user', content='first')])
        assert len(r1.tool_calls) == 1
        assert r1.tool_calls[0].name == 'tool_A'

        r2 = await target.respond([Message(role='user', content='second')])
        assert len(r2.tool_calls) == 1
        assert r2.tool_calls[0].name == 'tool_B'

    @pytest.mark.asyncio
    async def test_anthropic_tool_call_id_does_not_become_a_responses_item_id(self) -> None:
        """An Anthropic ``toolu_*`` id belongs in call_id only.

        Replayed as a Responses ``function_call.id`` it 400s every downstream
        call on that endpoint — the simulated user and the judge both run there.
        """
        from evaluatorq.contracts import render_tool_call
        from evaluatorq.openresponses.input_items import messages_to_responses_input

        tool_turn = AIMessage(
            content='',
            tool_calls=[{'name': 'search', 'args': {'q': 'x'}, 'id': 'toolu_01ABC', 'type': 'tool_call'}],
        )
        graph = MagicMock()
        graph.name = 'test'
        graph.ainvoke = AsyncMock(return_value={'messages': [tool_turn]})

        response = await LangGraphTarget(graph).respond([Message(role='user', content='hi')])
        item = response.tool_calls[0]
        assert item.call_id == 'toolu_01ABC'
        assert item.id.startswith('fc_')

        rendered = render_tool_call(item.model_copy(update={'result': 'done'}))
        assert rendered is not None
        tool_call, tool_message = rendered
        input_items = messages_to_responses_input(
            [Message(role='assistant', content=None, tool_calls=[tool_call]), tool_message]
        )
        function_call = next(i for i in input_items if i['type'] == 'function_call')
        # call_id carries the provider id; the item id is a locally minted fc_*.
        assert function_call['call_id'] == 'toolu_01ABC'
        assert function_call['id'].startswith('fc_')
        assert 'toolu_' not in function_call['id']

    @pytest.mark.asyncio
    async def test_reset_uses_different_thread_id(self) -> None:
        """After reset, respond must invoke ainvoke with a different thread_id."""
        graph = _make_graph()
        target = LangGraphTarget(graph)
        await target.respond([Message(role='user', content='first')])
        thread_id_before = graph.ainvoke.call_args[1]['config']['configurable']['thread_id']

        target.reset_conversation()
        await target.respond([Message(role='user', content='second')])
        thread_id_after = graph.ainvoke.call_args[1]['config']['configurable']['thread_id']

        assert thread_id_before != thread_id_after

    @pytest.mark.asyncio
    async def test_reset_then_send_extracts_all_tool_calls(self) -> None:
        """After reset, a fresh thread_id is used and _prev_msg_count=0, so tool calls are correctly extracted."""
        from langchain_core.messages import AIMessage

        msg_with_tool = AIMessage(
            content='result',
            tool_calls=[{'name': 'tool_X', 'args': {}, 'id': 'c3', 'type': 'tool_call'}],
        )

        graph = MagicMock()
        graph.name = 'test'
        graph.ainvoke = AsyncMock(return_value={'messages': [msg_with_tool]})

        target = LangGraphTarget(graph)
        await target.respond([Message(role='user', content='first')])  # _prev_msg_count becomes 1
        target.reset_conversation()  # _prev_msg_count resets to 0

        # Next send scans from index 0 — all returned messages are "new"
        result = await target.respond([Message(role='user', content='after reset')])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == 'tool_X'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('tool_content', 'expected'),
        [
            ('shipped', 'shipped'),
            # A tool that legitimately returns nothing is not a missing result.
            ('', ''),
            # Anthropic-backed graphs wrap even plain text in a block list.
            ([{'type': 'text', 'text': 'shipped'}], 'shipped'),
        ],
        ids=['text', 'empty', 'anthropic-blocks'],
    )
    async def test_tool_result_is_backfilled_from_tool_message(self, tool_content: Any, expected: str) -> None:
        """A tool call paired with its ToolMessage survives into the transcript.

        Without the pairing ``result`` stays None and `render_tool_call` drops the
        call, so a judge asked "did the agent use the tool?" sees nothing.
        """
        from evaluatorq.contracts import render_tool_call

        graph = _graph_returning(
            _tool_turn(), ToolMessage(content=tool_content, tool_call_id='call_1'), AIMessage(content='done')
        )

        response = await LangGraphTarget(graph).respond([Message(role='user', content='where is my order')])
        item = response.tool_calls[0]
        assert item.result == expected
        assert render_tool_call(item) is not None

    @pytest.mark.asyncio
    async def test_hand_built_dicts_pair_like_message_objects(self) -> None:
        """Graph state may hold plain ``{'role', 'content'}`` dicts rather than LangChain objects."""
        graph = _graph_returning(
            {'role': 'assistant', 'content': '', 'tool_calls': [{'name': 'order_status', 'args': {}, 'id': 'call_1'}]},
            {'role': 'tool', 'content': {'status': 'shipped'}, 'tool_call_id': 'call_1'},
            {'role': 'assistant', 'content': 'done'},
        )

        response = await LangGraphTarget(graph).respond([Message(role='user', content='hi')])
        assert response.tool_calls[0].result == '{"status": "shipped"}'

    @pytest.mark.asyncio
    async def test_tool_call_without_result_stays_none_and_is_dropped(self) -> None:
        """No ToolMessage this turn keeps result=None, so render_tool_call drops the call."""
        from evaluatorq.contracts import render_tool_call

        response = await LangGraphTarget(_graph_returning(_tool_turn())).respond([Message(role='user', content='hi')])
        assert response.tool_calls[0].result is None
        assert render_tool_call(response.tool_calls[0]) is None

    @pytest.mark.asyncio
    async def test_late_tool_message_warns_naming_the_call(self, caplog: pytest.LogCaptureFixture) -> None:
        """The interrupt/resume shape: the result lands a turn after the call, and must announce itself.

        Turn 1 emits the call and drops it (result=None). Turn 2 carries the
        ToolMessage, whose call is no longer in this turn's index — the tool's
        output is discarded, and the only signal anyone gets is this warning.
        """
        call = _tool_turn('refund', 'call_late')
        # A checkpointer returns accumulated state, so turn 2 repeats turn 1's message.
        graph = MagicMock()
        graph.name = 'test'
        graph.ainvoke = AsyncMock(
            side_effect=[
                {'messages': [call]},
                {'messages': [call, ToolMessage(content='refunded', tool_call_id='call_late'), AIMessage(content='ok')]},
            ]
        )

        target = LangGraphTarget(graph)
        first = await target.respond([Message(role='user', content='refund please')])
        assert first.tool_calls[0].result is None

        with caplog.at_level(logging.WARNING):
            second = await target.respond([Message(role='user', content='approved')])

        assert 'call_late' in caplog.text
        assert second.tool_calls == []

    @pytest.mark.asyncio
    async def test_tool_call_without_id_warns_at_the_point_of_loss(self, caplog: pytest.LogCaptureFixture) -> None:
        """A call the graph emits without an id can never be paired, so it warns where it happens.

        Its ToolCallOutputItem still mints a plausible ``call_*`` id, so downstream
        the loss is indistinguishable from a real call — the warning here is the
        only place the cause is visible.
        """
        from evaluatorq.contracts import render_tool_call

        graph = _graph_returning(_tool_turn('ghost', call_id=None), AIMessage(content='done'))

        with caplog.at_level(logging.WARNING):
            response = await LangGraphTarget(graph).respond([Message(role='user', content='hi')])

        assert 'ghost' in caplog.text
        assert response.tool_calls[0].result is None
        assert render_tool_call(response.tool_calls[0]) is None

    @pytest.mark.asyncio
    async def test_results_pair_by_id_not_by_arrival_order(self) -> None:
        """Two calls in one turn, results arriving reversed, must not cross-pair.

        Backfilling the most recently appended call would pass every single-call
        test and silently hand the judge tool ``a`` with tool ``b``'s output.
        """
        graph = _graph_returning(
            AIMessage(
                content='checking both',
                tool_calls=[
                    {'name': 'a', 'args': {}, 'id': 'call_a', 'type': 'tool_call'},
                    {'name': 'b', 'args': {}, 'id': 'call_b', 'type': 'tool_call'},
                ],
            ),
            ToolMessage(content='RESULT_B', tool_call_id='call_b'),
            ToolMessage(content='RESULT_A', tool_call_id='call_a'),
            AIMessage(content='done'),
        )

        response = await LangGraphTarget(graph).respond([Message(role='user', content='hi')])
        assert {(t.name, t.result) for t in response.tool_calls} == {('a', 'RESULT_A'), ('b', 'RESULT_B')}

    @pytest.mark.asyncio
    async def test_anthropic_block_list_tool_result_is_unwrapped(self) -> None:
        """An Anthropic-shaped block list renders as the tool's text, not its envelope."""
        graph = MagicMock()
        graph.name = 'test'
        graph.ainvoke = AsyncMock(
            return_value={
                'messages': [
                    AIMessage(
                        content='',
                        tool_calls=[{'name': 'order_status', 'args': {}, 'id': 'call_1', 'type': 'tool_call'}],
                    ),
                    ToolMessage(content=[{'type': 'text', 'text': 'shipped'}], tool_call_id='call_1'),
                    AIMessage(content='done'),
                ]
            }
        )

        response = await LangGraphTarget(graph).respond([Message(role='user', content='hi')])
        assert response.tool_calls[0].result == 'shipped'

    @pytest.mark.asyncio
    async def test_dumped_message_dicts_keep_their_tool_calls(self) -> None:
        """``BaseMessage.model_dump()`` keys the role on ``type``; those messages must still pair."""
        graph = _graph_returning(
            {'type': 'ai', 'content': '', 'tool_calls': [{'name': 'order_status', 'args': {}, 'id': 'c1'}]},
            {'type': 'tool', 'content': 'shipped', 'tool_call_id': 'c1'},
            {'type': 'ai', 'content': 'done'},
        )

        response = await LangGraphTarget(graph).respond([Message(role='user', content='hi')])
        assert response.tool_calls[0].result == 'shipped'

    @pytest.mark.asyncio
    async def test_messages_to_dict_envelope_keeps_its_tool_calls(self) -> None:
        """``messages_to_dict()`` nests fields under ``data``; reading the outer dict loses them silently."""
        graph = _graph_returning(
            {'type': 'ai', 'data': {'content': '', 'tool_calls': [{'name': 'order_status', 'args': {}, 'id': 'c1'}]}},
            {'type': 'tool', 'data': {'content': 'shipped', 'tool_call_id': 'c1'}},
            {'type': 'ai', 'data': {'content': 'done'}},
        )

        response = await LangGraphTarget(graph).respond([Message(role='user', content='hi')])
        assert response.tool_calls[0].name == 'order_status'
        assert response.tool_calls[0].result == 'shipped'

    @pytest.mark.asyncio
    async def test_unrecognized_message_shape_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A state message we cannot read is announced, not skipped in silence."""
        graph = _graph_returning({'kind': 'mystery', 'body': 'x'}, AIMessage(content='done'))

        with caplog.at_level(logging.WARNING):
            await LangGraphTarget(graph).respond([Message(role='user', content='hi')])

        assert 'unrecognized shape' in caplog.text

    def test_new_preserves_subclass(self) -> None:
        class Subclass(LangGraphTarget):
            pass

        assert type(Subclass(_make_graph()).new()) is Subclass

    @pytest.mark.asyncio
    async def test_extra_config_is_preserved(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph, config={'recursion_limit': 50})
        await target.respond([Message(role='user', content='hi')])

        config = graph.ainvoke.call_args[1]['config']
        assert config['recursion_limit'] == 50

    @pytest.mark.asyncio
    async def test_raises_on_missing_messages_key(self) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={'output': 'no messages here'})
        target = LangGraphTarget(graph)

        with pytest.raises(ValueError, match="'messages' key"):
            await target.respond([Message(role='user', content='hi')])

    @pytest.mark.asyncio
    async def test_raises_on_empty_messages_list(self) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={'messages': []})
        target = LangGraphTarget(graph)

        with pytest.raises(ValueError, match="empty 'messages' list"):
            await target.respond([Message(role='user', content='hi')])

    @pytest.mark.asyncio
    async def test_handles_dict_messages(self) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={'messages': [{'role': 'assistant', 'content': 'dict msg'}]})
        target = LangGraphTarget(graph)
        result = await target.respond([Message(role='user', content='hi')])
        assert result.text == 'dict msg'

    def test_clone_returns_independent_instance(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph, config={'recursion_limit': 50})
        cloned = target.new()
        assert cloned is not target
        assert cloned.memory_entity_id != target.memory_entity_id
        assert cloned._graph is graph
        assert cloned._extra_config is not target._extra_config
        assert cloned._extra_config == {'recursion_limit': 50}

    def test_clone_gets_fresh_memory_entity_id(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        cloned = target.new()
        assert cloned.memory_entity_id != target.memory_entity_id
        assert cloned.memory_entity_id  # non-empty

    def test_new_yields_distinct_keys(self) -> None:
        """Each call to new() must produce a unique _key for per-job isolation.

        Parallel red-team jobs rely on _key to identify the target instance;
        if all clones share the parent's _key, metrics and logs collide.
        """
        graph = _make_graph()
        target = LangGraphTarget(graph)
        clone1 = target.new()
        clone2 = target.new()
        # All three keys must be distinct
        assert clone1._key != target._key
        assert clone2._key != target._key
        assert clone1._key != clone2._key
        # Keys must still be non-empty
        assert clone1._key
        assert clone2._key

    @pytest.mark.asyncio
    async def test_configurable_key_collision_preserves_user_keys(self) -> None:
        """config={"configurable": {"custom_key": "val"}} must not be overwritten by memory_entity_id injection."""
        graph = _make_graph()
        target = LangGraphTarget(graph, config={'configurable': {'custom_key': 'val'}})
        await target.respond([Message(role='user', content='hi')])

        config = graph.ainvoke.call_args[1]['config']
        assert config['configurable']['custom_key'] == 'val'
        assert 'thread_id' in config['configurable']

    @pytest.mark.asyncio
    async def test_non_string_content_is_coerced_to_str(self) -> None:
        """List-type content (e.g. multimodal messages) must be coerced to str."""
        graph = MagicMock()
        msg = MagicMock()
        msg.content = [{'type': 'text', 'text': 'multimodal content'}]
        graph.ainvoke = AsyncMock(return_value={'messages': [msg]})
        target = LangGraphTarget(graph)
        result = await target.respond([Message(role='user', content='hi')])
        assert isinstance(result.text, str)
        assert 'multimodal content' in result.text


class TestLangGraphTargetAgentContext:
    @pytest.mark.asyncio
    async def test_get_agent_context_from_react_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tools bound via create_react_agent show up in the agent context."""
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        # NOTE: langgraph < 2.0 path. create_react_agent moved to
        # `langchain.agents.create_agent` in langgraph V1.0 and is removed in V2.0 —
        # update this import when bumping to langgraph 2.x.
        from langgraph.prebuilt import create_react_agent

        monkeypatch.setenv('OPENAI_API_KEY', 'sk-test-stub')

        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        graph = create_react_agent(ChatOpenAI(model='gpt-4o-mini'), tools=[add])
        target = LangGraphTarget(graph)

        ctx = await target.get_agent_context()
        assert ctx.key.startswith('LangGraph_')
        tool_names = ctx.get_tool_names()
        assert 'add' in tool_names
        # create_react_agent does not set a checkpointer by default → no memory entries
        assert ctx.memory_stores == []

    @pytest.mark.asyncio
    async def test_get_agent_context_handles_graph_without_tools_node(self) -> None:
        graph = _make_graph()
        graph.nodes = {}
        graph.checkpointer = None
        target = LangGraphTarget(graph)
        ctx = await target.get_agent_context()
        assert ctx.key.startswith('test_graph_')
        assert ctx.tools == []
        assert ctx.memory_stores == []

    @pytest.mark.asyncio
    async def test_get_agent_context_emits_memory_store_when_checkpointer_present(self) -> None:
        from evaluatorq.redteam.contracts import MemoryStoreInfo

        graph = _make_graph()
        graph.nodes = {}
        checkpointer = MagicMock()
        checkpointer.__class__.__name__ = 'InMemorySaver'
        graph.checkpointer = checkpointer
        target = LangGraphTarget(graph)

        ctx = await target.get_agent_context()
        assert len(ctx.memory_stores) == 1
        assert isinstance(ctx.memory_stores[0], MemoryStoreInfo)
        assert ctx.memory_stores[0].id == target.memory_entity_id

    @pytest.mark.asyncio
    async def test_get_agent_context_dedupes_tools_across_nodes(self) -> None:
        """Same tool registered in multiple ToolNodes must yield a single entry."""
        shared_tool = MagicMock()
        shared_tool.description = 'shared'
        bound_a = MagicMock()
        bound_a.tools_by_name = {'shared': shared_tool}
        bound_b = MagicMock()
        bound_b.tools_by_name = {'shared': shared_tool, 'extra': MagicMock(description=None)}
        node_a = MagicMock(bound=bound_a)
        node_b = MagicMock(bound=bound_b)

        graph = _make_graph()
        graph.nodes = {'tools_a': node_a, 'tools_b': node_b}
        graph.checkpointer = None
        target = LangGraphTarget(graph)

        ctx = await target.get_agent_context()
        names = [t.name for t in ctx.tools]
        assert names.count('shared') == 1
        assert sorted(names) == ['extra', 'shared']

    @pytest.mark.asyncio
    async def test_get_agent_context_override_returns_verbatim(self) -> None:
        from evaluatorq.redteam.contracts import AgentContext, ToolInfo

        override = AgentContext(
            key='my-custom-agent',
            tools=[ToolInfo(name='custom_tool')],
            description='explicitly-provided context',
        )
        graph = _make_graph()
        target = LangGraphTarget(graph, agent_context=override)

        ctx = await target.get_agent_context()
        assert ctx is override


class TestLangGraphTargetTokenUsage:
    """Tests for callback-handler-based token usage capture."""

    def test_direct_collector_accumulates_usage(self) -> None:
        """Collector accumulates tokens from a real LLMResult with ChatGeneration."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        collector = _TokenUsageCollector()
        msg = AIMessage(
            content='hi',
            usage_metadata=UsageMetadata(input_tokens=100, output_tokens=10, total_tokens=110),
        )
        gen = ChatGeneration(message=msg)
        result = LLMResult(generations=[[gen]])
        collector.on_llm_end(result)

        usage = collector.to_token_usage()
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 10
        assert usage.total_tokens == 110
        assert usage.calls == 1

    def test_total_tokens_zero_synthesized_from_components(self) -> None:
        """A contradictory total_tokens=0 with nonzero input/output is synthesized to
        input+output by the unified fallback rule (RES-906), so total reconciles."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        collector = _TokenUsageCollector()
        msg = AIMessage(
            content='hi',
            usage_metadata=UsageMetadata(input_tokens=50, output_tokens=50, total_tokens=0),
        )
        gen = ChatGeneration(message=msg)
        result = LLMResult(generations=[[gen]])
        collector.on_llm_end(result)

        usage = collector.to_token_usage()
        assert usage is not None
        assert usage.total_tokens == 100
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens

    def test_n_greater_than_1_no_double_count(self) -> None:
        """When n>1, only the first candidate in each inner list is counted."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        collector = _TokenUsageCollector()
        meta = UsageMetadata(input_tokens=20, output_tokens=5, total_tokens=25)
        msg1 = AIMessage(content='candidate 1', usage_metadata=meta)
        msg2 = AIMessage(content='candidate 2', usage_metadata=meta)
        gen1 = ChatGeneration(message=msg1)
        gen2 = ChatGeneration(message=msg2)
        # Both candidates are in the same inner list (same API call, n=2)
        result = LLMResult(generations=[[gen1, gen2]])
        collector.on_llm_end(result)

        # Only gen1 should be counted
        usage = collector.to_token_usage()
        assert usage is not None
        assert usage.calls == 1
        assert usage.input_tokens == 20
        assert usage.output_tokens == 5
        assert usage.total_tokens == 25

    def test_on_llm_error_does_not_crash(self) -> None:
        """on_llm_error must not raise; to_token_usage returns None when nothing captured."""
        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        collector = _TokenUsageCollector()
        collector.on_llm_error(Exception('boom'))
        assert collector.to_token_usage() is None

    def test_on_llm_error_after_success_preserves_prior_usage(self) -> None:
        """on_llm_error must not wipe usage captured by a prior on_llm_end call."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        collector = _TokenUsageCollector()
        msg = AIMessage(
            content='ok',
            usage_metadata=UsageMetadata(input_tokens=10, output_tokens=2, total_tokens=12),
        )
        result = LLMResult(generations=[[ChatGeneration(message=msg)]])
        collector.on_llm_end(result)

        collector.on_llm_error(Exception('later error'))

        usage = collector.to_token_usage()
        assert usage is not None
        assert usage.calls == 1
        assert usage.prompt_tokens == 10

    @pytest.mark.asyncio
    async def test_callbacks_as_list_integration(self) -> None:
        """Collector is appended after existing list callbacks; both reach ainvoke."""
        from langchain_core.callbacks import BaseCallbackHandler

        class SentinelHandler(BaseCallbackHandler):
            pass

        sentinel = SentinelHandler()
        graph = _make_graph('response')
        target = LangGraphTarget(graph, config={'callbacks': [sentinel]})
        await target.respond([Message(role='user', content='hi')])

        config_passed = graph.ainvoke.call_args[1]['config']
        callbacks = config_passed['callbacks']
        assert isinstance(callbacks, list)
        assert len(callbacks) == 2
        assert callbacks[0] is sentinel

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        assert isinstance(callbacks[1], _TokenUsageCollector)

    @pytest.mark.asyncio
    async def test_callbacks_as_manager_not_mutated(self) -> None:
        """Original BaseCallbackManager must not be mutated across respond calls.

        The implementation must copy the manager before adding the per-call collector
        so that stale collectors do not accumulate on the original instance or on
        .new() clones that share the same _extra_config reference.
        """
        from langchain_core.callbacks.manager import AsyncCallbackManager

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        original_manager = MagicMock(spec=AsyncCallbackManager)
        # copy() must return a fresh mock so we can assert the original was not touched.
        manager_copy = MagicMock(spec=AsyncCallbackManager)
        original_manager.copy.return_value = manager_copy

        graph = _make_graph('response')
        target = LangGraphTarget(graph, config={'callbacks': original_manager})

        await target.respond([Message(role='user', content='first')])
        await target.respond([Message(role='user', content='second')])

        # copy() called once per respond — never mutate the original.
        assert original_manager.copy.call_count == 2
        original_manager.add_handler.assert_not_called()

        # add_handler was called on the copy, not the original.
        assert manager_copy.add_handler.call_count == 2
        first_arg = manager_copy.add_handler.call_args_list[0][0][0]
        assert isinstance(first_arg, _TokenUsageCollector)

    @pytest.mark.asyncio
    async def test_new_yields_independent_instances(self) -> None:
        """Parent and clone are independent: separate memory_entity_id and fresh SendResult per call."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        def _fake_ainvoke(input_dict: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
            callbacks = config.get('callbacks', [])
            for cb in callbacks if isinstance(callbacks, list) else []:
                if isinstance(cb, _TokenUsageCollector):
                    meta = UsageMetadata(input_tokens=7, output_tokens=3, total_tokens=10)
                    msg = AIMessage(content='ok', usage_metadata=meta)
                    gen = ChatGeneration(message=msg)
                    cb.on_llm_end(LLMResult(generations=[[gen]]))
            return {'messages': [MagicMock(content='ok')]}

        graph = MagicMock()
        graph.name = 'test_graph'
        graph.ainvoke = AsyncMock(side_effect=_fake_ainvoke)

        parent = LangGraphTarget(graph)
        parent_result = await parent.respond([Message(role='user', content='p1')])

        clone = parent.new()
        clone_result = await clone.respond([Message(role='user', content='p2')])

        assert parent_result.usage is not None
        assert clone_result.usage is not None
        # Both captured usage independently
        assert parent_result.usage.prompt_tokens == 7
        assert clone_result.usage.prompt_tokens == 7
        # Instances are distinct
        assert parent is not clone
        assert parent.memory_entity_id != clone.memory_entity_id

    @pytest.mark.asyncio
    async def test_usage_collector_drains_when_ainvoke_raises(self) -> None:
        """Collector's finally block runs even when ainvoke raises.

        The inner try/finally in respond drains the collector;
        the exception propagates normally. This test verifies the finally runs
        without error and that the exception still reaches the caller.
        """
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        ai_msg = AIMessage(
            content='partial',
            usage_metadata={'input_tokens': 40, 'output_tokens': 5, 'total_tokens': 45},
        )
        gen = ChatGeneration(message=ai_msg)

        graph = MagicMock()
        graph.name = 'test_graph'

        async def failing_ainvoke(input: Any, config: Any) -> Any:  # noqa: A002
            handlers = config.get('callbacks') or []
            for h in handlers:
                if isinstance(h, _TokenUsageCollector):
                    h.on_llm_end(LLMResult(generations=[[gen]]))
            raise RuntimeError('provider error')

        graph.ainvoke = failing_ainvoke

        target = LangGraphTarget(graph)
        with pytest.raises(RuntimeError, match='provider error'):
            await target.respond([Message(role='user', content='hi')])

    @pytest.mark.asyncio
    async def test_no_usage_metadata_returns_none_in_send_result(self) -> None:
        """When graph fires no LLM callbacks, SendResult.usage is None."""
        graph = _make_graph('no usage')
        target = LangGraphTarget(graph)
        result = await target.respond([Message(role='user', content='hi')])
        # ainvoke mock doesn't fire callbacks, so collector gets no calls
        assert result.usage is None


class TestAnthropicBlockContent:
    """Claude-backed graphs return list-of-block content even for plain text.

    ``str()`` on that list yields its Python repr, which used to land in
    ``AgentResponse.text`` and flow on into transcripts and judges as if it were
    the model's answer.
    """

    @pytest.mark.asyncio
    async def test_text_blocks_are_extracted_not_repr(self) -> None:
        graph = MagicMock()
        graph.name = 'test_graph'
        msg = AIMessage(content=[{'type': 'text', 'text': 'Hello'}])
        graph.ainvoke = AsyncMock(return_value={'messages': [msg]})

        result = await LangGraphTarget(graph).respond([Message(role='user', content='hi')])
        assert result.text == 'Hello'

    @pytest.mark.asyncio
    async def test_interleaved_text_blocks_are_joined(self) -> None:
        graph = MagicMock()
        graph.name = 'test_graph'
        msg = AIMessage(
            content=[
                {'type': 'text', 'text': 'Let me check. '},
                {'type': 'tool_use', 'id': 't1', 'name': 'search', 'input': {}},
                {'type': 'text', 'text': 'Found it.'},
            ]
        )
        graph.ainvoke = AsyncMock(return_value={'messages': [msg]})

        result = await LangGraphTarget(graph).respond([Message(role='user', content='hi')])
        # tool_use is read separately from .tool_calls, so it contributes no text.
        assert result.text == 'Let me check. Found it.'

    @pytest.mark.asyncio
    async def test_unrenderable_block_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        graph = MagicMock()
        graph.name = 'test_graph'
        msg = AIMessage(content=[{'type': 'image', 'source': {}}, {'type': 'text', 'text': 'see above'}])
        graph.ainvoke = AsyncMock(return_value={'messages': [msg]})

        with caplog.at_level('WARNING'):
            result = await LangGraphTarget(graph).respond([Message(role='user', content='hi')])
        assert result.text == 'see above'
        assert 'image' in caplog.text
