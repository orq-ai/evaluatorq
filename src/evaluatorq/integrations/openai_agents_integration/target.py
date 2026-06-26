"""AgentTarget wrapper for OpenAI Agents SDK."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import Agent, Runner
from loguru import logger

from evaluatorq.contracts import (
    AgentContext,
    AgentResponse,
    AgentTarget,
    Message,
    OutputMessage,
    TextOutputItem,
    TokenUsage,
    ToolCallOutputItem,
    ToolInfo,
)


class OpenAIAgentTarget(AgentTarget):
    """Wraps an OpenAI Agents SDK Agent as an AgentTarget.

    Usage::

        from agents import Agent
        from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget

        agent = Agent(name="my-agent", instructions="You are a helpful assistant.")
        target = OpenAIAgentTarget(agent)

        # Pass to simulation or red teaming
        config = DynamicRunConfig(targets=[target])
    """

    def __init__(self, agent: Agent, *, run_kwargs: dict[str, Any] | None = None) -> None:
        """Create an OpenAI Agents SDK agent target.

        Args:
            agent: An OpenAI Agents SDK Agent instance.
            run_kwargs: Optional extra keyword arguments passed to ``Runner.run()``
                (e.g. ``{"max_turns": 10}``).
        """
        super().__init__(memory_entity_id=None)
        self._agent = agent
        self._run_kwargs = run_kwargs or {}

    async def respond(self, messages: list[Message]) -> AgentResponse:
        """Stateless: run the agent over the provided transcript.

        The OpenAI Agents SDK accepts a list of input items, so ``respond``
        renders each ``Message`` into Responses-API input items (preserving
        tool calls / tool results) and passes them in. The caller (orchestrator)
        owns conversation continuity. Only the items the run *adds* (sliced
        from ``to_input_list()`` past the input length) are passed to
        ``_build_response``, so the returned ``AgentResponse`` reflects just
        this turn's output.
        """
        input_data: list[Any] = [item for m in messages for item in _message_to_responses_input_items(m)]
        prev_len = len(input_data)
        try:
            result = await Runner.run(self._agent, input_data, **self._run_kwargs)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            raise
        except Exception as exc:
            raise RuntimeError(f'OpenAIAgentTarget: Runner.run() raised an error: {exc}') from exc

        if result.final_output is None:
            raise ValueError(
                'OpenAIAgentTarget: Runner.run() returned final_output=None. Ensure the agent produces a text response.'
            )

        full_list = result.to_input_list()
        if full_list[:prev_len] != input_data:
            logger.warning(
                'OpenAIAgentTarget: Runner.to_input_list() no longer echoes the input '
                'items 1:1 at the front (sent prev_len={n}, prefix differs); the '
                'new-items slice may misalign. Treat response output as approximate.',
                n=prev_len,
            )
        new_items = full_list[min(prev_len, len(full_list)) :]
        return self._build_response(new_items, result)

    def _build_response(self, new_items: list[Any], result: Any) -> AgentResponse:
        """Build an AgentResponse from the items a run added, plus its final output.

        Preserves original Responses API ``id``/``call_id`` fields for trace
        correlation. Iteration order is preserved so interleaved text/tool calls
        (e.g. reasoning -> tool_call -> reasoning) round-trip into
        ``AgentResponse.output``.

        ``function_call_output`` items returned by ``result.to_input_list()`` are
        merged into their matching ``ToolCallOutputItem`` (by ``call_id``) so the
        tool result survives transcript replay. Without this, a subsequent turn
        would emit a Responses-API ``function_call`` with no paired
        ``function_call_output`` and the API would reject the input list.
        """
        output_items: list[OutputMessage] = []
        # call_id -> index into output_items, for back-filling ``result`` when a
        # ``function_call_output`` item is encountered later in the stream.
        tool_call_index: dict[str, int] = {}

        def _record_tool_call(name: str, arguments: str, item_id: str, call_id: str) -> None:
            tc = (
                ToolCallOutputItem(name=name, arguments=arguments, id=item_id, call_id=call_id)
                if item_id and call_id
                else ToolCallOutputItem(name=name, arguments=arguments)
            )
            output_items.append(tc)
            if call_id:
                tool_call_index[call_id] = len(output_items) - 1

        def _attach_tool_output(call_id: str, output: str) -> None:
            idx = tool_call_index.get(call_id)
            if idx is None:
                return
            existing = output_items[idx]
            if isinstance(existing, ToolCallOutputItem):
                output_items[idx] = existing.model_copy(update={'result': output})

        for item in new_items:
            # Agents SDK to_input_list() returns Responses API items for tool calls,
            # e.g. {"type": "function_call", "id": "fc_...", "call_id": "call_...", "name": "...", "arguments": "..."}.
            if isinstance(item, dict) and item.get('type') == 'function_call':
                item_id = item.get('id', '')
                call_id = item.get('call_id', '')
                name = str(item.get('name', ''))
                arguments = _normalize_args_str(item.get('arguments', '{}'))
                _record_tool_call(name, arguments, item_id, call_id)
            # function_call_output items appear after their matching function_call in
            # ``result.to_input_list()`` and carry the tool's result. Merge into the
            # prior ToolCallOutputItem so transcript replay preserves the result.
            elif isinstance(item, dict) and item.get('type') == 'function_call_output':
                call_id = item.get('call_id', '')
                output_str = _extract_function_call_output(item.get('output'))
                _attach_tool_output(call_id, output_str)
            # Handle dict format (standard OpenAI message format)
            elif isinstance(item, dict) and item.get('role') == 'assistant':
                text = _extract_assistant_text(item.get('content'))
                if text:
                    output_items.append(TextOutputItem(text=text, annotations=[]))
                for tc in item.get('tool_calls') or []:
                    if isinstance(tc, dict):
                        func = tc.get('function', {})
                        name = func.get('name', '') if isinstance(func, dict) else ''
                        args_raw = _normalize_args_str(func.get('arguments', '{}') if isinstance(func, dict) else '{}')
                        tc_id = tc.get('id', '')
                        _record_tool_call(name, args_raw, tc_id, tc_id)
            # Handle typed SDK objects (future-proofing if to_input_list() returns non-dict items)
            elif not isinstance(item, dict):
                if getattr(item, 'type', None) == 'function_call':
                    item_id = getattr(item, 'id', '')
                    call_id = getattr(item, 'call_id', '')
                    name = str(getattr(item, 'name', ''))
                    arguments = _normalize_args_str(getattr(item, 'arguments', '{}'))
                    _record_tool_call(name, arguments, item_id, call_id)
                elif getattr(item, 'type', None) == 'function_call_output':
                    call_id = getattr(item, 'call_id', '')
                    output_str = _extract_function_call_output(getattr(item, 'output', None))
                    _attach_tool_output(call_id, output_str)
                elif getattr(item, 'role', None) == 'assistant':
                    text = _extract_assistant_text(getattr(item, 'content', None))
                    if text:
                        output_items.append(TextOutputItem(text=text, annotations=[]))
                    for tc in getattr(item, 'tool_calls', None) or []:
                        tc_name = getattr(tc, 'name', None) or getattr(getattr(tc, 'function', None), 'name', '') or ''
                        tc_args_raw = _normalize_args_str(
                            getattr(tc, 'arguments', None) or getattr(getattr(tc, 'function', None), 'arguments', '{}')
                        )
                        tc_id = getattr(tc, 'id', '')
                        _record_tool_call(str(tc_name), tc_args_raw, tc_id, tc_id)

        # Ensure final_output is reflected as a TextOutputItem. Avoid duplicating
        # if the last text emitted from history already matches.
        final_text = str(result.final_output)
        last_text = next(
            (item.text for item in reversed(output_items) if isinstance(item, TextOutputItem)),
            None,
        )
        if last_text != final_text:
            output_items.append(TextOutputItem(text=final_text, annotations=[]))

        ctx = getattr(result, 'context_wrapper', None)
        agent_usage = getattr(ctx, 'usage', None) if ctx is not None else None
        usage = TokenUsage.extract(agent_usage, calls=1)

        return AgentResponse(output=output_items, usage=usage)

    async def get_agent_context(self) -> AgentContext:
        """Return agent context derived from the wrapped Agent instance.

        Maps the SDK ``Agent`` fields onto :class:`AgentContext`:
        ``name`` → ``key``/``display_name``, ``instructions`` → ``system_prompt``,
        ``model`` → ``model``, ``tools`` → ``tools`` (via duck-typed introspection).
        There is no server-side memory, so ``memory_stores`` stays empty.
        """
        agent = self._agent
        key = str(getattr(agent, 'name', None) or 'openai_agent')
        instructions = getattr(agent, 'instructions', None)
        system_prompt = instructions if isinstance(instructions, str) else None

        model_attr = getattr(agent, 'model', None)
        if isinstance(model_attr, str):
            model = model_attr
        elif model_attr is None:
            model = None
        else:
            model = getattr(model_attr, 'model', None) or str(model_attr)

        tools: list[ToolInfo] = []
        for tool in getattr(agent, 'tools', []) or []:
            name = getattr(tool, 'name', None) or getattr(tool, '__name__', None)
            if not name:
                continue
            params = getattr(tool, 'params_json_schema', None)
            tools.append(
                ToolInfo(
                    name=str(name),
                    description=getattr(tool, 'description', None),
                    parameters=params if isinstance(params, dict) else None,
                )
            )

        return AgentContext(
            key=key,
            display_name=key,
            description='OpenAI Agents SDK target',
            system_prompt=system_prompt,
            tools=tools,
            model=model,
        )

    def clone(self) -> OpenAIAgentTarget:
        """Return a fresh independent instance for parallel job safety."""
        return OpenAIAgentTarget(self._agent, run_kwargs=dict(self._run_kwargs))

    def new(self) -> OpenAIAgentTarget:
        """Return an independent instance for parallel simulation/red-team jobs."""
        return self.clone()


def _extract_assistant_text(content: Any) -> str:
    """Pull plain text out of a Responses-API assistant content field.

    ``content`` may be a string or a list of content parts such as
    ``{"type": "output_text", "text": "..."}`` / ``{"type": "text", "text": "..."}``.
    Returns the concatenated text, or empty string if none found.
    """
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get('text')
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(part, 'text', None)
                if isinstance(text, str):
                    parts.append(text)
        return ''.join(parts)
    return ''


def _normalize_args_str(raw: Any) -> str:
    """Convert SDK tool-call arguments to a valid JSON string.

    Handles str (valid or invalid JSON), dict, or other types.
    Invalid JSON strings are wrapped as ``{"raw": "<original>"}`` so that the
    ``AgentResponse.tool_calls`` property can always parse ``arguments`` back
    to a dict without silently discarding the original value.
    """
    if isinstance(raw, dict):
        return json.dumps(raw)
    if not isinstance(raw, str):
        return '{}'
    try:
        json.loads(raw)
        return raw
    except (json.JSONDecodeError, ValueError):
        return json.dumps({'raw': raw})


def _extract_function_call_output(raw: Any) -> str:
    """Coerce a Responses-API ``function_call_output.output`` field to a string.

    The SDK may pass the output as a plain string, a JSON-serializable dict, or
    ``None`` (for tool errors that produced no payload). All branches collapse
    to ``str`` because :attr:`ToolCallOutputItem.result` is typed ``str | None``
    and the Responses API's ``function_call_output`` expects ``output: str``.
    """
    if raw is None:
        return ''
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw)
    except (TypeError, ValueError):
        return str(raw)


def _message_to_responses_input_items(m: Message) -> list[dict[str, Any]]:
    """Render a :class:`Message` as Responses-API input items.

    Inverse of :meth:`OpenAIAgentTarget._build_response`: an assistant turn with
    tool calls becomes a ``function_call`` item per call (plus a leading assistant
    text message when content is present); a ``tool`` result becomes a
    ``function_call_output``; anything else is a plain ``{"role", "content"}``
    message. This preserves multi-turn tool context that a naive flatten drops,
    matching what the SDK's ``to_input_list()`` round-trips.
    """
    if m.role == 'tool':
        return [{'type': 'function_call_output', 'call_id': m.tool_call_id or '', 'output': m.content or ''}]
    if m.role == 'assistant' and m.tool_calls:
        items: list[dict[str, Any]] = []
        if m.content:
            items.append({'role': 'assistant', 'content': m.content})
        for tc in m.tool_calls:
            fc: dict[str, Any] = {
                'type': 'function_call',
                'call_id': tc.id,
                'name': tc.function.name,
                'arguments': tc.function.arguments,
            }
            # Echo the Responses-API item id (fc_*) when available so the
            # function_call item round-trips intact across turns.
            if tc.item_id:
                fc['id'] = tc.item_id
            items.append(fc)
        return items
    # Multi-part content passes straight through as Responses-API content parts.
    if isinstance(m.content, list):
        return [{'role': m.role, 'content': [p.model_dump(mode='json') for p in m.content]}]
    return [{'role': m.role, 'content': m.content or ''}]
