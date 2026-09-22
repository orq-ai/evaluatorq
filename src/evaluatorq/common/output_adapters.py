"""Canonical adapters between agent output shapes and messages/text.

Every surface normalises third-party agent output here — a per-surface
``isinstance`` ladder is how the shapes drifted apart the last time.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.messages import coerce_content_text, messages_to_text
from evaluatorq.contracts import (
    AgentResponse,
    Message,
    OutputMessage,
    ReasoningOutputItem,
    TextOutputItem,
    ToolCallOutputItem,
)

if TYPE_CHECKING:
    from evaluatorq.types import Output


_MESSAGE_SHAPE_KEYS = frozenset({
    'role',
    'content',
    'tool_calls',
    'tool_call_id',
    'function_call',
    'function_call_output',
})


def _is_message_shape(item: Any) -> bool:
    """Whether a list element is a chat message rather than a structured item.

    A Responses output item (``{'type': 'output_text', ...}``) is a dict too, so
    dict-ness alone is not enough: read as a message it renders as nothing.
    """
    return isinstance(item, Message) or (isinstance(item, dict) and bool(_MESSAGE_SHAPE_KEYS.intersection(item)))


def output_to_text(output: Any) -> str:
    """Best-effort plain-text view of any Output. Total / fail-soft."""
    if output is None:
        return ''
    if isinstance(output, AgentResponse):
        # A tool-call-only response has empty .text, which has_meaningful_output still counts as a response.
        return output.text or messages_to_text([{'role': 'assistant', 'tool_calls': output.tool_calls}])
    if isinstance(output, str):
        return output
    if isinstance(output, list) and (not output or all(_is_message_shape(item) for item in output)):
        # messages_to_text, not a bare join: dropping tool calls hands the judge an empty string for an agent that acted.
        return messages_to_text(output)
    if isinstance(output, list):
        # A list of non-message mappings is not a transcript: every element would read as a content-less turn.
        try:
            return json.dumps(output, indent=2, default=str)
        except Exception as exc:
            logger.warning('output_to_text: json.dumps failed for a structured list, falling back to str(): {}', exc)
            return str(output)
    if isinstance(output, dict):
        if output.get('object') == 'response':
            try:
                return AgentResponse.from_openresponses(output).text
            except Exception as exc:
                logger.warning('output_to_text: from_openresponses failed, falling back to json: {}', exc)
        try:
            return json.dumps(output, indent=2, default=str)
        except Exception as exc:
            logger.warning('output_to_text: json.dumps failed, falling back to str(): {}', exc)
            return str(output)
    try:
        return str(output)
    except Exception as exc:
        # Degrading to '' here reads downstream as "the agent said nothing" —
        # always leave a trace so that is distinguishable from a real bug.
        logger.warning('output_to_text: str() failed for {}, degrading to empty text: {}', type(output), exc)
        return ''


def output_error_text(output: Any) -> str | None:
    """Target-level error message carried by an Output, if any."""
    if isinstance(output, AgentResponse):
        # Error presence, rather than message truthiness, marks a failed target.
        # Providers may legitimately return an empty message.
        return output.error.message if output.error is not None else None
    if isinstance(output, dict) and 'error' in output and output['error'] is not None:
        error = output['error']
        if isinstance(error, str):
            return error
        if isinstance(error, dict):
            message = error.get('message')
            return str(message) if message is not None else str(error)
        message = getattr(error, 'message', None)
        return str(message) if message is not None else str(error)
    return None


def _message_has_meaningful_output(message: Message | dict[str, Any]) -> bool:
    """Return whether a known message shape carries a response or tool action."""
    if isinstance(message, Message):
        return bool(message.tool_calls) or bool(coerce_content_text(message.content).strip())

    role = message.get('role')
    if not isinstance(role, str):
        # An unfamiliar provider shape is unknown structured output, not a blank one.
        return bool(message)
    if message.get('tool_calls') or message.get('function_call') or message.get('function_call_output'):
        return True
    if 'content' in message:
        content = message.get('content')
        if content is None:
            return any(key not in {'role', 'content'} for key in message)
        return bool(coerce_content_text(content).strip())
    # The role-specific branches that used to sit here computed exactly this.
    return any(key not in {'role', 'name'} for key in message)


def _output_message_has_meaningful_output(message: OutputMessage) -> bool:
    """Return whether a canonical output item carries text or a tool call."""
    if isinstance(message, ToolCallOutputItem):
        return True
    if isinstance(message, (TextOutputItem, ReasoningOutputItem)):
        return bool(message.text.strip())
    return True


def has_meaningful_output(output: Any) -> bool:
    """Return whether *output* is a non-blank recorded response.

    Known text-bearing message shapes reject whitespace-only assistant content, while
    tool calls and unfamiliar non-empty structured shapes count as meaningful. This
    lets trace replay fail on a genuinely blank response without discarding provider
    fields that evaluatorq does not know how to interpret yet.
    """
    if output is None:
        return False
    if isinstance(output, AgentResponse):
        if output.error is not None:
            return False
        return any(_output_message_has_meaningful_output(item) for item in output.output)
    if isinstance(output, str):
        return bool(output.strip())
    if isinstance(output, list):
        if not output:
            return False
        if all(isinstance(item, (Message, dict)) for item in output):
            return any(_message_has_meaningful_output(item) for item in output)
        if all(isinstance(item, (TextOutputItem, ToolCallOutputItem, ReasoningOutputItem)) for item in output):
            return any(_output_message_has_meaningful_output(item) for item in output)
        return True
    if isinstance(output, dict):
        if output.get('object') == 'response' and 'output' in output:
            response_output = output.get('output')
            return (
                has_meaningful_output(response_output) if isinstance(response_output, list) else bool(response_output)
            )
        if 'response' in output or 'tool_calls' in output:
            response = output.get('response')
            return bool(coerce_content_text(response).strip()) or bool(output.get('tool_calls'))
        return bool(output)
    return True


def _adapt_tool_call(tc: Any) -> ToolCallOutputItem:
    """Coerce a static-output tool-call entry into a ToolCallOutputItem."""
    if isinstance(tc, ToolCallOutputItem):
        return tc
    if isinstance(tc, dict):
        fn = tc.get('function', tc)
        raw_args = fn.get('arguments', '{}')
        arguments = raw_args if isinstance(raw_args, str) else json.dumps(raw_args or {})
        tid = str(tc.get('id', '') or '')
        return ToolCallOutputItem(
            id=tid, call_id=tid, name=str(fn.get('name', '')), arguments=arguments, result=tc.get('result')
        )
    # object with attributes (orchestrator item / test double)
    args_dict = getattr(tc, 'arguments_dict', None)
    if args_dict is not None:
        arguments = json.dumps(args_dict)
    else:
        raw = getattr(tc, 'arguments', '{}')
        arguments = raw if isinstance(raw, str) else json.dumps(raw or {})
    tid = str(getattr(tc, 'id', '') or '')
    return ToolCallOutputItem(
        id=tid, call_id=tid, name=str(getattr(tc, 'name', '')), arguments=arguments, result=getattr(tc, 'result', None)
    )


def _adapt_static_output(output: Any) -> list[OutputMessage]:
    """Adapt a static datapoint output ({response, tool_calls} dict, or a bare string)
    into structured OutputMessage records."""
    items: list[OutputMessage] = []
    if isinstance(output, dict):
        text = output.get('response', '')
        if text:
            items.append(TextOutputItem(text=str(text), annotations=[]))
        items.extend(_adapt_tool_call(tc) for tc in output.get('tool_calls') or [])
    elif output:
        items.append(TextOutputItem(text=str(output), annotations=[]))
    return items


def inputs_to_messages(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Coerce a DataPoint.inputs dict into a {role, content} message list. Fail-soft."""
    if isinstance(inputs, dict) and isinstance(inputs.get('messages'), list):
        out: list[dict[str, Any]] = []
        for m in inputs['messages']:
            if isinstance(m, dict):
                out.append({'role': str(m.get('role', 'user')), 'content': coerce_content_text(m.get('content', ''))})
            else:
                out.append({
                    'role': str(getattr(m, 'role', 'user')),
                    'content': coerce_content_text(getattr(m, 'content', '')),
                })
        return out
    if isinstance(inputs, dict) and 'input' in inputs:
        return [{'role': 'user', 'content': coerce_content_text(inputs['input'])}]
    try:
        body = json.dumps(inputs, indent=2, default=str)
    except Exception as exc:
        logger.warning('inputs_to_messages: json.dumps failed, falling back to str(): {}', exc)
        body = str(inputs)
    return [{'role': 'user', 'content': body}]


def output_to_messages(output: Output) -> list[OutputMessage]:
    """Convert any Output into structured OutputMessage records. Fail-soft."""
    if output is None:
        return []
    if isinstance(output, AgentResponse):
        return list(output.output)
    if isinstance(output, dict) and output.get('object') == 'response':
        try:
            return list(AgentResponse.from_openresponses(output).output)
        except Exception as exc:
            logger.warning('output_to_messages: from_openresponses failed, degrading to flat text: {}', exc)
            return [TextOutputItem(text=output_to_text(output), annotations=[])]
    if isinstance(output, str):
        return [TextOutputItem(text=output, annotations=[])] if output else []
    # dict / number / bool → reuse the static-output adapter, else text
    adapted = _adapt_static_output(output)
    return adapted or [TextOutputItem(text=output_to_text(output), annotations=[])]
