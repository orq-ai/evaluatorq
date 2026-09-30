"""ResponsesConversation: an agent run as OpenAI Responses items."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from openai.types.responses import Response, ResponseInputItem, ResponseOutputItem
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, ValidationInfo, field_validator

from evaluatorq.formats._ids import content_seed, stable_hex
from evaluatorq.formats._shared import atif_tool_arguments, json_arguments_text
from evaluatorq.openresponses.otel_messages import RESPONSES_ITEM_TYPES

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from evaluatorq.formats.atif import AtifTrajectory
    from evaluatorq.formats.chat import ChatConversation
    from evaluatorq.formats.otel import OtelTrace

_KNOWN_ITEM_TYPES = RESPONSES_ITEM_TYPES | {'message', 'compaction'}
_MODEL_OUTPUT_TYPES = frozenset({'reasoning', 'function_call', 'custom_tool_call', 'mcp_call'})
_INPUT_ITEM: TypeAdapter[ResponseInputItem] = TypeAdapter(ResponseInputItem)
_OUTPUT_ITEM: TypeAdapter[ResponseOutputItem] = TypeAdapter(ResponseOutputItem)


def item_type(item: dict[str, Any]) -> Any:
    """The item's `type`; a bare `{"role", "content"}` dict is a `message`."""
    kind = item.get('type')
    return 'message' if kind is None and 'role' in item else kind


def walk_items(
    items: list[dict[str, Any]], handlers: Mapping[str, Callable[[int, dict[str, Any]], None]], target: str
) -> None:
    """Call the handler for each item's `item_type` with its index; an item with no handler is warned and skipped."""
    for index, item in enumerate(items):
        kind = item_type(item)
        handler = handlers.get(kind) if isinstance(kind, str) else None
        if handler is None:
            logger.warning('Skipping Responses item of type {!r}: {} has no equivalent.', kind, target)
        else:
            handler(index, item)


def is_output_item(item: dict[str, Any]) -> bool:
    """Whether a model call produced this supported item, including custom and MCP tool calls."""
    kind = item_type(item)
    return kind in _MODEL_OUTPUT_TYPES or (kind == 'message' and item.get('role') == 'assistant')


def output_item(item: dict[str, Any]) -> ResponseOutputItem:
    """Type one output item as the SDK's `ResponseOutputItem`.

    A transcript may leave out what the API always sets, so a missing message or reasoning `id` becomes a hash of
    the item, a missing message `status` becomes `completed`, `output_text` parts get empty `annotations`, and
    function-call arguments that are not a string are written as JSON.

    Raises:
        ValueError: The item does not validate as a Responses output item.
    """
    data = {**item, 'type': item_type(item)}
    seed = content_seed(data)
    if data['type'] == 'message':
        data.setdefault('id', 'msg_' + stable_hex(seed, length=24))
        data.setdefault('status', 'completed')
        content = data.get('content')
        if content is None or isinstance(content, str):
            content = [{'type': 'output_text', 'text': content or ''}]
        if isinstance(content, list):
            parts = cast('list[Any]', content)
            content = [
                {'annotations': [], **p} if isinstance(p, dict) and p.get('type') == 'output_text' else p for p in parts
            ]
        data['content'] = content
    elif data['type'] == 'reasoning':
        data.setdefault('id', 'rs_' + stable_hex(seed, length=24))
        data.setdefault('summary', [])
    elif data['type'] == 'function_call' and not isinstance(data.get('arguments'), str):
        arguments, raw = atif_tool_arguments(data.get('arguments'), data.get('name'))
        data['arguments'] = raw if raw is not None else json_arguments_text(arguments)
    try:
        return _OUTPUT_ITEM.validate_python(data)
    except ValidationError as exc:
        msg = f'Responses {data["type"]!r} item is not a valid Responses output item: {exc}'
        raise ValueError(msg) from exc


def output_turns(items: list[dict[str, Any]]) -> list[list[int]]:
    """Indices of each run of consecutive output items; any other item ends the run."""
    turns: list[list[int]] = []
    open_turn = False
    for index, item in enumerate(items):
        if not is_output_item(item):
            open_turn = False
        elif open_turn:
            turns[-1].append(index)
        else:
            turns.append([index])
            open_turn = True
    return turns


def response_starts(items: list[dict[str, Any]], responses: list[Response]) -> dict[int, Response]:
    """Map the index of each response's first output item to the response (outputs already match `items`)."""
    order = [index for turn in output_turns(items) for index in turn]
    starts: dict[int, Response] = {}
    position = 0
    for response in responses:
        starts[order[position]] = response
        position += len(response.output)
    return starts


def _dump(item: ResponseOutputItem) -> dict[str, Any]:
    return item.model_dump(mode='json', exclude_none=True)


class ResponsesConversation(BaseModel):
    """A run as Responses items plus, optionally, the `Response` of each model call.

    `items` is the transcript: input and output items in order. Each `Response.output` holds the typed output
    items that call produced, and must match the transcript: responses built with an empty `output` get it
    filled from `items` (one response per run of consecutive output items), and responses that name their
    output must cover every output item in `items`, in order, none spanning an input item. A response only adds
    the model, usage, timing and status of its call.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    items: list[dict[str, Any]]
    responses: list[Response] | None = None

    @field_validator('items')
    @classmethod
    def _check_items(cls, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in items:
            kind = item_type(item)
            if kind not in _KNOWN_ITEM_TYPES:
                logger.warning('Unrecognised Responses item type {!r}; keeping it.', kind)
                continue
            try:
                if is_output_item(item):
                    output_item(item)
                else:
                    _INPUT_ITEM.validate_python(item)
            except (ValueError, ValidationError) as exc:
                logger.warning('Responses {!r} item does not validate as a Responses item; keeping it. ({})', kind, exc)
        return items

    @field_validator('responses')
    @classmethod
    def _outputs_match_items(cls, responses: list[Response] | None, info: ValidationInfo) -> list[Response] | None:
        items: list[dict[str, Any]] | None = info.data.get('items')
        if responses is None or items is None:
            return responses
        turns = output_turns(items)
        typed = {index: output_item(items[index]) for turn in turns for index in turn}
        if all(not response.output for response in responses):
            if len(responses) != len(turns):
                msg = (
                    f'Got {len(responses)} responses with no output for {len(turns)} runs of output items; '
                    'pass one Response per run, or set each Response.output'
                )
                raise ValueError(msg)
            return [
                response.model_copy(update={'output': [typed[index] for index in turn]})
                for response, turn in zip(responses, turns, strict=True)
            ]
        order = [index for turn in turns for index in turn]
        turn_of = {index: n for n, turn in enumerate(turns) for index in turn}
        position = 0
        for response in responses:
            span = order[position : position + len(response.output)]
            if (
                not response.output
                or len(span) != len(response.output)
                or len({turn_of[index] for index in span}) != 1
                or [_dump(o) for o in response.output] != [_dump(typed[index]) for index in span]
            ):
                msg = f'Response {response.id} output does not match the next output items of the transcript'
                raise ValueError(msg)
            position += len(response.output)
        if position != len(order):
            msg = f'{len(order) - position} output items of the transcript belong to no Response'
            raise ValueError(msg)
        return responses

    def to_chat(self) -> ChatConversation:
        """Render as chat messages.

        Lost: reasoning items (counted and warned once), media that is not text, image or file parts,
        `status`, annotations, and item ids that do not start with `fc_`.
        """
        from evaluatorq.formats import convert_chat_responses

        return convert_chat_responses.responses_to_chat(self)

    def to_atif(
        self, *, agent_name: str = 'unknown', agent_version: str = 'unknown', session_id: str | None = None
    ) -> AtifTrajectory:
        """Convert to an ATIF trajectory: user/system messages become steps, assistant output becomes agent steps.

        `session_id` defaults to a hash of the items, so converting the same items twice gives the same id.

        A tool result closes its agent step, and so does the first output item of the next `Response` when
        `responses` are set; each response enriches the agent step it starts. Kept in free-form slots: reasoning
        tokens and total tokens in `metrics.extra`; `status`, `error` and `incomplete_details` (non-default only),
        the response id, `fc_` item ids and encrypted-only reasoning in step `extra`; the developer role in
        `extra.original_role` of a system step. A `compaction` item becomes a system step whose `extra` holds
        `context_management` and the item under `evaluatorq.compaction`. Lost: encrypted reasoning content,
        non-text assistant parts, file parts (rendered as `[file: name]` markers), and item types ATIF has no step
        for (warned).
        """
        from evaluatorq.formats import convert_responses_atif

        return convert_responses_atif.responses_to_atif(
            self, agent_name=agent_name, agent_version=agent_version, session_id=session_id
        )

    def to_otel(self, *, agent_name: str = 'unknown', agent_version: str = 'unknown') -> OtelTrace:
        """Convert to one OTel trace through ATIF.

        Lost: everything lost by `ResponsesConversation.to_atif` and `AtifTrajectory.to_otel` (encrypted
        reasoning content, non-text assistant parts, file parts as markers, item types ATIF has no step for).
        """
        return self.to_atif(agent_name=agent_name, agent_version=agent_version).to_otel()
