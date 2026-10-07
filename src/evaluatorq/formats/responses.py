"""ResponsesConversation: an agent run as OpenAI Responses items."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast, get_args, get_origin

from loguru import logger
from openai.types.responses import Response, ResponseInputItem, ResponseOutputItem
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, ValidationInfo, field_validator

from evaluatorq.formats._ids import content_seed, stable_hex
from evaluatorq.formats._shared import atif_tool_arguments, json_arguments_text
from evaluatorq.openresponses.otel_messages import RESPONSES_ITEM_TYPES

if TYPE_CHECKING:
    from evaluatorq.formats.atif import AtifTrajectory
    from evaluatorq.formats.chat import ChatConversation
    from evaluatorq.formats.otel import OtelTrace


def _literal_string_values(annotation: Any) -> set[str]:
    """Read string values from a Literal annotation, allowing Annotated and union wrappers."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _literal_string_values(args[0]) if args else set()
    if origin is Literal:
        return {value for value in args if isinstance(value, str)}
    if args:
        return set().union(*(_literal_string_values(arg) for arg in args))
    return set()


def _sdk_output_item_types(output_item_type: Any) -> frozenset[str]:
    """Discover output item discriminators exposed by this OpenAI SDK, skipping unknown union members."""
    item_types: set[str] = set()
    pending = [output_item_type]
    while pending:
        member = pending.pop()
        origin = get_origin(member)
        args = get_args(member)
        if origin is Annotated:
            if args:
                pending.append(args[0])
            continue
        if args:
            pending.extend(args)
            continue
        model_fields = getattr(member, 'model_fields', None)
        if not isinstance(model_fields, Mapping):
            continue
        type_field = model_fields.get('type')
        annotation = getattr(type_field, 'annotation', None)
        item_types.update(_literal_string_values(annotation))
    return frozenset(item_types)


# Keep output recognition aligned with the installed OpenAI SDK. An SDK union
# member we cannot inspect stays on the raw preservation path. Result items
# ending in `_output` are transcript inputs, not model-response boundaries;
# compaction has its own system-step mapping and is excluded from model outputs.
_SDK_OUTPUT_TYPES = _sdk_output_item_types(ResponseOutputItem)
_RESPONSE_RESULT_ITEM_TYPES = frozenset(kind for kind in _SDK_OUTPUT_TYPES if kind.endswith('_output'))
_RESTORABLE_OUTPUT_TYPES = _SDK_OUTPUT_TYPES - _RESPONSE_RESULT_ITEM_TYPES - {'compaction'}
_KNOWN_ITEM_TYPES = RESPONSES_ITEM_TYPES | _SDK_OUTPUT_TYPES | {'message', 'compaction'}
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
    if kind == 'function_call':
        call_id = item.get('call_id')
        if not isinstance(call_id, str) or not call_id:
            return False
    return (isinstance(kind, str) and kind in _RESTORABLE_OUTPUT_TYPES and kind != 'message') or (
        kind == 'message' and item.get('role') == 'assistant'
    )


def is_sdk_output_item(item: dict[str, Any]) -> bool:
    """Whether the installed SDK has a typed output model for this item discriminator."""
    kind = item_type(item)
    return isinstance(kind, str) and kind in _SDK_OUTPUT_TYPES


def is_raw_tool_call(item: dict[str, Any]) -> bool:
    """Whether an opaque/future call item should be preserved outside the SDK output union."""
    kind = item_type(item)
    return isinstance(kind, str) and kind.endswith('_call') and not is_output_item(item)


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
                logger.warning(
                    'Responses {!r} item does not validate as a Responses item; keeping it. ({})',
                    kind,
                    type(exc).__name__,
                )
        return items

    @field_validator('responses')
    @classmethod
    def _outputs_match_items(cls, responses: list[Response] | None, info: ValidationInfo) -> list[Response] | None:
        items: list[dict[str, Any]] | None = info.data.get('items')
        if responses is None or items is None:
            return responses
        unsupported = sorted({
            item.type
            for response in responses
            for item in response.output
            if item.type not in _RESTORABLE_OUTPUT_TYPES and not (item.type == 'message' and item.role == 'assistant')
        })
        if unsupported:
            msg = f'Response.output item types {unsupported} are not supported by ResponsesConversation.responses'
            raise ValueError(msg)
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
