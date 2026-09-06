"""Conversion functions from SimulationResult to OpenResponses format."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_core import to_jsonable_python

from evaluatorq.contracts import content_to_text
from evaluatorq.openresponses.convert_models import (
    FunctionCall,
    FunctionCallOutput,
    FunctionCallStatus,
    IncompleteDetails,
    InputFileContent,
    InputImageContent,
    InputTextContent,
    InputTokensDetails,
    Message,
    MessageRole,
    MessageStatus,
    OutputTextContent,
    OutputTokensDetails,
    Usage,
)
from evaluatorq.openresponses.input_items import responses_function_call_item_id

# Message.content accepts any content part; simulation only emits text parts, but
# the variable annotation must match the (invariant) list element union.
_ContentList = list[InputTextContent | InputImageContent | InputFileContent | OutputTextContent]

if TYPE_CHECKING:
    from evaluatorq.simulation.types import SimulationResult


def _generate_item_id(prefix: str) -> str:
    return f'{prefix}_{uuid.uuid4().hex[:24]}'


def _jsonable(value: Any, key: str) -> Any:
    """Serialise a free-form ``SimulationResult.metadata`` value for the trace payload.

    A type pydantic cannot serialise degrades to its ``repr()`` rather than raising —
    one exotic value must not take down the conversion of an otherwise complete run —
    and the fallback logs a warning naming the key and the type, so the degradation is
    not silent.
    """

    def _unserialisable(unknown: object) -> str:
        logger.warning(
            f'simulation metadata key {key!r} holds a {type(unknown).__name__} that cannot be '
            f'serialised to JSON; publishing its repr() instead'
        )
        return repr(unknown)

    return to_jsonable_python(value, fallback=_unserialisable)


def to_open_responses(
    result: SimulationResult,
    model: str = 'simulation',
) -> dict[str, Any]:
    """Convert a SimulationResult to OpenResponses format.

    Mapping:

    - messages with role "user"      -> input[] as Message with input_text content
    - messages with role "assistant"  -> output[] as Message with output_text content
    - messages with role "system"     -> input[] as Message with input_text content
    - token_usage                     -> Usage
    - terminated_by                   -> status
    - goal_achieved, rules_broken, criteria_results, turn_metrics -> metadata

    ``criteria_results`` never travels alone: ``criteria_verified``,
    ``criteria_meta``, ``criteria_errors``, ``scorer_errors`` and ``datapoint_id``
    are always present in the metadata block, ``None`` when the run has none.

    ``criteria_errors`` and ``scorer_errors`` report the state of ``result.metadata``
    **at conversion time**, which is not the same as the end of the run. On the
    evaluatorq job path (``simulation/api.py``) the conversion happens inside the job,
    before the scorers run, and both keys are written during scoring — so on that path
    they are always ``None`` and a ``null`` there means "not yet known", not "no errors".
    Only the persisted-results export path (``eq sim`` report generation) converts after
    scoring and can show them populated.

    ``token_usage_known`` sits beside ``usage`` for the same reason ``criteria_verified``
    sits beside ``criteria_results``: ``False`` means the usage numbers are partial and
    must be read as unknown, never as a cheap run.
    """
    now = int(time.time())

    input_items: list[dict[str, Any]] = []
    output_items: list[dict[str, Any]] = []

    for msg in result.messages:
        if msg.role in ('user', 'system'):
            in_content: _ContentList = [InputTextContent(type='input_text', text=content_to_text(msg.content))]
            message = Message(
                type='message',
                id=_generate_item_id('msg'),
                role=MessageRole(msg.role),
                status=MessageStatus.completed,
                content=in_content,
            )
            input_items.append(message.model_dump(mode='json'))
        elif msg.role == 'assistant':
            # An assistant turn can carry text and/or tool_calls. Emit the text
            # message when there is content, then a function_call item per call
            # (separate Responses output items). A tool-only turn skips the empty
            # text message. Mirrors the langchain integration's mapping.
            if msg.content:
                out_content: _ContentList = [OutputTextContent(text=content_to_text(msg.content), annotations=[])]
                message = Message(
                    type='message',
                    id=_generate_item_id('msg'),
                    role=MessageRole.assistant,
                    status=MessageStatus.completed,
                    content=out_content,
                )
                output_items.append(message.model_dump(mode='json'))
            for tc in msg.tool_calls or []:
                function_call = FunctionCall(
                    type='function_call',
                    id=responses_function_call_item_id(tc.item_id) or _generate_item_id('fc'),
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                    status=FunctionCallStatus.completed,
                )
                output_items.append(function_call.model_dump(mode='json'))
        elif msg.role == 'tool':
            function_call_output = FunctionCallOutput(
                type='function_call_output',
                id=_generate_item_id('fco'),
                call_id=msg.tool_call_id or '',
                output=content_to_text(msg.content),
                status=FunctionCallStatus.completed,
            )
            output_items.append(function_call_output.model_dump(mode='json'))

    # Map terminated_by to status
    if result.terminated_by.value == 'judge':
        status = 'completed'
    elif result.terminated_by.value == 'error':
        status = 'failed'
    else:
        status = 'incomplete'

    incomplete_details = (
        IncompleteDetails(reason=f'{result.terminated_by.value}: {result.reason}').model_dump(mode='json')
        if status == 'incomplete'
        else None
    )

    # Build usage from token_usage
    usage_data = None
    if result.token_usage.total_tokens > 0:
        usage_data = Usage(
            input_tokens=result.token_usage.input_tokens,
            output_tokens=result.token_usage.output_tokens,
            total_tokens=result.token_usage.total_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=result.token_usage.cached_tokens),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=result.token_usage.reasoning_tokens),
        ).model_dump(mode='json')

    metadata: dict[str, Any] = {
        'framework': 'simulation',
        'goal_achieved': result.goal_achieved,
        'goal_completion_score': result.goal_completion_score,
        'terminated_by': result.terminated_by.value,
        'reason': result.reason,
        'turn_count': result.turn_count,
        'rules_broken': result.rules_broken,
        # criteria_results is the lossy dict; criteria_verified is its warning label
        # ("False means the whole block is unaudited: treat criteria_results as unknown,
        # not met"), and criteria_meta is the only place per-criterion `audited` survives.
        # All three are emitted unconditionally — a missing key reads as a clean run.
        'criteria_verified': result.criteria_verified,
        'criteria_meta': _jsonable(result.metadata.get('criteria_meta'), 'criteria_meta'),
        'criteria_errors': _jsonable(result.metadata.get('criteria_errors'), 'criteria_errors'),
        'scorer_errors': _jsonable(result.metadata.get('scorer_errors'), 'scorer_errors'),
        'datapoint_id': _jsonable(result.metadata.get('datapoint_id'), 'datapoint_id'),
    }
    # `is not None`, not truthiness: {} is a scenario with zero criteria, not an absent block.
    if result.criteria_results is not None:
        metadata['criteria_results'] = result.criteria_results
    if result.turn_metrics:
        metadata['turn_metrics'] = [tm.model_dump(mode='json') for tm in result.turn_metrics]

    return {
        'id': _generate_item_id('resp'),
        'object': 'response',
        'created_at': now,
        'completed_at': now if status == 'completed' else None,
        'status': status,
        'incomplete_details': incomplete_details,
        'model': model,
        'previous_response_id': None,
        'instructions': None,
        'input': input_items,
        'output': output_items,
        'error': {'message': result.reason} if result.terminated_by.value == 'error' else None,
        'tools': [],
        'tool_choice': 'auto',
        'truncation': 'disabled',
        'parallel_tool_calls': False,
        'text': {'format': {'type': 'text'}},
        'top_p': 1,
        'presence_penalty': 0,
        'frequency_penalty': 0,
        'top_logprobs': 0,
        'temperature': 1,
        'reasoning': None,
        'user': None,
        'usage': usage_data,
        'token_usage_known': result.token_usage_known,
        'max_output_tokens': None,
        'max_tool_calls': None,
        'store': False,
        'background': False,
        'service_tier': 'default',
        'metadata': metadata,
        'safety_identifier': None,
        'prompt_cache_key': None,
    }
