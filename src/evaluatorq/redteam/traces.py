"""Turn imported traces into dynamic red-team seed datapoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.trace_input import fetch_traces
from evaluatorq.contracts import Message, StrEnum
from evaluatorq.types import DataPoint, Trace, TraceInput

if TYPE_CHECKING:
    from collections.abc import Sequence


class TraceStart(StrEnum):
    """The point in an imported trace where a red-team run starts."""

    FIRST_USER = 'first_user'
    LAST_ASSISTANT = 'last_assistant'


def _message_dump(message: Message) -> dict[str, object]:
    """Serialize a canonical message without dropping structured fields."""
    return message.model_dump(mode='json', exclude_none=True)


def _seed_messages(trace: Trace, start_from: TraceStart) -> list[dict[str, object]]:
    """Select the replay prefix for one trace, validating its boundaries."""
    if trace.import_error:
        raise ValueError(f'Trace {trace.trace_id!r} could not be imported: {trace.import_error}')

    messages = trace.messages
    first_user = next((index for index, message in enumerate(messages) if message.role == 'user'), None)
    if first_user is None:
        raise ValueError(f'Trace {trace.trace_id!r} has no user turn to seed a red-team attack.')

    if start_from is TraceStart.FIRST_USER:
        return [_message_dump(messages[first_user])]

    last_assistant = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index].role == 'assistant'),
        None,
    )
    if last_assistant is None:
        raise ValueError(f'Trace {trace.trace_id!r} has no assistant turn for last_assistant continuation.')
    return [_message_dump(message) for message in messages[: last_assistant + 1]]


def _seed_datapoint(trace: Trace, start_from: TraceStart) -> DataPoint:
    """Project one trace through the shared trace datapoint mapping."""
    datapoint = trace.to_datapoint()
    datapoint.inputs['trace_seed_messages'] = _seed_messages(trace, start_from)
    datapoint.inputs['trace_start_from'] = start_from.value
    return datapoint


async def datapoints_from_traces(
    source: TraceInput | Sequence[Trace],
    *,
    start_from: TraceStart | str = TraceStart.FIRST_USER,
) -> list[DataPoint]:
    """Build dynamic red-team seed rows from imported traces.

    ``first_user`` seeds only the first user turn. ``last_assistant`` preserves
    the imported transcript through its final assistant turn so a caller-owned
    target can continue that conversation.
    """
    traces = await fetch_traces(source) if isinstance(source, TraceInput) else list(source)
    replay_point = TraceStart(start_from)
    return [_seed_datapoint(trace, replay_point) for trace in traces]


__all__ = ['TraceStart', 'datapoints_from_traces']
