"""Turn imported traces into dynamic red-team seed datapoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import ValidationError

from evaluatorq.common.trace_input import fetch_traces, partition_traces
from evaluatorq.contracts import Message, StrEnum
from evaluatorq.types import DataPoint, Trace, TraceInput

if TYPE_CHECKING:
    from collections.abc import Sequence


class TraceStart(StrEnum):
    """The point in an imported trace where a red-team run starts."""

    FIRST_USER = 'first_user'
    LAST_ASSISTANT = 'last_assistant'


# The two keys that carry trace-replay state on a `DataPoint.inputs` mapping.
# Exported so the producer (`_seed_datapoint`) and both readers (the
# `red_team()` API-boundary precheck in `runner.py` and the per-attack read in
# `adaptive/pipeline.py`) share one name instead of three copies that can drift.
TRACE_SEED_MESSAGES_KEY = 'trace_seed_messages'
TRACE_START_FROM_KEY = 'trace_start_from'


def _message_dump(message: Message) -> dict[str, object]:
    """Serialize a canonical message without dropping structured fields."""
    return message.model_dump(mode='json', exclude_none=True)


def _seed_messages(trace: Trace, start_from: TraceStart) -> list[dict[str, object]] | None:
    """Select the replay prefix for one trace, or ``None`` when it cannot seed one.

    A missing user (or, for ``last_assistant``, assistant) turn is a per-trace
    skip, not a batch failure — the caller decides whether "some traces had no
    seedable turn" is fatal (see `datapoints_from_traces`).
    """
    messages = trace.messages
    first_user = next((index for index, message in enumerate(messages) if message.role == 'user'), None)
    if first_user is None:
        logger.warning('Trace {!r} has no user turn to seed a red-team attack; skipping.', trace.trace_id)
        return None

    if start_from is TraceStart.FIRST_USER:
        return [_message_dump(messages[first_user])]

    last_assistant = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index].role == 'assistant'),
        None,
    )
    if last_assistant is None:
        logger.warning('Trace {!r} has no assistant turn for last_assistant continuation; skipping.', trace.trace_id)
        return None
    return [_message_dump(message) for message in messages[: last_assistant + 1]]


def _seed_datapoint(trace: Trace, start_from: TraceStart) -> DataPoint | None:
    """Project one trace through the shared trace datapoint mapping, or ``None`` to skip it."""
    seed_messages = _seed_messages(trace, start_from)
    if seed_messages is None:
        return None
    datapoint = trace.to_datapoint()
    datapoint.inputs[TRACE_SEED_MESSAGES_KEY] = seed_messages
    datapoint.inputs[TRACE_START_FROM_KEY] = start_from.value
    return datapoint


async def datapoints_from_traces(
    source: TraceInput | Sequence[Trace],
    *,
    start_from: TraceStart | str = TraceStart.FIRST_USER,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[DataPoint]:
    """Build dynamic red-team seed rows from imported traces.

    ``first_user`` seeds only the first user turn. ``last_assistant`` preserves
    the imported transcript through its final assistant turn so a caller-owned
    target can continue that conversation.

    ``api_key`` / ``base_url`` are forwarded to `fetch_traces` for a non-default
    workspace or a self-hosted deployment; ignored when ``source`` is already a
    sequence of traces.

    A trace with an unreadable import (`Trace.import_error`) or with no user
    turn to seed (or, for ``last_assistant``, no assistant turn) is skipped
    with a logged warning rather than aborting the whole batch — the other
    traces in the batch are still usable. This raises only when nothing in the
    batch produced a usable seed: either every trace failed to import, or every
    importable trace had no seedable turn.

    This is a different function from `evaluatorq.simulation.datapoints_from_traces`:
    that one bills a summarize call and an inference call per trace and returns
    `SimulationDatapoint` rows for the agent simulation surface. This one makes
    no LLM calls and returns red-team seed `DataPoint` rows.
    """
    if isinstance(source, TraceInput):
        traces = await fetch_traces(source, api_key=api_key, base_url=base_url)
    else:
        traces = list(source)
    usable, failed = partition_traces(traces, caller='redteam.datapoints_from_traces')
    if not usable:
        if failed:
            raise ValueError(f'All {len(failed)} imported trace(s) failed to import; no trace seeds could be built.')
        raise ValueError('No traces were supplied; no trace seeds could be built.')

    replay_point = TraceStart(start_from)
    seeds = [datapoint for trace in usable if (datapoint := _seed_datapoint(trace, replay_point)) is not None]
    if not seeds:
        raise ValueError(
            f'None of the {len(usable)} imported trace(s) had a seedable turn for start_from={replay_point.value!r}.'
        )
    return seeds


def parse_trace_seed(
    inputs: Mapping[str, object],
    *,
    label: str,
    required: bool = True,
) -> tuple[list[Message], TraceStart] | tuple[None, None]:
    """Parse and validate one datapoint's trace-seed replay metadata.

    Canonical for every reader of ``trace_seed_messages`` / ``trace_start_from``:
    the API-boundary precheck in ``red_team()`` and the per-attack read in the
    dynamic pipeline. Before this, the precheck required both keys while the
    pipeline silently defaulted a missing ``trace_start_from`` to
    ``first_user`` — a ``last_assistant`` transcript that reached the pipeline
    without going through the precheck replayed as one opening turn with the
    imported context dropped, and no error.

    ``required=True`` (the default, used where every datapoint is meant to be a
    trace seed) raises when ``trace_seed_messages`` is absent. ``required=False``
    (the pipeline's per-row read, where most rows are not trace seeds) returns
    ``(None, None)`` in that case instead. Either way, once ``trace_seed_messages``
    is present, ``trace_start_from`` is required — it is never defaulted.
    """
    if TRACE_SEED_MESSAGES_KEY not in inputs:
        if required:
            raise ValueError(
                f'{label} is missing trace seed metadata: '
                f"expected '{TRACE_SEED_MESSAGES_KEY}' and '{TRACE_START_FROM_KEY}'."
            )
        return None, None

    raw_messages = inputs[TRACE_SEED_MESSAGES_KEY]
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f'{label}.{TRACE_SEED_MESSAGES_KEY} must be a non-empty list of Message mappings.')

    parsed_messages: list[Message] = []
    for message_index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping):
            raise TypeError(f'{label}.{TRACE_SEED_MESSAGES_KEY}[{message_index}] must be a Message mapping.')
        try:
            parsed_messages.append(Message.model_validate(message))
        except ValidationError as exc:
            raise ValueError(
                f'{label}.{TRACE_SEED_MESSAGES_KEY}[{message_index}] is not parseable as a Message: {exc}'
            ) from exc

    if TRACE_START_FROM_KEY not in inputs:
        raise ValueError(
            f'{label} has {TRACE_SEED_MESSAGES_KEY} but no {TRACE_START_FROM_KEY}; '
            'it is never defaulted — a default would silently replay a last_assistant '
            'transcript as a single first_user turn and drop the imported context.'
        )
    try:
        start_from = TraceStart(inputs[TRACE_START_FROM_KEY])
    except ValueError as exc:
        raise ValueError(f'{label}.{TRACE_START_FROM_KEY} is not a valid TraceStart value.') from exc

    if start_from is TraceStart.LAST_ASSISTANT and parsed_messages[-1].role != 'assistant':
        # The replay prefix is sent as-is and the attack is appended as a user
        # turn, so a prefix ending in a user or tool message produces two
        # consecutive user turns (or an unanswered tool call) for a
        # caller-owned-history target.
        raise ValueError(
            f'{label}.{TRACE_SEED_MESSAGES_KEY} must end with an assistant message when '
            f"{TRACE_START_FROM_KEY} is '{TraceStart.LAST_ASSISTANT.value}'; "
            f'it ends with a {parsed_messages[-1].role} message.'
        )

    return parsed_messages, start_from


__all__ = [
    'TRACE_SEED_MESSAGES_KEY',
    'TRACE_START_FROM_KEY',
    'TraceStart',
    'datapoints_from_traces',
    'parse_trace_seed',
]
