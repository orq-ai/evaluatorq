import asyncio
import inspect
import os
from collections.abc import Awaitable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from .common.llm_limit import llm_concurrency_limit
from .common.messages import coerce_content_text
from .common.parallelism import resolve_datapoint_parallelism
from .common.trace_input import fetch_traces
from .fetch_data import (
    fetch_dataset_batches,
    fetch_experiment_datapoints,
    setup_orq_client,
)
from .processings import process_data_point
from .progress import Phase, ProgressService, safe_update_progress, with_progress
from .send_results import send_results_to_orq
from .table_display import display_results_table
from .tracing import capture_parent_context, tracing_session
from .types import (
    DataPoint,
    DataPointComplete,
    DataPointInput,
    DataPointResult,
    DatasetIdInput,
    Evaluator,
    EvaluatorParams,
    EvaluatorqResult,
    ExperimentInput,
    Job,
    TraceInput,
)

if TYPE_CHECKING:
    from orq_ai_sdk import Orq

    from .tracing.context import TracingContext


class _StreamingEvaluationError(RuntimeError):
    """Report multiple failures collected from a streaming evaluation's tasks."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = errors
        details = '; '.join(f'{type(error).__name__}: {error}' for error in errors)
        super().__init__(f'Streaming evaluation failed with {len(errors)} errors: {details}')


async def _notify_datapoint_complete(
    callback: DataPointComplete | None,
    results: list[DataPointResult],
) -> None:
    if callback is None:
        return
    for result in results:
        returned = callback(result)
        if inspect.isawaitable(returned):
            await returned


def check_pass_failures(results: EvaluatorqResult, *, treat_errors_as_failure: bool = False) -> bool:
    """
    Check if any evaluator returned pass_=False.

    Args:
        results: The evaluation results to check
        treat_errors_as_failure: When True, a row whose datapoint or job errored (e.g. a
            missing recorded response in no-inference mode), or whose evaluator errored
            (e.g. every judge call raised), also counts as a failure. Without this, an
            errored job has no evaluator scores and an errored evaluator leaves ``pass_``
            unset, so both would be invisible here, letting a run with no usable
            responses or no usable scores exit successfully. Neither errored path is scored:
            a job that raised has no output to score, and one that reported its own
            failure through a top-level ``error`` key keeps its output but skips its
            evaluators — so this flag is the only thing that fails such a row.

    Returns:
        True if any evaluator failed (pass_=False), False otherwise
    """
    for data_point_result in results:
        if treat_errors_as_failure and data_point_result.error:
            return True
        if data_point_result.job_results:
            for job_result in data_point_result.job_results:
                if treat_errors_as_failure and job_result.error:
                    return True
                if job_result.evaluator_scores:
                    for evaluator_score in job_result.evaluator_scores:
                        if treat_errors_as_failure and evaluator_score.error:
                            return True
                        if evaluator_score.score.pass_ is False:
                            return True
    return False


def extract_recorded_response(messages: Any) -> str:
    """Return the last recorded assistant response from a row's ``messages`` column.

    Used by the no-inference path: the pre-recorded conversation already contains the
    response we want to score, so we surface the last assistant message's text rather
    than generating a new one.

    Raises:
        ValueError: if ``messages`` is empty or holds no assistant message with text.
    """
    if not messages:
        raise ValueError(
            "inference=False requires a recorded response in the 'messages' column, but this row has no messages."
        )
    for message in reversed(list(messages)):
        role = message.get('role') if isinstance(message, dict) else getattr(message, 'role', None)
        if role != 'assistant':
            continue
        content = message.get('content') if isinstance(message, dict) else getattr(message, 'content', None)
        text = coerce_content_text(content)
        if text.strip():
            return text
    raise ValueError(
        "inference=False requires a recorded response in the 'messages' column, "
        'but this row has no assistant message with text content.'
    )


# Must be async to satisfy the Job protocol (an Awaitable-returning callable), even
# though replaying a recorded response involves no awaiting.
async def _replay_recorded_response(data_point: DataPoint, _row_index: int) -> dict[str, Any]:  # noqa: RUF029
    """Synthetic job for the no-inference path: replays the pre-recorded response."""
    if import_error := data_point.inputs.get('trace_import_error'):
        raise ValueError(f'The source trace could not be imported: {import_error}')
    if 'recorded_output' in data_point.inputs:
        return {'name': 'recorded', 'output': data_point.inputs['recorded_output']}
    response = extract_recorded_response(data_point.inputs.get('messages'))
    return {'name': 'recorded', 'output': response}


@dataclass
class _StreamingProgress:
    """Counters the streaming fetch, the datapoint workers and the poller share.

    One mutable object rather than three boxed locals: the two helpers below run
    outside ``evaluatorq``'s frame, so they cannot rebind its locals, and a
    single owner beats a mix of one-element lists and a stringly-keyed dict.
    Mutated from several concurrent tasks, but only by ``+= 1`` on a distinct
    field per writer and a single set of ``stop``, all inside one event loop.
    """

    total: int = 0
    processed: int = 0
    stop: bool = False


async def _process_with_semaphore(
    index: int,
    data_promise: DataPoint,
    data_point_semaphore: asyncio.Semaphore,
    jobs: list[Job],
    evaluators_list: list[Evaluator],
    datapoint_parallelism: int,
    state: _StreamingProgress,
    tracing_context: 'TracingContext | None',
    on_datapoint_complete: DataPointComplete | None,
) -> list[Any]:
    """Process one datapoint under the concurrency bound, then count it as done."""
    async with data_point_semaphore:
        result = await process_data_point(
            data_promise,
            index,
            jobs,
            evaluators_list,
            datapoint_parallelism,
            None,  # Don't pass progress in streaming mode - use polling instead
            tracing_context,
        )
        await _notify_datapoint_complete(on_datapoint_complete, result)
        state.processed += 1
        return result


async def _poll_progress(progress: ProgressService, state: _StreamingProgress) -> None:
    """Redraw the progress display until ``state.stop`` is set by the fetch loop.

    A display fault is logged and ends polling; it never fails a run that has
    otherwise completed.
    """
    try:
        while not state.stop:
            await progress.update_progress(
                total_data_points=state.total,
                current_data_point=state.processed,
                phase=Phase.PROCESSING if state.processed > 0 else Phase.FETCHING,
            )
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - display faults must not fail a completed run
        logger.warning(
            'Progress polling stopped after display failure: {}: {}',
            type(exc).__name__,
            exc,
        )


@dataclass(frozen=True)
class _StreamingInputs:
    progress: ProgressService
    datapoint_parallelism: int
    orq_client: 'Orq'
    dataset_id: str
    include_messages: bool
    jobs: list[Job]
    evaluators_list: list[Evaluator]
    tracing_context: 'TracingContext'
    on_datapoint_complete: DataPointComplete | None


@dataclass(frozen=True)
class _EvaluationInputs:
    params: EvaluatorParams | dict[str, Any] | None
    data: DatasetIdInput | ExperimentInput | Sequence[Awaitable[DataPoint] | DataPointInput] | None
    jobs: list[Job] | None
    evaluators: list[Evaluator] | None
    datapoint_parallelism: int
    llm_parallelism: int | None
    print_results: bool
    description: str | None
    path: str | None
    inference: bool
    single_trace: bool
    on_datapoint_complete: DataPointComplete | None


@dataclass(frozen=True)
class _ResolvedEvaluationInputs:
    data: DatasetIdInput | ExperimentInput | Sequence[Awaitable[DataPoint] | DataPointInput]
    inference: bool
    jobs: list[Job]
    evaluators_list: list[Evaluator]
    datapoint_parallelism: int
    llm_parallelism: int | None
    print_results: bool
    description: str | None
    path: str | None
    single_trace: bool
    on_datapoint_complete: DataPointComplete | None


def _normalise_params(inputs: _EvaluationInputs) -> _ResolvedEvaluationInputs:
    """Validate ``evaluatorq``'s two calling conventions into one settled shape.

    The first phase of a run: either a ``params`` object (validated when handed a
    dict) or the loose keyword arguments, never a mix, and one of the two is
    required. Under ``inference=False`` any caller-supplied ``jobs`` is dropped
    with a warning and replaced by the recorded-response replay job, because that
    mode reads answers from the dataset rather than generating them.

    Returns:
        The resolved inputs every later phase reads, with ``jobs`` non-empty.

    Raises:
        ValueError: Neither ``params`` nor the ``data``/``jobs`` pair was given.
    """
    params = inputs.params
    data = inputs.data
    jobs = inputs.jobs
    evaluators = inputs.evaluators
    datapoint_parallelism = inputs.datapoint_parallelism
    llm_parallelism = inputs.llm_parallelism
    print_results = inputs.print_results
    description = inputs.description
    path = inputs.path
    inference = inputs.inference
    single_trace = inputs.single_trace
    on_datapoint_complete = inputs.on_datapoint_complete

    # Handle params dict/object vs kwargs
    if params is not None:
        # Validate params if passed as dict
        validated = EvaluatorParams.model_validate(params) if isinstance(params, dict) else params
    elif data is not None and (jobs is not None or not inference):
        # Use kwargs ('jobs' is optional when inference=False, since responses are replayed).
        validated = EvaluatorParams(
            data=data,
            jobs=jobs,
            evaluators=evaluators,
            datapoint_parallelism=datapoint_parallelism,
            llm_parallelism=llm_parallelism,
            print_results=print_results,
            description=description,
            path=path,
            inference=inference,
            single_trace=single_trace,
            on_datapoint_complete=on_datapoint_complete,
        )
    else:
        raise ValueError(
            "Either 'params' or both 'data' and 'jobs' keyword arguments are required "
            "(omit 'jobs' only when inference=False)"
        )

    # Extract validated values
    data = validated.data
    inference = validated.inference
    if inference:
        # The validator guarantees jobs is non-empty whenever inference=True.
        jobs = cast('list[Job]', validated.jobs)
    else:
        # No-inference mode: skip generation and replay each row's recorded response.
        if validated.jobs:
            logger.warning(
                "inference=False: ignoring the provided 'jobs'; responses are replayed from recorded output when "
                "available, otherwise from the 'messages' column."
            )
        jobs = [_replay_recorded_response]
    evaluators_list = validated.evaluators or []
    datapoint_parallelism = validated.datapoint_parallelism
    llm_parallelism = validated.llm_parallelism
    print_results = validated.print_results
    description = validated.description
    path = validated.path
    single_trace = validated.single_trace
    on_datapoint_complete = validated.on_datapoint_complete

    return _ResolvedEvaluationInputs(
        data=data,
        inference=inference,
        jobs=cast('list[Job]', jobs),
        evaluators_list=evaluators_list,
        datapoint_parallelism=datapoint_parallelism,
        llm_parallelism=llm_parallelism,
        print_results=print_results,
        description=description,
        path=path,
        single_trace=single_trace,
        on_datapoint_complete=on_datapoint_complete,
    )


async def _enter_single_trace(
    span_stack: AsyncExitStack,
    tracing_context: 'TracingContext',
    name: str,
    trace_type: str,
) -> None:
    """Open the one run span that all of a single-trace run's rows hang under.

    Enters the span on ``span_stack`` so it closes with the caller's exit stack,
    then writes the span's context back onto ``tracing_context.parent_context``:
    ``tracing_session`` captured the ambient context before this span existed,
    and the per-row job spans pass ``parent_context`` explicitly rather than
    reading the ambient one, so without that write they would parent elsewhere.

    Args:
        span_stack: Exit stack owning the span for the rest of the run.
        tracing_context: Context whose ``parent_context`` this rebinds in place.
        name: Run name recorded on the span.
        trace_type: Trace type recorded on the span.
    """
    from evaluatorq.tracing.spans import RunSpanOptions, with_run_span

    run_span = await span_stack.enter_async_context(
        with_run_span(
            RunSpanOptions(
                run_id=tracing_context.run_id,
                run_name=name,
                parent_context=tracing_context.parent_context,
                trace_type=trace_type,
            )
        )
    )
    # Re-point the context the per-row job spans parent to. tracing_session
    # captured the ambient context *before* this span existed, and jobs pass
    # parent_context explicitly rather than reading the ambient one.
    if run_span is not None:
        tracing_context.parent_context = await capture_parent_context()


async def _resolve_experiment_input(
    data: DatasetIdInput | ExperimentInput | Sequence[Awaitable[DataPoint] | DataPointInput] | None,
    orq_api_key: str | None,
    base_url: str | None,
) -> DatasetIdInput | Sequence[Awaitable[DataPoint] | DataPointInput] | None:
    """Turn an experiment reference into the rows it recorded, before the data phase.

    Only ``ExperimentInput`` is touched; every other source is returned unchanged
    so the caller can keep dispatching on its type. The validator already
    guarantees ``inference=False`` whenever this substitution happens.

    Returns:
        The experiment's recorded datapoints, or ``data`` untouched.

    Raises:
        ValueError: An experiment was requested without ``orq_api_key``.
    """
    # Experiment source (no-inference only): replace the input with the experiment's
    # recorded responses, then fall through to the in-memory data path below. The
    # validator already guarantees inference=False here.
    if isinstance(data, ExperimentInput):
        if not orq_api_key:
            raise ValueError('ORQ_API_KEY environment variable must be set to load responses from an Orq experiment.')
        data = await fetch_experiment_datapoints(
            orq_api_key,
            data.experiment_id,
            data.run_id,
            base_url=base_url,
        )
    return data


def _resolve_streaming_inputs(
    data: DatasetIdInput,
    orq_api_key: str | None,
    *,
    inference: bool,
) -> tuple['Orq', str, bool]:
    """Build the Orq client and fetch settings the streaming data phase needs.

    ``include_messages`` is forced on under ``inference=False`` regardless of what
    the dataset input asked for, since that mode replays the recorded responses
    and they arrive only in the ``messages`` column; the override is logged.

    Returns:
        The client, the dataset id, and the ``include_messages`` flag to fetch with.

    Raises:
        ValueError: No ``orq_api_key``, so no client could be built.
    """
    orq_client: Orq | None = None

    if orq_api_key:
        orq_client = setup_orq_client(orq_api_key)

    if not orq_api_key or not orq_client:
        raise ValueError('ORQ_API_KEY environment variable must be set to fetch datapoints from Orq platform.')
    dataset_id = data.dataset_id
    # No-inference mode needs the recorded responses, which arrive in the
    # 'messages' column only when include_messages is enabled.
    include_messages = data.include_messages or not inference
    if include_messages and not data.include_messages:
        logger.debug(
            'inference=False: enabling include_messages to load recorded responses '
            'despite include_messages=False on the dataset input.'
        )
    return orq_client, dataset_id, include_messages


@dataclass(frozen=True)
class _EvaluationFinishInputs:
    print_results: bool
    orq_api_key: str | None
    send_results: bool
    name: str
    description: str | None
    dataset_id: str | None
    results: EvaluatorqResult
    start_time: datetime
    path: str | None
    base_url: str | None
    experiment_url_out: list[str] | None


async def _finish_evaluation(inputs: _EvaluationFinishInputs) -> None:
    """Display and upload a finished run's results, the last phase of ``evaluatorq``.

    Prints the table when asked, then uploads to Orq only when both an API key and
    ``send_results`` are present. Callers that opted in by passing a sink list get
    the created experiment's URL appended to ``inputs.experiment_url_out`` — that
    list is the only value this writes back, since simulation persists the URL on
    its ``SimulationRun`` report. A run with no sink, no key or no URL writes
    nothing and is not an error.
    """
    print_results = inputs.print_results
    orq_api_key = inputs.orq_api_key
    send_results = inputs.send_results
    name = inputs.name
    description = inputs.description
    dataset_id = inputs.dataset_id
    results = inputs.results
    start_time = inputs.start_time
    path = inputs.path
    base_url = inputs.base_url
    experiment_url_out = inputs.experiment_url_out

    # Display results table
    if print_results:
        await display_results_table(results)

    # Upload results to Orq platform if API key is available
    if orq_api_key and send_results:
        upload_response = await send_results_to_orq(
            orq_api_key,
            name,
            description,
            dataset_id,
            results,
            start_time,
            datetime.now(timezone.utc),
            path=path,
            base_url=base_url,
        )
        # Hand the created experiment's URL back to callers that opted in with a
        # sink list (e.g. simulation persists it on the SimulationRun report).
        if experiment_url_out is not None and upload_response is not None and upload_response.experiment_url:
            experiment_url_out.append(upload_response.experiment_url)


async def _run_streaming_evaluation(inputs: _StreamingInputs) -> EvaluatorqResult:
    """Run the processing phase against a dataset fetched batch by batch.

    Each datapoint starts as soon as its batch arrives rather than after the
    fetch completes, so a ``_StreamingProgress`` carries the counters between the
    fetch loop, the workers and the polling task, and the total is only final
    once the fetch ends. A fetch failure cancels the in-flight datapoints; those
    cancellations are ours and are not reported, whereas a caller's cancellation
    wins outright and the task errors it raced are logged and discarded.

    Returns:
        Every row's results, flattened in datapoint order.

    Raises:
        BaseException: The single fetch or task error, re-raised as it arrived.
        _StreamingEvaluationError: Several failures, collected together.
    """
    progress = inputs.progress
    datapoint_parallelism = inputs.datapoint_parallelism
    orq_client = inputs.orq_client
    dataset_id = inputs.dataset_id
    include_messages = inputs.include_messages
    jobs = inputs.jobs
    evaluators_list = inputs.evaluators_list
    tracing_context = inputs.tracing_context
    on_datapoint_complete = inputs.on_datapoint_complete

    all_results: EvaluatorqResult = []
    processing_tasks: list[asyncio.Task[list[Any]]] = []
    datapoint_index = 0

    # Counters shared with the datapoint workers and the poller.
    state = _StreamingProgress()

    # Semaphore bounding concurrent datapoints
    data_point_semaphore = asyncio.Semaphore(datapoint_parallelism)

    # Initialize progress with unknown total (streaming mode)
    await safe_update_progress(
        progress,
        operation='streaming initialization',
        total_data_points=0,
        current_data_point=0,
        phase=Phase.FETCHING,
    )

    # Start a background task to poll and update progress
    polling_task = asyncio.create_task(_poll_progress(progress, state))
    fetch_error: BaseException | None = None
    cancelled_for_fetch: set[asyncio.Task[Any]] = set()

    try:
        # Fetch and process batches
        async for batch in fetch_dataset_batches(orq_client, dataset_id, include_messages=include_messages):
            state.total += len(batch.datapoints)

            # Start processing this batch immediately
            for datapoint in batch.datapoints:
                task = asyncio.create_task(
                    _process_with_semaphore(
                        datapoint_index,
                        datapoint,
                        data_point_semaphore,
                        jobs,
                        evaluators_list,
                        datapoint_parallelism,
                        state,
                        tracing_context,
                        on_datapoint_complete,
                    )
                )
                processing_tasks.append(task)
                datapoint_index += 1

    except asyncio.CancelledError as exc:
        fetch_error = exc
    except Exception as exc:  # noqa: BLE001 - collect fetch failures with task failures
        fetch_error = exc
    finally:
        # A fetch failure also cancels in-flight processing.
        state.stop = True
        if fetch_error is not None:
            for task in processing_tasks:
                if not task.done():
                    task.cancel()
                    cancelled_for_fetch.add(task)
        _ = polling_task.cancel()
        processing_results = await asyncio.gather(
            *processing_tasks,
            return_exceptions=True,
        )
        # Deliberately cancelled above; poll_progress logs the rest.
        _ = await asyncio.gather(polling_task, return_exceptions=True)

    # Tasks we cancelled ourselves are not errors the caller should see.
    task_errors = [
        result
        for task, result in zip(processing_tasks, processing_results, strict=True)
        if isinstance(result, BaseException)
        and not (isinstance(result, asyncio.CancelledError) and task in cancelled_for_fetch)
    ]
    if isinstance(fetch_error, asyncio.CancelledError):
        for error in task_errors:
            logger.warning(
                'Discarding processing task error while caller cancellation wins: {}: {}',
                type(error).__name__,
                error,
            )
        raise fetch_error

    errors = ([fetch_error] if fetch_error is not None else []) + task_errors
    if errors:
        if len(errors) == 1:
            raise errors[0]
        raise _StreamingEvaluationError(errors)
    results_nested = cast('list[list[Any]]', processing_results)

    # Final progress update
    await safe_update_progress(
        progress,
        operation='streaming final update',
        total_data_points=state.total,
        current_data_point=state.processed,
        phase=Phase.PROCESSING,
    )

    # Flatten results
    for result_list in results_nested:
        all_results.extend(result_list)

    return all_results


async def _run_in_memory_evaluation(
    data_promises: list[DataPoint],
    progress: ProgressService,
    datapoint_parallelism: int,
    jobs: list[Job],
    evaluators_list: list[Evaluator],
    tracing_context: 'TracingContext',
    on_datapoint_complete: DataPointComplete | None,
) -> EvaluatorqResult:
    """Run the processing phase against datapoints already held in memory.

    The non-streaming counterpart: the total is known up front, so the progress
    service is updated directly by each row instead of by a polling task, and the
    rows are gathered in one call rather than accumulated batch by batch. A row
    that raises propagates immediately; there is no error collection here.

    Returns:
        Every row's results, flattened in datapoint order.
    """
    # Initialize progress
    await safe_update_progress(
        progress,
        operation='evaluation initialization',
        total_data_points=len(data_promises),
        current_data_point=0,
        phase=Phase.INITIALIZING,
    )

    # Process data points with controlled concurrency
    data_point_semaphore = asyncio.Semaphore(datapoint_parallelism)

    async def process_with_semaphore(index: int, data_promise: Awaitable[DataPoint] | DataPoint) -> list[Any]:
        async with data_point_semaphore:
            result = await process_data_point(
                data_promise,
                index,
                jobs,
                evaluators_list,
                datapoint_parallelism,
                progress,
                tracing_context,
            )
            await _notify_datapoint_complete(on_datapoint_complete, result)
            return result

    tasks = [
        asyncio.create_task(process_with_semaphore(index, data_promise))
        for index, data_promise in enumerate(data_promises)
    ]

    # Gather all results
    try:
        results_nested = await asyncio.gather(*tasks)
    except BaseException:
        # Keep this run's work inside its tracing/client contexts even
        # when a callback fails or the caller cancels the evaluation.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    # Flatten results
    results: EvaluatorqResult = []
    for result_list in results_nested:
        results.extend(result_list)

    return results


async def evaluatorq(
    name: str,
    params: EvaluatorParams | dict[str, Any] | None = None,
    *,
    data: DatasetIdInput | ExperimentInput | TraceInput | Sequence[Awaitable[DataPoint] | DataPointInput] | None = None,
    jobs: list[Job] | None = None,
    evaluators: list[Evaluator] | None = None,
    datapoint_parallelism: int | None = None,
    llm_parallelism: int | None = None,
    parallelism: int | None = None,
    print_results: bool = True,
    description: str | None = None,
    path: str | None = None,
    inference: bool = True,
    single_trace: bool = False,
    on_datapoint_complete: DataPointComplete | None = None,
    _send_results: bool = True,
    _base_url: str | None = None,
    _trace_type: str = 'evaluatorq',
    _experiment_url_out: list[str] | None = None,
) -> EvaluatorqResult:
    """
    Run an evaluation with the given parameters.

    Can be called with either a params dict/object or keyword arguments:

    ```python
    # Using keyword arguments (recommended):
    await evaluatorq("name", data=[...], jobs=[...], datapoint_parallelism=5)

    # Using a dict:
    await evaluatorq("name", {"data": [...], "jobs": [...], "datapoint_parallelism": 5})

    # Using EvaluatorParams:
    await evaluatorq("name", EvaluatorParams(data=[...], jobs=[...]))
    ```

    Args:
        name: Name of the evaluation run
        params: Optional EvaluatorParams instance or dict with all parameters.
        data: The data to evaluate. A DatasetIdInput to fetch from Orq platform, an
              ExperimentInput to replay an experiment's recorded responses (requires
              inference=False), a TraceInput to import recorded trace conversations
              (requires inference=False), or a list of DataPoint instances/awaitables.
        jobs: The jobs to run on the data.
        evaluators: The evaluators to use. If not provided, only jobs will run.
        datapoint_parallelism: Task concurrency, applied at two levels: datapoints run
              at most ``datapoint_parallelism`` at a time, and within one datapoint a
              single shared budget of the same size bounds its jobs and then its
              evaluators (the budget is not split between them — a job releases its
              slot before its evaluators take theirs). Defaults to 10; set to 1 for
              sequential execution. Bounds tasks, not requests: see ``llm_parallelism``.
        parallelism: Deprecated alias for ``datapoint_parallelism``.
        llm_parallelism: Ceiling on in-flight LLM requests for the whole run,
              counted per request rather than per task, so it holds however the
              datapoint/job/evaluator/jury fan-out nests. Unbounded by default. This
              is the knob to set against a provider concurrency limit; ``datapoint_parallelism``
              bounds tasks, and one task can issue many requests. Only requests routed
              through evaluatorq are counted — wrap a job's own provider calls in
              ``evaluatorq.common.llm_limit.llm_slot()`` to include them.
        print_results: Whether to print results table to console. Defaults to True.
        description: Optional description for the evaluation run.
        path: Optional path (e.g. "MyProject/MyFolder") to place the experiment
              in a specific project and folder on the Orq platform.
        inference: When True (default) jobs run to generate responses. When False,
              generation is skipped and evaluators score each row's recorded output
              when present, falling back to its ``messages`` column; ``jobs`` is then
              optional and ignored.
        single_trace: Group every row under one ``evaluatorq.run`` span so the whole
              evaluation is a single trace. Defaults to False, which leaves each row's
              ``orq.job`` as its own root — an N-row run is then N separate traces.
        on_datapoint_complete: Optional sync or async callback invoked exactly once after each
              DataPointResult reaches a terminal state. Exceptions propagate and abort the run.

    Returns:
        List of DataPointResult objects

    Raises:
        ValidationError: If parameters fail validation.
        ValueError: If neither params nor required kwargs are provided.

    Example:
        ```python
        from evaluatorq import DataPoint, EvaluationResult, evaluatorq, job

        @job("uppercase")
        async def uppercase_job(data: DataPoint, row: int):
            return data.inputs["text"].upper()

        async def matches_expected(params):
            return EvaluationResult(value=1 if params["output"] == params["data"].expected_output else 0)

        await evaluatorq(
            "uppercase-eval",
            data=[DataPoint(inputs={"text": "hi"}, expected_output="HI")],
            jobs=[uppercase_job],
            evaluators=[{"name": "matches-expected", "scorer": matches_expected}],
        )
        ```
    """
    datapoint_parallelism = resolve_datapoint_parallelism(
        datapoint_parallelism, parallelism, default=10, caller='evaluatorq'
    )
    resolved = _normalise_params(
        _EvaluationInputs(
            params=params,
            data=data,
            jobs=jobs,
            evaluators=evaluators,
            datapoint_parallelism=datapoint_parallelism,
            llm_parallelism=llm_parallelism,
            print_results=print_results,
            description=description,
            path=path,
            inference=inference,
            single_trace=single_trace,
            on_datapoint_complete=on_datapoint_complete,
        )
    )
    data = resolved.data
    inference = resolved.inference
    jobs = resolved.jobs
    evaluators_list = resolved.evaluators_list
    datapoint_parallelism = resolved.datapoint_parallelism
    llm_parallelism = resolved.llm_parallelism
    print_results = resolved.print_results
    description = resolved.description
    path = resolved.path
    single_trace = resolved.single_trace
    on_datapoint_complete = resolved.on_datapoint_complete

    async with (
        tracing_session(name, trace_type=_trace_type) as tracing_context,
        AsyncExitStack() as span_stack,
    ):
        # Before any fan-out, so every task created below inherits the budget.
        _ = span_stack.enter_context(llm_concurrency_limit(llm_parallelism))
        if single_trace:
            await _enter_single_trace(span_stack, tracing_context, name, _trace_type)

        orq_api_key = os.environ.get('ORQ_API_KEY')

        start_time = datetime.now(timezone.utc)

        dataset_id: str | None = None

        data = await _resolve_experiment_input(data, orq_api_key, _base_url)

        if isinstance(data, TraceInput):
            traces = await fetch_traces(
                data,
                api_key=orq_api_key,
                base_url=_base_url,
            )
            data = [trace.to_datapoint() for trace in traces]

        # Create progress service
        progress = ProgressService()

        # Handle dataset_id case - use streaming fetch
        if isinstance(data, DatasetIdInput):
            orq_client, dataset_id, include_messages = _resolve_streaming_inputs(data, orq_api_key, inference=inference)

            # Stream fetch and process batches concurrently
            results = await with_progress(
                _run_streaming_evaluation(
                    _StreamingInputs(
                        progress=progress,
                        datapoint_parallelism=datapoint_parallelism,
                        orq_client=orq_client,
                        dataset_id=dataset_id,
                        include_messages=include_messages,
                        jobs=jobs,
                        evaluators_list=evaluators_list,
                        tracing_context=tracing_context,
                        on_datapoint_complete=on_datapoint_complete,
                    )
                ),
                progress,
                show_progress=print_results,
            )

        else:
            # Non-streaming case: process all data at once
            data_promises = cast('list[DataPoint]', data)

            results = await with_progress(
                _run_in_memory_evaluation(
                    data_promises,
                    progress,
                    datapoint_parallelism,
                    jobs,
                    evaluators_list,
                    tracing_context,
                    on_datapoint_complete,
                ),
                progress,
                show_progress=print_results,
            )

        await _finish_evaluation(
            _EvaluationFinishInputs(
                print_results=print_results,
                orq_api_key=orq_api_key,
                send_results=_send_results,
                name=name,
                description=description,
                dataset_id=dataset_id,
                results=results,
                start_time=start_time,
                path=path,
                base_url=_base_url,
                experiment_url_out=_experiment_url_out,
            )
        )

        return results
