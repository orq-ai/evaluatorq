import asyncio
import os
import sys
from collections.abc import Awaitable, Sequence
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from itertools import starmap
from typing import Any, cast

from loguru import logger

from .common.messages import coerce_content_text
from .fetch_data import (
    fetch_dataset_batches,
    fetch_experiment_datapoints,
    setup_orq_client,
)
from .processings import process_data_point
from .progress import Phase, ProgressService, with_progress
from .send_results import send_results_to_orq
from .table_display import display_results_table
from .tracing import capture_parent_context, tracing_session
from .types import (
    DataPoint,
    DataPointInput,
    DatasetIdInput,
    Evaluator,
    EvaluatorParams,
    EvaluatorqResult,
    ExperimentInput,
    Job,
)


class _StreamingEvaluationError(RuntimeError):
    """Report multiple failures collected from a streaming evaluation's tasks."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = errors
        details = '; '.join(f'{type(error).__name__}: {error}' for error in errors)
        super().__init__(f'Streaming evaluation failed with {len(errors)} errors: {details}')


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
            responses or no usable scores exit successfully.

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
    response = extract_recorded_response(data_point.inputs.get('messages'))
    return {'name': 'recorded', 'output': response}


async def evaluatorq(
    name: str,
    params: EvaluatorParams | dict[str, Any] | None = None,
    *,
    data: DatasetIdInput | ExperimentInput | Sequence[Awaitable[DataPoint] | DataPointInput] | None = None,
    jobs: list[Job] | None = None,
    evaluators: list[Evaluator] | None = None,
    parallelism: int = 10,
    print_results: bool = True,
    description: str | None = None,
    path: str | None = None,
    inference: bool = True,
    single_trace: bool = False,
    _exit_on_failure: bool = False,
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
    await evaluatorq("name", data=[...], jobs=[...], parallelism=5)

    # Using a dict:
    await evaluatorq("name", {"data": [...], "jobs": [...], "parallelism": 5})

    # Using EvaluatorParams:
    await evaluatorq("name", EvaluatorParams(data=[...], jobs=[...]))
    ```

    Args:
        name: Name of the evaluation run
        params: Optional EvaluatorParams instance or dict with all parameters.
        data: The data to evaluate. A DatasetIdInput to fetch from Orq platform, an
              ExperimentInput to replay an experiment's recorded responses (requires
              inference=False), or a list of DataPoint instances/awaitables.
        jobs: The jobs to run on the data.
        evaluators: The evaluators to use. If not provided, only jobs will run.
        parallelism: Maximum concurrency for each of three scopes: datapoints,
              jobs per datapoint, and evaluators per job. Defaults to 10; set to 1 for
              sequential execution, or lower it if your provider rate-limits.
        print_results: Whether to print results table to console. Defaults to True.
        description: Optional description for the evaluation run.
        path: Optional path (e.g. "MyProject/MyFolder") to place the experiment
              in a specific project and folder on the Orq platform.
        inference: When True (default) jobs run to generate responses. When False,
              generation is skipped and evaluators score the pre-recorded response in
              each row's ``messages`` column; ``jobs`` is then optional and ignored.
        single_trace: Group every row under one ``evaluatorq.run`` span so the whole
              evaluation is a single trace. Defaults to False, which leaves each row's
              ``orq.job`` as its own root — an N-row run is then N separate traces.
        _exit_on_failure: Whether to exit the process when an evaluator fails.
              Defaults to False for library callers, who receive the failed result
              with ``pass_=False`` and no ``SystemExit``. CLI users still get a
              non-zero exit from the redteam and simulation command failure gates.

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
            parallelism=parallelism,
            print_results=print_results,
            description=description,
            path=path,
            inference=inference,
            single_trace=single_trace,
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
                "inference=False: ignoring the provided 'jobs'; responses are replayed from the 'messages' column."
            )
        jobs = [_replay_recorded_response]
    evaluators_list = validated.evaluators or []
    parallelism = validated.parallelism
    print_results = validated.print_results
    description = validated.description
    path = validated.path
    single_trace = validated.single_trace

    async with tracing_session(name, trace_type=_trace_type) as tracing_context, AsyncExitStack() as span_stack:
        if single_trace:
            from .tracing.spans import RunSpanOptions, with_run_span

            run_span = await span_stack.enter_async_context(
                with_run_span(
                    RunSpanOptions(
                        run_id=tracing_context.run_id,
                        run_name=name,
                        parent_context=tracing_context.parent_context,
                        trace_type=_trace_type,
                    )
                )
            )
            # Re-point the context the per-row job spans parent to. tracing_session
            # captured the ambient context *before* this span existed, and jobs pass
            # parent_context explicitly rather than reading the ambient one.
            if run_span is not None:
                tracing_context.parent_context = await capture_parent_context()

        orq_api_key = os.environ.get('ORQ_API_KEY')

        start_time = datetime.now(timezone.utc)

        dataset_id: str | None = None

        # Experiment source (no-inference only): replace the input with the experiment's
        # recorded responses, then fall through to the in-memory data path below. The
        # validator already guarantees inference=False here.
        if isinstance(data, ExperimentInput):
            if not orq_api_key:
                raise ValueError(
                    'ORQ_API_KEY environment variable must be set to load responses from an Orq experiment.'
                )
            data = await fetch_experiment_datapoints(
                orq_api_key,
                data.experiment_id,
                data.run_id,
                base_url=_base_url,
            )

        # Create progress service
        progress = ProgressService()

        # Handle dataset_id case - use streaming fetch
        if isinstance(data, DatasetIdInput):
            orq_client = None

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

            # Stream fetch and process batches concurrently
            async def run_streaming_evaluation() -> EvaluatorqResult:
                all_results: EvaluatorqResult = []
                processing_tasks: list[asyncio.Task[list[Any]]] = []
                total_datapoints = 0
                datapoint_index = 0

                # Shared progress state for tracking processed count
                progress_ref = {'processed': 0}

                # Semaphore for controlling parallelism
                data_point_semaphore = asyncio.Semaphore(parallelism)

                async def process_with_semaphore(index: int, data_promise: DataPoint) -> list[Any]:
                    async with data_point_semaphore:
                        result = await process_data_point(
                            data_promise,
                            index,
                            jobs,
                            evaluators_list,
                            parallelism,
                            None,  # Don't pass progress in streaming mode - use polling instead
                            tracing_context,
                        )
                        progress_ref['processed'] += 1
                        return result

                # Initialize progress with unknown total (streaming mode)
                await progress.update_progress(
                    total_data_points=0,
                    current_data_point=0,
                    phase=Phase.FETCHING,
                )

                # Start a background task to poll and update progress
                stop_polling = False

                async def poll_progress():
                    while not stop_polling:
                        await progress.update_progress(
                            total_data_points=total_datapoints,
                            current_data_point=progress_ref['processed'],
                            phase=Phase.PROCESSING if progress_ref['processed'] > 0 else Phase.FETCHING,
                        )
                        await asyncio.sleep(0.1)

                polling_task = asyncio.create_task(poll_progress())
                fetch_error: BaseException | None = None
                cancelled_for_fetch: set[asyncio.Task[Any]] = set()

                try:
                    # Fetch and process batches
                    async for batch in fetch_dataset_batches(orq_client, dataset_id, include_messages=include_messages):
                        total_datapoints += len(batch.datapoints)

                        # Start processing this batch immediately
                        for datapoint in batch.datapoints:
                            task = asyncio.create_task(process_with_semaphore(datapoint_index, datapoint))
                            processing_tasks.append(task)
                            datapoint_index += 1

                except asyncio.CancelledError as exc:
                    fetch_error = exc
                except Exception as exc:  # noqa: BLE001 - collect fetch failures with task failures
                    fetch_error = exc
                finally:
                    # Stop polling on both success and failure. A fetch failure also
                    # cancels in-flight processing, but their exceptions are still
                    # collected separately from the deliberately cancelled poller.
                    stop_polling = True
                    if fetch_error is not None:
                        for task in processing_tasks:
                            if not task.done() and task.cancelling() == 0:
                                task.cancel()
                                cancelled_for_fetch.add(task)
                    _ = polling_task.cancel()
                    processing_results = await asyncio.gather(
                        *processing_tasks,
                        return_exceptions=True,
                    )
                    polling_result = await asyncio.gather(
                        polling_task,
                        return_exceptions=True,
                    )

                task_errors = [
                    result
                    for task, result in zip(processing_tasks, processing_results, strict=True)
                    if isinstance(result, BaseException)
                    and not (isinstance(result, asyncio.CancelledError) and task in cancelled_for_fetch)
                ]
                task_errors.extend(
                    result
                    for result in polling_result
                    if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError)
                )
                errors = ([fetch_error] if fetch_error is not None else []) + task_errors
                if errors:
                    if len(errors) == 1:
                        raise errors[0]
                    raise _StreamingEvaluationError(errors)
                results_nested = cast('list[list[Any]]', processing_results)

                # Final progress update
                await progress.update_progress(
                    total_data_points=total_datapoints,
                    current_data_point=progress_ref['processed'],
                    phase=Phase.PROCESSING,
                )

                # Flatten results
                for result_list in results_nested:
                    all_results.extend(result_list)

                return all_results

            results = await with_progress(run_streaming_evaluation(), progress, show_progress=print_results)

        else:
            # Non-streaming case: process all data at once
            data_promises = cast('list[DataPoint]', data)

            async def run_evaluation() -> EvaluatorqResult:
                # Initialize progress
                await progress.update_progress(
                    total_data_points=len(data_promises),
                    current_data_point=0,
                    phase=Phase.INITIALIZING,
                )

                # Process data points with controlled concurrency
                data_point_semaphore = asyncio.Semaphore(parallelism)

                async def process_with_semaphore(
                    index: int, data_promise: Awaitable[DataPoint] | DataPoint
                ) -> list[Any]:
                    async with data_point_semaphore:
                        return await process_data_point(
                            data_promise,
                            index,
                            jobs,
                            evaluators_list,
                            parallelism,
                            progress,
                            tracing_context,
                        )

                tasks = list(starmap(process_with_semaphore, enumerate(data_promises)))

                # Gather all results
                results_nested = await asyncio.gather(*tasks)

                # Flatten results
                results: EvaluatorqResult = []
                for result_list in results_nested:
                    results.extend(result_list)

                return results

            results = await with_progress(run_evaluation(), progress, show_progress=print_results)

        # Display results table
        if print_results:
            await display_results_table(results)

        # Upload results to Orq platform if API key is available
        if orq_api_key and _send_results:
            upload_response = await send_results_to_orq(
                orq_api_key,
                name,
                description,
                dataset_id,
                results,
                start_time,
                datetime.now(timezone.utc),
                path=path,
                base_url=_base_url,
            )
            # Hand the created experiment's URL back to callers that opted in with a
            # sink list (e.g. simulation persists it on the SimulationRun report).
            if _experiment_url_out is not None and upload_response is not None and upload_response.experiment_url:
                _experiment_url_out.append(upload_response.experiment_url)

        # Check for pass failures and exit if any. In no-inference mode a row that has no
        # usable recorded response surfaces as a job error rather than an evaluator score,
        # so count those errors as failures too (the mode must fail loudly, not exit 0).
        has_failures = check_pass_failures(results, treat_errors_as_failure=not inference)
        if has_failures and _exit_on_failure:
            sys.exit(1)

        return results
