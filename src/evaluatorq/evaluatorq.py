import asyncio
import os
from collections.abc import Awaitable, Sequence
from datetime import datetime, timezone
from typing import Any, cast

from .fetch_data import fetch_dataset_batches, setup_orq_client
from .processings import process_data_point
from .progress import Phase, ProgressService, with_progress
from .send_results import send_results_to_orq
from .table_display import display_results_table
from .tracing import (
    TracingContext,
    capture_parent_context,
    flush_tracing,
    generate_run_id,
    init_tracing_if_needed,
    shutdown_tracing,
)
from .types import (
    DataPoint,
    DataPointInput,
    DatasetIdInput,
    Evaluator,
    EvaluatorParams,
    EvaluatorqResult,
    Job,
)


def check_pass_failures(results: EvaluatorqResult) -> bool:
    """
    Check if any evaluator returned pass_=False.

    Args:
        results: The evaluation results to check

    Returns:
        True if any evaluator failed (pass_=False), False otherwise
    """
    for data_point_result in results:
        if data_point_result.job_results:
            for job_result in data_point_result.job_results:
                if job_result.evaluator_scores:
                    for evaluator_score in job_result.evaluator_scores:
                        if evaluator_score.score.pass_ is False:
                            return True
    return False


async def evaluatorq(
    name: str,
    params: EvaluatorParams | dict[str, Any] | None = None,
    *,
    data: DatasetIdInput | Sequence[Awaitable[DataPoint] | DataPointInput] | None = None,
    jobs: list[Job] | None = None,
    evaluators: list[Evaluator] | None = None,
    parallelism: int = 1,
    print_results: bool = True,
    description: str | None = None,
    path: str | None = None,
) -> EvaluatorqResult:
    """
    Run an evaluation with the given parameters.

    Can be called with either a params dict/object or keyword arguments:

        # Using keyword arguments (recommended):
        await evaluatorq("name", data=[...], jobs=[...], parallelism=5)

        # Using a dict:
        await evaluatorq("name", {"data": [...], "jobs": [...], "parallelism": 5})

        # Using EvaluatorParams:
        await evaluatorq("name", EvaluatorParams(data=[...], jobs=[...]))

    Args:
        name: Name of the evaluation run
        params: Optional EvaluatorParams instance or dict with all parameters.
        data: The data to evaluate. Either a DatasetIdInput to fetch from Orq platform,
              or a list of DataPoint instances/awaitables.
        jobs: The jobs to run on the data.
        evaluators: The evaluators to use. If not provided, only jobs will run.
        parallelism: Number of jobs to run in parallel. Defaults to 1 (sequential).
        print_results: Whether to print results table to console. Defaults to True.
        description: Optional description for the evaluation run.
        path: Optional path (e.g. "MyProject/MyFolder") to place the experiment
              in a specific project and folder on the Orq platform.

    Returns:
        List of DataPointResult objects

    Raises:
        ValidationError: If parameters fail validation.
        ValueError: If neither params nor required kwargs are provided.
    """
    # Handle params dict/object vs kwargs
    if params is not None:
        # Validate params if passed as dict
        if isinstance(params, dict):
            validated = EvaluatorParams.model_validate(params)
        else:
            validated = params
    elif data is not None and jobs is not None:
        # Use kwargs
        validated = EvaluatorParams(
            data=data,
            jobs=jobs,
            evaluators=evaluators,
            parallelism=parallelism,
            print_results=print_results,
            description=description,
            path=path,
        )
    else:
        raise ValueError(
            "Either 'params' or both 'data' and 'jobs' keyword arguments are required"
        )

    # Extract validated values
    data = validated.data
    jobs = validated.jobs
    evaluators_list = validated.evaluators or []
    parallelism = validated.parallelism
    print_results = validated.print_results
    description = validated.description
    path = validated.path

    # Initialize tracing if OTEL is configured
    tracing_enabled = await init_tracing_if_needed()
    parent_context = await capture_parent_context() if tracing_enabled else None
    tracing_context: TracingContext | None = None
    if tracing_enabled:
        tracing_context = TracingContext(
            run_id=generate_run_id(),
            run_name=name,
            enabled=True,
            parent_context=parent_context,
        )

    orq_api_key = os.environ.get("ORQ_API_KEY")

    start_time = datetime.now(timezone.utc)

    dataset_id: str | None = None

    # Create progress service
    progress = ProgressService()

    # Handle dataset_id case - use streaming fetch
    if isinstance(data, DatasetIdInput):
        orq_client = None

        if orq_api_key:
            orq_client = setup_orq_client(orq_api_key)

        if not orq_api_key or not orq_client:
            raise ValueError(
                "ORQ_API_KEY environment variable must be set to fetch datapoints from Orq platform."
            )
        dataset_id = data.dataset_id
        include_messages = data.include_messages

        # Stream fetch and process batches concurrently
        async def run_streaming_evaluation() -> EvaluatorqResult:
            all_results: EvaluatorqResult = []
            processing_tasks: list[asyncio.Task[list[Any]]] = []
            total_datapoints = 0
            datapoint_index = 0

            # Shared progress state for tracking processed count
            progress_ref = {"processed": 0}

            # Semaphore for controlling parallelism
            data_point_semaphore = asyncio.Semaphore(parallelism)

            async def process_with_semaphore(
                index: int, data_promise: DataPoint
            ) -> list[Any]:
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
                    progress_ref["processed"] += 1
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
                        current_data_point=progress_ref["processed"],
                        phase=Phase.PROCESSING
                        if progress_ref["processed"] > 0
                        else Phase.FETCHING,
                    )
                    await asyncio.sleep(0.1)

            polling_task = asyncio.create_task(poll_progress())

            try:
                # Fetch and process batches
                async for batch in fetch_dataset_batches(orq_client, dataset_id, include_messages=include_messages):
                    total_datapoints += len(batch.datapoints)

                    # Start processing this batch immediately
                    for datapoint in batch.datapoints:
                        task = asyncio.create_task(
                            process_with_semaphore(datapoint_index, datapoint)
                        )
                        processing_tasks.append(task)
                        datapoint_index += 1

                # Wait for all processing tasks to complete
                results_nested = await asyncio.gather(*processing_tasks)
            finally:
                # Stop the polling task
                stop_polling = True
                _ = polling_task.cancel()
                try:
                    await polling_task
                except asyncio.CancelledError:
                    pass

            # Final progress update
            await progress.update_progress(
                total_data_points=total_datapoints,
                current_data_point=progress_ref["processed"],
                phase=Phase.PROCESSING,
            )

            # Flatten results
            for result_list in results_nested:
                all_results.extend(result_list)

            return all_results

        results = await with_progress(
            run_streaming_evaluation(), progress, show_progress=print_results
        )

    else:
        # Non-streaming case: process all data at once
        data_promises = cast(list[DataPoint], data)

        async def run_evaluation() -> EvaluatorqResult:
            # Initialize progress
            await progress.update_progress(
                total_data_points=len(data_promises),
                current_data_point=0,
                phase=Phase.INITIALIZING,
            )

            # Process data points with controlled concurrency
            data_point_semaphore = asyncio.Semaphore(max(1, parallelism // len(jobs)))

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

            tasks = [
                process_with_semaphore(index, data_promise)
                for index, data_promise in enumerate(data_promises)
            ]

            # Gather all results
            results_nested = await asyncio.gather(*tasks)

            # Flatten results
            results: EvaluatorqResult = []
            for result_list in results_nested:
                results.extend(result_list)

            return results

        results = await with_progress(
            run_evaluation(), progress, show_progress=print_results
        )

    # Display results table
    if print_results:
        await display_results_table(results)

    # Upload results to Orq platform if API key is available
    if orq_api_key:
        _ = await send_results_to_orq(
            orq_api_key,
            name,
            description,
            dataset_id,
            results,
            start_time,
            datetime.now(timezone.utc),
            path=path,
        )

    # Shutdown tracing gracefully
    if tracing_enabled:
        await flush_tracing()
        await asyncio.sleep(2)  # Give additional time for network operations
        await shutdown_tracing()

    # Check for pass failures and exit if any
    has_failures = check_pass_failures(results)
    if has_failures:
        import sys

        sys.exit(1)

    return results
