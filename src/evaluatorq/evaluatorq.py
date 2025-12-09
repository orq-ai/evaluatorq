import asyncio
import os
from collections.abc import Awaitable, Sequence
from datetime import datetime, timezone
from typing import Any, cast

from .fetch_data import fetch_dataset_as_datapoints, setup_orq_client
from .processings import process_data_point
from .progress import Phase, ProgressService, with_progress
from .send_results import send_results_to_orq
from .table_display import display_results_table
from .types import (
    DataPoint,
    DatasetIdInput,
    Evaluator,
    EvaluatorParams,
    EvaluatorqResult,
    Job,
)


async def evaluatorq(
    name: str,
    params: EvaluatorParams | dict[str, Any] | None = None,
    *,
    data: DatasetIdInput | Sequence[Awaitable[DataPoint] | DataPoint] | None = None,
    jobs: list[Job] | None = None,
    evaluators: list[Evaluator] | None = None,
    parallelism: int = 1,
    print_results: bool = True,
    description: str | None = None,
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

    orq_api_key = os.environ.get("ORQ_API_KEY")

    start_time = datetime.now(timezone.utc)

    data_promises: Sequence[Awaitable[DataPoint] | DataPoint]
    dataset_id: str | None = None

    # Handle dataset_id case
    if isinstance(data, DatasetIdInput):
        orq_client = None

        if orq_api_key:
            orq_client = setup_orq_client(orq_api_key)

        if not orq_api_key or not orq_client:
            raise ValueError(
                "ORQ_API_KEY environment variable must be set to fetch datapoints from Orq platform."
            )
        dataset_id = data.dataset_id
        data_promises = await fetch_dataset_as_datapoints(orq_client, dataset_id)

    else:
        data_promises = cast(list[DataPoint], data)

    # Create progress service
    progress = ProgressService()

    # Define the main evaluation coroutine
    async def run_evaluation() -> EvaluatorqResult:
        # Initialize progress
        await progress.update_progress(
            total_data_points=len(data_promises),
            current_data_point=0,
            phase=Phase.INITIALIZING,
        )

        # Process data points with controlled concurrency
        # Use a semaphore to limit concurrent data points to avoid overwhelming the system
        # This allows parallelism within each data point (controlled by the parallelism param)
        # while also having multiple data points in flight
        data_point_semaphore = asyncio.Semaphore(max(1, parallelism // len(jobs)))

        async def process_with_semaphore(
            index: int, data_promise: Awaitable[DataPoint] | DataPoint
        ):
            async with data_point_semaphore:
                return await process_data_point(
                    data_promise, index, jobs, evaluators_list, parallelism, progress
                )

        tasks = [
            process_with_semaphore(index, data_promise)
            for index, data_promise in enumerate(data_promises)
        ]

        # Gather all results
        results_nested = await asyncio.gather(*tasks)

        # Flatten results (each process_data_point returns a list)
        results: EvaluatorqResult = []
        for result_list in results_nested:
            results.extend(result_list)

        return results

    # Run evaluation with progress tracking
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
        )

    return results
