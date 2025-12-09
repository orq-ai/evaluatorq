import asyncio
from collections.abc import Awaitable
from inspect import isawaitable
from typing import cast

from .job_helper import JobError
from .progress import Phase, ProgressService
from .types import (
    DataPoint,
    DataPointResult,
    EvaluationResult,
    Evaluator,
    EvaluatorScore,
    Job,
    JobResult,
    Output,
    ScorerParameter,
)


async def process_data_point(
    data_promise: DataPoint | Awaitable[DataPoint],
    row_index: int,
    jobs: list[Job],
    evaluators: list[Evaluator] | None,
    parallelism: int,
    progress_service: ProgressService | None = None,
) -> list[DataPointResult]:
    """
    Process a single data point through all jobs and evaluators.

    Args:
        data_promise: A DataPoint or an awaitable that resolves to a DataPoint
        row_index: Index of this data point in the dataset
        jobs: List of jobs to execute
        evaluators: List of evaluators to run on job outputs
        parallelism: Number of jobs to run in parallel
        progress_service: Optional progress tracking service

    Returns:
        List containing a single DataPointResult with job results and evaluator scores
    """
    try:
        # Resolve the data point (await if it's awaitable, otherwise use directly)
        if isawaitable(data_promise):
            data_point = await data_promise
        else:
            data_point = data_promise

        # Update progress for this data point
        if progress_service:
            await progress_service.update_progress(
                current_data_point=row_index + 1, phase=Phase.PROCESSING
            )

        # Process jobs with concurrency control
        semaphore = asyncio.Semaphore(parallelism)

        async def run_job_with_semaphore(job: Job) -> JobResult:
            async with semaphore:
                return await process_job(
                    job, data_point, row_index, evaluators, progress_service
                )

        # Execute all jobs with controlled parallelism
        job_results = await asyncio.gather(
            *[run_job_with_semaphore(job) for job in jobs],
            return_exceptions=False,
        )

        return [
            DataPointResult(
                data_point=data_point,
                job_results=job_results,
                error=None,
            )
        ]

    except Exception as error:
        # Return error result with placeholder data point
        return [
            DataPointResult(
                data_point=DataPoint(inputs={}, expected_output=None),
                error=str(error),
                job_results=None,
            )
        ]


async def process_job(
    job: Job,
    data_point: DataPoint,
    row_index: int,
    evaluators: list[Evaluator] | None = None,
    progress_service: ProgressService | None = None,
) -> JobResult:
    """
    Process a single job and optionally run evaluators on its output.

    Args:
        job: The job function to execute
        data_point: The data point to pass to the job
        row_index: Index of the data point
        evaluators: List of evaluators to run on the job output
        progress_service: Optional progress tracking service

    Returns:
        JobResult containing job output and evaluator scores
    """
    job_name = "job"  # Default name
    output: Output = None
    error: str | None = None

    try:
        # Execute the job
        result = await job(data_point, row_index)
        job_name = cast(str, result["name"])
        output = cast(Output, result["output"])

        # Update progress with current job name
        if progress_service:
            await progress_service.update_progress(current_job=job_name)

    except JobError as e:
        # Extract job name from JobError
        job_name = e.job_name
        error = str(e.original_error)

        # Return early with error if job failed
        return JobResult(
            job_name=job_name,
            output=None,
            error=error,
            evaluator_scores=[],
        )
    except Exception as e:
        error = str(e)

        # Return early with error if job failed
        return JobResult(
            job_name=job_name,
            output=None,
            error=error,
            evaluator_scores=[],
        )

    # Process evaluators if any and job was successful
    evaluator_scores: list[EvaluatorScore] = []

    if evaluators:
        # Update phase to evaluating
        if progress_service:
            await progress_service.update_progress(phase=Phase.EVALUATING)

        # Run all evaluators concurrently (unbounded concurrency)
        # Using create_task for better event loop scheduling
        tasks = [
            asyncio.create_task(
                process_evaluator(evaluator, data_point, output, progress_service)
            )
            for evaluator in evaluators
        ]

        evaluator_scores = await asyncio.gather(*tasks)

    return JobResult(
        job_name=job_name,
        output=output,
        error=None,
        evaluator_scores=evaluator_scores,
    )


async def process_evaluator(
    evaluator: Evaluator,
    data_point: DataPoint,
    output: Output,
    progress_service: ProgressService | None = None,
) -> EvaluatorScore:
    """
    Process a single evaluator.

    Args:
        evaluator: The evaluator configuration with name and scorer function
        data_point: The original data point
        output: The job output to evaluate
        progress_service: Optional progress tracking service

    Returns:
        EvaluatorScore with the evaluation result or error
    """
    evaluator_name = evaluator["name"]

    try:
        # Update current evaluator in progress
        if progress_service:
            await progress_service.update_progress(current_evaluator=evaluator_name)

        # Execute the scorer
        scorer_param: ScorerParameter = {
            "data": data_point,
            "output": output,
        }

        result = await evaluator["scorer"](scorer_param)

        # Convert dict to EvaluationResult if needed
        if isinstance(result, dict):
            score = EvaluationResult.model_validate(result)
        else:
            score = result

        return EvaluatorScore(
            evaluator_name=evaluator_name,
            score=score,
            error=None,
        )

    except Exception as error:
        # Return error result with empty score
        return EvaluatorScore(
            evaluator_name=evaluator_name,
            score=EvaluationResult(value=""),
            error=str(error),
        )
