import asyncio
from collections.abc import Awaitable
from inspect import isawaitable
from typing import TYPE_CHECKING, cast

from loguru import logger

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

if TYPE_CHECKING:
    from .tracing.context import TracingContext


async def process_data_point(
    data_promise: DataPoint | Awaitable[DataPoint],
    row_index: int,
    jobs: list[Job],
    evaluators: list[Evaluator] | None,
    parallelism: int,
    progress_service: ProgressService | None = None,
    tracing_context: 'TracingContext | None' = None,
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
        tracing_context: Optional tracing context for OTEL spans

    Returns:
        List containing a single DataPointResult with job results and evaluator scores
    """
    data_point: DataPoint | None = None
    try:
        # Resolve the data point (await if it's awaitable, otherwise use directly)
        if isawaitable(data_promise):
            data_point = await data_promise
        else:
            data_point = data_promise

        # Update progress for this data point
        if progress_service:
            await progress_service.update_progress(current_data_point=row_index + 1, phase=Phase.PROCESSING)

        # Process jobs with concurrency control
        semaphore = asyncio.Semaphore(parallelism)

        async def run_job_with_semaphore(job: Job) -> JobResult:
            return await process_job(
                job,
                data_point,
                row_index,
                evaluators,
                progress_service,
                tracing_context,
                semaphore,
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
        logger.warning('Data point {} failed before job execution: {}', row_index, error)
        # Preserve the resolved DataPoint's identity when available; only fall
        # back to a placeholder (tagged with row_index, so concurrent failures
        # don't collide) when resolution itself failed.
        fallback = data_point if isinstance(data_point, DataPoint) else DataPoint(inputs={'row_index': row_index})
        return [
            DataPointResult(
                data_point=fallback,
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
    tracing_context: 'TracingContext | None' = None,
    semaphore: asyncio.Semaphore | None = None,
) -> JobResult:
    """
    Process a single job and optionally run evaluators on its output.

    Args:
        job: The job function to execute
        data_point: The data point to pass to the job
        row_index: Index of the data point
        evaluators: List of evaluators to run on the job output
        progress_service: Optional progress tracking service
        tracing_context: Optional tracing context for OTEL spans
        semaphore: Optional per-datapoint semaphore shared by the job and its evaluators

    Returns:
        JobResult containing job output and evaluator scores
    """
    # Import tracing utilities lazily to avoid import errors when OTEL is not installed
    from .tracing.spans import (
        JobSpanOptions,
        set_job_name_attribute,
        with_job_span,
    )

    job_name = 'job'  # Default name
    output: Output = None
    error: str | None = None

    # Wrap the job execution in a span if tracing is enabled
    async with with_job_span(
        JobSpanOptions(
            run_id=tracing_context.run_id if tracing_context else '',
            row_index=row_index,
            parent_context=tracing_context.parent_context if tracing_context else None,
            trace_type=tracing_context.trace_type if tracing_context else 'evaluatorq',
        )
    ) as job_span:
        try:
            # Execute the job
            if semaphore is None:
                result = await job(data_point, row_index)
            else:
                async with semaphore:
                    result = await job(data_point, row_index)
            job_name = cast('str', result['name'])
            output = cast('Output', result['output'])

            # Set job name on span after execution
            set_job_name_attribute(job_span, job_name)

            # Update progress with current job name
            if progress_service:
                await progress_service.update_progress(current_job=job_name)

        except JobError as e:
            # Extract job name from JobError
            job_name = e.job_name
            error = str(e.original_error)
            set_job_name_attribute(job_span, job_name)

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

            async def run_evaluator_with_semaphore(evaluator: Evaluator) -> EvaluatorScore:
                if semaphore is None:
                    return await process_evaluator(
                        evaluator, data_point, output, progress_service, tracing_context, row_index=row_index
                    )
                async with semaphore:
                    return await process_evaluator(
                        evaluator, data_point, output, progress_service, tracing_context, row_index=row_index
                    )

            # Run all evaluators concurrently, bounded by the per-datapoint semaphore.
            tasks = [asyncio.create_task(run_evaluator_with_semaphore(evaluator)) for evaluator in evaluators]

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
    tracing_context: 'TracingContext | None' = None,
    row_index: int | None = None,
) -> EvaluatorScore:
    """
    Process a single evaluator.

    Args:
        evaluator: The evaluator configuration with name and scorer function
        data_point: The original data point
        output: The job output to evaluate
        progress_service: Optional progress tracking service
        tracing_context: Optional tracing context for OTEL spans
        row_index: Dataset index of the data point, forwarded to the scorer as
            ``ScorerParameter['row']`` so per-item logic (e.g. cyclic judge
            assignment) can key on dataset position instead of arrival order

    Returns:
        EvaluatorScore with the evaluation result or error
    """
    # Import tracing utilities lazily to avoid import errors when OTEL is not installed
    from .tracing.spans import (
        EvaluationSpanOptions,
        set_evaluation_attributes,
        with_evaluation_span,
    )

    evaluator_name = evaluator['name']

    # Wrap the evaluator execution in a span if tracing is enabled
    async with with_evaluation_span(
        EvaluationSpanOptions(
            run_id=tracing_context.run_id if tracing_context else '',
            evaluator_name=evaluator_name,
        )
    ) as eval_span:
        try:
            # Update current evaluator in progress
            if progress_service:
                await progress_service.update_progress(current_evaluator=evaluator_name)

            # Execute the scorer
            scorer_param: ScorerParameter = {
                'data': data_point,
                'output': output,
            }
            if row_index is not None:
                scorer_param['row'] = row_index

            result = await evaluator['scorer'](scorer_param)

            # Convert dict to EvaluationResult if needed
            score = EvaluationResult.model_validate(result) if isinstance(result, dict) else result

            # Set evaluation attributes on span
            set_evaluation_attributes(
                eval_span,
                score.value,
                explanation=score.explanation,
                pass_=score.pass_,
                evaluator_name=evaluator_name,
                evaluator_type=evaluator.get('evaluator_type'),
            )

            return EvaluatorScore(
                evaluator_name=evaluator_name,
                score=score,
                error=None,
            )

        except Exception as error:
            # Return error result with empty score
            return EvaluatorScore(
                evaluator_name=evaluator_name,
                score=EvaluationResult(value=''),
                error=str(error),
            )
