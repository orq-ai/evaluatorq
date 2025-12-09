"""Helper function for creating named jobs with error handling."""

from collections.abc import Awaitable
from inspect import isawaitable
from typing import Any, Callable, overload

from .types import DataPoint, Job, Output


class JobError(Exception):
    """Exception that preserves the job name when a job fails."""

    def __init__(self, job_name: str, original_error: Exception):
        self.job_name: str = job_name
        self.original_error: Exception = original_error
        super().__init__(str(original_error))


@overload
def job(
    name: str,
) -> Callable[[Callable[[DataPoint, int], Awaitable[Output] | Output]], Job]:
    """Decorator form: @job("name")"""
    ...


@overload
def job(
    name: str,
    fn: Callable[[DataPoint, int], Awaitable[Output] | Output],
) -> Job:
    """Functional form: job("name", fn)"""
    ...


def job(
    name: str,
    fn: Callable[[DataPoint, int], Awaitable[Output] | Output] | None = None,
) -> Job | Callable[[Callable[[DataPoint, int], Awaitable[Output] | Output]], Job]:
    """
    Helper function/decorator to create a named job that ensures the job name is preserved
    even when errors occur during execution.

    This wrapper:
    - Automatically formats the return value as {"name": ..., "output": ...}
    - Attaches the job name to errors for better error tracking
    - Can be used as a decorator (@job("name")) or function (job("name", fn))

    Args:
        name: The name of the job
        fn: The job function that returns the output (optional when used as decorator)

    Returns:
        A Job function that always includes the job name

    Example:
        ```python
        # As a decorator:
        @job("text-analyzer")
        async def analyze_text(data: DataPoint, row: int):
            return {"length": len(data.inputs["text"])}

        # As a function wrapper:
        my_job = job("my-job", async_function)

        # With lambda for simple cases:
        uppercase_job = job("uppercase", lambda data, row: data.inputs["text"].upper())
        ```
    """

    def create_wrapper(
        func: Callable[[DataPoint, int], Awaitable[Output] | Output],
    ) -> Job:
        async def wrapper(data: DataPoint, row: int) -> dict[str, Any]:
            try:
                # Execute the job function
                result = func(data, row)

                # Await if it's a coroutine, otherwise use directly
                if isawaitable(result):
                    output: Output = await result
                else:
                    output = result  # type: ignore

                job_return: dict[str, Any] = {
                    "name": name,
                    "output": output,
                }
                return job_return

            except Exception as error:
                # Wrap error with job name for better tracking
                # This allows process_job to extract the name even on failure
                raise JobError(name, error) from error

        return wrapper

    # If fn is provided, use functional form
    if fn is not None:
        return create_wrapper(fn)

    # Otherwise, return decorator
    return create_wrapper
