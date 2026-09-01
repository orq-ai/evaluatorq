"""Wraps the simulation framework as an evaluatorq Job.

Follows the same pattern as wrap_langchain_agent().
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from evaluatorq.simulation._datapoint_io import _extract_single_datapoint
from evaluatorq.simulation.adapters import from_orq_deployment
from evaluatorq.simulation.convert import to_open_responses
from evaluatorq.simulation.evaluators.scorers import UNEVALUATED_TERMINATIONS
from evaluatorq.simulation.types import (
    DEFAULT_MODEL,
    Message,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from evaluatorq.contracts import AgentResponse, LLMCallConfig
    from evaluatorq.simulation.agents.base import BaseAgent
    from evaluatorq.types import DataPoint

logger = logging.getLogger(__name__)


def wrap_simulation_agent(
    *,
    name: str = 'simulation',
    target: Callable[[list[Message]], str | Awaitable[str] | Awaitable[AgentResponse]] | None = None,
    agent_key: str | None = None,
    max_turns: int = 10,
    model: str | None = None,
    llm_config: LLMCallConfig | None = None,
    user_simulator: BaseAgent | None = None,
    judge: BaseAgent | None = None,
    **deprecated_kwargs: Any,
) -> Callable[[DataPoint, int], Awaitable[dict[str, Any]]]:
    """Create an evaluatorq Job that runs agent simulations.

    Each DataPoint should have inputs containing simulation data:
    - ``persona`` and ``scenario``, or
    - ``datapoint`` (full SimulationDatapoint object), or
    - ``datapoints`` / ``personas`` + ``scenarios`` each of length one

    ``llm_config`` is the fuller surface behind ``model``: it configures every
    simulation-side call (user simulator and judge), never the target under test.
    ``model`` stays the shorthand for setting only the model; when both name one,
    ``llm_config.model`` wins and the contradiction is logged.

    The returned callable owns a long-lived ``SimulationRunner`` (and its
    underlying HTTP client). Call ``await job_fn.aclose()`` after your
    ``evaluatorq()`` run finishes to release the connection pool — otherwise
    it leaks until process exit. Example:

    ```python
    job = wrap_simulation_agent(target=cb)
    try:
        await evaluatorq("run", data=[...], jobs=[job], evaluators=[...])
    finally:
        await job.aclose()
    ```
    """
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    if 'evaluators' in deprecated_kwargs:
        # Removed in RES-594: scoring belongs on the evaluatorq() call, not
        # the job that produces the output. Raise loud — silently dropping
        # scoring would be worse than a TypeError.
        raise TypeError(
            "wrap_simulation_agent() no longer accepts 'evaluators='. Pass your "
            'evaluator list to evaluatorq(..., evaluators=...) instead. See CHANGELOG.'
        )
    if deprecated_kwargs:
        raise TypeError(
            f'wrap_simulation_agent() got unexpected keyword argument(s): {", ".join(sorted(deprecated_kwargs))}'
        )

    resolved_target = target
    if not resolved_target and agent_key:
        resolved_target = from_orq_deployment(agent_key)
    if not resolved_target:
        raise ValueError('wrap_simulation_agent requires either target or agent_key')

    runner = SimulationRunner(
        target=resolved_target,
        model=model or DEFAULT_MODEL,
        llm_config=llm_config,
        max_turns=max_turns,
        user_simulator=user_simulator,
        judge=judge,
    )

    async def job_fn(data: DataPoint, _row: int) -> dict[str, Any]:
        sim_dp = _extract_single_datapoint(data)
        result = await runner.run(datapoint=sim_dp, max_turns=max_turns)
        # Emitted unconditionally, None on success. A run the runner ended in error
        # or timeout never reached the judge, so it has no verdict to score; without
        # this key process_job sees a returned dict, counts the row as a success,
        # and a conversation that never happened reports a 100% pass rate.
        failed = result.terminated_by in UNEVALUATED_TERMINATIONS
        return {
            'name': name,
            # The runner's, not `model`: `llm_config.model` wins the resolution and is what ran.
            'output': to_open_responses(result, runner.model),
            'error': result.reason if failed else None,
        }

    async def aclose() -> None:
        await runner.close()

    setattr(job_fn, 'aclose', aclose)  # noqa: B010
    return job_fn
