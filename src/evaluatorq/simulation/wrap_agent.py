"""Wraps the simulation framework as an evaluatorq Job.

Follows the same pattern as wrap_langchain_agent().
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from evaluatorq.simulation._datapoint_io import _extract_single_datapoint
from evaluatorq.simulation.adapters import from_orq_deployment
from evaluatorq.simulation.convert import to_open_responses
from evaluatorq.simulation.types import (
    DEFAULT_MODEL,
    ChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from evaluatorq.simulation.agents.base import BaseAgent
    from evaluatorq.types import DataPoint

logger = logging.getLogger(__name__)


def wrap_simulation_agent(
    *,
    name: str = "simulation",
    target_callback: Callable[[list[ChatMessage]], str | Awaitable[str]] | None = None,
    agent_key: str | None = None,
    max_turns: int = 10,
    model: str | None = None,
    user_simulator: BaseAgent | None = None,
    judge: BaseAgent | None = None,
    **deprecated_kwargs: Any,
) -> Callable[[DataPoint, int], Awaitable[dict[str, Any]]]:
    """Create an evaluatorq Job that runs agent simulations.

    Each DataPoint should have inputs containing simulation data:
    - ``persona`` and ``scenario``, or
    - ``datapoint`` (full Datapoint object), or
    - ``datapoints`` / ``personas`` + ``scenarios`` each of length one

    The returned callable owns a long-lived ``SimulationRunner`` (and its
    underlying HTTP client). Call ``await job_fn.aclose()`` after your
    ``evaluatorq()`` run finishes to release the connection pool — otherwise
    it leaks until process exit. Example::

        job = wrap_simulation_agent(target_callback=cb)
        try:
            await evaluatorq("run", data=[...], jobs=[job], evaluators=[...])
        finally:
            await job.aclose()
    """
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    if "evaluators" in deprecated_kwargs:
        # Removed in RES-594: scoring belongs on the evaluatorq() call, not
        # the job that produces the output. Raise loud — silently dropping
        # scoring would be worse than a TypeError.
        raise TypeError(
            "wrap_simulation_agent() no longer accepts 'evaluators='. Pass your "
            "evaluator list to evaluatorq(..., evaluators=...) instead. See CHANGELOG."
        )
    if deprecated_kwargs:
        raise TypeError(
            f"wrap_simulation_agent() got unexpected keyword arguments: "
            f"{sorted(deprecated_kwargs)}"
        )

    resolved_callback = target_callback
    if not resolved_callback and agent_key:
        resolved_callback = from_orq_deployment(agent_key)
    if not resolved_callback:
        raise ValueError(
            "wrap_simulation_agent requires either target_callback or agent_key"
        )

    effective_model = model or DEFAULT_MODEL

    runner = SimulationRunner(
        target_callback=resolved_callback,
        model=effective_model,
        max_turns=max_turns,
        user_simulator=user_simulator,
        judge=judge,
    )

    async def job_fn(data: DataPoint, _row: int) -> dict[str, Any]:
        sim_dp = _extract_single_datapoint(data)
        result = await runner.run(datapoint=sim_dp, max_turns=max_turns)
        return {
            "name": name,
            "output": to_open_responses(result, effective_model),
        }

    async def aclose() -> None:
        await runner.close()

    setattr(job_fn, "aclose", aclose)  # noqa: B010
    return job_fn
