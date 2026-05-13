"""Auto-upload helper for the direct simulate() path.

Converts ``list[SimulationResult]`` into ``list[DataPointResult]`` (the
shape ``send_results_to_orq`` expects) and posts them to the Orq platform.

Used by ``simulate()`` and ``generate_and_simulate()`` when their
``upload_results=True`` flag is set and ``ORQ_API_KEY`` is present in
the environment. The ``wrap_simulation_agent()`` → ``evaluatorq()``
routing path already gets upload for free via the framework; this
helper covers the standalone entry points (RES-598).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from evaluatorq.send_results import send_results_to_orq
from evaluatorq.simulation.convert import to_open_responses
from evaluatorq.types import (
    DataPoint,
    DataPointResult,
    EvaluationResult,
    EvaluatorScore,
    JobResult,
)

if TYPE_CHECKING:
    from datetime import datetime

    from evaluatorq.simulation.types import SimulationResult

logger = logging.getLogger(__name__)


JOB_NAME = "simulation"


def _to_data_point_result(
    result: SimulationResult, model: str
) -> DataPointResult:
    """Convert one SimulationResult into a DataPointResult."""
    metadata = dict(result.metadata or {})
    persona = metadata.get("persona")
    scenario = metadata.get("scenario")
    raw_error = metadata.get("error")
    error = raw_error if isinstance(raw_error, str) else None

    inputs: dict[str, Any] = {}
    if persona:
        inputs["persona"] = persona
    if scenario:
        inputs["scenario"] = scenario

    raw_scores = metadata.get("evaluator_scores") or {}
    evaluator_scores: list[EvaluatorScore] = []
    if isinstance(raw_scores, dict):
        for name, score in raw_scores.items():
            try:
                evaluator_scores.append(
                    EvaluatorScore(
                        evaluator_name=str(name),
                        score=EvaluationResult(value=score),
                    )
                )
            except Exception as e:
                logger.warning(
                    "Skipping evaluator score %r (could not coerce value=%r): %s",
                    name,
                    score,
                    e,
                )

    job_result = JobResult(
        job_name=JOB_NAME,
        output=to_open_responses(result, model),
        error=error,
        evaluator_scores=evaluator_scores or None,
    )

    return DataPointResult(
        data_point=DataPoint(inputs=inputs),
        error=error,
        job_results=[job_result],
    )


async def upload_simulation_results(
    *,
    api_key: str,
    evaluation_name: str,
    evaluation_description: str | None = None,
    results: list[SimulationResult],
    start_time: datetime,
    end_time: datetime,
    model: str,
    path: str | None = None,
) -> None:
    """Convert simulation results and upload them to the Orq platform.

    Mirrors the contract of evaluatorq's own auto-upload: failures are
    logged but not raised, so a network glitch never breaks a successful
    simulation run.
    """
    if not results:
        logger.debug("upload_simulation_results: no results to send")
        return

    # Convert per-result so one malformed SimulationResult doesn't drop the
    # whole batch. Skipped results are logged with their index for diagnosis.
    data_point_results: list[DataPointResult] = []
    for i, r in enumerate(results):
        try:
            data_point_results.append(_to_data_point_result(r, model))
        except Exception as e:
            logger.warning(
                "Skipping simulation result %d during upload conversion: %s",
                i,
                e,
            )

    if not data_point_results:
        logger.error(
            "All %d simulation results failed conversion; nothing to upload.",
            len(results),
        )
        return

    try:
        await send_results_to_orq(
            api_key=api_key,
            evaluation_name=evaluation_name,
            evaluation_description=evaluation_description,
            dataset_id=None,
            results=data_point_results,
            start_time=start_time,
            end_time=end_time,
            path=path,
        )
    except Exception as e:
        logger.error(
            "Failed to upload %d simulation results to Orq platform: %s. "
            "Results have been kept in memory.",
            len(data_point_results),
            e,
        )
