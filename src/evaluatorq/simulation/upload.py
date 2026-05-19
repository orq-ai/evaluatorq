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

from typing import TYPE_CHECKING, Any

from loguru import logger

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
            evaluator_scores.append(
                EvaluatorScore(
                    evaluator_name=str(name),
                    score=EvaluationResult(value=score),
                )
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

    Raises conversion or upload errors so ``upload_results=True`` cannot
    report a successful simulation shape when the requested upload failed.
    """
    if not results:
        logger.debug("upload_simulation_results: no results to send")
        return

    data_point_results = [_to_data_point_result(r, model) for r in results]

    await send_results_to_orq(
        api_key=api_key,
        evaluation_name=evaluation_name,
        evaluation_description=evaluation_description,
        dataset_id=None,
        results=data_point_results,
        start_time=start_time,
        end_time=end_time,
        path=path,
        raise_on_error=True,
    )
