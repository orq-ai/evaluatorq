"""Rebuild a prior red-team run's datapoints so it can be replayed verbatim.

Saved runs carry a ``datapoints`` key (the exact ``DataPoint.inputs`` fed to the
pipeline, written by ``_auto_save_run``). Replaying reads them back and hands
them to the runner as pre-built datapoints, so no strategy planning, attack
generation, or dataset load happens: the same attacks run again, against
whatever target and evaluators the new invocation specifies.

Runs saved before this existed have no ``datapoints`` key and cannot be
replayed — the report alone does not carry the dynamic strategy definitions the
orchestrator needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from evaluatorq.common.replay import ReplayError, load_run_payload
from evaluatorq.redteam.contracts import Pipeline

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.types import DataPoint

SURFACE = 'red team'

DATAPOINTS_KEY = 'datapoints'
"""Key under which a saved run stores the raw inputs of every datapoint it ran."""


@dataclass(frozen=True)
class RedTeamReplay:
    """A prior run's cases, ready to be re-run."""

    datapoints: list[DataPoint]
    pipeline: Pipeline
    categories: list[str]
    run_name: str
    path: Path


def _infer_pipeline(datapoints: list[DataPoint], stored: Any) -> Pipeline:
    """Trust the stored pipeline; fall back to the datapoints' hybrid tags."""
    try:
        return Pipeline(stored)
    except ValueError:
        sources = {dp.inputs.get('hybrid_source') for dp in datapoints}
        if sources == {'static'}:
            return Pipeline.STATIC
        return Pipeline.HYBRID if 'static' in sources else Pipeline.DYNAMIC


def load_redteam_replay(reference: str, runs_dir: Path) -> RedTeamReplay:
    """Resolve *reference* to a saved run and rebuild its datapoints.

    Raises:
        ReplayError: The run cannot be found, read, or has no stored datapoints.
    """
    from evaluatorq.types import DataPoint

    payload, path = load_run_payload(reference, runs_dir, surface=SURFACE)

    raw = payload.get(DATAPOINTS_KEY)
    if not isinstance(raw, list) or not raw:
        raise ReplayError(
            f'Previous red team run {path.name} stores no datapoints, so it cannot be replayed. '
            'Only runs saved by evaluatorq 1.4.0+ record the cases they ran; re-run the original '
            'configuration once to produce a replayable run.'
        )

    datapoints: list[DataPoint] = []
    for i, inputs in enumerate(raw):
        if not isinstance(inputs, dict):
            raise ReplayError(f'Previous red team run {path.name}: datapoint {i} is not an object.')
        datapoints.append(DataPoint(inputs=dict(inputs)))

    categories = list(dict.fromkeys(str(c) for dp in datapoints if (c := dp.inputs.get('category'))))

    return RedTeamReplay(
        datapoints=datapoints,
        pipeline=_infer_pipeline(datapoints, payload.get('pipeline')),
        categories=categories,
        run_name=str(payload.get('run_name') or path.stem),
        path=path,
    )
