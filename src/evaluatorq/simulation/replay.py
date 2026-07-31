"""Rebuild a prior simulation run's datapoints so it can be replayed verbatim.

Saved runs carry the ``SimulationDatapoint`` list they simulated. Replaying
reads it back and hands it to the runner as the resolved datapoint set, so no
persona/scenario generation and no first-message generation happen: the same
conversations start again, against whatever target and evaluators the new
invocation specifies.

Runs saved before the ``datapoints`` field existed cannot be replayed — their
results keep persona and scenario *names*, not the objects the simulator needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.replay import ReplayError, load_run_payload
from evaluatorq.simulation.types import SimulationDatapoint

if TYPE_CHECKING:
    from pathlib import Path

SURFACE = 'simulation'


def load_simulation_replay(reference: str, runs_dir: Path | None = None) -> list[SimulationDatapoint]:
    """Resolve *reference* to a saved run and return the datapoints it ran.

    Raises:
        ReplayError: The run cannot be found, read, or stores no datapoints.
    """
    from pydantic import ValidationError

    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    resolved_dir = runs_dir if runs_dir is not None else get_sim_runs_dir()
    payload, path = load_run_payload(reference, resolved_dir, surface=SURFACE)

    raw = payload.get('datapoints')
    if not isinstance(raw, list) or not raw:
        raise ReplayError(
            f'Previous simulation run {path.name} stores no datapoints, so it cannot be replayed. '
            'Only runs saved by evaluatorq 1.4.0+ record the cases they ran; re-run the original '
            'configuration once to produce a replayable run.'
        )

    try:
        return [SimulationDatapoint.model_validate(row) for row in raw]
    except ValidationError as exc:
        raise ReplayError(f'Previous simulation run {path.name} has unreadable datapoints: {exc}') from exc
