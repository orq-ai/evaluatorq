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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from evaluatorq.common.replay import ReplayError, load_run_payload
from evaluatorq.simulation.types import SimulationDatapoint

if TYPE_CHECKING:
    from pathlib import Path

SURFACE = 'simulation'


@dataclass(frozen=True)
class SimulationReplay:
    """A prior run's cases, plus the settings they were run under."""

    datapoints: list[SimulationDatapoint]
    max_turns: int | None = None
    """Turn cap the original run used, restored unless the caller overrides it.
    A conversation that ran to 12 turns is not the same case re-run at 10."""


def load_simulation_replay(reference: str, runs_dir: Path | None = None) -> SimulationReplay:
    """Resolve *reference* to a saved run and return the cases it ran.

    Raises:
        ReplayError: The run cannot be found, read, or stores no datapoints.
    """
    from pydantic import ValidationError

    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    resolved_dir = runs_dir if runs_dir is not None else get_sim_runs_dir()
    payload, path = load_run_payload(reference, resolved_dir, surface=SURFACE)

    # Check the discriminator first, unconditionally: a red team run saved by
    # this same version carries a non-empty top-level ``datapoints`` too, so
    # gating this on an empty list would let it through to fail as a pydantic
    # dump instead of the redirect. ``pipeline`` is red-team-only.
    if payload.get('pipeline') is not None:
        raise ReplayError(
            f'{path.name} is a red team run, not a simulation run. '
            f'Replay it with: eq redteam run --from-run {path.name} --target <target>'
        )

    raw = payload.get('datapoints')
    if not isinstance(raw, list) or not raw:
        raise ReplayError(
            f'{path.name} records no simulation datapoints, so it cannot be replayed. Replay needs the '
            'cases themselves, which only runs saved after replay support existed carry. Re-run the '
            'original configuration once to produce a replayable run.'
        )

    try:
        datapoints = [SimulationDatapoint.model_validate(row) for row in raw]
    except ValidationError as exc:
        raise ReplayError(f'Previous simulation run {path.name} has unreadable datapoints: {exc}') from exc

    stored_turns = payload.get('max_turns')
    return SimulationReplay(datapoints=datapoints, max_turns=stored_turns if isinstance(stored_turns, int) else None)
