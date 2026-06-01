"""Domain-specific exceptions for the evaluatorq.simulation package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluatorq.simulation.types import SimulationResult


class SimulationError(Exception):
    """Base exception for all agent-simulation errors."""


class SimulationCancelledError(SimulationError):
    """Simulation run was declined/cancelled by the user via the on_confirm hook."""


class SimulationDroppedError(SimulationError):
    """Raised when simulation job(s) produced no result and were dropped.

    Subclass of ``SimulationError`` so ``except SimulationError:`` catches the
    cache-miss path (parity with ``SimulationCancelledError``).

    ``partial_results`` carries the ``SimulationResult`` objects for the rows
    that *did* succeed, so ``_simulate_core`` can still hand them to
    ``on_run_complete`` instead of an empty list when the run is aborted.
    """

    def __init__(self, message: str, partial_results: list[SimulationResult] | None = None) -> None:
        super().__init__(message)
        self.partial_results: list[SimulationResult] = partial_results or []
