"""Domain-specific exceptions for the evaluatorq.simulation package."""


class SimulationError(Exception):
    """Base exception for all agent-simulation errors."""


class SimulationCancelledError(SimulationError):
    """Simulation run was declined/cancelled by the user via the on_confirm hook."""
