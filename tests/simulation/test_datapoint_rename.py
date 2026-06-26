"""RES-911: simulation Datapoint renamed to SimulationDatapoint (no alias)."""

from __future__ import annotations

import pytest


def test_simulation_datapoint_is_canonical_name() -> None:
    from evaluatorq.simulation import SimulationDatapoint

    assert SimulationDatapoint.__name__ == "SimulationDatapoint"


def test_old_datapoint_name_is_gone() -> None:
    """Agent Simulation is unreleased, so the old name is dropped outright — no
    backward-compat alias."""
    import evaluatorq.simulation as sim
    import evaluatorq.simulation.types as sim_types

    assert "Datapoint" not in sim.__all__
    with pytest.raises(AttributeError):
        getattr(sim, "Datapoint")
    with pytest.raises(AttributeError):
        getattr(sim_types, "Datapoint")


def test_simulation_datapoint_constructs() -> None:
    from evaluatorq.simulation import SimulationDatapoint
    from evaluatorq.simulation.types import CommunicationStyle, Persona, Scenario

    dp = SimulationDatapoint(
        id="dp-1",
        persona=Persona(
            name="P",
            patience=0.5,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.5,
            communication_style=CommunicationStyle.casual,
            background="b",
        ),
        scenario=Scenario(name="S", goal="g"),
        user_system_prompt="sys",
        first_message="hi",
    )
    assert dp.id == "dp-1"
