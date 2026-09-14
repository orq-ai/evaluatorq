"""Validation tests for the internal simulation configuration."""

from __future__ import annotations

import pytest

from evaluatorq.simulation._config import SimulationConfig


def test_simulation_config_rejects_duplicate_evaluator_names() -> None:
    with pytest.raises(ValueError, match="duplicate evaluator name.*'criteria_met'"):
        SimulationConfig(evaluator_names=['criteria_met', 'goal_achieved', 'criteria_met'])
