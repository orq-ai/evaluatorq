"""The ``recommendations=`` flag resolves the same way on both surfaces."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluatorq.common.recommendations import resolve_recommendations
from evaluatorq.redteam.contracts import RedTeamRecommendationConfig
from evaluatorq.simulation.reports import SimulationRecommendationConfig


@pytest.mark.parametrize('config_cls', [RedTeamRecommendationConfig, SimulationRecommendationConfig])
def test_true_means_defaults(config_cls: type) -> None:
    assert resolve_recommendations(True, config_cls) == config_cls()


@pytest.mark.parametrize('config_cls', [RedTeamRecommendationConfig, SimulationRecommendationConfig])
def test_false_means_skip(config_cls: type) -> None:
    assert resolve_recommendations(False, config_cls) is None


def test_instance_passes_through() -> None:
    tuned = RedTeamRecommendationConfig(max_areas=2, max_traces=3)
    assert resolve_recommendations(tuned, RedTeamRecommendationConfig) is tuned

    tuned_sim = SimulationRecommendationConfig(max_suggestions=7)
    assert resolve_recommendations(tuned_sim, SimulationRecommendationConfig) is tuned_sim


@pytest.mark.parametrize('config_cls', [RedTeamRecommendationConfig, SimulationRecommendationConfig])
def test_unknown_field_rejected(config_cls: type) -> None:
    """A misspelled knob must raise, not silently fall back to the default."""
    with pytest.raises(ValidationError):
        config_cls(max_are4s=2)


@pytest.mark.parametrize('entry_point', ['simulate', 'generate_and_simulate'])
@pytest.mark.asyncio
async def test_public_simulation_entry_points_forward_recommendations(
    entry_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag is reachable from the SDK, not just the CLI — a saved run carries
    suggestions without dropping to the private ``_*_run`` helpers."""
    from unittest.mock import MagicMock

    from evaluatorq.simulation import api

    captured: dict[str, object] = {}

    async def fake_run(**kwargs: object) -> object:
        captured.update(kwargs)
        return MagicMock(results=[])

    monkeypatch.setattr(api, f'_{entry_point}_run', fake_run)

    tuned = SimulationRecommendationConfig(max_suggestions=2)
    await getattr(api, entry_point)(target=lambda _messages: 'ok', recommendations=tuned)

    assert captured['recommendations'] is tuned
