"""The ``recommendations=`` flag resolves the same way on both surfaces."""

from __future__ import annotations

import pytest

from evaluatorq.common.recommendations import resolve_recommendations
from evaluatorq.redteam.contracts import RecommendationConfig as RedTeamRecommendationConfig
from evaluatorq.simulation.reports import RecommendationConfig as SimRecommendationConfig


@pytest.mark.parametrize('config_cls', [RedTeamRecommendationConfig, SimRecommendationConfig])
def test_true_means_defaults(config_cls: type) -> None:
    assert resolve_recommendations(True, config_cls) == config_cls()


@pytest.mark.parametrize('config_cls', [RedTeamRecommendationConfig, SimRecommendationConfig])
def test_false_means_skip(config_cls: type) -> None:
    assert resolve_recommendations(False, config_cls) is None


def test_instance_passes_through() -> None:
    tuned = RedTeamRecommendationConfig(max_areas=2, max_traces=3)
    assert resolve_recommendations(tuned, RedTeamRecommendationConfig) is tuned

    tuned_sim = SimRecommendationConfig(max_suggestions=7)
    assert resolve_recommendations(tuned_sim, SimRecommendationConfig) is tuned_sim
