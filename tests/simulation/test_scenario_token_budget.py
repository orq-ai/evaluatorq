from evaluatorq.simulation.generators.scenario_generator import _scenario_token_budget


def test_budget_scales_past_the_flat_cap():
    assert _scenario_token_budget(5) == 6000
    assert _scenario_token_budget(30) == 15000
