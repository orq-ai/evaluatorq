from evaluatorq.simulation.types import DEFAULT_MODEL


def test_default_model_is_openai_gpt_5_4_mini():
    assert DEFAULT_MODEL == "openai/gpt-5.4-mini"
