# tests/redteam/test_target_parsing.py
import pytest
from evaluatorq.redteam.runner import _parse_target
from evaluatorq.redteam.contracts import TargetKind


def test_agent_prefix():
    kind, value = _parse_target('agent:my-key')
    assert kind == TargetKind.AGENT
    assert value == 'my-key'


def test_deployment_prefix():
    kind, value = _parse_target('deployment:my-dep')
    assert kind == TargetKind.DEPLOYMENT
    assert value == 'my-dep'


def test_no_prefix_defaults_to_agent():
    kind, value = _parse_target('my-key')
    assert kind == TargetKind.AGENT
    assert value == 'my-key'


def test_llm_prefix_raises_with_migration_hint():
    with pytest.raises(ValueError, match='OpenAIModelTarget'):
        _parse_target('llm:gpt-4o')


def test_openai_prefix_raises_with_migration_hint():
    with pytest.raises(ValueError, match='OpenAIModelTarget'):
        _parse_target('openai:gpt-4o')


def test_backend_not_a_param():
    import inspect
    from evaluatorq.redteam.runner import red_team
    assert 'backend' not in inspect.signature(red_team).parameters


def test_attack_model_not_a_param():
    import inspect
    from evaluatorq.redteam.runner import red_team
    assert 'attack_model' not in inspect.signature(red_team).parameters


def test_evaluator_model_not_a_param():
    import inspect
    from evaluatorq.redteam.runner import red_team
    assert 'evaluator_model' not in inspect.signature(red_team).parameters


def test_llm_kwargs_not_a_param():
    import inspect
    from evaluatorq.redteam.runner import red_team
    assert 'llm_kwargs' not in inspect.signature(red_team).parameters
