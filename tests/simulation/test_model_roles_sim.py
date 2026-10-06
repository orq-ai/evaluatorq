"""The simulated user, generators and summaries run on the fast role; the judge on the smart role."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from evaluatorq.common.model_roles import set_cli_models

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    for name in (
        'EVALUATORQ_FAST_MODEL',
        'EVALUATORQ_SMART_MODEL',
        'EVALUATORQ_CLASSIFIER_MODEL',
        'EVALUATORQ_COMPILER_MODEL',
        'EVALUATORQ_APPLY_MODEL',
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    set_cli_models()
    yield
    set_cli_models()


def _persona():
    from evaluatorq.simulation.types import CommunicationStyle, Persona

    return Persona(
        name='Test User',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='bg',
    )


def _scenario():
    from evaluatorq.simulation.types import Scenario

    return Scenario(name='S', goal='g')


def test_default_user_fast_judge_smart(monkeypatch):
    monkeypatch.setenv('EVALUATORQ_FAST_MODEL', 'env/fast')
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    from evaluatorq.simulation._config import SimulationConfig, sim_role_config

    cfg = SimulationConfig()
    assert sim_role_config(cfg.llm_config, 'sim.user').model == 'env/fast'
    assert sim_role_config(cfg.llm_config, 'sim.judge').model == 'env/smart'


def test_explicit_model_applies_to_every_role():
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation._config import sim_role_config

    cfg = LLMCallConfig(model='x/y', temperature=0.3)
    assert sim_role_config(cfg, 'sim.user').model == 'x/y'
    assert sim_role_config(cfg, 'sim.judge').model == 'x/y'
    assert sim_role_config(cfg, 'sim.judge').temperature == 0.3


def test_none_config_resolves_role():
    from evaluatorq.simulation._config import sim_role_config

    assert sim_role_config(None, 'sim.user').model == 'openai/gpt-6-luna'
    assert sim_role_config(None, 'sim.judge').model == 'openai/gpt-6-sol'


def test_agents_built_directly_pick_their_role(monkeypatch):
    monkeypatch.setenv('EVALUATORQ_FAST_MODEL', 'env/fast')
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    from evaluatorq.simulation._config import SimulationConfig
    from evaluatorq.simulation.agents.judge import JudgeAgent
    from evaluatorq.simulation.agents.user_simulator import UserSimulatorAgent

    cfg = SimulationConfig().llm_config
    assert UserSimulatorAgent(cfg)._model == 'env/fast'
    assert JudgeAgent(cfg)._model == 'env/smart'
    assert UserSimulatorAgent()._model == 'env/fast'
    assert JudgeAgent()._model == 'env/smart'


def test_runner_builds_user_fast_and_judge_smart(monkeypatch):
    monkeypatch.setenv('EVALUATORQ_FAST_MODEL', 'env/fast')
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    runner = SimulationRunner(target=lambda messages: 'ok')
    user, judge = runner._resolve_simulator_and_judge(persona=_persona(), scenario=_scenario(), system_prompt='')
    assert user._model == 'env/fast'
    assert judge._model == 'env/smart'


def test_runner_explicit_model_reaches_judge(monkeypatch):
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    runner = SimulationRunner(target=lambda messages: 'ok', model='x/y', llm_config=LLMCallConfig(temperature=0.2))
    user, judge = runner._resolve_simulator_and_judge(persona=_persona(), scenario=_scenario(), system_prompt='')
    assert user._model == 'x/y'
    assert judge._model == 'x/y'


def test_generators_default_to_fast_role(monkeypatch):
    monkeypatch.setenv('EVALUATORQ_FAST_MODEL', 'env/fast')
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    from evaluatorq.simulation.generators import (
        DatapointGenerator,
        FirstMessageGenerator,
        PersonaGenerator,
        ScenarioGenerator,
    )

    for cls in (PersonaGenerator, ScenarioGenerator, FirstMessageGenerator, DatapointGenerator):
        assert cls()._model == 'env/fast', cls.__name__
    assert PersonaGenerator(model='x/y')._model == 'x/y'
