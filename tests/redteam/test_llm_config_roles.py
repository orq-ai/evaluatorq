"""An attacker config that never names a model follows the smart role."""

from __future__ import annotations

import pytest

from evaluatorq.common.model_roles import set_cli_models
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.redteam.contracts import LLMConfig


@pytest.fixture(autouse=True)
def _smart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')


def test_default_attacker_follows_smart() -> None:
    assert LLMConfig().attacker.model == 'env/smart'


def test_bare_call_config_follows_smart() -> None:
    assert LLMConfig(attacker=LLMCallConfig()).attacker.model == 'env/smart'


def test_call_config_with_only_temperature_follows_smart_and_keeps_temperature() -> None:
    attacker = LLMConfig(attacker=LLMCallConfig(temperature=0.3)).attacker
    assert attacker.model == 'env/smart'
    assert attacker.temperature == 0.3


def test_dict_form_follows_smart() -> None:
    assert LLMConfig(attacker={'temperature': 0.7}).attacker.model == 'env/smart'  # ty: ignore[invalid-argument-type]


def test_explicit_model_is_kept() -> None:
    assert LLMConfig(attacker=LLMCallConfig(model='mine/x')).attacker.model == 'mine/x'


def test_task_override_beats_smart_for_an_unset_model() -> None:
    set_cli_models(overrides={'redteam.attacker': 'task/a'})
    assert LLMConfig(attacker=LLMCallConfig()).attacker.model == 'task/a'
