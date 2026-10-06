from __future__ import annotations

from pathlib import Path

import pytest

from evaluatorq.common.model_roles import set_cli_models


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
    set_cli_models()
    yield
    set_cli_models()


def test_llm_config_defaults_to_smart_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam.contracts import LLMConfig

    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    cfg = LLMConfig()
    assert cfg.attacker.model == 'env/smart'
    assert cfg.evaluator.judges == ['env/smart']


def test_attacker_task_override() -> None:
    from evaluatorq.redteam.contracts import LLMConfig

    set_cli_models(overrides={'redteam.attacker': 'a/b'})
    assert LLMConfig().attacker.model == 'a/b'
    assert LLMConfig().evaluator.judges[0] == 'openai/gpt-6-sol'


def test_explicit_model_wins() -> None:
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.redteam.contracts import LLMConfig

    assert LLMConfig(attacker=LLMCallConfig(model='x/y')).attacker.model == 'x/y'


def test_jury_default_is_smart(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.llm_jury import _resolve_panel

    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    assert _resolve_panel(None, None) == ['env/smart']
