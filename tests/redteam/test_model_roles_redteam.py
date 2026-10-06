from __future__ import annotations

from pathlib import Path
from typing import Any

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


@pytest.mark.parametrize(
    ('cfg_kwargs', 'expected'),
    [
        ({'evaluator': {'model': 'cfg/judge'}}, 'cfg/judge'),
        ({'call': {'model': 'cfg/judge'}}, 'cfg/judge'),
        ({'call': {'timeout_ms': 5}}, 'env/smart'),
    ],
)
def test_owasp_evaluator_takes_the_judge_a_cfg_names(
    monkeypatch: pytest.MonkeyPatch, cfg_kwargs: dict[str, dict[str, Any]], expected: str
) -> None:
    from unittest.mock import AsyncMock

    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator
    from evaluatorq.redteam.contracts import EvaluatorConfig

    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    kind, fields = next(iter(cfg_kwargs.items()))
    cfg = EvaluatorConfig(**fields) if kind == 'evaluator' else LLMCallConfig(**fields)

    assert OWASPEvaluator(cfg=cfg, llm_client=AsyncMock()).panel[0] == expected


@pytest.mark.asyncio
async def test_create_owasp_evaluator_judges_with_the_cfg_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    from evaluatorq import DataPoint
    from evaluatorq.redteam.contracts import EvaluatorConfig
    from evaluatorq.redteam.frameworks.owasp import evaluatorq_bridge

    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    asked: list[str] = []
    monkeypatch.setattr(evaluatorq_bridge, 'get_evaluator_for_category', lambda _category, model: asked.append(model))

    scorer = evaluatorq_bridge.create_owasp_evaluator(cfg=EvaluatorConfig(model='cfg/judge'), llm_client=AsyncMock())
    await scorer['scorer']({'data': DataPoint(inputs={'category': 'ASI01', 'messages': []}), 'output': {'response': 'x'}})

    assert asked == ['cfg/judge']


def test_an_evaluator_without_cfg_keeps_the_pipeline_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    from evaluatorq.redteam import contracts
    from evaluatorq.redteam.adaptive import evaluator

    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    tuned = contracts.PIPELINE_CONFIG.evaluator.model_copy(update={'timeout_ms': 1234})
    monkeypatch.setattr(contracts.PIPELINE_CONFIG, 'evaluator', tuned)

    built = evaluator.OWASPEvaluator(llm_client=AsyncMock())

    assert built._call_cfg.timeout_ms == 1234
    assert built.evaluator_model == 'env/smart'
