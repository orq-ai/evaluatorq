"""Insights, signals and apply resolve their models through `role_model`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluatorq.common.model_roles import ROLE_ENV, role_model
from evaluatorq.insights.models import InsightsConfig
from evaluatorq.signals.config import ClassifierConfig


@pytest.fixture(autouse=True)
def isolated_roles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in (*ROLE_ENV.values(), 'EVALUATORQ_APPLY_MODEL', 'EVALUATORQ_COMPILER_MODEL'):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / 'dashboard-settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))
    return path


def test_insights_config_resolves_models_by_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    config = InsightsConfig(labels=[], dimensions=[])
    assert config.summary_model == 'env/smart'
    assert config.embedding_model == 'openai/text-embedding-3-small'
    assert config.classifier_model == role_model('classifier', task='insights.labels')


def test_signals_classifier_reads_the_settings_file(isolated_roles: Path) -> None:
    isolated_roles.write_text(json.dumps({'classifier_model': 'file/classifier'}))
    assert ClassifierConfig().model == 'file/classifier'


def test_apply_model_is_the_smart_role(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.apply_ui import apply_model

    assert apply_model() == role_model('smart', task='apply')
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    assert apply_model() == 'env/smart'


@pytest.mark.asyncio
async def test_insights_run_labels_on_insights_labels_and_searches_on_finder_classifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.common.model_roles import set_cli_models
    from evaluatorq.insights import pipeline
    from evaluatorq.insights.models import InsightsPopulation
    from tests.insights.test_pipeline import _labels, _patch_clients, _resolved, _trace

    traces = [_trace(1)]
    seen: dict[str, str] = {}

    async def resolve(*args, classifier_model: str, **kwargs):
        seen['population'] = classifier_model
        return _resolved(traces)

    async def label(*args, model: str, **kwargs):
        seen['labels'] = model
        return _labels(traces)

    _patch_clients(monkeypatch)
    monkeypatch.setattr(pipeline, 'resolve_population', resolve)
    monkeypatch.setattr(pipeline, 'label_traces', label)
    set_cli_models(overrides={'insights.labels': 'override/labels', 'finder.classifier': 'override/finder'})
    try:
        run = await pipeline.insights(
            InsightsPopulation(query='refund requests'),
            dimensions=(),
            labels=('user_frustration',),
            runs_dir=tmp_path / 'runs',
        )
    finally:
        set_cli_models()

    assert seen == {'population': 'override/finder', 'labels': 'override/labels'}
    assert run.config.classifier_model == 'override/labels'
    assert run.population['finder_classifier_model'] == 'override/finder'
