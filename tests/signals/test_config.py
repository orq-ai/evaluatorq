"""Config defaults, file loading, versioning and the bundled thresholds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluatorq.signals import SignalsConfig, TagThresholds
from evaluatorq.signals.config import ClassifierConfig, default_tool_roles


def test_defaults_carry_the_research_tool_roles() -> None:
    roles = SignalsConfig().tool_roles
    assert {roles[n] for n in ('Bash', 'bash', 'exec', 'exec_command', 'orq_shell', 'run_shell_command')} == {'bash'}
    assert roles['WebFetch'] == 'webview'
    assert roles['Skill'] == 'skill'
    assert {roles[n] for n in ('Agent', 'Task', 'call_sub_agent')} == {'subagent'}
    assert SignalsConfig().tool_role('Read') == 'other'
    assert SignalsConfig().error_detection == 'status'
    assert SignalsConfig().retry_window == 3
    assert SignalsConfig().empty_values == frozenset({'', '[]', '{}', 'null'})


def test_from_file_sets_a_subset_and_keeps_the_rest(tmp_path: Path) -> None:
    path = tmp_path / 'cfg.json'
    path.write_text(json.dumps({'error_detection': 'status_and_content', 'retry_window': 5}))
    config = SignalsConfig.from_file(path)
    assert (config.error_detection, config.retry_window) == ('status_and_content', 5)
    assert config.canonicalisation == 'sorted_keys'
    assert config.tool_roles == default_tool_roles()
    assert config.alert_thresholds == SignalsConfig().alert_thresholds


def test_from_file_merges_tool_roles_over_the_defaults(tmp_path: Path) -> None:
    path = tmp_path / 'cfg.json'
    path.write_text(json.dumps({'tool_roles': {'my_shell': 'bash', 'Skill': 'other'}}))
    roles = SignalsConfig.from_file(path).tool_roles
    assert roles['my_shell'] == 'bash'
    assert roles['Skill'] == 'other'
    assert roles['Bash'] == 'bash'


def test_from_file_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / 'cfg.json'
    path.write_text(json.dumps({'error_detecton': 'status'}))
    with pytest.raises(ValidationError):
        SignalsConfig.from_file(path)


def test_direct_construction_replaces_tool_roles() -> None:
    assert SignalsConfig(tool_roles={'x': 'bash'}).tool_roles == {'x': 'bash'}


@pytest.mark.parametrize('value', [-0.1, 100.1, float('nan'), float('inf')])
def test_tag_percentiles_must_be_within_percent_range(value: float) -> None:
    with pytest.raises(ValidationError, match='between 0 and 100'):
        SignalsConfig(tag_percentiles={'error_heavy.tool_error_rate': value})


def test_classifier_model_is_read_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('EVALUATORQ_CLASSIFIER_MODEL', raising=False)
    assert ClassifierConfig().model == 'typesafe/jev-latest'
    assert ClassifierConfig().enabled is False
    monkeypatch.setenv('EVALUATORQ_CLASSIFIER_MODEL', 'other/model')
    assert ClassifierConfig().model == 'other/model'
    assert SignalsConfig().classifier.model == 'other/model'


def test_bundled_thresholds_load_once() -> None:
    bundled = TagThresholds.bundled()
    again = TagThresholds.bundled()
    assert bundled.rule_version == 'v3-local-cc-2026-09-29'
    assert bundled.thresholds['error_heavy.tool_error_rate'] == 0.0909
    assert len(bundled.percentiles['tool_error_rate']) == 101
    assert again == bundled
    assert again is not bundled


def test_bundled_thresholds_are_isolated_from_caller_mutation() -> None:
    bundled = TagThresholds.bundled()
    expected_threshold = bundled.thresholds['error_heavy.tool_error_rate']
    expected_percentile = bundled.percentiles['tool_error_rate'][0]

    bundled.thresholds['error_heavy.tool_error_rate'] = 123.0
    bundled.percentiles['tool_error_rate'][0] = 123.0

    fresh = TagThresholds.bundled()
    assert fresh.thresholds['error_heavy.tool_error_rate'] == expected_threshold
    assert fresh.percentiles['tool_error_rate'][0] == expected_percentile
    assert fresh.thresholds is not bundled.thresholds
    assert fresh.percentiles['tool_error_rate'] is not bundled.percentiles['tool_error_rate']


def test_version_is_suffixed_only_when_customised() -> None:
    bundled = TagThresholds.bundled()
    assert SignalsConfig().version == bundled.rule_version
    assert SignalsConfig(tag_thresholds=bundled).version == bundled.rule_version
    assert (
        SignalsConfig(tag_percentiles={'error_heavy.tool_error_rate': 80}).version == bundled.rule_version + '+custom'
    )
    other = bundled.model_copy(update={'thresholds': {**bundled.thresholds, 'error_heavy.tool_error_rate': 0.5}})
    assert SignalsConfig(tag_thresholds=other).version == bundled.rule_version + '+custom'
    assert SignalsConfig(tag_thresholds=other).resolved_tag_thresholds() is other


def test_cohort_threshold_interpolates_linearly() -> None:
    table = TagThresholds(rule_version='t', thresholds={}, percentiles={'m': [float(i) for i in range(101)]})
    assert table.cohort_threshold('m', 50) == 50.0
    assert table.cohort_threshold('m', 12.5) == 12.5
    assert table.cohort_threshold('m', 100) == 100.0
    assert table.cohort_threshold('m', 150) == 100.0
    assert table.cohort_threshold('m', -5) == 0.0
    with pytest.raises(KeyError):
        table.cohort_threshold('missing', 50)


@pytest.mark.parametrize(
    'values',
    [
        [],
        [0.0] * 100,
        [0.0] * 102,
        [0.0] * 100 + [float('nan')],
        [0.0] * 100 + [float('inf')],
        [1.0] + [0.0] * 100,
    ],
    ids=['empty', 'short', 'long', 'nan', 'infinite', 'decreasing'],
)
def test_invalid_percentile_tables_are_rejected(values: list[float]) -> None:
    with pytest.raises(ValidationError, match='percentile table'):
        TagThresholds(rule_version='custom', thresholds={}, percentiles={'m': values})
