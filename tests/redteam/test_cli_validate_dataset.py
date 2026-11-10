"""Tests for the redteam dataset validation CLI command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evaluatorq.redteam.cli import app


# huggingface-hub ships only with the `redteam` extra; skip rather than fail collection.
huggingface_hub = pytest.importorskip('huggingface_hub')


runner = CliRunner()


def _write_dataset(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def _valid_sample(sample_id: str = 'sample-1') -> dict[str, Any]:
    return {
        'input': {
            'id': sample_id,
            'category': 'ASI01',
            'attack_technique': 'direct-injection',
            'delivery_method': 'direct-request',
            'severity': 'high',
            'vulnerability_domain': 'agent',
            'framework': 'OWASP-ASI',
            'turn_type': 'single',
            'source': 'test',
        },
        'messages': [{'role': 'user', 'content': 'Test attack prompt.'}],
    }


def test_rejects_dataset_without_top_level_samples_key(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, {'not_samples': []})

    result = runner.invoke(app, ['validate-dataset', str(dataset)])

    assert result.exit_code == 1, result.output
    assert "FAIL: Expected top-level object with 'samples' key." in result.stderr


def test_reports_sample_validation_errors(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, {'samples': [{}]})

    result = runner.invoke(app, ['validate-dataset', str(dataset)])

    assert result.exit_code == 1, result.output
    assert 'FAIL: 2 validation error(s):' in result.stderr
    assert 'sample[0].input: Field required' in result.stderr
    assert 'sample[0].messages: Field required' in result.stderr


def test_truncates_sample_validation_errors_after_twenty(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, {'samples': [{} for _ in range(11)]})

    result = runner.invoke(app, ['validate-dataset', str(dataset)])

    assert result.exit_code == 1, result.output
    assert 'FAIL: 22 validation error(s):' in result.stderr
    assert '  ... and 2 more' in result.stderr


@pytest.mark.parametrize(
    ('category', 'expected'),
    [
        ('sensitive_information_disclosure', "'sensitive_information_disclosure' is not in the default category set"),
        ('LLM03', "Use 'ASI04'."),
        ('owasp-llm03', "Use 'ASI04'."),
    ],
)
def test_rejects_category_a_run_would_not_request(tmp_path: Path, category: str, expected: str) -> None:
    sample = _valid_sample('row-7')
    sample['input']['category'] = category
    dataset = _write_dataset(tmp_path, {'samples': [_valid_sample(), sample]})

    result = runner.invoke(app, ['validate-dataset', str(dataset)])

    assert result.exit_code == 1, result.output
    assert 'FAIL: 1 validation error(s):' in result.stderr
    assert "sample[1] (id='row-7').input.category" in result.stderr
    assert expected in result.stderr


def test_accepts_lowercase_and_prefixed_category(tmp_path: Path) -> None:
    lower, prefixed = _valid_sample('a'), _valid_sample('b')
    lower['input']['category'] = 'asi01'
    prefixed['input']['category'] = 'OWASP-LLM01'
    dataset = _write_dataset(tmp_path, {'samples': [lower, prefixed]})

    result = runner.invoke(app, ['validate-dataset', str(dataset)])

    assert result.exit_code == 0, result.output
    assert 'OK: All 2 samples are valid.' in result.output


def test_rejects_unresolvable_vulnerability(tmp_path: Path) -> None:
    sample = _valid_sample()
    del sample['input']['category']
    sample['input']['vulnerability'] = 'sensitive_information_disclosure'
    dataset = _write_dataset(tmp_path, {'samples': [sample]})

    result = runner.invoke(app, ['validate-dataset', str(dataset)])

    assert result.exit_code == 1, result.output
    assert (
        "sample[0] (id='sample-1').input: Value error, Unknown vulnerability 'sensitive_information_disclosure'"
        in result.stderr
    )


def test_reports_default_huggingface_download_failure(monkeypatch: Any) -> None:
    def raise_download(**_: object) -> str:
        raise RuntimeError('network down')

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', raise_download)

    result = runner.invoke(app, ['validate-dataset'])

    assert result.exit_code == 1, result.output
    assert (
        'Failed to download dataset from HuggingFace '
        '(orq/redteam-vulnerabilities/redteam_dataset.v2.json): network down. Check your network '
        'connection, that the repository exists, and your access '
        '(set HF_TOKEN for gated/private datasets).'
    ) in result.stderr


def test_reports_named_huggingface_download_failure(monkeypatch: Any) -> None:
    def raise_download(**_: object) -> str:
        raise RuntimeError('network down')

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', raise_download)

    result = runner.invoke(app, ['validate-dataset', 'hf:org/repo/file.json'])

    assert result.exit_code == 1, result.output
    assert (
        'Failed to download dataset from HuggingFace (org/repo/file.json): network down. '
        'Check your network connection, that the repository exists, and your '
        'access (set HF_TOKEN for gated/private datasets).'
    ) in result.stderr


def test_reports_missing_huggingface_before_malformed_source(monkeypatch: Any) -> None:
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)

    result = runner.invoke(app, ['validate-dataset', 'hf:malformed'])

    assert result.exit_code == 1, result.output
    assert 'huggingface-hub not installed' in result.stderr


def test_reports_unreadable_downloaded_file(monkeypatch: Any) -> None:
    missing_path = '/does/not/exist/dataset.json'

    def return_missing_path(**_: object) -> str:
        return missing_path

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', return_missing_path)

    result = runner.invoke(app, ['validate-dataset', 'hf:org/repo/file.json'])

    assert result.exit_code == 1, result.output
    assert 'Failed to read dataset file:' in result.stderr
    assert missing_path in result.stderr


def test_accepts_valid_local_dataset(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, {'samples': [_valid_sample()]})

    result = runner.invoke(app, ['validate-dataset', str(dataset)])

    assert result.exit_code == 0, result.output
    assert 'OK: All 1 samples are valid.' in result.stdout
