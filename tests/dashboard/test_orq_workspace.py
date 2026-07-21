"""Tests for env-based workspace/host resolution (CLI/JWT machinery removed)."""

from __future__ import annotations

import pytest

from evaluatorq.dashboard import orq_workspace as ow


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ('ORQ_WORKSPACE', 'ORQ_WORKSPACE_SLUG', 'ORQ_BASE_URL'):
        monkeypatch.delenv(var, raising=False)


# --- workspace slug ---------------------------------------------------------


def test_resolve_slug_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_WORKSPACE', 'orq-research')
    assert ow.resolve_slug() == 'orq-research'


def test_resolve_slug_alias_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'alias-ws')
    assert ow.resolve_slug() == 'alias-ws'


def test_resolve_slug_none_when_unset() -> None:
    assert ow.resolve_slug() is None


# --- host -------------------------------------------------------------------


def test_resolve_base_url_default() -> None:
    assert ow.resolve_base_url() == 'https://my.orq.ai'


def test_resolve_base_url_env_strips_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_BASE_URL', 'https://staging.orq.ai/')
    assert ow.resolve_base_url() == 'https://staging.orq.ai'


@pytest.mark.parametrize(
    ('url', 'label'),
    [
        ('https://my.orq.ai', 'Production'),
        ('https://my.staging.orq.ai', 'Staging'),
        ('https://orq.internal.acme.com', 'On-prem'),
        (None, 'Production'),
    ],
)
def test_classify_host(url: str | None, label: str) -> None:
    assert ow.classify_host(url) == label


# --- settings page renders read-only (no editable panels / POST routes) -----


def test_settings_page_is_read_only(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from starlette.testclient import TestClient

    from evaluatorq.dashboard.app import build_app

    monkeypatch.setenv('ORQ_WORKSPACE', 'orq-research')
    client = TestClient(build_app(roots=[tmp_path]), follow_redirects=False)

    page = client.get('/settings').text
    assert 'Orq workspace' in page
    assert 'orq-research' in page
    assert 'Orq host' in page
    # Editable controls and their POST routes are gone.
    assert 'action="/settings/workspace"' not in page
    assert client.post('/settings/workspace', data={'workspace': 'x'}).status_code == 404
    assert client.post('/settings/host', data={'base_url': 'x'}).status_code == 404
