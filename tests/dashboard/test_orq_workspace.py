"""Tests for environment workspace links and saved profile host resolution."""

from __future__ import annotations

import asyncio
import threading

import pytest

from evaluatorq.dashboard import orq_workspace as ow
from evaluatorq.dashboard.trace_links import thread_trace_url


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for var in ('ORQ_WORKSPACE', 'ORQ_WORKSPACE_SLUG', 'ORQ_BASE_URL', 'ORQ_API_KEY'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'empty-settings.json'))
    monkeypatch.setattr(ow.shutil, 'which', lambda _name: None)
    ow._cli_slug_cache.clear()


# --- workspace slug ---------------------------------------------------------


def test_resolve_slug_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_WORKSPACE', 'orq-research')
    assert ow.resolve_slug() == 'orq-research'


def test_resolve_slug_alias_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'alias-ws')
    assert ow.resolve_slug() == 'alias-ws'


def test_resolve_slug_none_when_unset() -> None:
    assert ow.resolve_slug() is None


def test_resolve_slug_from_authenticated_cli_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.orq_scope import OrqScope

    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    monkeypatch.setenv('ORQ_API_KEY', 'test-project-key')
    calls: list[tuple[str | None, bool]] = []

    def discover(profile: str | None, *, use_cli_session: bool = False) -> OrqScope:
        calls.append((profile, use_cli_session))
        return OrqScope(workspace_key='orq-research', workspace_id='research-id')

    monkeypatch.setattr('evaluatorq.dashboard.orq_scope.discover_orq_scope', discover)

    assert ow.resolve_slug() == 'orq-research'
    assert ow.resolve_slug() == 'orq-research'
    assert calls == [(None, False)]


def test_resolve_slug_from_cli_session_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.orq_scope import OrqScope

    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    calls: list[tuple[str | None, bool]] = []

    def discover(profile: str | None, *, use_cli_session: bool = False) -> OrqScope:
        calls.append((profile, use_cli_session))
        return OrqScope(workspace_key='orq-research', workspace_id='research-id')

    monkeypatch.setattr('evaluatorq.dashboard.orq_scope.discover_orq_scope', discover)
    assert ow.resolve_slug() == 'orq-research'
    assert calls == [(None, True)]


def test_cli_session_scope_is_not_cached_across_profile_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.orq_scope import OrqScope

    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    slugs = iter(('first-workspace', 'second-workspace'))
    calls = 0

    def discover(profile: str | None, *, use_cli_session: bool = False) -> OrqScope:
        nonlocal calls
        calls += 1
        assert profile is None
        assert use_cli_session is True
        return OrqScope(workspace_key=next(slugs))

    monkeypatch.setattr('evaluatorq.dashboard.orq_scope.discover_orq_scope', discover)
    assert ow.resolve_slug() == 'first-workspace'
    assert ow.resolve_slug() == 'second-workspace'
    assert calls == 2


def test_cli_session_slug_is_cached_only_for_one_render(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.orq_scope import OrqScope

    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    slugs = iter(('first-workspace', 'second-workspace'))
    calls = 0

    def discover(profile: str | None, *, use_cli_session: bool = False) -> OrqScope:
        nonlocal calls
        calls += 1
        assert profile is None
        assert use_cli_session is True
        return OrqScope(workspace_key=next(slugs))

    monkeypatch.setattr('evaluatorq.dashboard.orq_scope.discover_orq_scope', discover)
    with ow.cli_slug_render_scope():
        first = thread_trace_url('thread-1')
        second = thread_trace_url('thread-2')
    assert calls == 1
    assert first is not None and '/first-workspace/traces?' in first
    assert second is not None and '/first-workspace/traces?' in second

    # A new render resolves the CLI's newly selected profile again.
    with ow.cli_slug_render_scope():
        third = thread_trace_url('thread-3')
    assert calls == 2
    assert third is not None and '/second-workspace/traces?' in third


def test_uncached_slug_lookup_does_not_block_running_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.orq_scope import OrqScope

    monkeypatch.setenv('ORQ_API_KEY', 'slow-key')
    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    ow._cli_slug_cache.clear()
    started = threading.Event()
    release = threading.Event()

    def discover(profile: str | None, *, use_cli_session: bool = False) -> OrqScope:
        started.set()
        assert release.wait(timeout=2)
        return OrqScope(workspace_key='orq-research')

    monkeypatch.setattr('evaluatorq.dashboard.orq_scope.discover_orq_scope', discover)

    async def exercise() -> None:
        lookup = asyncio.create_task(asyncio.to_thread(ow.resolve_slug))
        assert await asyncio.to_thread(started.wait, 1)
        # The loop can run this coroutine while the CLI lookup remains blocked.
        await asyncio.sleep(0)
        release.set()
        assert await lookup == 'orq-research'

    asyncio.run(exercise())
@pytest.mark.parametrize('method', ['environment', 'cli_profile'])
def test_legacy_saved_workspace_does_not_create_trace_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path, method: str
) -> None:
    from evaluatorq.trace_finder.settings import DashboardSettings, save_settings

    path = tmp_path / 'settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))
    monkeypatch.setattr(ow.shutil, 'which', lambda _name: None)
    save_settings(DashboardSettings.model_validate({
        'orq_auth_method': method, 'orq_profile': 'staging', 'orq_workspace': 'old-workspace',
    }), path)

    assert ow.resolve_slug() is None


# --- host -------------------------------------------------------------------


def test_resolve_base_url_default() -> None:
    assert ow.resolve_base_url() == 'https://my.orq.ai'


def test_resolve_base_url_env_strips_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_BASE_URL', 'https://staging.orq.ai/')
    assert ow.resolve_base_url() == 'https://staging.orq.ai'


def test_saved_profile_host_wins_over_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from evaluatorq.trace_finder.settings import DashboardSettings, save_settings

    path = tmp_path / 'settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))
    monkeypatch.setenv('ORQ_BASE_URL', 'https://environment.orq.ai')
    save_settings(DashboardSettings.model_validate({
        'orq_profile': 'staging', 'orq_profile_host': 'https://staging.orq.ai/',
    }), path)

    assert ow.resolve_base_url() == 'https://staging.orq.ai'


@pytest.mark.parametrize('unsafe_host', ['javascript:alert(1)', 'https://host.example/path', 'https://host.example?x=1', 'https://user@host.example', 'https://host.example\\evil'])
def test_unsafe_saved_profile_host_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch, tmp_path, unsafe_host: str) -> None:
    from evaluatorq.trace_finder.settings import DashboardSettings, save_settings

    monkeypatch.setenv('ORQ_BASE_URL', 'https://environment.orq.ai')
    save_settings(
        DashboardSettings.model_validate({'orq_profile': 'research', 'orq_profile_host': unsafe_host}),
        tmp_path / 'settings.json',
    )
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    assert ow.resolve_base_url() == 'https://environment.orq.ai'


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


# --- workspace and host remain read-only on the editable settings page ------


def test_settings_page_keeps_workspace_and_host_read_only(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from starlette.testclient import TestClient

    from evaluatorq.dashboard.app import build_app

    monkeypatch.setenv('ORQ_WORKSPACE', 'orq-research')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'dashboard-settings.json'))
    client = TestClient(build_app(roots=[tmp_path]), follow_redirects=False)

    page = client.get('/settings').text
    assert 'Orq workspace' in page
    assert 'orq-research' in page
    assert 'Orq host' in page
    assert 'action="/settings"' in page
    # The old standalone workspace and host POST routes stay gone.
    assert 'action="/settings/workspace"' not in page
    assert client.post('/settings/workspace', data={'workspace': 'x'}).status_code == 404
    assert client.post('/settings/host', data={'base_url': 'x'}).status_code == 404
