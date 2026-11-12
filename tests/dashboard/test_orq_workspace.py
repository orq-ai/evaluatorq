"""Tests for environment workspace links and saved profile host resolution."""

from __future__ import annotations

import os
import threading
import time

import pytest

from evaluatorq.dashboard import orq_workspace as ow
from evaluatorq.dashboard.trace_links import thread_trace_url
from tests.dashboard.conftest import settle_workspace_lookup


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for var in ('ORQ_WORKSPACE', 'ORQ_WORKSPACE_SLUG', 'ORQ_BASE_URL', 'ORQ_API_KEY'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'empty-settings.json'))
    monkeypatch.setattr(ow.shutil, 'which', lambda _name: None)


# --- workspace slug ---------------------------------------------------------


def test_resolve_slug_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_WORKSPACE', 'orq-research')
    assert ow.resolve_slug() == 'orq-research'


def test_resolve_slug_alias_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'alias-ws')
    assert ow.resolve_slug() == 'alias-ws'


def test_resolve_slug_none_when_unset() -> None:
    assert ow.resolve_slug() is None


def test_cli_oauth_selected_server_and_workspace_drive_fallback_links(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.trace_finder.settings import DashboardSettings, save_settings

    monkeypatch.setenv('ORQ_WORKSPACE', 'unrelated-environment-workspace')
    save_settings(DashboardSettings.model_validate({
        'orq_auth_method': 'cli_oauth',
        'orq_oauth_server': 'https://selected.orq.ai',
        'orq_workspace': 'selected-workspace',
    }))

    assert ow.resolve_base_url() == 'https://selected.orq.ai'
    assert ow.resolve_slug() == 'selected-workspace'


def test_resolve_slug_from_authenticated_cli_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.orq_scope import OrqScope

    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    monkeypatch.setenv('ORQ_API_KEY', 'test-project-key')
    calls: list[tuple[str | None, bool]] = []

    def discover(profile: str | None, *, use_cli_session: bool = False) -> OrqScope:
        calls.append((profile, use_cli_session))
        return OrqScope(workspace_key='orq-research', workspace_id='research-id')

    monkeypatch.setattr('evaluatorq.dashboard.orq_scope.discover_orq_scope', discover)

    assert ow.resolve_slug() is None
    settle_workspace_lookup()
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
    ow.resolve_slug()
    settle_workspace_lookup()
    assert ow.resolve_slug() == 'orq-research'
    assert calls == [(None, True)]


def test_first_render_does_not_wait_for_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.orq_scope import OrqScope

    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    release = threading.Event()

    def discover(profile: str | None, *, use_cli_session: bool = False) -> OrqScope:
        assert release.wait(timeout=5)
        return OrqScope(workspace_key='orq-research')

    monkeypatch.setattr('evaluatorq.dashboard.orq_scope.discover_orq_scope', discover)
    started = time.monotonic()
    first = thread_trace_url('thread-1')
    assert time.monotonic() - started < 0.5
    assert first is None
    assert ow.slug_lookup_pending()

    release.set()
    settle_workspace_lookup()
    assert not ow.slug_lookup_pending()
    later = thread_trace_url('thread-1')
    assert later is not None and '/orq-research/traces?' in later


def test_a_cli_account_switch_starts_a_new_lookup_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.orq_scope import OrqScope

    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    slugs = iter(('first-workspace', 'second-workspace'))
    monkeypatch.setattr(
        'evaluatorq.dashboard.orq_scope.discover_orq_scope',
        lambda _profile, *, use_cli_session=False: OrqScope(workspace_key=next(slugs)),
    )
    ow.resolve_slug()
    settle_workspace_lookup()
    assert ow.resolve_slug() == 'first-workspace'

    # `orq` rewrites its config when the account changes; the cached slug must not survive it.
    ow._CLI_CONFIG_DIR.mkdir()
    (ow._CLI_CONFIG_DIR / 'config.json').write_text('{}')
    assert ow.resolve_slug() is None
    settle_workspace_lookup()
    assert ow.resolve_slug() == 'second-workspace'


def test_a_dashboard_profile_switch_does_not_reuse_the_old_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.orq_scope import OrqScope
    from evaluatorq.trace_finder.settings import DashboardSettings, save_settings

    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    monkeypatch.setattr(
        'evaluatorq.dashboard.orq_scope.discover_orq_scope',
        lambda profile, *, use_cli_session=False: OrqScope(workspace_key=f'{profile}-workspace'),
    )
    for profile in ('first', 'second'):
        save_settings(DashboardSettings.model_validate({'orq_auth_method': 'cli_profile', 'orq_profile': profile}))
        assert ow.resolve_slug() is None
        settle_workspace_lookup()
        assert ow.resolve_slug() == f'{profile}-workspace'


def test_report_pages_poll_until_the_lookup_lands(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.orq_scope import OrqScope
    from evaluatorq.dashboard.shell import WORKSPACE_PENDING_ROUTE, page

    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    release = threading.Event()

    def discover(profile: str | None, *, use_cli_session: bool = False) -> OrqScope:
        assert release.wait(timeout=5)
        return OrqScope(workspace_key='orq-research')

    monkeypatch.setattr('evaluatorq.dashboard.orq_scope.discover_orq_scope', discover)
    assert WORKSPACE_PENDING_ROUTE in page('Run', '', active_surface='sim')
    assert WORKSPACE_PENDING_ROUTE not in page('Settings', '', active_nav='settings')

    release.set()
    settle_workspace_lookup()
    assert WORKSPACE_PENDING_ROUTE not in page('Run', '', active_surface='sim')


def test_pending_route_reloads_only_when_the_lookup_found_a_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.testclient import TestClient

    from evaluatorq.dashboard.app import build_app
    from evaluatorq.dashboard.orq_scope import OrqScope
    from evaluatorq.dashboard.shell import WORKSPACE_PENDING_ROUTE

    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    release = threading.Event()
    found: list[str | None] = ['orq-research']

    def discover(profile: str | None, *, use_cli_session: bool = False) -> OrqScope:
        assert release.wait(timeout=5)
        return OrqScope(workspace_key=found[0])

    monkeypatch.setattr('evaluatorq.dashboard.orq_scope.discover_orq_scope', discover)
    client = TestClient(build_app([]))
    ow.resolve_slug()
    assert client.get(WORKSPACE_PENDING_ROUTE).status_code == 204
    release.set()
    settle_workspace_lookup()
    assert client.get(WORKSPACE_PENDING_ROUTE).headers['HX-Refresh'] == 'true'

    # A lookup that finds nothing stops the poll (htmx 286) instead of reloading for nothing.
    found[0] = None
    ow._cli_slug_cache.clear()
    ow.resolve_slug()
    settle_workspace_lookup()
    stopped = client.get(WORKSPACE_PENDING_ROUTE)
    assert stopped.status_code == 286
    assert 'HX-Refresh' not in stopped.headers


def test_a_lookup_for_another_credential_does_not_count_as_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    monkeypatch.setattr(ow, '_pending', {('other-profile', None, None): threading.Thread()})
    assert not ow.slug_lookup_pending()


def test_a_new_cli_session_replaces_the_old_cache_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.orq_scope import OrqScope

    monkeypatch.setattr(ow.shutil, 'which', lambda _name: '/usr/bin/orq')
    monkeypatch.setattr(
        'evaluatorq.dashboard.orq_scope.discover_orq_scope',
        lambda _profile, *, use_cli_session=False: OrqScope(workspace_key='ws'),
    )
    ow._CLI_CONFIG_DIR.mkdir()
    for content in ('{}', '{"current": "b"}'):
        (ow._CLI_CONFIG_DIR / 'config.json').write_text(content)
        os.utime(ow._CLI_CONFIG_DIR / 'config.json', ns=(len(content), len(content)))
        ow.resolve_slug()
        settle_workspace_lookup()
    assert len(ow._cli_slug_cache) == 1


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
