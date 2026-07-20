"""Orq traces deep-link builders — URL formatting and the hide-when-unconfigured
rule (no workspace slug ⇒ no button, never a broken link)."""

from __future__ import annotations

import pytest

from evaluatorq.dashboard import trace_links


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ('ORQ_UI_BASE_URL', 'ORQ_BASE_URL', 'ORQ_WORKSPACE_SLUG', 'ORQ_WORKSPACE'):
        monkeypatch.delenv(var, raising=False)


def test_urls_none_without_workspace_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    # No slug configured → both builders return None so no button renders.
    assert trace_links.thread_trace_url('run1:0') is None
    assert trace_links.run_trace_url('run1') is None
    assert trace_links.trace_link_button(None, 'x') == ''


def test_thread_url_is_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'orq-research')
    url = trace_links.thread_trace_url('run1:3')
    assert url == 'https://my.orq.ai/orq-research/traces?query=thread_id%3Ais%3Arun1%3A3'


def test_run_url_contains_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'orq-research')
    url = trace_links.run_trace_url('run1')
    assert url == 'https://my.orq.ai/orq-research/traces?query=thread_id%3Acontains%3Arun1'


def test_ui_base_falls_back_to_orq_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_BASE_URL', 'https://acme.orq.ai/')
    monkeypatch.setenv('ORQ_WORKSPACE', 'acme')
    url = trace_links.thread_trace_url('t:0')
    assert url is not None and url.startswith('https://acme.orq.ai/acme/traces?query=')


def test_empty_ids_return_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'orq-research')
    assert trace_links.thread_trace_url(None) is None
    assert trace_links.thread_trace_url('') is None
    assert trace_links.run_trace_url(None) is None


def test_button_renders_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    html = trace_links.trace_link_button('https://x/y?query=a%3Ab', 'View traces')
    assert 'href="https://x/y?query=a%3Ab"' in html
    assert 'View traces' in html
    assert 'target="_blank"' in html


def test_button_renders_extra_attributes() -> None:
    html = trace_links.trace_link_button(
        'https://x/y',
        'View traces',
        extra_attributes={'data-no-drawer': None, 'data-origin': 'conversation'},
    )

    assert 'data-no-drawer' in html
    assert 'data-origin="conversation"' in html


if __name__ == '__main__':
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-q']))
