"""Deep links from the dashboard into Orq trace observability.

Builds ``…/<workspace-slug>/traces?query=…`` URLs so a conversation or a whole
run can be opened in the Orq traces UI. Each run mints a ``run_id`` and its
conversations get a run-scoped ``thread_id`` built by
``evaluatorq.common.thread_context.build_thread_id`` — ``{run_id}:{index}`` for
simulations, ``{run_id}:{agent_key}:{index}`` for red-team attacks. Both share
the ``run_id`` prefix, so:

* a single conversation is matched with ``thread_id:is:<thread_id>``;
* every conversation in a run is matched with ``thread_id:contains:<run_id>``;
* a single known trace is matched directly with ``trace_id:is:<trace_id>`` — used
  to deep-link the last successful target-agent response of a conversation.

Source of host + workspace:

* The run's own ``experiment_url`` (``{host}/{workspace}/experiments/{id}``) when
  passed — the web app resolves that path for anyone with access.
* Otherwise the environment: ``ORQ_UI_BASE_URL`` / ``ORQ_BASE_URL`` for the host
  and ``ORQ_WORKSPACE`` / ``ORQ_WORKSPACE_SLUG`` for the slug. When the slug is
  unresolvable the buttons are hidden (the builders return ``None``), so we never
  render a broken link.
"""

from __future__ import annotations

import os
from urllib.parse import quote

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.orq_links import parse_experiment_url

_DEFAULT_UI_BASE = 'https://my.orq.ai'


def ui_base_url() -> str:
    """Return the Orq UI base URL (no trailing slash)."""
    base = os.environ.get('ORQ_UI_BASE_URL') or os.environ.get('ORQ_BASE_URL') or _DEFAULT_UI_BASE
    return base.rstrip('/')


def workspace_slug() -> str | None:
    """Workspace slug from ``ORQ_WORKSPACE`` / ``ORQ_WORKSPACE_SLUG`` env, or None."""
    from evaluatorq.dashboard.orq_workspace import resolve_slug

    slug = (resolve_slug() or '').strip().strip('/')
    return slug or None


def _traces_url(query: str, experiment_url: str | None) -> str | None:
    # Prefer the run's own experiment_url (host + workspace); fall back to env.
    host, slug = parse_experiment_url(experiment_url)
    if not (host and slug):
        host, slug = ui_base_url(), workspace_slug()
    if not slug:
        return None
    return f'{host.rstrip("/")}/{quote(slug, safe="")}/traces?query={quote(query, safe="")}'


def thread_trace_url(thread_id: str | None, experiment_url: str | None = None) -> str | None:
    """URL that filters traces to a single conversation's thread.

    Returns ``None`` when *thread_id* is empty or no workspace can be resolved.
    """
    if not thread_id:
        return None
    return _traces_url(f'thread_id:is:{thread_id}', experiment_url)


def single_trace_url(trace_id: str | None, experiment_url: str | None = None) -> str | None:
    """URL that opens a single known trace directly (by its trace id).

    Used to deep-link the last successful target-agent response of a
    conversation/attack, rather than filtering the whole thread. Returns ``None``
    when *trace_id* is empty or no workspace can be resolved.
    """
    if not trace_id:
        return None
    return _traces_url(f'trace_id:is:{trace_id}', experiment_url)


def run_trace_url(run_id: str | None, experiment_url: str | None = None) -> str | None:
    """URL that filters traces to every conversation in a run.

    Uses the ``contains`` operator so it matches all ``{run_id}:{seq}`` threads.
    Returns ``None`` when *run_id* is empty or no workspace can be resolved.
    """
    if not run_id:
        return None
    return _traces_url(f'thread_id:contains:{run_id}', experiment_url)


def trace_link_button(
    url: str | None,
    label: str,
    *,
    onclick: str | None = None,
    extra_attributes: dict[str, str | None] | None = None,
) -> str:
    """Render a trace deep-link as a secondary button, or '' when *url* is None.

    ``onclick`` is for callers that embed the button inside a ``<summary>`` and
    need ``event.stopPropagation()`` so clicking the link doesn't toggle the
    surrounding ``<details>``. ``extra_attributes`` adds escaped HTML
    attributes; use a ``None`` value for a boolean attribute.
    """
    if not url:
        return ''
    onclick_attr = f' onclick="{esc(onclick)}"' if onclick else ''
    extra_attrs = ''.join(
        f' {esc(name)}' if value is None else f' {esc(name)}="{esc(value)}"'
        for name, value in (extra_attributes or {}).items()
    )
    return (
        f'<a class="btn-secondary trace-link" href="{esc(url)}" '
        f'target="_blank" rel="noopener noreferrer"{extra_attrs}{onclick_attr}>{esc(label)}</a>'
    )
