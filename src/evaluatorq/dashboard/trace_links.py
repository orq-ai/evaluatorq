"""Deep links from the dashboard into Orq trace observability.

Builds ``…/<workspace-slug>/traces?query=…`` URLs so a conversation or a whole
run can be opened in the Orq traces UI. Each run mints a ``run_id`` and its
conversations get a run-scoped ``thread_id`` built by
``evaluatorq.common.thread_context.build_thread_id`` — ``{run_id}:{index}`` for
simulations, ``{run_id}:{agent_key}:{index}`` for red-team attacks. Both share
the ``run_id`` prefix, so:

* a single conversation is matched with ``thread_id:is:<thread_id>``;
* every conversation in a run is matched with ``thread_id:contains:<run_id>``.

Configuration (env):

* ``ORQ_UI_BASE_URL`` — UI host (default: ``ORQ_BASE_URL`` env, else
  ``https://my.orq.ai``).
* ``ORQ_WORKSPACE`` / ``ORQ_WORKSPACE_SLUG`` — workspace slug for the URL path.
  When neither is set the buttons are hidden (the URL builders return ``None``),
  so we never render a broken link.
"""

from __future__ import annotations

import os
from urllib.parse import quote

from evaluatorq.common.reports import esc

_DEFAULT_UI_BASE = 'https://my.orq.ai'


def ui_base_url() -> str:
    """Return the Orq UI base URL (no trailing slash)."""
    base = os.environ.get('ORQ_UI_BASE_URL') or os.environ.get('ORQ_BASE_URL') or _DEFAULT_UI_BASE
    return base.rstrip('/')


def workspace_slug() -> str | None:
    """Return the configured workspace slug, or ``None`` when unset."""
    slug = os.environ.get('ORQ_WORKSPACE') or os.environ.get('ORQ_WORKSPACE_SLUG') or ''
    slug = slug.strip().strip('/')
    return slug or None


def _traces_url(query: str) -> str | None:
    slug = workspace_slug()
    if not slug:
        return None
    return f'{ui_base_url()}/{quote(slug, safe="")}/traces?query={quote(query, safe="")}'


def thread_trace_url(thread_id: str | None) -> str | None:
    """URL that filters traces to a single conversation's thread.

    Returns ``None`` when *thread_id* is empty or no workspace slug is set.
    """
    if not thread_id:
        return None
    return _traces_url(f'thread_id:is:{thread_id}')


def run_trace_url(run_id: str | None) -> str | None:
    """URL that filters traces to every conversation in a run.

    Uses the ``contains`` operator so it matches all ``{run_id}:{seq}`` threads.
    Returns ``None`` when *run_id* is empty or no workspace slug is set.
    """
    if not run_id:
        return None
    return _traces_url(f'thread_id:contains:{run_id}')


def trace_link_button(url: str | None, label: str, *, onclick: str | None = None) -> str:
    """Render a trace deep-link as a secondary button, or '' when *url* is None.

    ``onclick`` is for callers that embed the button inside a ``<summary>`` and
    need ``event.stopPropagation()`` so clicking the link doesn't toggle the
    surrounding ``<details>``.
    """
    if not url:
        return ''
    onclick_attr = f' onclick="{esc(onclick)}"' if onclick else ''
    return (
        f'<a class="btn-secondary trace-link" href="{esc(url)}" '
        f'target="_blank" rel="noopener noreferrer"{onclick_attr}>{esc(label)}</a>'
    )
