"""Resolve the Orq workspace slug + host for deep-links — from env only.

Deep-links now derive their host + workspace from each run's own
``experiment_url`` (``{host}/{workspace}/experiments/{id}``; see
``orq_links.parse_experiment_url``), which the web app resolves correctly for
anyone with access — no API key, no workspace config, no ``orq`` CLI. This module
is only the **fallback** for runs that never uploaded an experiment (so have no
``experiment_url``): it reads ``ORQ_WORKSPACE`` / ``ORQ_BASE_URL`` from the
environment. When those are unset, links are simply hidden.

(Historically this reverse-engineered the slug from the API key via the local
``orq`` CLI — a blocking subprocess that could 404. That is gone.)
"""

from __future__ import annotations

import os

DEFAULT_BASE_URL = 'https://my.orq.ai'


def resolve_slug() -> str | None:
    """Workspace slug from ``ORQ_WORKSPACE`` / ``ORQ_WORKSPACE_SLUG`` env, or None."""
    env = os.environ.get('ORQ_WORKSPACE') or os.environ.get('ORQ_WORKSPACE_SLUG')
    return env.strip() or None if env and env.strip() else None


def resolve_base_url() -> str:
    """Orq host from ``ORQ_BASE_URL`` env, or the prod default (no trailing slash)."""
    return (os.environ.get('ORQ_BASE_URL') or DEFAULT_BASE_URL).rstrip('/')


def classify_host(url: str | None) -> str:
    """Label a host as 'Production' / 'Staging' / 'On-prem' for display.

    Environment can only be told from the host we talk to (a workspace UUID
    carries none): ``my.orq.ai`` is prod; any ``*.orq.ai`` marked staging/dev is
    shared pre-prod; anything else is a self-hosted (on-prem) deployment.
    """
    host = (url or DEFAULT_BASE_URL).strip().lower()
    if 'my.orq.ai' in host and 'staging' not in host and 'dev' not in host:
        return 'Production'
    if '.orq.ai' in host and ('staging' in host or 'dev' in host):
        return 'Staging'
    return 'On-prem'
