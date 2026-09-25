"""Resolve the Orq workspace slug and host for dashboard deep-links.

Deep-links now derive their host + workspace from each run's own
``experiment_url`` (``{host}/{workspace}/experiments/{id}``; see
``orq_links.parse_experiment_url``), which the web app resolves correctly for
anyone with access — no API key, no workspace config, no ``orq`` CLI. This module
is the fallback for runs without an ``experiment_url``: it reads the saved
dashboard profile and workspace first, then the environment. Links are hidden
when no workspace slug is available.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from loguru import logger

DEFAULT_BASE_URL = 'https://my.orq.ai'


def resolve_slug() -> str | None:
    """Workspace slug saved in dashboard settings, or the environment fallback."""
    from evaluatorq.trace_finder.settings import load_settings

    saved = load_settings().orq_workspace
    if saved:
        return saved
    env = os.environ.get('ORQ_WORKSPACE') or os.environ.get('ORQ_WORKSPACE_SLUG')
    return env.strip() or None if env and env.strip() else None


def resolve_base_url() -> str:
    """Return a safe Orq origin from the saved profile, environment, or prod default."""
    from evaluatorq.trace_finder.settings import load_settings

    saved = load_settings()
    host = saved.orq_profile_host if saved.orq_profile else None
    for source, candidate in (
        ('saved profile', host),
        ('ORQ_BASE_URL', os.environ.get('ORQ_BASE_URL')),
        ('default', DEFAULT_BASE_URL),
    ):
        if not candidate:
            continue
        origin = candidate.strip().rstrip('/')
        try:
            parsed = urlsplit(origin)
            valid = (
                parsed.scheme in {'http', 'https'}
                and parsed.hostname is not None
                and parsed.port != 0
                and parsed.path == ''
                and not parsed.query
                and not parsed.fragment
                and '?' not in origin
                and '#' not in origin
                and parsed.username is None
                and parsed.password is None
                and not any(char.isspace() for char in origin)
                and '\\' not in origin
            )
        except ValueError:
            valid = False
        if valid:
            return origin
        logger.warning('Ignoring invalid Orq UI host origin from {}', source)
    return DEFAULT_BASE_URL


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
