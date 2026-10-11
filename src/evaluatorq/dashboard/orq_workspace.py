"""Resolve the Orq workspace slug and host for dashboard deep-links.

Deep-links now derive their host + workspace from each run's own
``experiment_url`` (``{host}/{workspace}/experiments/{id}``; see
``orq_links.parse_experiment_url``), which the web app resolves correctly for
anyone with access — no API key, no workspace config, no ``orq`` CLI. This module
is the fallback for runs without an ``experiment_url``: it reads the
workspace from the environment, then discovers the active CLI credential's
slug. Links are hidden when no workspace slug is available.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_BASE_URL = 'https://my.orq.ai'
_cli_slug_cache: dict[tuple[str | None, str | None, bool], tuple[float, str | None]] = {}
_render_cli_slug_cache: ContextVar[dict[tuple[str | None, str | None], str | None] | None] = ContextVar(
    'render_cli_slug_cache', default=None
)


@contextmanager
def cli_slug_render_scope() -> Iterator[None]:
    """Reuse CLI-session slug discovery for one render, then discard it."""
    token = _render_cli_slug_cache.set({})
    try:
        yield
    finally:
        _render_cli_slug_cache.reset(token)


def _cli_slug(profile: str | None, fingerprint: str | None, *, use_cli_session: bool = False) -> str | None:
    """Cache credential lookups and reuse CLI-session discovery within one render."""
    key = (profile, fingerprint, use_cli_session)
    render_cache = _render_cli_slug_cache.get() if use_cli_session else None
    if render_cache is not None and (profile, fingerprint) in render_cache:
        return render_cache[profile, fingerprint]
    cached = _cli_slug_cache.get(key) if not use_cli_session else None
    if cached is not None and cached[0] > monotonic():
        return cached[1]

    slug = _discover_cli_slug(profile, fingerprint, use_cli_session=use_cli_session)
    if render_cache is not None:
        render_cache[profile, fingerprint] = slug
    return slug


def _discover_cli_slug(profile: str | None, fingerprint: str | None, *, use_cli_session: bool) -> str | None:
    from evaluatorq.dashboard.orq_scope import discover_orq_scope

    scope = discover_orq_scope(profile, use_cli_session=use_cli_session)
    slug = scope.workspace_key
    if not slug:
        logger.warning(
            'Could not resolve the Orq workspace slug from the CLI: {}',
            scope.error or 'the credential has no matching listed workspace',
        )
    if not use_cli_session:
        _cli_slug_cache[profile, fingerprint, False] = (monotonic() + (300 if slug and fingerprint else 30), slug)
    return slug


def resolve_slug() -> str | None:
    """Workspace slug from the selected OAuth scope, environment, or active CLI credential."""
    from evaluatorq.trace_finder.settings import credential_fingerprint, load_settings

    settings = load_settings()
    if settings.orq_auth_method == 'cli_oauth' and settings.orq_workspace:
        return settings.orq_workspace
    env = os.environ.get('ORQ_WORKSPACE') or os.environ.get('ORQ_WORKSPACE_SLUG')
    if env and env.strip():
        return env.strip()
    if not shutil.which('orq'):
        return None
    if settings.orq_auth_method == 'cli_profile' and settings.orq_profile:
        return _cli_slug(settings.orq_profile, settings.orq_credential_fingerprint)
    api_key = os.environ.get('ORQ_API_KEY', '').strip()
    if api_key:
        return _cli_slug(None, credential_fingerprint(api_key, os.environ.get('ORQ_BASE_URL')))
    return _cli_slug(None, None, use_cli_session=True)


def resolve_base_url() -> str:
    """Return a safe Orq origin from the saved profile, environment, or prod default."""
    from evaluatorq.trace_finder.settings import load_settings

    saved = load_settings()
    host = (
        saved.orq_oauth_server
        if saved.orq_auth_method == 'cli_oauth'
        else saved.orq_profile_host
        if saved.orq_profile
        else None
    )
    for source, candidate in (
        ('selected credential', host),
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
