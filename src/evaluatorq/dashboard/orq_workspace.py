"""Resolve the Orq workspace slug and host for dashboard deep-links.

Deep-links now derive their host + workspace from each run's own
``experiment_url`` (``{host}/{workspace}/experiments/{id}``; see
``orq_links.parse_experiment_url``), which the web app resolves correctly for
anyone with access — no API key, no workspace config, no ``orq`` CLI. This module
is the fallback for runs without an ``experiment_url``: it reads the
workspace from the environment, then discovers the active CLI credential's
slug. Links are hidden when no workspace slug is available.

CLI discovery runs on a background thread and is cached per credential, so a
report render never waits on the ``orq`` CLI (#271).
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

from loguru import logger

from evaluatorq.common.env_config import DEFAULT_ORQ_BASE_URL, env_raw, env_text

# Retained for backward-compatible imports; the canonical value lives in common.env_config.
DEFAULT_BASE_URL = DEFAULT_ORQ_BASE_URL
_CacheKey = tuple[str | None, str | None, tuple[object, ...] | None]
_cli_slug_cache: dict[_CacheKey, tuple[float, str | None]] = {}
_pending: dict[_CacheKey, threading.Thread] = {}
_lock = threading.Lock()
_CLI_CONFIG_DIR = Path.home() / '.orq'


def _cli_session_stamp() -> tuple[object, ...]:
    """Identify the CLI's active session: switching it rewrites these files, so the key changes at once."""
    files = (_CLI_CONFIG_DIR / 'config.json', _CLI_CONFIG_DIR / 'credentials.json')
    return tuple(path.stat().st_mtime_ns if path.exists() else None for path in files)


def _cli_slug(profile: str | None, fingerprint: str | None, *, use_cli_session: bool = False) -> str | None:
    """Return the cached slug for this credential and refresh it in the background when missing or stale.

    Never runs the CLI on the caller's thread: a first render shows no fallback links and
    `discovery_pending` tells the page to reload once the lookup lands.
    """
    key: _CacheKey = (profile, fingerprint, _cli_session_stamp() if use_cli_session else None)
    with _lock:
        cached = _cli_slug_cache.get(key)
        if cached is not None and cached[0] > monotonic():
            return cached[1]
        if key not in _pending:
            thread = threading.Thread(
                target=_discover_in_background,
                args=(key, profile, use_cli_session),
                name='orq-workspace-slug',
                daemon=True,
            )
            _pending[key] = thread
            thread.start()
    # ponytail: a stale value for the same credential is still that credential's workspace.
    return cached[1] if cached is not None else None


def discovery_pending() -> bool:
    """Whether a workspace lookup is still running."""
    with _lock:
        return bool(_pending)


def _discover_in_background(key: _CacheKey, profile: str | None, use_cli_session: bool) -> None:  # noqa: FBT001
    slug = None
    try:
        slug = _discover_cli_slug(profile, use_cli_session=use_cli_session)
    finally:
        with _lock:
            # Only the registered lookup for this key may publish; an orphaned one is dropped.
            if _pending.get(key) is threading.current_thread():
                # A found slug holds for 5 minutes; a failed lookup is retried after 30 seconds.
                _cli_slug_cache[key] = (monotonic() + (300 if slug else 30), slug)
                del _pending[key]


def _discover_cli_slug(profile: str | None, *, use_cli_session: bool) -> str | None:
    from evaluatorq.dashboard.orq_scope import discover_orq_scope

    scope = discover_orq_scope(profile, use_cli_session=use_cli_session)
    slug = scope.workspace_key
    if not slug:
        logger.warning(
            'Could not resolve the Orq workspace slug from the CLI: {}',
            scope.error or 'the credential has no matching listed workspace',
        )
    return slug


def resolve_slug() -> str | None:
    """Workspace slug from the selected OAuth scope, environment, or active CLI credential."""
    from evaluatorq.trace_finder.settings import credential_fingerprint, load_settings

    settings = load_settings()
    if settings.orq_auth_method == 'cli_oauth' and settings.orq_workspace:
        return settings.orq_workspace
    env = env_raw('ORQ_WORKSPACE') or env_raw('ORQ_WORKSPACE_SLUG')
    if env and env.strip():
        return env.strip()
    if not shutil.which('orq'):
        return None
    if settings.orq_auth_method == 'cli_profile' and settings.orq_profile:
        return _cli_slug(settings.orq_profile, settings.orq_credential_fingerprint)
    api_key = env_text('ORQ_API_KEY')
    if api_key:
        return _cli_slug(None, credential_fingerprint(api_key, env_raw('ORQ_BASE_URL')))
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
        ('ORQ_BASE_URL', env_raw('ORQ_BASE_URL')),
        ('default', DEFAULT_ORQ_BASE_URL),
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
    return DEFAULT_ORQ_BASE_URL


def classify_host(url: str | None) -> str:
    """Label a host as 'Production' / 'Staging' / 'On-prem' for display.

    Environment can only be told from the host we talk to (a workspace UUID
    carries none): ``my.orq.ai`` is prod; any ``*.orq.ai`` marked staging/dev is
    shared pre-prod; anything else is a self-hosted (on-prem) deployment.
    """
    host = (url or DEFAULT_ORQ_BASE_URL).strip().lower()
    if 'my.orq.ai' in host and 'staging' not in host and 'dev' not in host:
        return 'Production'
    if '.orq.ai' in host and ('staging' in host or 'dev' in host):
        return 'Staging'
    return 'On-prem'
