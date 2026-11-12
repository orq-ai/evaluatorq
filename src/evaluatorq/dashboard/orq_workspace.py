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

import json
import shutil
import threading
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

from loguru import logger

from evaluatorq.common.env_config import DEFAULT_ORQ_BASE_URL, env_raw, env_text

# Retained for backward-compatible imports; the canonical value lives in common.env_config.
DEFAULT_BASE_URL = DEFAULT_ORQ_BASE_URL
_CacheKey = tuple[str | None, str | None, str | None]
_FOUND_TTL_S = 300
_MISSING_TTL_S = 30
_cli_slug_cache: dict[_CacheKey, tuple[float, str | None]] = {}
_pending: dict[_CacheKey, threading.Thread] = {}
_lock = threading.Lock()
_CLI_CONFIG_DIR = Path.home() / '.orq'
_session_memo: tuple[tuple[int | None, int | None], str] | None = None


def _mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _cache_key(profile: str | None, fingerprint: str | None, *, use_cli_session: bool) -> _CacheKey:
    """Key a lookup by credential; for the CLI session, by the account the CLI is signed in as."""
    return (profile, fingerprint, _cli_session_identity() if use_cli_session else None)


def _cli_session_identity() -> str:
    """Identify the CLI's active account, re-reading its files only when one of them changed."""
    global _session_memo
    stamp = (_mtime_ns(_CLI_CONFIG_DIR / 'config.json'), _mtime_ns(_CLI_CONFIG_DIR / 'credentials.json'))
    if _session_memo is None or _session_memo[0] != stamp:
        _session_memo = (stamp, _read_cli_session_identity())
    return _session_memo[1]


def _read_cli_session_identity() -> str:
    """Hash the active profile's key and host, so other writes to the files (path bindings, flags) keep the cache.

    A re-login under the same profile name changes the key and so the identity.
    """
    from evaluatorq.trace_finder.settings import credential_fingerprint

    def profile_entry(data: object, name: str) -> dict:
        profiles = data.get('profiles') if isinstance(data, dict) else None
        entry = profiles.get(name) if isinstance(profiles, dict) else None
        return entry if isinstance(entry, dict) else {}

    def load(name: str) -> object:
        try:
            return json.loads((_CLI_CONFIG_DIR / name).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None

    config = load('config.json')
    current = config.get('current') if isinstance(config, dict) else None
    if not isinstance(current, str):
        return ''
    entry = profile_entry(load('credentials.json'), current) or profile_entry(config, current)
    key, host = entry.get('api_key'), entry.get('server') or entry.get('base_url')
    if not isinstance(key, str):
        return f'profile:{current}'
    return credential_fingerprint(key, host if isinstance(host, str) else None) or f'profile:{current}'


def _cli_slug(profile: str | None, fingerprint: str | None, *, use_cli_session: bool = False) -> str | None:
    """Return the cached slug for this credential, starting a background lookup when it is missing or expired.

    Never runs the CLI on the caller's thread. An expired slug is still returned while it
    refreshes, because it belongs to the same credential; a missing one returns ``None``.
    """
    key = _cache_key(profile, fingerprint, use_cli_session=use_cli_session)
    with _lock:
        cached = _cli_slug_cache.get(key)
        if cached is not None and cached[0] > monotonic():
            return cached[1]
        if key not in _pending:
            thread = threading.Thread(
                target=_discover_in_background, args=(key,), name='orq-workspace-slug', daemon=True
            )
            _pending[key] = thread
            thread.start()
    return cached[1] if cached is not None else None


def _discover_in_background(key: _CacheKey) -> None:
    profile, _fingerprint, stamp = key
    slug = None
    try:
        slug = _discover_cli_slug(profile, use_cli_session=stamp is not None)
    except Exception:  # noqa: BLE001 - a failed lookup is cached as missing and retried
        logger.opt(exception=True).warning('Orq workspace lookup failed')
    finally:
        with _lock:
            # Tests swap `_pending` between runs; a lookup no longer registered must not publish.
            if _pending.get(key) is threading.current_thread():
                for old in [k for k in _cli_slug_cache if k[:2] == key[:2]]:
                    del _cli_slug_cache[old]
                _cli_slug_cache[key] = (monotonic() + (_FOUND_TTL_S if slug else _MISSING_TTL_S), slug)
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


def _slug_source() -> str | tuple[str | None, str | None, bool] | None:
    """The configured slug, or the credential the CLI is asked about, or ``None`` when neither applies."""
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
        return (settings.orq_profile, settings.orq_credential_fingerprint, False)
    api_key = env_text('ORQ_API_KEY')
    if api_key:
        return (None, credential_fingerprint(api_key, env_raw('ORQ_BASE_URL')), False)
    return (None, None, True)


def resolve_slug() -> str | None:
    """Workspace slug from the selected OAuth scope, environment, or active CLI credential."""
    source = _slug_source()
    if isinstance(source, tuple):
        profile, fingerprint, use_cli_session = source
        if use_cli_session:
            return _cli_slug(profile, fingerprint, use_cli_session=True)
        return _cli_slug(profile, fingerprint)
    return source


def slug_lookup_pending() -> bool:
    """Whether a lookup is running for the credential the dashboard currently uses."""
    source = _slug_source()
    if not isinstance(source, tuple):
        return False
    key = _cache_key(source[0], source[1], use_cli_session=source[2])
    with _lock:
        return key in _pending


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
