"""Canonical Orq SDK client construction — do not build ``Orq(...)`` elsewhere.

Five call sites each carried their own copy of the same four steps: lazy-import
the optional SDK, translate ``ModuleNotFoundError`` into an install hint, read
``ORQ_API_KEY``, and apply the ``ORQ_BASE_URL`` default. Two of the five had
drifted and never passed ``server_url`` at all, so a self-hosted deployment was
silently ignored on those paths.

Callers that need a domain-specific exception (e.g. red team's
``CredentialError``) should check the key themselves and pass it in — this
module raises ``ValueError`` for a missing key and ``ImportError`` for a missing
SDK, nothing else.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

if TYPE_CHECKING:
    from orq_ai_sdk import Orq

_INSTALL_HINT = (
    'The orq_ai_sdk package is not installed. Install it with: '
    'uv add "evaluatorq[orq]" (or: python -m pip install "evaluatorq[orq]")'
)

DEFAULT_ORQ_BASE_URL = 'https://my.orq.ai'


def orq_server_url() -> str:
    """Return the Orq API base URL, honouring ``ORQ_BASE_URL``."""
    return os.environ.get('ORQ_BASE_URL', DEFAULT_ORQ_BASE_URL)


def resolve_orq_client(api_key: str | None = None, *, server_url: str | None = None) -> Orq:
    """Build an Orq SDK client from ``api_key`` or ``ORQ_API_KEY``.

    Raises:
        ImportError: the optional ``orq-ai-sdk`` dependency is not installed.
        ValueError: no API key was passed and ``ORQ_API_KEY`` is unset.
    """
    key = api_key or os.environ.get('ORQ_API_KEY')
    if not key:
        raise ValueError('ORQ_API_KEY environment variable must be set to reach the Orq API.')

    try:
        from orq_ai_sdk import Orq
    except ModuleNotFoundError as e:  # pragma: no cover - extra not installed
        raise ImportError(_INSTALL_HINT) from e

    return Orq(api_key=key, server_url=server_url or orq_server_url())


async def close_orq_client(client: Orq) -> None:
    """Close both transports owned by an Orq SDK client, when they were created."""
    async_exit = getattr(client, '__aexit__', None)
    sync_exit = getattr(client, '__exit__', None)
    try:
        if async_exit is not None:
            await async_exit(None, None, None)
    finally:
        if sync_exit is not None:
            sync_exit(None, None, None)


class OrqProfile(NamedTuple):
    """One credentials profile from the ``orq`` CLI."""

    name: str
    api_key: str
    server: str | None
    active: bool


def list_orq_profiles(timeout: float = 5.0) -> tuple[OrqProfile, ...]:
    """Profiles the installed ``orq`` CLI knows, or none when it is missing, fails, or has no keys.

    Runs ``orq auth profile list -o json`` so the CLI stays the only reader of its
    credential store. Profiles without an API key (device logins) are skipped.
    """
    binary = shutil.which('orq')
    if binary is None:
        return ()
    try:
        result = subprocess.run(
            [binary, 'auth', 'profile', 'list', '-o', 'json', '--no-input'],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        listing = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        logger.warning('Could not read profiles from the installed orq CLI: {}', exc)
        return ()
    rows = listing.get('profiles') if isinstance(listing, dict) else listing
    if not isinstance(rows, list):
        logger.warning(
            'Could not read profiles from the installed orq CLI (exit {}): {}',
            result.returncode,
            result.stderr.strip()[:200] or 'unexpected JSON response',
        )
        return ()
    profiles: list[OrqProfile] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name, api_key = row.get('name'), row.get('api_key')
        if isinstance(name, str) and name and isinstance(api_key, str) and api_key:
            server = row.get('server')
            profiles.append(
                OrqProfile(
                    name, api_key, server if isinstance(server, str) and server else None, bool(row.get('active'))
                )
            )
    return tuple(profiles)


def apply_orq_profile(name: str, profiles: tuple[OrqProfile, ...] | None = None) -> bool:
    """Point ``ORQ_API_KEY`` and ``ORQ_BASE_URL`` at the named CLI profile; False when it is unknown."""
    # ponytail: every client in the process resolves from the environment, so the
    # profile lands there; thread explicit credentials through if a second consumer appears.
    for profile in profiles if profiles is not None else list_orq_profiles():
        if profile.name == name:
            os.environ['ORQ_API_KEY'] = profile.api_key
            if profile.server:
                os.environ['ORQ_BASE_URL'] = profile.server
            else:
                os.environ.pop('ORQ_BASE_URL', None)
            return True
    logger.warning('Orq profile {} is not known to the orq CLI; keeping the environment credentials', name)
    return False
