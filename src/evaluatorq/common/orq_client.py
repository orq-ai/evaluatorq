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

import os
from typing import TYPE_CHECKING

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


def resolve_orq_client(api_key: str | None = None) -> Orq:
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

    return Orq(api_key=key, server_url=orq_server_url())
