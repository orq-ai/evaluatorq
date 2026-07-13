"""Build Orq Studio deep-links for report entities.

Pure string helper — no network, no SDK. Studio routes use the configured
``ORQ_WORKSPACE`` key, not the API's UUID-shaped ``workspace_id``:
``{base}/{workspace_key}/{entity}/{id}``.
"""

from __future__ import annotations

import os
from uuid import UUID

_ENTITY = {'agent': 'agents', 'deployment': 'deployments'}


def is_studio_workspace_key(value: str | None) -> bool:
    """Whether a value can safely occupy Studio's ``:workspaceKey`` route.

    Orq entity APIs return a UUID in ``workspace_id``. Putting that UUID in a
    Studio URL leads to an inaccessible route, even when the entity itself is
    readable through the API.
    """
    if not value:
        return False
    try:
        UUID(value)
    except ValueError:
        return True
    return False


def studio_workspace_key(fallback: str | None = None) -> str | None:
    """Return the configured Studio workspace key, falling back when valid.

    ``ORQ_WORKSPACE`` is the source of truth for links made by this process.
    The fallback supports older report records that already captured a Studio
    key, but UUID API workspace IDs are never accepted as route slugs.
    """
    configured = os.getenv('ORQ_WORKSPACE')
    if is_studio_workspace_key(configured):
        return configured
    return fallback if is_studio_workspace_key(fallback) else None


def orq_studio_url(
    *,
    target_kind: str | None,
    entity_id: str | None,
    workspace_id: str | None,
    base_url: str,
) -> str | None:
    """Return the Studio deep-link, or None if not linkable.

    ``workspace_id`` is retained as the parameter name for compatibility.
    ``ORQ_WORKSPACE`` takes precedence because entity APIs only return a UUID
    workspace ID, which cannot be used in Studio routes.
    """
    entity = _ENTITY.get(target_kind or '')
    workspace_key = studio_workspace_key(workspace_id)
    if entity is None or not entity_id or not workspace_key or not base_url:
        return None
    return f'{base_url.rstrip("/")}/{workspace_key}/{entity}/{entity_id}'
