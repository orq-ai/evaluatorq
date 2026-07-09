"""Build Orq Studio deep-links for report entities.

Pure string helper — no network, no SDK. Studio URL shape:
``{base}/{workspace_id}/{entity}/{id}`` where entity is the pluralised kind.
"""

from __future__ import annotations

_ENTITY = {'agent': 'agents', 'deployment': 'deployments'}


def orq_studio_url(
    *,
    target_kind: str | None,
    entity_id: str | None,
    workspace_id: str | None,
    base_url: str,
) -> str | None:
    """Return the Studio deep-link, or None if not linkable.

    Returns None for non-orq targets (openai/direct) or when any required
    field is missing — callers hide the button in that case.
    """
    entity = _ENTITY.get(target_kind or '')
    if entity is None or not entity_id or not workspace_id or not base_url:
        return None
    return f'{base_url.rstrip("/")}/{workspace_id}/{entity}/{entity_id}'
