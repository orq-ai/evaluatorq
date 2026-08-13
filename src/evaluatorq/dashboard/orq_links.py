"""Build Orq Studio deep-links for report entities.

Pure string helper — no network, no SDK. The preferred source of a link's host +
workspace is the run's own ``experiment_url`` (``{host}/{workspace}/experiments/
{id}``), parsed by `parse_experiment_url`. When a run has no experiment
(upload skipped/failed), we fall back to the ``ORQ_WORKSPACE`` / ``ORQ_BASE_URL``
environment. Studio routes use the workspace **key** (slug), never the API's
UUID-shaped ``workspace_id``: ``{base}/{workspace_key}/{entity}/{id}``.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse
from uuid import UUID

from evaluatorq.dashboard.orq_workspace import resolve_base_url

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


def parse_experiment_url(url: str | None) -> tuple[str | None, str | None]:
    """Extract ``(host, workspace_key)`` from an Orq ``experiment_url``.

    Experiment URLs look like ``https://my.orq.ai/orq-research/experiments/{id}``
    — host + workspace slug, no project. Returns ``(None, None)`` for anything
    that isn't a recognisable experiment URL, or whose first segment is a UUID
    (not a valid Studio route key).
    """
    if not url:
        return None, None
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return None, None
    segments = [s for s in parsed.path.split('/') if s]
    # Expect /{workspace}/experiments/{id}
    if len(segments) >= 2 and segments[1] == 'experiments' and is_studio_workspace_key(segments[0]):
        return f'{parsed.scheme}://{parsed.netloc}', segments[0]
    return None, None


def studio_workspace_key(fallback: str | None = None) -> str | None:
    """Return the Studio workspace key (slug) from env, falling back when valid.

    Order: ``ORQ_WORKSPACE`` env (explicit override) → the captured-record
    ``fallback`` (the entity's own workspace slug). UUID workspace IDs are never
    accepted as route slugs.
    """
    env = os.getenv('ORQ_WORKSPACE')
    if is_studio_workspace_key(env):
        return env
    return fallback if is_studio_workspace_key(fallback) else None


def orq_studio_url(
    *,
    target_kind: str | None,
    entity_id: str | None,
    experiment_url: str | None = None,
    workspace_id: str | None = None,
    base_url: str | None = None,
) -> str | None:
    """Return the Studio deep-link for an entity, or None if not linkable.

    Host + workspace come from the run's ``experiment_url`` when available (the
    web app resolves that path for anyone with access); otherwise they fall back
    to ``ORQ_WORKSPACE`` env / the ``workspace_id`` slug and ``base_url`` / the
    ``ORQ_BASE_URL`` env. ``workspace_id`` UUIDs are rejected as route keys.
    """
    entity = _ENTITY.get(target_kind or '')
    if entity is None or not entity_id:
        return None

    host, workspace_key = parse_experiment_url(experiment_url)
    if not (host and workspace_key):
        workspace_key = studio_workspace_key(workspace_id)
        host = base_url or resolve_base_url()
    if not workspace_key or not host:
        return None
    return f'{host.rstrip("/")}/{workspace_key}/{entity}/{entity_id}'
