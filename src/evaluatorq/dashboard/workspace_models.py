"""Switching catalogue models on and off for the dashboard's workspace (`/v2/workspace-models`).

A CLI OAuth sign-in names a person, so it may manage models when that person holds the workspace Admin role. An
API key names no one; it may manage models when Orq lets it (a Management Key with the workspace-model permission).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx

from evaluatorq.common.cli_oauth import OrqCLIError, is_workspace_admin, set_workspace_model
from evaluatorq.common.orq_client import close_orq_client, resolve_orq_client

if TYPE_CHECKING:
    from evaluatorq.dashboard.auth import DashboardAuth

_TIMEOUT_MS = 15_000


class WorkspaceModelError(RuntimeError):
    """Orq refused or could not be reached; the message is what the row shows."""


async def can_manage_models(auth: DashboardAuth) -> bool:
    """Whether *auth* may enable and disable workspace models."""
    if auth.method == 'cli_oauth':
        return await is_workspace_admin(auth.base_url)
    if not auth.api_key:
        return False
    # Orq checks the permission before the body, so an empty enable is a probe that changes nothing:
    # 403 means the key may not manage models, 400 (invalid body) means it may. Orq has no read-only way to ask: the
    # profile endpoint `orq whoami` reads answers only a browser sign-in (ENG-3656).
    from orq_ai_sdk.models import OrqError

    try:
        await _set(auth, '', enabled=True)
    except OrqError as exc:
        return exc.status_code == 400
    except httpx.HTTPError:
        return False
    return True


async def set_model_enabled(auth: DashboardAuth, model_id: str, *, enabled: bool) -> None:
    """Enable or disable *model_id* for the workspace; raises `WorkspaceModelError` with Orq's reason."""
    if auth.method == 'cli_oauth':
        try:
            await set_workspace_model(auth.base_url, model_id, enabled=enabled)
        except OrqCLIError as exc:
            raise WorkspaceModelError(str(exc)) from exc
        return
    from orq_ai_sdk.models import OrqError

    try:
        await _set(auth, model_id, enabled=enabled)
    except OrqError as exc:
        raise WorkspaceModelError(f'HTTP {exc.status_code}: {exc.body[:300]}') from exc
    except httpx.HTTPError as exc:
        raise WorkspaceModelError(f'Orq could not be reached: {exc}') from exc


async def _set(auth: DashboardAuth, model_id: str, *, enabled: bool) -> None:
    client = resolve_orq_client(api_key=auth.api_key, base_url=auth.base_url)
    try:
        if enabled:
            await client.models.enable_async(model_id=model_id, timeout_ms=_TIMEOUT_MS)
        else:
            # The SDK puts the id into the path unescaped, so a slash in it would end the path segment.
            await client.models.disable_async(model_id=quote(model_id, safe=''), timeout_ms=_TIMEOUT_MS)
    finally:
        await close_orq_client(client)
