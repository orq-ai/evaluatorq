"""Switching catalogue models on and off for the dashboard's workspace (`/v2/workspace-models`).

A CLI OAuth sign-in names a person, so it may manage models when that person holds the workspace Admin role. An
API key names no one; it may manage models when Orq lets it (a Management Key with the workspace-model permission).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx

from evaluatorq.common.cli_oauth import OrqCLIError, is_workspace_admin, set_workspace_model

if TYPE_CHECKING:
    from evaluatorq.dashboard.auth import DashboardAuth

_TIMEOUT_S = 15.0


class WorkspaceModelError(RuntimeError):
    """Orq refused or could not be reached; the message is what the row shows."""


async def can_manage_models(auth: DashboardAuth) -> bool:
    """Whether *auth* may enable and disable workspace models."""
    if auth.method == 'cli_oauth':
        return await is_workspace_admin(auth.base_url)
    if not auth.api_key:
        return False
    # Orq checks the permission before the body, so an empty enable is a probe that changes nothing:
    # 403 means the key may not manage models, 400 (invalid body) means it may. Orq has no read-only way to ask
    # (RES-1973).
    try:
        response = await _request(auth, 'POST', '/v2/workspace-models', json={'model_id': ''})
    except httpx.HTTPError:
        return False
    return response.status_code == 400


async def set_model_enabled(auth: DashboardAuth, model_id: str, *, enabled: bool) -> None:
    """Enable or disable *model_id* for the workspace; raises `WorkspaceModelError` with Orq's reason."""
    if auth.method == 'cli_oauth':
        try:
            await set_workspace_model(auth.base_url, model_id, enabled=enabled)
        except OrqCLIError as exc:
            raise WorkspaceModelError(str(exc)) from exc
        return
    try:
        if enabled:
            response = await _request(auth, 'POST', '/v2/workspace-models', json={'model_id': model_id})
        else:
            response = await _request(auth, 'DELETE', f'/v2/workspace-models/{quote(model_id, safe="")}')
    except httpx.HTTPError as exc:
        raise WorkspaceModelError(f'Orq could not be reached: {exc}') from exc
    if response.is_error:
        raise WorkspaceModelError(f'HTTP {response.status_code}: {response.text[:300]}')


async def _request(auth: DashboardAuth, method: str, path: str, **kwargs: object) -> httpx.Response:
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as http:
        return await http.request(
            method, f'{auth.base_url.rstrip("/")}{path}', headers={'Authorization': f'Bearer {auth.api_key}'}, **kwargs
        )
