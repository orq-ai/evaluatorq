"""Switching catalogue models on and off for the dashboard's workspace (`/v2/workspace-models`).

A CLI OAuth sign-in names a person, so it may manage models when that person holds the workspace Admin role. An
API key names no one; it counts as an admin when Orq lets it list the workspace users.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx
from loguru import logger

from evaluatorq.common.cli_oauth import OrqCLIError, is_workspace_admin, set_workspace_model
from evaluatorq.common.orq_client import close_orq_client, resolve_orq_client

if TYPE_CHECKING:
    from evaluatorq.dashboard.auth import DashboardAuth

_TIMEOUT_MS = 15_000
# A sign-in's admin rights rarely change, so a yes is kept for half an hour; a no may be a failed check, so it expires
# sooner. Keyed like the catalogue cache in `common.model_catalogue`: per host and credential, for the process.
_ADMIN_YES_TTL_S = 30 * 60.0
_ADMIN_NO_TTL_S = 60.0
_admin_answers: dict[tuple[object, ...], tuple[float, bool]] = {}


class WorkspaceModelError(RuntimeError):
    """Orq refused or could not be reached; the message is what the row shows."""


async def can_manage_models(auth: DashboardAuth) -> bool:
    """Whether *auth* may enable and disable workspace models."""
    if auth.method == 'cli_oauth':
        return await is_workspace_admin(auth.base_url)
    if not auth.api_key:
        return False
    # A key is no workspace member, so it has no role to read. Listing the members needs the user read permission,
    # which admin keys hold; a key with workspace-model write but no user read reads as no (ENG-3656).
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_MS / 1000) as http:
            response = await http.get(
                f'{auth.base_url.rstrip("/")}/v3/users',
                params={'limit': 1},
                headers={'Authorization': f'Bearer {auth.api_key}'},
            )
    except httpx.HTTPError:
        return False
    return response.is_success


async def may_manage_models(auth: DashboardAuth) -> bool:
    """`can_manage_models`, remembered per sign-in (a CLI sign-in's check costs two `orq` runs, about 1 s).

    A failed check reads as no, so it hides the switches rather than failing the page.
    """
    secret = hashlib.sha256(auth.api_key.encode()).hexdigest() if auth.api_key else None
    key = (auth.method, auth.base_url, secret, auth.profile.name if auth.profile else None)
    hit = _admin_answers.get(key)
    if hit is not None and hit[0] > time.monotonic():
        return hit[1]
    try:
        allowed = await can_manage_models(auth)
    except (httpx.HTTPError, OSError, RuntimeError):
        logger.exception('The workspace admin check failed')
        allowed = False
    _admin_answers[key] = (time.monotonic() + (_ADMIN_YES_TTL_S if allowed else _ADMIN_NO_TTL_S), allowed)
    return allowed


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
