"""Resolve the dashboard's saved Orq authentication choice in one place."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.orq_client import DEFAULT_ORQ_BASE_URL, list_orq_profiles, resolve_orq_client

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evaluatorq.common.orq_client import OrqProfile
    from evaluatorq.trace_finder.settings import DashboardSettings


@dataclass(frozen=True)
class DashboardAuth:
    """One resolved credential source; ``api_key`` is never serialized."""

    method: str
    api_key: str | None
    base_url: str
    profile: OrqProfile | None = None

    @property
    def label(self) -> str:
        if self.method == 'cli_profile' and self.profile is not None:
            return f'CLI profile {self.profile.name}'
        return {
            'environment': 'ORQ_API_KEY',
            'cli_oauth': 'CLI OAuth',
            'stored_api_key': 'Saved API key',
        }.get(self.method, 'Orq credentials')


def resolve_dashboard_auth(
    settings: DashboardSettings, *, profiles: Sequence[OrqProfile] | None = None
) -> DashboardAuth:
    """Resolve the selected source or raise a human-readable action request."""

    method = settings.orq_auth_method
    if method == 'environment':
        key = os.environ.get('ORQ_API_KEY', '').strip()
        return DashboardAuth(method, key or None, os.environ.get('ORQ_BASE_URL', DEFAULT_ORQ_BASE_URL))
    if method == 'cli_profile':
        name = settings.orq_profile
        if not name:
            raise ValueError('Choose an Orq CLI API-key profile in Settings.')
        available = list_orq_profiles() if profiles is None else profiles
        profile = next((item for item in available if item.name == name), None)
        if profile is None:
            raise ValueError(f'Orq CLI profile {name} is unavailable. Choose a credential in Settings.')
        if not profile.api_key or '*' in profile.api_key:
            raise ValueError(f'Orq CLI profile {name} has no usable API key. Choose a credential in Settings.')
        return DashboardAuth(method, profile.api_key, profile.server or DEFAULT_ORQ_BASE_URL, profile)
    if method == 'stored_api_key':
        from evaluatorq.trace_finder.settings import read_stored_api_key

        try:
            key = read_stored_api_key(settings)
        except (OSError, RuntimeError) as exc:
            raise ValueError(str(exc)) from exc
        if not key:
            raise ValueError('No saved API key is available. Enter one in Settings.')
        return DashboardAuth(method, key, settings.orq_profile_host or DEFAULT_ORQ_BASE_URL)
    if method == 'cli_oauth':
        return DashboardAuth(method, None, settings.orq_oauth_server or DEFAULT_ORQ_BASE_URL)
    raise ValueError('Unknown dashboard authentication method. Choose one in Settings.')


def _account_fields(auth: DashboardAuth) -> dict[str, Any]:
    fields: dict[str, Any] = {
        'method': auth.method,
        'base_url': auth.base_url.rstrip('/'),
        'profile': auth.profile.name if auth.profile else None,
        'key_fingerprint': sha256(auth.api_key.encode()).hexdigest() if auth.api_key else None,
    }
    if auth.method == 'cli_oauth':
        from evaluatorq.common.cli_oauth import oauth_subject

        fields['oauth_subject'] = oauth_subject(auth.base_url)
    return fields


def _hash_fields(fields: dict[str, Any]) -> str:
    return sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def auth_identity(auth: DashboardAuth, settings: DashboardSettings) -> str:
    """Hash the selected account and scope without putting a secret in a worker payload."""

    return _hash_fields({
        **_account_fields(auth),
        'workspace': settings.orq_workspace,
        'project': settings.orq_project_id,
    })


def account_identity(auth: DashboardAuth) -> str:
    """Hash only the selected account, without workspace, project, or any secret."""

    return _hash_fields(_account_fields(auth))


async def verified_account_identity(auth: DashboardAuth, client: Any) -> str:
    """Hash an Orq-verified user, workspace, or project principal without credential secrets."""
    parsed = urlparse(auth.base_url)
    base_url = f'{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip("/")}'
    if auth.method == 'cli_oauth':
        from evaluatorq.common.cli_oauth import oauth_subject

        subject = await asyncio.to_thread(oauth_subject, auth.base_url)
        user_id = subject.get('user_id') if isinstance(subject, dict) else None
        if not isinstance(user_id, str) or not user_id:
            raise ValueError('Orq OAuth did not return a verified user ID.')
        principal = {'user_id': user_id}
    else:
        try:
            response = await client.workspaces.list_async()
        except Exception as exc:
            if getattr(exc, 'status_code', None) != 403:
                raise
            response = None
        if response is not None:
            data = getattr(response, 'data', None)
            ids = {getattr(item, 'id', None) for item in data or ()}
            ids = {value for value in ids if isinstance(value, str) and value}
            if getattr(response, 'has_more', False):
                raise ValueError('Orq credential can access multiple workspaces; cannot verify one workspace.')
            if len(ids) != 1:
                raise ValueError('Orq credential did not verify exactly one workspace.')
            workspace_id = next(iter(ids))
            principal = {'workspace_id': workspace_id}
        else:
            page = await client.projects.list_async(limit=100)
            rows = list(getattr(page, 'data', None) or ())
            project_ids = {
                value
                for value in (getattr(item, 'project_id', None) for item in rows)
                if isinstance(value, str) and value
            }
            if getattr(page, 'has_more', False):
                raise ValueError('Orq credential can access multiple projects; cannot verify one project.')
            if len(project_ids) != 1:
                raise ValueError('Orq credential did not verify exactly one project.')
            principal = {'project_id': next(iter(project_ids))}
    return _hash_fields({'version': 2, 'base_url': base_url, **principal})


def account_label(auth: DashboardAuth) -> str:
    """Name the account for display: profile or method name plus host, never a secret."""

    name = auth.profile.name if auth.profile else auth.label
    host = urlparse(auth.base_url).netloc or auth.base_url.rstrip('/')
    return f'{name} on {host}'


def build_auth_clients(
    auth: DashboardAuth, *, workspace: str | None = None, project: str | None = None
) -> tuple[Any, Any]:
    """Build Orq and model clients for a resolved dashboard credential."""

    if auth.method == 'cli_oauth':
        from evaluatorq.common.cli_oauth import build_cli_oauth_clients

        return build_cli_oauth_clients(server_url=auth.base_url, workspace=workspace, project=project)
    llm = resolve_llm_client(
        extra_api_key=auth.api_key,
        orq_host=auth.base_url,
        require_orq=True,
        max_retries=0,
    ).client
    orq = resolve_orq_client(auth.api_key, base_url=auth.base_url)
    return orq, llm


def build_orq_client(auth: DashboardAuth, *, workspace: str | None = None, project: str | None = None) -> Any:
    """Build only the Orq client for a resolved dashboard credential."""

    if auth.method == 'cli_oauth':
        from evaluatorq.common.cli_oauth import build_cli_oauth_clients

        return build_cli_oauth_clients(server_url=auth.base_url, workspace=workspace, project=project)[0]
    return resolve_orq_client(auth.api_key, base_url=auth.base_url)
