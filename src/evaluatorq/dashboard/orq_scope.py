"""Discover the workspace and projects available to the dashboard credential."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from time import monotonic
from typing import Any

from loguru import logger

from evaluatorq.common.orq_client import DEFAULT_ORQ_BASE_URL


@dataclass(frozen=True)
class OrqProject:
    """A project visible to the selected credential."""

    id: str
    name: str
    workspace_id: str


@dataclass(frozen=True)
class OrqScope:
    """The one workspace and its projects available to a dashboard credential."""

    workspace_key: str | None = None
    workspace_id: str | None = None
    projects: tuple[OrqProject, ...] = ()
    error: str | None = None


def _cli_json(args: list[str], *, profile: str | None, timeout: float) -> dict[str, Any] | None:
    binary = shutil.which('orq')
    if binary is None:
        return None
    command = [binary]
    environment = os.environ.copy()
    environment.pop('ORQ_VERBOSE', None)
    environment.pop('ORQ_JMESPATH', None)
    if profile:
        command.extend(('--profile', profile))
        # The named profile must determine scope, even when this shell has an unrelated key.
        for name in ('ORQ_API_KEY', 'ORQ_BASE_URL', 'ORQ_SERVER', 'ORQ_WORKSPACE', 'ORQ_WORKSPACE_SLUG', 'ORQ_PROJECT'):
            environment.pop(name, None)
    else:
        for name in ('ORQ_WORKSPACE', 'ORQ_WORKSPACE_SLUG', 'ORQ_PROJECT'):
            environment.pop(name, None)
        environment['ORQ_SERVER'] = os.environ.get('ORQ_BASE_URL', DEFAULT_ORQ_BASE_URL)
    command.extend((*args, '-o', 'json', '--no-input'))
    try:
        result = subprocess.run(
            command,
            env=environment,
            cwd=tempfile.gettempdir(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            return None
        value = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        logger.warning('Could not inspect Orq scope through the CLI: {}', exc)
        return None
    return value if isinstance(value, dict) else None


def _remaining_timeout(deadline: float, per_call_timeout: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError('Orq scope discovery timed out.')
    return min(per_call_timeout, remaining)


def _profile_workspace_rows(
    profile: str, workspace_id: str | None, timeout: float, deadline: float
) -> list[dict[str, Any]]:
    """Page through the profile's workspaces until the selected workspace is found."""
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(50):
        args = ['workspaces', 'list', '--limit', '200']
        if cursor:
            args.extend(('--starting-after', cursor))
        listing = _cli_json(args, profile=profile, timeout=_remaining_timeout(deadline, timeout))
        page = listing.get('data') if listing is not None else None
        if not isinstance(page, list):
            raise TypeError('The orq CLI could not list workspaces for this credential.')
        rows.extend(row for row in page if isinstance(row, dict))
        if any(row.get('id') == workspace_id for row in rows):
            return rows
        if listing is not None and not listing.get('has_more'):
            return rows
        cursor = page[-1].get('id') if page and isinstance(page[-1], dict) else None
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            raise ValueError('The orq CLI returned an invalid workspace page cursor.')
        seen.add(cursor)
    raise ValueError('The orq CLI returned too many workspace pages to choose safely.')


def discover_orq_scope(profile: str | None = None, *, timeout: float = 5.0) -> OrqScope:  # noqa: C901
    """Find the workspace and all visible projects without exposing a credential."""
    if profile is None and not os.environ.get('ORQ_API_KEY', '').strip():
        return OrqScope(error='Set ORQ_API_KEY to discover projects.')
    deadline = monotonic() + 3 * timeout
    projects: list[OrqProject] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(50):
        args = ['projects', 'list', '--limit', '200']
        if cursor:
            args.extend(('--starting-after', cursor))
        try:
            page = _cli_json(args, profile=profile, timeout=_remaining_timeout(deadline, timeout))
        except TimeoutError as exc:
            return OrqScope(error=str(exc))
        if page is None or not isinstance(page.get('data'), list):
            return OrqScope(error='The orq CLI could not list projects for this credential.')
        rows = page['data']
        for row in rows:
            if not isinstance(row, dict):
                continue
            project_id, name, workspace_id = row.get('project_id'), row.get('name'), row.get('workspace_id')
            if all(isinstance(value, str) and value for value in (project_id, name, workspace_id)):
                projects.append(OrqProject(project_id, name, workspace_id))
        if not page.get('has_more'):
            break
        cursor = rows[-1].get('project_id') if rows and isinstance(rows[-1], dict) else None
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            return OrqScope(error='The orq CLI returned an invalid project page cursor.')
        seen.add(cursor)
    else:
        return OrqScope(error='The orq CLI returned too many project pages to choose safely.')

    workspace_ids = {project.workspace_id for project in projects}
    if len(workspace_ids) > 1:
        return OrqScope(error='This credential returned projects from multiple workspaces.')
    workspace_id = next(iter(workspace_ids), None)
    if profile:
        try:
            rows = _profile_workspace_rows(profile, workspace_id, timeout, deadline)
        except (TypeError, ValueError, TimeoutError) as exc:
            return OrqScope(error=str(exc))
    else:
        try:
            status = _cli_json(['status'], profile=None, timeout=_remaining_timeout(deadline, timeout))
        except TimeoutError as exc:
            return OrqScope(error=str(exc))
        credential = status.get('credential') if status else None
        if isinstance(credential, dict) and credential.get('workspace_id') != workspace_id:
            return OrqScope(error='The CLI session and API key point at different workspaces.')
        rows = status.get('workspaces') if status else None
    workspace_key = None
    if isinstance(rows, list):
        workspace_key = next(
            (
                row['key']
                for row in rows
                if isinstance(row, dict) and row.get('id') == workspace_id and isinstance(row.get('key'), str)
            ),
            None,
        )
    if workspace_id and not workspace_key:
        logger.warning('The orq CLI listed projects but could not resolve their workspace slug')
    return OrqScope(workspace_key, workspace_id, tuple(sorted(projects, key=lambda project: project.name.casefold())))
