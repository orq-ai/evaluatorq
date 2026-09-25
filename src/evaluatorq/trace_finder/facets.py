"""Load a defensive, live catalogue of Orq trace facet values."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from loguru import logger

from .models import FACET_NAMES, FacetCatalogue, FacetName

if TYPE_CHECKING:
    from datetime import datetime

    from orq_ai_sdk import Orq

_FACET_FIELDS: Mapping[FacetName, str] = MappingProxyType({
    'project': 'project_id',
    'model': 'model',
    'provider': 'provider',
    'status': 'status',
    'product': 'product',
    'trace_type': 'attributes.orq.leading_span.span_type',
    'agent_name': 'agent_name',
    'tool_name': 'tool_name',
})
if set(_FACET_FIELDS) != set(FACET_NAMES):
    raise RuntimeError('trace facet field mapping is out of sync with FACET_NAMES')


async def load_facet_catalogue(
    client: Orq,
    *,
    start: datetime,
    end: datetime,
    limit: int = 50,
) -> FacetCatalogue:
    """Load available facet values, retaining a bounded frequent subset on overflow."""

    tasks = [
        asyncio.create_task(_safe_facet_values(client, field, start=start, end=end, limit=limit))
        for field in _FACET_FIELDS.values()
    ]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    values_by_name = {name: values for name, (values, _) in zip(_FACET_FIELDS, results, strict=True)}
    truncated_facets = frozenset(name for name, (_, truncated) in zip(_FACET_FIELDS, results, strict=True) if truncated)
    if truncated_facets:
        logger.warning(
            'Orq facet values exceed the {}-value limit for {}; using returned values ranked by frequency',
            limit,
            ', '.join(sorted(truncated_facets)),
        )

    project_names = await _project_names(client)
    missing_projects = set(values_by_name['project']) - set(project_names)
    if missing_projects:
        raise ValueError(f'cannot resolve facet project ids: {sorted(missing_projects)}')
    labels = project_labels(project_names)
    values_by_name['project'] = tuple(labels[project_id] for project_id in values_by_name['project'])
    return FacetCatalogue.model_validate({**values_by_name, 'truncated_facets': truncated_facets})


def project_labels(project_names: Mapping[str, str]) -> dict[str, str]:
    """Disambiguate equal project names using their stable IDs."""
    counts: dict[str, int] = {}
    for name in project_names.values():
        counts[name] = counts.get(name, 0) + 1
    return {
        project_id: f'{name} ({project_id})' if counts[name] > 1 else name for project_id, name in project_names.items()
    }


async def _safe_facet_values(
    client: Orq,
    field: str,
    *,
    start: datetime,
    end: datetime,
    limit: int,
) -> tuple[tuple[str, ...], bool]:
    response = await client.traces.list_facet_values_async(field=field, from_=start, to=end, limit=limit)
    raw_values = _read(response, 'values')
    if not isinstance(raw_values, (list, tuple)):
        raise TypeError(f'trace facet field {field} returned no readable values')
    ranked: list[tuple[str, int, int]] = []
    for index, item in enumerate(raw_values):
        value = _read(item, 'value')
        if not isinstance(value, str) or not value or value == 'unknown':
            continue
        count = _read(item, 'count')
        ranked.append((value, count if type(count) is int and count >= 0 else 0, index))
    ranked.sort(key=lambda item: (-item[1], item[2]))
    return tuple(dict.fromkeys(value for value, _, _ in ranked)), bool(_read(response, 'has_more'))


async def _project_names(client: Orq) -> dict[str, str]:
    """Resolve every visible project id to its human-readable name."""

    projects: dict[str, str] = {}
    starting_after: str | None = None
    seen_cursors: set[str] = set()
    while True:
        response = await client.projects.list_async(limit=100, starting_after=starting_after)
        page = _read(response, 'data')
        page_items = list(page) if isinstance(page, (list, tuple)) else []
        for project in page_items:
            project_id = _read(project, 'project_id')
            name = _read(project, 'name')
            if project_id and name:
                projects[str(project_id)] = str(name)
        if not _read(response, 'has_more'):
            return projects
        if not page_items:
            raise RuntimeError('project response reported more pages without projects')
        cursor = str(_read(page_items[-1], 'project_id') or '')
        if not cursor or cursor in seen_cursors:
            raise RuntimeError(f'repeated project page cursor {cursor!r}')
        seen_cursors.add(cursor)
        starting_after = cursor


def _read(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
