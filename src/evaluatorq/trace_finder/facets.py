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
    """Load all facet values concurrently, degrading an unavailable field to empty."""

    results = await asyncio.gather(
        *(
            _safe_facet_values(client, name, field, start=start, end=end, limit=limit)
            for name, field in _FACET_FIELDS.items()
        )
    )
    values_by_name = dict(zip(_FACET_FIELDS, results, strict=True))

    try:
        project_names = await _project_names(client)
    except Exception as error:  # noqa: BLE001 - the catalogue must remain usable if projects fail
        logger.warning('Trace facet field project_id project lookup failed: {}; using an empty facet', error)
        project_names = {}
    values_by_name['project'] = tuple(
        sorted({project_names[project_id] for project_id in values_by_name['project'] if project_id in project_names})
    )
    return FacetCatalogue.model_validate(values_by_name)


async def _safe_facet_values(
    client: Orq,
    name: FacetName,
    field: str,
    *,
    start: datetime,
    end: datetime,
    limit: int,
) -> tuple[str, ...]:
    try:
        response = await client.traces.list_facet_values_async(field=field, from_=start, to=end, limit=limit)
        raw_values = _read(response, 'values')
        if not isinstance(raw_values, (list, tuple)):
            logger.warning('Trace facet field {} returned no readable values; using an empty facet', field)
            return ()
        values = {
            value
            for item in raw_values
            if isinstance(value := _read(item, 'value'), str) and value and value != 'unknown'
        }
        return tuple(sorted(values))
    except Exception as error:  # noqa: BLE001 - one field must not break the catalogue
        logger.warning('Trace facet field {} failed: {}; using an empty facet', field, error)
        return ()


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
