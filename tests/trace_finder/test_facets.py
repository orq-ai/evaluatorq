"""Tests for the live Orq facet catalogue adapter."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest
from evaluatorq.trace_finder.facets import load_facet_catalogue


class FakeTraces:
    def __init__(self, *, fail_status: bool = False, has_more: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_status = fail_status
        self.has_more = has_more

    async def list_facet_values_async(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        field = cast(str, kwargs['field'])
        if field == 'status' and self.fail_status:
            raise RuntimeError('status service unavailable')
        values = {
            'project_id': ['project-1'],
            'model': ['gpt-5'],
            'provider': ['openai'],
            'status': ['error'],
            'product': ['agents'],
            'attributes.orq.leading_span.span_type': ['span.responses'],
            'agent_name': ['support-agent'],
            'tool_name': ['lookup'],
        }[field]
        return SimpleNamespace(
            values=[SimpleNamespace(value=value) for value in values],
            has_more=self.has_more and field == 'status',
        )


class FakeProjects:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def list_async(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(project_id='project-1', name='Research')],
            has_more=False,
        )


class FakeClient:
    def __init__(self, *, fail_status: bool = False, has_more: bool = False) -> None:
        self.traces = FakeTraces(fail_status=fail_status, has_more=has_more)
        self.projects = FakeProjects()


@pytest.mark.asyncio
async def test_load_facet_catalogue_gathers_all_fields_and_resolves_projects() -> None:
    client = FakeClient()
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 22, tzinfo=timezone.utc)

    catalogue = await load_facet_catalogue(cast(Any, client), start=start, end=end)

    assert catalogue.project == ('Research',)
    assert catalogue.model == ('gpt-5',)
    assert catalogue.provider == ('openai',)
    assert catalogue.status == ('error',)
    assert catalogue.product == ('agents',)
    assert catalogue.trace_type == ('span.responses',)
    assert catalogue.agent_name == ('support-agent',)
    assert catalogue.tool_name == ('lookup',)
    assert {call['field'] for call in client.traces.calls} == {
        'project_id',
        'model',
        'provider',
        'status',
        'product',
        'attributes.orq.leading_span.span_type',
        'agent_name',
        'tool_name',
    }
    assert all(call['from_'] == start and call['to'] == end and call['limit'] == 50 for call in client.traces.calls)
    assert client.projects.calls


@pytest.mark.asyncio
async def test_duplicate_project_names_keep_distinct_id_labels() -> None:
    client = FakeClient()
    original_facets = client.traces.list_facet_values_async

    async def facets(**kwargs: object) -> object:
        if kwargs['field'] == 'project_id':
            return SimpleNamespace(values=[SimpleNamespace(value=name) for name in ('project-1', 'project-2')], has_more=False)
        return await original_facets(**kwargs)

    async def projects(**kwargs: object) -> object:
        return SimpleNamespace(
            data=[SimpleNamespace(project_id=name, name='Research') for name in ('project-1', 'project-2')],
            has_more=False,
        )

    client.traces.list_facet_values_async = facets
    client.projects.list_async = projects
    now = datetime(2026, 9, 22, tzinfo=timezone.utc)

    catalogue = await load_facet_catalogue(cast(Any, client), start=now, end=now)

    assert catalogue.project == ('Research (project-1)', 'Research (project-2)')


@pytest.mark.parametrize(('fail_status', 'has_more'), [(True, False), (False, True)])
@pytest.mark.asyncio
async def test_load_facet_catalogue_rejects_unavailable_or_incomplete_values(
    fail_status: bool, has_more: bool
) -> None:
    client = FakeClient(fail_status=fail_status, has_more=has_more)
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    with pytest.raises((RuntimeError, ValueError), match='status'):
        await load_facet_catalogue(cast(Any, client), start=start, end=start)


@pytest.mark.asyncio
async def test_load_facet_catalogue_cancels_sibling_requests_on_failure() -> None:
    started = asyncio.Event()
    cancelled: set[str] = set()

    class FailingTraces(FakeTraces):
        async def list_facet_values_async(self, **kwargs: object) -> object:
            field = cast(str, kwargs['field'])
            self.calls.append(kwargs)
            if len(self.calls) == 8:
                started.set()
            await started.wait()
            if field == 'status':
                raise RuntimeError('status service unavailable')
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.add(field)
                raise

    client = FakeClient()
    client.traces = FailingTraces()
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)

    with pytest.raises(RuntimeError, match='status service unavailable'):
        await load_facet_catalogue(cast(Any, client), start=start, end=start)

    assert len(client.traces.calls) == 8
    assert len(cancelled) == 7
