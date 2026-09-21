"""Tests for the live Orq facet catalogue adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest
from loguru import logger

from evaluatorq.trace_finder.facets import load_facet_catalogue


class FakeTraces:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def list_facet_values_async(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        field = cast(str, kwargs['field'])
        if field == 'status':
            raise RuntimeError('status service unavailable')
        values = {
            'project_id': ['project-1'],
            'model': ['gpt-5'],
            'provider': ['openai'],
            'product': ['agents'],
            'attributes.orq.leading_span.span_type': ['span.responses'],
            'agent_name': ['support-agent'],
            'tool_name': ['lookup'],
        }[field]
        return SimpleNamespace(values=[SimpleNamespace(value=value) for value in values])


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
    def __init__(self) -> None:
        self.traces = FakeTraces()
        self.projects = FakeProjects()


@pytest.mark.asyncio
async def test_load_facet_catalogue_gathers_all_fields_resolves_projects_and_degrades_one_failure() -> None:
    client = FakeClient()
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 22, tzinfo=timezone.utc)

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
    try:
        catalogue = await load_facet_catalogue(cast(Any, client), start=start, end=end)
    finally:
        logger.remove(sink_id)

    assert catalogue.project == ('Research',)
    assert catalogue.model == ('gpt-5',)
    assert catalogue.provider == ('openai',)
    assert catalogue.status == ()
    assert catalogue.product == ('agents',)
    assert catalogue.trace_type == ('span.responses',)
    assert catalogue.agent_name == ('support-agent',)
    assert catalogue.tool_name == ('lookup',)
    assert any('status' in message for message in messages)
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
