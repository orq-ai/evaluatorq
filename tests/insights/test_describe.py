"""Unit tests for `evaluatorq.insights.describe` — LLM cluster naming."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from evaluatorq.common.structured_output import StructuredResult
from evaluatorq.insights import describe as describe_module
from evaluatorq.insights.describe import ClusterName, describe_clusters, describe_top_level

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class _FakeClient:
    """Placeholder object — `generate_structured` is monkeypatched, so this is never called."""


def fake_client() -> AsyncOpenAI:
    return cast('AsyncOpenAI', cast(object, _FakeClient()))


@pytest.mark.asyncio
async def test_describe_clusters_calls_generate_structured_per_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[ClusterName]:
        calls.append(kwargs)
        return StructuredResult(parsed=ClusterName(name='Refund requests', description='Users asked about refunds.'), raw='')

    monkeypatch.setattr(describe_module, 'generate_structured', fake_generate_structured)

    members = {
        0: ['refund my order', 'i want my money back'],
        1: ['reset my password', 'cannot log in'],
    }
    neighbours = {0: [1], 1: [0]}

    result = await describe_clusters(members, neighbours=neighbours, dimension='intent', client=fake_client(), model='m')

    assert len(calls) == 2
    assert result[0] == ClusterName(name='Refund requests', description='Users asked about refunds.')
    assert result[1] == ClusterName(name='Refund requests', description='Users asked about refunds.')
    for kwargs in calls:
        assert kwargs['response_format'] is ClusterName
        assert kwargs['max_tokens'] == 400
        assert kwargs['label'] == 'insights.describe'


@pytest.mark.asyncio
async def test_describe_clusters_includes_member_and_contrastive_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[int, str] = {}

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[ClusterName]:
        content = kwargs['messages'][0]['content']
        captured[len(captured)] = content
        return StructuredResult(parsed=ClusterName(name='n', description='d'), raw='')

    monkeypatch.setattr(describe_module, 'generate_structured', fake_generate_structured)

    members = {0: ['refund my order'], 1: ['reset my password'], 2: ['cancel my subscription']}
    neighbours = {0: [1, 2]}

    await describe_clusters({0: members[0]}, neighbours=neighbours, dimension='intent', client=fake_client(), model='m')

    # Only cluster 0 was described (members only has key 0), so its neighbours
    # 1 and 2 are absent from `members` and contribute no contrastive text —
    # this test only checks that the member text made it into the prompt.
    content = next(iter(captured.values()))
    assert 'refund my order' in content
    assert '<examples>' in content
    assert '<contrastive>' in content


@pytest.mark.asyncio
async def test_describe_clusters_uses_up_to_3_neighbours_and_3_examples_each(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[ClusterName]:
        captured.append(kwargs['messages'][0]['content'])
        return StructuredResult(parsed=ClusterName(name='n', description='d'), raw='')

    monkeypatch.setattr(describe_module, 'generate_structured', fake_generate_structured)

    members = {
        0: ['target text'],
        1: ['n1-a', 'n1-b', 'n1-c', 'n1-d'],
        2: ['n2-a', 'n2-b'],
        3: ['n3-a'],
        4: ['n4-a'],  # a 4th neighbour must be dropped (cap is 3 neighbour clusters)
    }
    neighbours = {0: [1, 2, 3, 4]}

    await describe_clusters(members, neighbours=neighbours, dimension='intent', client=fake_client(), model='m')

    content = captured[0]
    assert 'n1-a' in content
    assert 'n1-b' in content
    assert 'n1-c' in content
    assert 'n1-d' not in content  # 4th example of neighbour 1 dropped (cap is 3 per neighbour)
    assert 'n2-a' in content
    assert 'n3-a' in content
    assert 'n4-a' not in content  # 4th neighbour cluster dropped (cap is 3 neighbour clusters)


@pytest.mark.asyncio
async def test_describe_clusters_exception_yields_error_string_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    from loguru import logger

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[ClusterName]:
        raise RuntimeError('provider down')

    monkeypatch.setattr(describe_module, 'generate_structured', fake_generate_structured)

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
    try:
        result = await describe_clusters(
            {0: ['hello']}, neighbours={0: []}, dimension='intent', client=fake_client(), model='m'
        )
    finally:
        logger.remove(sink_id)

    assert isinstance(result[0], str)
    assert 'provider down' in result[0]
    assert any('0' in message and 'provider down' in message for message in messages)


@pytest.mark.asyncio
async def test_describe_clusters_unparseable_output_yields_error_string(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[ClusterName]:
        return StructuredResult(parsed=None, raw='not json')

    monkeypatch.setattr(describe_module, 'generate_structured', fake_generate_structured)

    result = await describe_clusters({0: ['hello']}, neighbours={0: []}, dimension='intent', client=fake_client(), model='m')

    assert isinstance(result[0], str)
    assert 'unparseable' in result[0]


@pytest.mark.asyncio
async def test_describe_clusters_dimension_selects_the_right_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[ClusterName]:
        captured.append(kwargs['messages'][0]['content'])
        return StructuredResult(parsed=ClusterName(name='n', description='d'), raw='')

    monkeypatch.setattr(describe_module, 'generate_structured', fake_generate_structured)

    await describe_clusters({0: ['x']}, neighbours={0: []}, dimension='failure', client=fake_client(), model='m')
    await describe_clusters({0: ['x']}, neighbours={0: []}, dimension='sentiment', client=fake_client(), model='m')

    assert 'failure mechanism' in captured[0]
    assert 'sentiment pattern' in captured[1]


@pytest.mark.asyncio
async def test_describe_top_level_names_group_from_children(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[ClusterName]:
        captured.append(kwargs['messages'][0]['content'])
        assert kwargs['response_format'] is ClusterName
        assert kwargs['max_tokens'] == 400
        return StructuredResult(parsed=ClusterName(name='Billing support', description='Covers refunds and billing questions.'), raw='')

    monkeypatch.setattr(describe_module, 'generate_structured', fake_generate_structured)

    children = {
        0: [
            ClusterName(name='Refund requests', description='Users asked about refunds.'),
            ClusterName(name='Invoice questions', description='Users asked about invoices.'),
        ]
    }

    result = await describe_top_level(children, client=fake_client(), model='m')

    assert result[0] == ClusterName(name='Billing support', description='Covers refunds and billing questions.')
    assert 'Refund requests' in captured[0]
    assert 'Invoice questions' in captured[0]


@pytest.mark.asyncio
async def test_describe_top_level_failure_yields_error_string(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[ClusterName]:
        raise RuntimeError('boom')

    monkeypatch.setattr(describe_module, 'generate_structured', fake_generate_structured)

    children = {0: [ClusterName(name='a', description='b')]}
    result = await describe_top_level(children, client=fake_client(), model='m')

    assert isinstance(result[0], str)
    assert 'boom' in result[0]


@pytest.mark.parametrize('function, args', [
    (describe_clusters, ({0: ['member']},)),
    (describe_top_level, ({0: [ClusterName(name='child', description='details')]},)),
])
@pytest.mark.asyncio
async def test_describe_apis_reject_non_positive_parallelism(function: Any, args: Any) -> None:
    kwargs: dict[str, Any] = {'client': fake_client(), 'model': 'm', 'parallelism': 0}
    if function is describe_clusters:
        kwargs.update(neighbours={0: []}, dimension='intent')

    with pytest.raises(ValueError, match='parallelism must be positive'):
        await function(*args, **kwargs)
