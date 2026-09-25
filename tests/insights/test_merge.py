"""Unit tests for `evaluatorq.insights.merge` — classifier-based merge of near-duplicate clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from loguru import logger

from evaluatorq.common.judge import ClassifyAnswer, ClassifyOutcome, ClassifyRequest, ClassifyResponse, JudgeError
from evaluatorq.insights import merge as merge_module
from evaluatorq.insights.describe import ClusterName
from evaluatorq.insights.merge import merge_similar

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class _FakeClient:
    """Placeholder object — `run_classify` is monkeypatched, so this is never called."""


def fake_client() -> AsyncOpenAI:
    return cast('AsyncOpenAI', cast(object, _FakeClient()))


def _names(*ids: int) -> dict[int, ClusterName]:
    return {i: ClusterName(name=f'cluster-{i}', description=f'description-{i}') for i in ids}


@pytest.mark.asyncio
async def test_merges_pair_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        return ClassifyOutcome(response=ClassifyResponse(answers={'same': ClassifyAnswer(type='noul', noul=0.6)}))

    monkeypatch.setattr(merge_module, 'run_classify', fake_run_classify)

    names = _names(0, 1)
    result = await merge_similar(names, examples={0: ['a'], 1: ['b']}, neighbours={0: [1], 1: [0]}, client=fake_client(), model='m', threshold=0.5)

    assert result[0] == result[1]


@pytest.mark.asyncio
async def test_does_not_merge_pair_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        return ClassifyOutcome(response=ClassifyResponse(answers={'same': ClassifyAnswer(type='noul', noul=0.4)}))

    monkeypatch.setattr(merge_module, 'run_classify', fake_run_classify)

    names = _names(0, 1)
    result = await merge_similar(names, examples={0: ['a'], 1: ['b']}, neighbours={0: [1], 1: [0]}, client=fake_client(), model='m', threshold=0.5)

    assert result[0] == 0
    assert result[1] == 1


@pytest.mark.asyncio
async def test_transitive_merge_collapses_chain_to_one_representative(monkeypatch: pytest.MonkeyPatch) -> None:
    # a~b (0,1) and b~c (1,2) merge above threshold; a~c is never asked (not neighbours)
    # but must still collapse onto one representative via union-find.
    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        return ClassifyOutcome(response=ClassifyResponse(answers={'same': ClassifyAnswer(type='noul', noul=0.9)}))

    monkeypatch.setattr(merge_module, 'run_classify', fake_run_classify)

    names = _names(0, 1, 2)
    neighbours = {0: [1], 1: [0, 2], 2: [1]}
    result = await merge_similar(names, examples={}, neighbours=neighbours, client=fake_client(), model='m', threshold=0.5)

    assert len({result[0], result[1], result[2]}) == 1


@pytest.mark.asyncio
async def test_failed_pair_keeps_both_clusters_unmerged_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        return ClassifyOutcome(error_kind=JudgeError.API_STATUS, error_message='500 from router')

    monkeypatch.setattr(merge_module, 'run_classify', fake_run_classify)

    names = _names(0, 1)
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
    try:
        result = await merge_similar(
            names, examples={0: ['a'], 1: ['b']}, neighbours={0: [1], 1: [0]}, client=fake_client(), model='m', threshold=0.5
        )
    finally:
        logger.remove(sink_id)

    assert result[0] == 0
    assert result[1] == 1
    assert any('0' in message and '1' in message for message in messages)


@pytest.mark.asyncio
async def test_missing_answer_keeps_both_unmerged(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        return ClassifyOutcome(response=ClassifyResponse(answers={}))

    monkeypatch.setattr(merge_module, 'run_classify', fake_run_classify)

    names = _names(0, 1)
    result = await merge_similar(
        names, examples={0: ['a'], 1: ['b']}, neighbours={0: [1], 1: [0]}, client=fake_client(), model='m', threshold=0.5
    )

    assert result[0] == 0
    assert result[1] == 1


@pytest.mark.asyncio
async def test_pair_deduplicated_regardless_of_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        nonlocal calls
        calls += 1
        return ClassifyOutcome(response=ClassifyResponse(answers={'same': ClassifyAnswer(type='noul', noul=0.9)}))

    monkeypatch.setattr(merge_module, 'run_classify', fake_run_classify)

    names = _names(0, 1)
    # Both directions listed as neighbours — must be asked once, not twice.
    await merge_similar(names, examples={}, neighbours={0: [1], 1: [0]}, client=fake_client(), model='m', threshold=0.5)

    assert calls == 1


@pytest.mark.asyncio
async def test_neighbour_missing_from_names_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        nonlocal calls
        calls += 1
        return ClassifyOutcome(response=ClassifyResponse(answers={'same': ClassifyAnswer(type='noul', noul=0.9)}))

    monkeypatch.setattr(merge_module, 'run_classify', fake_run_classify)

    names = _names(0)  # cluster 1 was never described (e.g. it failed) so it's absent
    await merge_similar(names, examples={}, neighbours={0: [1]}, client=fake_client(), model='m', threshold=0.5)

    assert calls == 0


@pytest.mark.asyncio
async def test_request_carries_both_clusters_name_description_and_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[ClassifyRequest] = []

    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        captured.append(request)
        return ClassifyOutcome(response=ClassifyResponse(answers={'same': ClassifyAnswer(type='noul', noul=0.1)}))

    monkeypatch.setattr(merge_module, 'run_classify', fake_run_classify)

    names = _names(0, 1)
    examples = {0: ['ex-a1', 'ex-a2', 'ex-a3', 'ex-a4'], 1: ['ex-b1']}
    await merge_similar(names, examples=examples, neighbours={0: [1], 1: [0]}, client=fake_client(), model='m', threshold=0.5)

    assert len(captured) == 1
    state = captured[0].state
    assert isinstance(state, dict)
    assert state['cluster_a']['name'] == 'cluster-0'
    assert state['cluster_a']['description'] == 'description-0'
    assert state['cluster_a']['examples'] == ['ex-a1', 'ex-a2', 'ex-a3']  # capped at 3
    assert state['cluster_b']['examples'] == ['ex-b1']
    question = captured[0].questions['same']
    assert question.kind == 'noul'
    assert 'same category' in question.instructions
