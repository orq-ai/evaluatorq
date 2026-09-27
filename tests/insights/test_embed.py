"""Unit tests for `evaluatorq.insights.embed` — batched embeddings with a vector cache."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from loguru import logger
from openai import APITimeoutError

from evaluatorq.common import model_catalogue
from evaluatorq.common.model_catalogue import ModelInfo
from evaluatorq.contracts import LLMCallConfig, Usage
from evaluatorq.insights.cache import InsightsCache
from evaluatorq.insights.embed import EmbeddingError, embed_texts
from evaluatorq.insights.usage import UsageLedger

MODEL = 'openai/text-embedding-3-small'


@pytest.fixture
def cache(tmp_path: Path) -> InsightsCache:
    return InsightsCache(tmp_path / 'insights.sqlite')


class _Embedding:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _Usage:
    def __init__(self, prompt_tokens: int = 1, total_tokens: int = 1) -> None:
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens


class _Response:
    def __init__(
        self,
        data: list[_Embedding],
        *,
        prompt_tokens: int = 1,
        model: str | None = MODEL,
    ) -> None:
        self.data = data
        self.usage = _Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)
        self.model = model


def _vector_for(text: str) -> list[float]:
    return [float(len(text)), float(sum(ord(c) for c in text) % 97)]


class _FakeEmbeddingsResource:
    def __init__(
        self,
        *,
        fail_batches: set[int] | None = None,
        wrong_count_batches: set[int] | None = None,
        prompt_tokens: int = 1,
        response_model: str | None = MODEL,
    ) -> None:
        self.calls: list[list[str]] = []
        self._fail_batches = fail_batches or set()
        self._wrong_count_batches = wrong_count_batches or set()
        self._prompt_tokens = prompt_tokens
        self._response_model = response_model

    async def create(self, *, model: str, input: list[str]) -> _Response:  # noqa: A002
        self.calls.append(list(input))
        if len(self.calls) - 1 in self._fail_batches:
            raise RuntimeError('embedding API down')
        data = [_Embedding(_vector_for(text)) for text in input]
        if len(self.calls) - 1 in self._wrong_count_batches:
            data = data[:-1]
        return _Response(data, prompt_tokens=self._prompt_tokens, model=self._response_model)


class _FakeClient:
    def __init__(
        self,
        *,
        fail_batches: set[int] | None = None,
        wrong_count_batches: set[int] | None = None,
        prompt_tokens: int = 1,
        response_model: str | None = MODEL,
    ) -> None:
        self.embeddings = _FakeEmbeddingsResource(
            fail_batches=fail_batches,
            wrong_count_batches=wrong_count_batches,
            prompt_tokens=prompt_tokens,
            response_model=response_model,
        )


def fake_client(
    *,
    fail_batches: set[int] | None = None,
    wrong_count_batches: set[int] | None = None,
    prompt_tokens: int = 1,
    response_model: str | None = MODEL,
) -> Any:
    return _FakeClient(
        fail_batches=fail_batches,
        wrong_count_batches=wrong_count_batches,
        prompt_tokens=prompt_tokens,
        response_model=response_model,
    )


@pytest.mark.asyncio
async def test_embed_texts_records_priced_usage(
    cache: InsightsCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def price(
        usage: Usage | None, model: str, client: Any, *, served_model: str | None = None
    ) -> Usage | None:
        assert usage is not None
        return usage.model_copy(update={'total_cost': 0.000012, 'input_cost': 0.000012, 'priced_calls': 1})

    monkeypatch.setattr('evaluatorq.insights.embed.price_usage', price)
    ledger = UsageLedger()
    client = fake_client(prompt_tokens=12)

    await embed_texts(['hello'], client=client, model=MODEL, cache=cache, usage=ledger)

    recorded = ledger.totals()['embed']
    assert recorded is not None
    assert recorded.input_tokens == 12
    assert recorded.total_cost == 0.000012


@pytest.mark.asyncio
async def test_embed_texts_records_failed_retry_as_unknown_then_success(
    cache: InsightsCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RetryOnce:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, *, model: str, input: list[str]) -> _Response:  # noqa: A002
            self.calls += 1
            if self.calls == 1:
                raise APITimeoutError(request=httpx.Request('POST', 'https://example.com'))
            return _Response([_Embedding(_vector_for(text)) for text in input], prompt_tokens=7)

    client: Any = _FakeClient()
    client.embeddings = RetryOnce()

    async def price(usage: Usage | None, model: str, client: Any, *, served_model: str | None = None) -> Usage | None:
        assert usage is not None
        return usage.model_copy(update={'total_cost': 0.00001, 'input_cost': 0.00001, 'priced_calls': 1})

    monkeypatch.setattr('evaluatorq.insights.embed.price_usage', price)

    async def no_wait(*_: Any) -> None:
        return None

    monkeypatch.setattr('evaluatorq.common.retry.asyncio.sleep', no_wait)
    ledger = UsageLedger()

    await embed_texts(['hello'], client=client, model=MODEL, cache=cache, usage=ledger)

    recorded = ledger.totals()['embed']
    assert recorded is not None
    assert recorded.calls == 2
    assert recorded.priced_calls == 1
    assert recorded.prompt_tokens == 7
    assert recorded.total_cost == 0.00001


@pytest.mark.asyncio
@pytest.mark.parametrize(('retry_count', 'expected_calls'), [(0, 1), (1, 2)])
async def test_embed_retry_count_bounds_attempts_and_records_unknown_usage(
    cache: InsightsCache, retry_count: int, expected_calls: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    class AlwaysTimeout:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, *, model: str, input: list[str]) -> _Response:  # noqa: A002
            self.calls += 1
            raise APITimeoutError(request=httpx.Request('POST', 'https://example.com'))

    client: Any = _FakeClient()
    client.embeddings = AlwaysTimeout()

    async def no_wait(*_: Any) -> None:
        return None

    monkeypatch.setattr('evaluatorq.common.retry.asyncio.sleep', no_wait)
    ledger = UsageLedger()
    with pytest.raises(EmbeddingError):
        await embed_texts(
            ['hello'],
            client=client,
            model=MODEL,
            cache=cache,
            usage=ledger,
            cfg=LLMCallConfig(model=MODEL, retry_count=retry_count),
        )

    assert client.embeddings.calls == expected_calls
    recorded = ledger.totals()['embed']
    assert recorded is not None
    assert recorded.calls == expected_calls


@pytest.mark.asyncio
async def test_embed_texts_prices_at_the_response_served_model(
    cache: InsightsCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    served_model = 'openai/served-embedding-model'

    async def catalogue(client: Any = None) -> dict[str, ModelInfo]:
        return {'served-embedding-model': ModelInfo(0.005, 0.0, 'openai', supports_responses=False)}

    monkeypatch.setattr(model_catalogue, '_load_catalogue', catalogue)
    ledger = UsageLedger()

    await embed_texts(
        ['hello'],
        client=fake_client(prompt_tokens=12, response_model=served_model),
        model=MODEL,
        cache=cache,
        usage=ledger,
    )

    recorded = ledger.totals()['embed']
    assert recorded is not None
    assert recorded.input_cost == pytest.approx(0.00006)
    assert recorded.total_cost == pytest.approx(0.00006)


@pytest.mark.asyncio
async def test_embed_texts_warns_once_when_model_is_unpriced(
    cache: InsightsCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def catalogue_miss(
        usage: Usage | None, model: str, client: Any, *, served_model: str | None = None
    ) -> Usage | None:
        return usage

    monkeypatch.setattr('evaluatorq.insights.embed.price_usage', catalogue_miss)
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
    ledger = UsageLedger()
    client = fake_client()
    try:
        await embed_texts(['one', 'two', 'three'], client=client, model=MODEL, cache=cache, usage=ledger, batch_size=1)
        await embed_texts(['four', 'five'], client=client, model=MODEL, cache=cache, usage=ledger, batch_size=1)
    finally:
        logger.remove(sink_id)

    recorded = ledger.totals()['embed']
    assert recorded is not None
    assert recorded.total_cost is None
    warnings = [message for message in messages if 'not priced in the Orq catalogue' in message]
    assert warnings == [f'Insights embedding model {MODEL} is not priced in the Orq catalogue; embed cost stays unknown']


@pytest.mark.asyncio
async def test_embed_texts_without_ledger_warns_once_per_call(
    cache: InsightsCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def catalogue_miss(
        usage: Usage | None, model: str, client: Any, *, served_model: str | None = None
    ) -> Usage | None:
        return usage

    monkeypatch.setattr('evaluatorq.insights.embed.price_usage', catalogue_miss)
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
    try:
        await embed_texts(['one', 'two'], client=fake_client(), model=MODEL, cache=cache, batch_size=1)
    finally:
        logger.remove(sink_id)

    warnings = [message for message in messages if 'not priced in the Orq catalogue' in message]
    assert warnings == [f'Insights embedding model {MODEL} is not priced in the Orq catalogue; embed cost stays unknown']


@pytest.mark.asyncio
async def test_embed_texts_cache_hit_avoids_the_call(cache: InsightsCache) -> None:
    cache.put_vectors(MODEL, {'hello world': [1.0, 2.0]})
    client = fake_client()

    result = await embed_texts(['hello world'], client=client, model=MODEL, cache=cache)

    assert result == {'hello world': [1.0, 2.0]}
    assert client.embeddings.calls == []


@pytest.mark.asyncio
async def test_embed_texts_batching_splits_into_multiple_calls(cache: InsightsCache) -> None:
    texts = [f'text-{i}' for i in range(600)]
    client = fake_client()

    result = await embed_texts(texts, client=client, model=MODEL, cache=cache, batch_size=256)

    assert len(client.embeddings.calls) == 3
    assert [len(batch) for batch in client.embeddings.calls] == [256, 256, 88]
    assert set(result.keys()) == set(texts)
    for text in texts:
        assert result[text] == _vector_for(text)

    # Newly-embedded vectors are cached for next time.
    cached = cache.get_vectors(MODEL, texts)
    assert len(cached) == 600


@pytest.mark.asyncio
async def test_embed_texts_failure_raises_embedding_error(cache: InsightsCache) -> None:
    client = fake_client(fail_batches={0})

    with pytest.raises(EmbeddingError):
        await embed_texts(['a', 'b'], client=client, model=MODEL, cache=cache)


@pytest.mark.asyncio
async def test_malformed_embedding_response_raises_embedding_error(cache: InsightsCache) -> None:
    client = fake_client(wrong_count_batches={0})

    with pytest.raises(EmbeddingError, match='embedding batch failed'):
        await embed_texts(['a', 'b'], client=client, model=MODEL, cache=cache)


@pytest.mark.asyncio
async def test_embed_texts_duplicate_texts_embedded_once(cache: InsightsCache) -> None:
    client = fake_client()

    result = await embed_texts(['dup', 'dup', 'unique'], client=client, model=MODEL, cache=cache)

    assert client.embeddings.calls == [['dup', 'unique']]
    assert result == {'dup': _vector_for('dup'), 'unique': _vector_for('unique')}


@pytest.mark.asyncio
async def test_embed_texts_empty_input_returns_empty_dict(cache: InsightsCache) -> None:
    client = fake_client()

    result = await embed_texts([], client=client, model=MODEL, cache=cache)

    assert result == {}
    assert client.embeddings.calls == []
