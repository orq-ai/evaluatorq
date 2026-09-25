"""Unit tests for `evaluatorq.insights.embed` — batched embeddings with a vector cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evaluatorq.insights.cache import InsightsCache
from evaluatorq.insights.embed import EmbeddingError, embed_texts

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
    def __init__(self, data: list[_Embedding]) -> None:
        self.data = data
        self.usage = _Usage()


def _vector_for(text: str) -> list[float]:
    return [float(len(text)), float(sum(ord(c) for c in text) % 97)]


class _FakeEmbeddingsResource:
    def __init__(self, *, fail_batches: set[int] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._fail_batches = fail_batches or set()

    async def create(self, *, model: str, input: list[str]) -> _Response:  # noqa: A002
        self.calls.append(list(input))
        if len(self.calls) - 1 in self._fail_batches:
            raise RuntimeError('embedding API down')
        return _Response([_Embedding(_vector_for(text)) for text in input])


class _FakeClient:
    def __init__(self, *, fail_batches: set[int] | None = None) -> None:
        self.embeddings = _FakeEmbeddingsResource(fail_batches=fail_batches)


def fake_client(*, fail_batches: set[int] | None = None) -> Any:
    return _FakeClient(fail_batches=fail_batches)


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
