"""Embedding pass: batched text embeddings through the orq router, with a vector cache.

One retry layer: `with_retry` wraps each batch's `client.embeddings.create` call,
and `embed_texts` disarms the SDK's own budget with `without_client_retries` so
the two never multiply.

Unlike a per-trace summary or label, a whole embedding batch failing after
retries is not a per-trace failure: it takes an entire dimension's clustering
down with it, so it raises `EmbeddingError` rather than degrading silently
(review focus / global constraints: "embedding service down for a dimension"
marks the run `status='error'` with a `StageFailure`, the caller's job).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.common.retry import with_retry, without_client_retries
from evaluatorq.common.tracing import record_token_usage, with_llm_span

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types import CreateEmbeddingResponse

    from evaluatorq.insights.cache import InsightsCache


class EmbeddingError(RuntimeError):
    """A whole embedding batch failed after retries — a stage failure, not a per-trace one."""


async def _embed_batch(client: AsyncOpenAI, *, model: str, batch: list[str]) -> CreateEmbeddingResponse:
    async def attempt() -> CreateEmbeddingResponse:
        async with with_llm_span(
            model=model,
            operation='embeddings',
            attributes={'orq.llm.purpose': 'insights.embed'},
        ) as span:
            response = await client.embeddings.create(model=model, input=batch)
            record_token_usage(
                span,
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
                calls=1,
            )
            return response

    return await with_retry(attempt, label='insights embed')


async def embed_texts(
    texts: list[str],
    *,
    client: AsyncOpenAI,
    model: str,
    cache: InsightsCache,
    batch_size: int = 256,
) -> dict[str, list[float]]:
    """Embed every unique text in `texts`, keyed by the original text.

    Cache lookup first (`InsightsCache.get_vectors`), then the remaining misses
    are embedded in batches of `batch_size` through `client.embeddings.create`.
    Duplicate texts are embedded once. Raises `EmbeddingError` if any batch
    fails after retries — a whole-dimension failure is a stage failure, never
    silently dropped or treated as an empty vector.
    """
    unique_texts = list(dict.fromkeys(texts))
    if not unique_texts:
        return {}
    client = without_client_retries(client)

    vectors: dict[str, list[float]] = dict(cache.get_vectors(model, unique_texts))
    missing = [text for text in unique_texts if text not in vectors]

    new_vectors: dict[str, list[float]] = {}
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        try:
            response = await _embed_batch(client, model=model, batch=batch)
            for text, item in zip(batch, response.data, strict=True):
                new_vectors[text] = item.embedding
        except Exception as exc:
            message = str(exc)
            logger.warning('Insights embedding batch of {} text(s) failed: {}', len(batch), message)
            raise EmbeddingError(f'embedding batch failed: {message}') from exc

    if new_vectors:
        cache.put_vectors(model, new_vectors)
    vectors.update(new_vectors)

    return {text: vectors[text] for text in unique_texts}
