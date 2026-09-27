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

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.common.model_catalogue import price_usage
from evaluatorq.common.retry import with_retry, without_client_retries
from evaluatorq.common.structured_output import warn_config_redirect, warn_unread_config_fields
from evaluatorq.common.tracing import record_token_usage, with_llm_span
from evaluatorq.contracts import LLMCallConfig, Usage

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types import CreateEmbeddingResponse

    from evaluatorq.insights.cache import InsightsCache
    from evaluatorq.insights.usage import UsageLedger


class EmbeddingError(RuntimeError):
    """A whole embedding batch failed after retries — a stage failure, not a per-trace one."""


async def _embed_batch(
    client: AsyncOpenAI, *, model: str, batch: list[str], cfg: LLMCallConfig, usage_ledger: UsageLedger | None = None
) -> tuple[CreateEmbeddingResponse, Usage | None]:
    async def attempt() -> tuple[CreateEmbeddingResponse, Usage | None]:
        try:
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
                priced_usage = await price_usage(
                    Usage.extract(response.usage), model, client, served_model=getattr(response, 'model', None)
                )
                if usage_ledger is not None:
                    usage_ledger.add('embed', priced_usage)
                return response, priced_usage
        except Exception:
            if usage_ledger is not None:
                usage_ledger.add('embed', None)
            raise

    return await with_retry(attempt, max_attempts=cfg.retry_count + 1, label='insights embed')


async def embed_texts(
    texts: list[str],
    *,
    client: AsyncOpenAI,
    model: str,
    cache: InsightsCache,
    usage: UsageLedger | None = None,
    batch_size: int = 256,
    cfg: LLMCallConfig | None = None,
) -> dict[str, list[float]]:
    """Embed every unique text in `texts`, keyed by the original text.

    Cache lookup first (`InsightsCache.get_vectors`), then the remaining misses
    are embedded in batches of `batch_size` through `client.embeddings.create`.
    Duplicate texts are embedded once. Raises `EmbeddingError` if any batch
    fails after retries — a whole-dimension failure is a stage failure, never
    silently dropped or treated as an empty vector.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    unique_texts = list(dict.fromkeys(texts))
    if not unique_texts:
        return {}
    client = without_client_retries(client)
    call_cfg = cfg if cfg is not None else LLMCallConfig(model=model)
    if cfg is not None:
        warn_config_redirect(cfg, resolved_model=model, caller='insights.embed_texts')
        warn_unread_config_fields(cfg, frozenset({'retry_count'}), caller='insights.embed_texts')

    # SQLite is synchronous; keep cache I/O off the event loop, just like the
    # other blocking local-store operations in the insights pipeline.
    vectors: dict[str, list[float]] = dict(await asyncio.to_thread(cache.get_vectors, model, unique_texts))
    missing = [text for text in unique_texts if text not in vectors]

    new_vectors: dict[str, list[float]] = {}
    unpriced = False
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        try:
            response, batch_usage = await _embed_batch(
                client,
                model=model,
                batch=batch,
                cfg=call_cfg,
                usage_ledger=usage,
            )
            if batch_usage is not None and batch_usage.total_cost is None:
                unpriced = True
            for text, item in zip(batch, response.data, strict=True):
                new_vectors[text] = item.embedding
        except Exception as exc:
            message = str(exc)
            logger.warning('Insights embedding batch of {} text(s) failed: {}', len(batch), message)
            raise EmbeddingError(f'embedding batch failed: {message}') from exc

    if unpriced and (usage is None or usage.claim_warning('unpriced_embedding_cost')):
        logger.warning(
            'Insights embedding model {} is not priced in the Orq catalogue; embed cost stays unknown', model
        )

    if new_vectors:
        await asyncio.to_thread(cache.put_vectors, model, new_vectors)
    vectors.update(new_vectors)

    return {text: vectors[text] for text in unique_texts}
