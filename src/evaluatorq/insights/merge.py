"""Classifier-based merge of near-duplicate clusters.

One retry layer: mirrors `labeling.py`'s `_classify_with_retry` for a different
call path — `run_classify` never raises, so a retryable outcome
(`JudgeError.TIMEOUT`/`API_CONNECTION`) re-raises its own `error_exc` for
`with_retry` to retry; the final attempt's outcome is returned as-is. Do not
add a second retry layer around this call path.

Per-pair failure never raises and never merges: a classify call that comes
back with an error, a missing answer, or an unreadable probability skips the
merge for that pair only and logs a warning, per the "a degraded path
announces itself" house rule.
"""

from __future__ import annotations

import asyncio
from itertools import starmap
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.judge import ClassifyQuestion, ClassifyRequest, JudgeError, run_classify
from evaluatorq.common.retry import with_retry
from evaluatorq.contracts import LLMCallConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from evaluatorq.common.judge import ClassifyOutcome
    from evaluatorq.insights.describe import ClusterName

_SAME_KEY = 'same'
_MAX_EXAMPLES = 3

# Only these two kinds carry a live `error_exc` we can safely re-raise for `with_retry`
# to classify; PARSE/UNKNOWN failures are not transient and must not be retried.
_RETRYABLE_ERROR_KINDS = frozenset({JudgeError.TIMEOUT, JudgeError.API_CONNECTION})


def _candidate_pairs(names: dict[int, ClusterName], neighbours: dict[int, list[int]]) -> list[tuple[int, int]]:
    """Unordered, deduplicated candidate pairs: every cluster with each of its listed neighbours.

    A neighbour id absent from `names` (e.g. it failed description) is skipped —
    there is nothing to compare it against.
    """
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for cluster_id in sorted(names):
        for neighbour_id in neighbours.get(cluster_id, []):
            if neighbour_id == cluster_id or neighbour_id not in names:
                continue
            pair = (cluster_id, neighbour_id) if cluster_id < neighbour_id else (neighbour_id, cluster_id)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def _cluster_state(name: ClusterName, cluster_examples: list[str]) -> dict[str, Any]:
    return {'name': name.name, 'description': name.description, 'examples': cluster_examples[:_MAX_EXAMPLES]}


async def _classify_pair_with_retry(
    *, client: AsyncOpenAI, model: str, cfg: LLMCallConfig, request: ClassifyRequest
) -> ClassifyOutcome:
    """Call `run_classify`, retrying only a transient failure — see module docstring."""
    holder: list[ClassifyOutcome] = []

    async def attempt() -> ClassifyOutcome:
        outcome = await run_classify(client=client, model=model, cfg=cfg, request=request)
        holder.append(outcome)
        if outcome.error_kind in _RETRYABLE_ERROR_KINDS and outcome.error_exc is not None:
            raise outcome.error_exc
        return outcome

    try:
        return await with_retry(attempt, label='insights merge classify')
    except Exception:  # noqa: BLE001 - the retryable branch above always leaves a recorded outcome behind
        return holder[-1]


def _find(parent: dict[int, int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: dict[int, int], a: int, b: int) -> None:
    root_a, root_b = _find(parent, a), _find(parent, b)
    if root_a == root_b:
        return
    # Deterministic: the smaller id wins as representative, so the result does
    # not depend on the (concurrent, unordered) completion order of the pairs.
    if root_a < root_b:
        parent[root_b] = root_a
    else:
        parent[root_a] = root_b


async def merge_similar(
    names: dict[int, ClusterName],
    examples: dict[int, list[str]],
    neighbours: dict[int, list[int]],
    *,
    client: AsyncOpenAI,
    model: str,
    threshold: float = 0.5,
    parallelism: int = 20,
) -> dict[int, int]:
    """Merge near-duplicate clusters via one classifier `noul` question per candidate pair.

    Candidate pairs are every cluster and each of its `neighbours`
    (`cluster.nearest_neighbours`), unordered and deduplicated. Each pair asks
    a single `noul` question — "Do these two clusters describe the same
    category of user conversation?" — over both clusters' names, descriptions
    and up to 3 example texts each; a probability >= `threshold` merges the
    pair (union-find), so a chain of pairwise merges (a~b, b~c) collapses
    transitively onto one representative. A failed or unreadable pair is
    logged and left unmerged (`merge_similar` never raises).

    Returns every cluster id in `names` mapped to its representative id (a
    cluster that merged into nothing maps to itself).
    """
    parent = {cluster_id: cluster_id for cluster_id in names}
    pairs = _candidate_pairs(names, neighbours)
    semaphore = asyncio.Semaphore(parallelism)
    cfg = LLMCallConfig(model=model, timeout_ms=90_000)

    async def _check_pair(a: int, b: int) -> None:
        state = {
            'cluster_a': _cluster_state(names[a], examples.get(a, [])),
            'cluster_b': _cluster_state(names[b], examples.get(b, [])),
        }
        question = ClassifyQuestion(
            kind='noul',
            instructions='Do these two clusters describe the same category of user conversation?',
            state=state,
        )
        request = ClassifyRequest(state=state, questions={_SAME_KEY: question})

        async with semaphore:
            outcome = await _classify_pair_with_retry(client=client, model=model, cfg=cfg, request=request)

        if outcome.error_kind is not None or outcome.response is None:
            message = outcome.error_message or (
                outcome.error_kind.value if outcome.error_kind else 'classify reply produced no response'
            )
            logger.warning('Insights merge classify failed for clusters {} vs {}: {}', a, b, message)
            return

        answer = outcome.response.answers.get(_SAME_KEY)
        if answer is None or answer.noul is None:
            logger.warning(
                'Insights merge classify reply for clusters {} vs {} is missing the {!r} probability', a, b, _SAME_KEY
            )
            return

        if answer.noul >= threshold:
            _union(parent, a, b)

    await asyncio.gather(*starmap(_check_pair, pairs))

    return {cluster_id: _find(parent, cluster_id) for cluster_id in names}
