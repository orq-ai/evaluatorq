"""Local sqlite cache for trace summaries and embedding vectors.

One connection per `InsightsCache` instance, used only from the event loop thread that
constructs it; each `put_*` call is a single batched transaction. No retry layer here —
a cache miss is always safe (the caller recomputes), so `sqlite3.Error` degrades to a
miss with a `logger.warning` rather than being retried or raised into the run.
"""

from __future__ import annotations

import hashlib
import sqlite3
from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.insights.models import TraceSummary

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_CACHE_PATH = Path('.evaluatorq/cache/insights.sqlite')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (trace_id, span_id, model, prompt_hash)
);
CREATE TABLE IF NOT EXISTS vectors (
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (model, text_hash)
);
"""


def prompt_hash(text: str) -> str:
    """Sha256 hex digest of `text`, truncated to 16 hex chars — used to key summary cache rows."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _pack_vector(vector: list[float]) -> bytes:
    return array('f', vector).tobytes()


def _unpack_vector(blob: bytes) -> list[float]:
    values = array('f')
    values.frombytes(blob)
    return list(values)


class InsightsCache:
    """Sqlite-backed cache of per-trace summaries and per-text embedding vectors.

    `enabled=False` (or a connection that fails to open/migrate) makes every `get_*` a miss
    and every `put_*` a no-op — the cache never raises into a run. A summary is only reused
    when `model` and `prompt_hash` both match the row that produced it (constraint 4 of the
    global review focus: a summary from a different prompt or model must not be reused).
    """

    def __init__(self, path: Path | None = None, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._conn: sqlite3.Connection | None = None
        if not enabled:
            return

        resolved = path if path is not None else DEFAULT_CACHE_PATH
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(resolved, check_same_thread=False)
            conn.executescript(_SCHEMA)
            self._conn = conn
        except sqlite3.Error as exc:
            logger.warning(f'InsightsCache: failed to open/migrate {resolved}: {exc}; caching disabled for this run')
            self._conn = None

    def get_summary(self, trace_id: str, span_id: str, model: str, prompt_hash_value: str) -> TraceSummary | None:
        """Look up a cached summary; a miss on any of trace/span/model/prompt-hash returns None."""
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                'SELECT payload FROM summaries WHERE trace_id = ? AND span_id = ? AND model = ? AND prompt_hash = ?',
                (trace_id, span_id, model, prompt_hash_value),
            ).fetchone()
        except sqlite3.Error as exc:
            logger.warning(f'InsightsCache.get_summary: {exc}; treating as a miss')
            return None
        if row is None:
            return None
        try:
            return TraceSummary.model_validate_json(row[0])
        except ValueError as exc:
            logger.warning(f'InsightsCache.get_summary: corrupt cached payload for {trace_id}/{span_id}: {exc}')
            return None

    def put_summary(
        self, trace_id: str, span_id: str, model: str, prompt_hash_value: str, summary: TraceSummary
    ) -> None:
        """Cache a summary keyed on (trace_id, span_id, model, prompt_hash); no-op if disabled."""
        if self._conn is None:
            return
        try:
            with self._conn:
                self._conn.execute(
                    'INSERT OR REPLACE INTO summaries (trace_id, span_id, model, prompt_hash, payload) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (trace_id, span_id, model, prompt_hash_value, summary.model_dump_json()),
                )
        except sqlite3.Error as exc:
            logger.warning(f'InsightsCache.put_summary: {exc}; summary for {trace_id}/{span_id} not cached')

    def get_vectors(self, model: str, texts: Iterable[str]) -> dict[str, list[float]]:
        """Return the subset of `texts` found in the cache for `model`, keyed by the original text."""
        if self._conn is None:
            return {}
        texts = list(texts)
        if not texts:
            return {}
        hash_to_text = {_text_hash(text): text for text in texts}
        try:
            placeholders = ','.join('?' for _ in hash_to_text)
            # Only '?' placeholders are interpolated here, one per hashed text — no
            # caller-supplied value ever reaches the query string itself.
            rows = self._conn.execute(
                f'SELECT text_hash, vector FROM vectors WHERE model = ? AND text_hash IN ({placeholders})',  # noqa: S608
                (model, *hash_to_text.keys()),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.warning(f'InsightsCache.get_vectors: {exc}; treating as a miss')
            return {}
        result: dict[str, list[float]] = {}
        for text_hash, blob in rows:
            text = hash_to_text.get(text_hash)
            if text is None:
                continue
            result[text] = _unpack_vector(blob)
        return result

    def put_vectors(self, model: str, vectors: dict[str, list[float]]) -> None:
        """Cache embedding vectors keyed on (model, sha256(text)); no-op if disabled."""
        if self._conn is None:
            return
        if not vectors:
            return
        try:
            with self._conn:
                self._conn.executemany(
                    'INSERT OR REPLACE INTO vectors (model, text_hash, vector) VALUES (?, ?, ?)',
                    [(model, _text_hash(text), _pack_vector(vector)) for text, vector in vectors.items()],
                )
        except sqlite3.Error as exc:
            logger.warning(f'InsightsCache.put_vectors: {exc}; {len(vectors)} vector(s) not cached')

    def close(self) -> None:
        """Close the underlying connection; safe to call on a disabled or already-closed cache."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
