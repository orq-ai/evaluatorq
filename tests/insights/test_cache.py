"""Unit tests for the local sqlite cache of trace summaries and embedding vectors."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from typing import cast

from evaluatorq.insights.cache import InsightsCache, prompt_hash
from evaluatorq.insights.models import TraceSummary


def test_prompt_hash_is_full_sha256_digest():
    text = 'summary prompt'
    assert prompt_hash(text) == hashlib.sha256(text.encode('utf-8')).hexdigest()
    assert len(prompt_hash(text)) == 64


def test_default_cache_follows_run_store_directory(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    cache = InsightsCache()
    try:
        cache.put_vectors('e', {'a': [0.5]})
        assert (tmp_path / 'cache' / 'insights.sqlite').is_file()
        assert cache.get_vectors('e', ['a']) == {'a': [0.5]}
    finally:
        cache.close()


def test_vector_payload_uses_portable_float_byte_order():
    from evaluatorq.insights.cache import _pack_vector, _unpack_vector

    blob = _pack_vector([0.5, 1.0])
    assert blob.startswith(b'EQV2')
    assert blob[-8:] == struct.pack('!2f', 0.5, 1.0)
    assert _unpack_vector(blob) == [0.5, 1.0]


def test_summary_hit_requires_same_model_and_prompt(tmp_path):
    c = InsightsCache(tmp_path / 'c.sqlite')
    s = TraceSummary(summary='s', request='r', task=None, topic=None, sentiment_explanation=None)
    c.put_summary('t', 'sp', 'm1', prompt_hash('p1'), s)
    assert c.get_summary('t', 'sp', 'm1', prompt_hash('p1')) == s
    assert c.get_summary('t', 'sp', 'm1', prompt_hash('p2')) is None
    assert c.get_summary('t', 'sp', 'm2', prompt_hash('p1')) is None


def test_vectors_per_text_hit_and_miss(tmp_path):
    c = InsightsCache(tmp_path / 'c.sqlite')
    c.put_vectors('e', {'a': [0.5, 1.0]})
    assert c.get_vectors('e', ['a', 'b']) == {'a': [0.5, 1.0]}


def test_large_vector_lookup_stays_below_sqlite_bind_limit(tmp_path):
    cache = InsightsCache(tmp_path / 'c.sqlite')
    vectors = {f'text-{index}': [float(index)] for index in range(1201)}
    cache.put_vectors('e', vectors)
    connection = cache._conn
    assert connection is not None

    class LimitedConnection:
        queries = 0

        def execute(self, query, params):
            assert len(params) <= 501
            self.queries += 1
            return connection.execute(query, params)

    limited = LimitedConnection()
    cache._conn = cast('sqlite3.Connection', cast('object', limited))
    try:
        assert cache.get_vectors('e', vectors) == vectors
        assert limited.queries == 3
    finally:
        cache._conn = connection
        cache.close()


def test_disabled_cache_never_hits(tmp_path):
    c = InsightsCache(tmp_path / 'c.sqlite', enabled=False)
    c.put_vectors('e', {'a': [1.0]})
    assert c.get_vectors('e', ['a']) == {}


def test_corrupt_file_degrades_to_miss(tmp_path, caplog):
    p = tmp_path / 'c.sqlite'
    p.write_bytes(b'not sqlite')
    assert InsightsCache(p).get_vectors('e', ['a']) == {}


def test_unwritable_cache_directory_disables_cache(tmp_path, monkeypatch):
    def fail_mkdir(*args, **kwargs):
        raise PermissionError('read only')

    monkeypatch.setattr('pathlib.Path.mkdir', fail_mkdir)
    cache = InsightsCache(tmp_path / 'cache.sqlite')

    assert cache.get_vectors('e', ['a']) == {}
    cache.put_vectors('e', {'a': [1.0]})


def test_corrupt_vector_row_is_miss_but_valid_rows_survive(tmp_path):
    path = tmp_path / 'cache.sqlite'
    cache = InsightsCache(path)
    cache.put_vectors('e', {'good': [0.5, 1.0]})
    with sqlite3.connect(path) as conn:
        conn.execute('INSERT INTO vectors (model, text_hash, vector) VALUES (?, ?, ?)', ('e', 'bad-hash', b'bad'))
    from evaluatorq.insights.cache import _text_hash

    with sqlite3.connect(path) as conn:
        conn.execute('UPDATE vectors SET text_hash = ? WHERE text_hash = ?', (_text_hash('bad'), 'bad-hash'))

    assert cache.get_vectors('e', ['good', 'bad']) == {'good': [0.5, 1.0]}


def test_aligned_truncation_and_same_length_corruption_are_cache_misses(tmp_path):
    path = tmp_path / 'cache.sqlite'
    cache = InsightsCache(path)
    cache.put_vectors('e', {'good': [0.5, 1.0], 'short': [0.25, 0.75], 'tampered': [1.0, 2.0]})

    from evaluatorq.insights.cache import _text_hash

    with sqlite3.connect(path) as conn:
        for text in ('short', 'tampered'):
            key = _text_hash(text)
            blob = conn.execute('SELECT vector FROM vectors WHERE text_hash = ?', (key,)).fetchone()[0]
            damaged = blob[:-4] if text == 'short' else blob[:-1] + bytes([blob[-1] ^ 1])
            conn.execute('UPDATE vectors SET vector = ? WHERE text_hash = ?', (damaged, key))

    assert cache.get_vectors('e', ['good', 'short', 'tampered']) == {'good': [0.5, 1.0]}


def test_invalid_vectors_are_skipped_without_aborting_valid_writes(tmp_path):
    cache = InsightsCache(tmp_path / 'cache.sqlite')
    vectors = {
        'good': [0.5, 1.0],
        'not-numeric': cast('list[float]', ['bad']),
        'overflow': [1e100],
        'empty': [],
        'non-finite': [float('nan')],
    }

    cache.put_vectors('e', vectors)

    assert cache.get_vectors('e', list(vectors)) == {'good': [0.5, 1.0]}
