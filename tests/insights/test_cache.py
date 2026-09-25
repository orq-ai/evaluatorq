"""Unit tests for the local sqlite cache of trace summaries and embedding vectors."""

from __future__ import annotations

from evaluatorq.insights.cache import InsightsCache, prompt_hash
from evaluatorq.insights.models import TraceSummary


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


def test_disabled_cache_never_hits(tmp_path):
    c = InsightsCache(tmp_path / 'c.sqlite', enabled=False)
    c.put_vectors('e', {'a': [1.0]})
    assert c.get_vectors('e', ['a']) == {}


def test_corrupt_file_degrades_to_miss(tmp_path, caplog):
    p = tmp_path / 'c.sqlite'
    p.write_bytes(b'not sqlite')
    assert InsightsCache(p).get_vectors('e', ['a']) == {}
