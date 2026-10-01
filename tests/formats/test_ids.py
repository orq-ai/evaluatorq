"""Deterministic SHA-256 ids."""

# ruff: noqa: S101

from __future__ import annotations

from evaluatorq.formats._ids import content_seed, stable_hex


def test_stable_hex_is_deterministic_and_sized() -> None:
    assert stable_hex('a', 'b', length=16) == stable_hex('a', 'b', length=16)
    assert len(stable_hex('a', length=32)) == 32


def test_stable_hex_separates_parts_with_nul() -> None:
    assert stable_hex('a', 'bc', length=64) != stable_hex('ab', 'c', length=64)


def test_content_seed_ignores_key_order() -> None:
    assert content_seed({'a': 1, 'b': [2]}) == content_seed({'b': [2], 'a': 1})
    assert content_seed({'a': 1}) != content_seed({'a': 2})
