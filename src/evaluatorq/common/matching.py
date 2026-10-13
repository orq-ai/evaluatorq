"""Shared positional matching by key for ordered collections."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

T = TypeVar('T')
K = TypeVar('K', bound='Hashable')


def pair_indices_by_key(
    entries_a: Sequence[T], entries_b: Sequence[T], *, key: Callable[[T], K]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Pair equal keys in A order, consuming B occurrences positionally.

    Returned indexes preserve source order for matched A entries and leftovers.
    Indexes, rather than object identities, ensure repeated references count as
    distinct occurrences.
    """
    b_by_key: dict[K, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries_b):
        b_by_key[key(entry)].append(index)

    cursor: dict[K, int] = defaultdict(int)
    matched: list[tuple[int, int]] = []
    a_only: list[int] = []
    used_b: set[int] = set()
    for index_a, entry in enumerate(entries_a):
        entry_key = key(entry)
        bucket = b_by_key.get(entry_key, [])
        position = cursor[entry_key]
        if position < len(bucket):
            index_b = bucket[position]
            cursor[entry_key] = position + 1
            matched.append((index_a, index_b))
            used_b.add(index_b)
        else:
            a_only.append(index_a)

    b_only = [index for index in range(len(entries_b)) if index not in used_b]
    return matched, a_only, b_only
