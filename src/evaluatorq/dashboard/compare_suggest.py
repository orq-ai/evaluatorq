"""Which earlier runs a report's compare menu suggests.

A run is suggested when it is older than the current run and either shares its
name once version and date endings are dropped, or, when no name matches,
tested the same agent. Default names (``sim``, ``simulation-<stamp>-<hex>``)
never match by name: every unnamed run would match every other one.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from evaluatorq.dashboard import library, metrics

if TYPE_CHECKING:
    from datetime import datetime

# How many runs the "any other run" picker lists. Suggestions search every run.
MAX_CANDIDATES = 100

# One trailing version or date ending, with the separators before it. A bare
# number counts only after a separator: `gpt4` is a model name, `agent 2` a version.
_TAIL = re.compile(
    r'[\s_.#-]*(?:v\d+|\(\d+\)|(?<=[\s_#])#?\d{1,3}|\d{4}-\d{2}-\d{2}(?:[t ]\d{2}:?\d{2}(?::?\d{2})?)?|\d{8}(?:-\d{6})?|[0-9a-f]{8})$'
)
# Side labels that name a role, not an agent: two runs sharing them tested nothing in common.
_PLACEHOLDER_LABELS = frozenset({'a', 'b', 'baseline', 'candidate', 'control', 'treatment'})
_DEFAULT_KEYS = frozenset({'', 'sim', 'red-team', 'simulation'})
_GENERATED_NAME = re.compile(r'simulation-\d{8}-\d{6}-[0-9a-f]{8}')


@dataclass(frozen=True)
class CompareCandidate:
    """One run that can be offered as the other side of a comparison."""

    rid: str
    name: str
    when: datetime
    target: str | None  # agent (or label pair) under test; None when unknown
    headline: str


def name_key(name: str) -> str:
    """*name* lowercased with trailing version and date endings removed."""
    key = name.strip().lower()
    while (stripped := _TAIL.sub('', key)) != key:
        key = stripped
    return key.rstrip('-_ .#')


def is_default_name(name: str) -> bool:
    """Whether *name* is one a run gets when nobody names it."""
    return name_key(name) in _DEFAULT_KEYS or bool(_GENERATED_NAME.fullmatch(name.strip().lower()))


def suggest(current: CompareCandidate, others: list[CompareCandidate], limit: int = 3) -> list[CompareCandidate]:
    """Earlier runs to compare *current* with, newest first: name matches, else same target."""
    earlier = sorted(
        (c for c in others if c.when < current.when and c.rid != current.rid), key=lambda c: c.when, reverse=True
    )
    if not is_default_name(current.name):
        key = name_key(current.name)
        by_name = [c for c in earlier if not is_default_name(c.name) and name_key(c.name) == key]
        if by_name:
            return by_name[:limit]
    if current.target is None:
        return []
    return [c for c in earlier if c.target == current.target][:limit]


def candidates(surface: Literal['sim', 'pairwise'], roots: list[Path] | None) -> list[CompareCandidate]:
    """Comparable runs of *surface*, newest first, errored runs left out."""
    if surface == 'sim':
        return [_sim_candidate(r) for r in metrics.sim_run_rows(roots) if not r.error]
    key = tuple(str(r.resolve()) for r in (roots or library.default_roots()))
    return list(_pairwise_candidates(key, library.fingerprint(roots)))


@functools.lru_cache(maxsize=8)
def _pairwise_candidates(roots_key: tuple[str, ...], fingerprint: tuple[int, int, int]) -> tuple[CompareCandidate, ...]:
    """Pairwise candidates for *roots_key*, cached until the stores' *fingerprint* changes."""
    del fingerprint  # only part of the cache key
    cards = [c for c in library.scan([Path(r) for r in roots_key]) if c.surface == 'pairwise' and not c.error]
    found = []
    # ponytail: reads every pairwise report once per store change; carry the labels on ReportCard if that gets slow.
    for card in cards:
        if card.path is None:
            continue
        try:
            found.append(_pairwise_candidate(card, card.path))
        except (OSError, ValueError):  # removed or rewritten since the scan: leave it out of the menu
            continue
    return tuple(found)


def _sim_candidate(row: metrics.SimRunRow) -> CompareCandidate:
    target = row.targets[0][0] if len(row.targets) == 1 else None
    # A kind label ("Orq agent") or the run's own name (what old reports fall back
    # to) says nothing about which agent ran, so it must not match other runs.
    if target in metrics.GENERIC_TARGET_LABELS or target == row.name:
        target = None
    rate = row.achieved / row.cases if row.cases > row.errors else None
    goal = '—' if rate is None else f'{rate:.0%}'
    noun = 'conversation' if row.cases == 1 else 'conversations'
    return CompareCandidate(row.rid, row.name, row.when, target, f'goal {goal} · {row.cases:,} {noun}')


def _pairwise_candidate(card: library.ReportCard, path: Path) -> CompareCandidate:
    data = library.read_json_cached(path)
    a, b = str(data.get('label_a') or 'A'), str(data.get('label_b') or 'B')
    target = None if {a.lower(), b.lower()} <= _PLACEHOLDER_LABELS else f'{a} vs {b}'
    return CompareCandidate(card.id, card.name, card.created_at, target, card.headline)
