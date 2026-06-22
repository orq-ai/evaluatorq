"""Discovery + identity + single-field kind sniff for the report dashboard."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

_ARTIFACT_PREFIXES = ('01_', '02_', '03_')


def report_id(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).digest()[:12]
    return base64.urlsafe_b64encode(digest).decode().rstrip('=')


def sniff_kind(data: dict[str, object]) -> str | None:
    """Surface from a single required-unique field. sim ('mode') checked first."""
    if 'mode' in data:
        return 'sim'
    if 'pipeline' in data:
        return 'redteam'
    return None


def load_surface(path: Path) -> tuple[str | None, dict[str, object]]:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None, {}
    return sniff_kind(data), data


@dataclass(frozen=True)
class ReportCard:
    id: str
    surface: str
    name: str
    created_at: datetime
    headline: str
    path: Path
    error: str | None = None


def _default_roots() -> list[Path]:
    from evaluatorq.redteam.runner import get_runs_dir
    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    return [get_runs_dir(), get_sim_runs_dir()]


def _iter_report_files(roots: list[Path]) -> Iterator[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob('*.json')):
            if not p.name.startswith(_ARTIFACT_PREFIXES):
                yield p


def _card(path: Path) -> ReportCard | None:
    surface, data = load_surface(path)
    if surface is None:
        return None
    created = data.get('created_at')
    try:
        created_at = datetime.fromisoformat(str(created)) if created else datetime.now(tz=timezone.utc)
    except (TypeError, ValueError):
        created_at = datetime.now(tz=timezone.utc)
    # Normalize naive timestamps (hand-edited / third-party JSON) to UTC so the
    # index sort never mixes naive and aware datetimes (TypeError on compare).
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    name = data.get('run_name') or data.get('description') or path.stem
    error = None
    if surface == 'redteam' and 'summary' not in data:
        error = 'missing required field: summary'
    elif surface == 'sim' and 'scorer_averages' not in data:
        error = 'missing required field: scorer_averages'
    headline = '' if error else f"{data.get('total_results', 0)} {'attacks' if surface == 'redteam' else 'conversations'}"
    return ReportCard(report_id(path), surface, str(name), created_at, headline, path, error)


def scan(roots: list[Path] | None = None) -> list[ReportCard]:
    roots = roots or _default_roots()
    cards = [c for p in _iter_report_files(roots) if (c := _card(p)) is not None]
    return sorted(cards, key=lambda c: c.created_at, reverse=True)


def resolve(rid: str, roots: list[Path] | None = None) -> Path | None:
    roots = roots or _default_roots()
    for p in _iter_report_files(roots):
        if report_id(p) == rid:
            return p
    logger.debug('report id not found after rescan: {}', rid)
    return None
