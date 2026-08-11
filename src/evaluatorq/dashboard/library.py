"""Discovery + identity + single-field kind sniff for the report dashboard."""

from __future__ import annotations

import base64
import functools
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from loguru import logger

from evaluatorq.common.run_manifest import MANIFESTS_DIR_NAME, iter_report_files, summary_is_complete

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from evaluatorq.contracts import RunManifest

_Model = TypeVar('_Model')


def report_id(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).digest()[:12]
    return base64.urlsafe_b64encode(digest).decode().rstrip('=')


@functools.lru_cache(maxsize=64)
def read_json(path_str: str, mtime_ns: int) -> dict[str, object]:
    """Parse and cache the JSON at *path_str*.

    The cache key includes *mtime_ns* (nanosecond modification time) so that
    any on-disk change automatically produces a cache miss — stale data is
    never served.  The *mtime_ns* argument is derived by the caller from
    ``Path.stat().st_mtime_ns`` and must not be fabricated.

    Args:
        path_str:  Absolute path string (``str(path.resolve())``).
        mtime_ns:  Nanosecond mtime of the file at the time of the call.

    Returns:
        Parsed JSON object as a ``dict``.

    Raises:
        json.JSONDecodeError: When the file content is not valid JSON.
        OSError: When the file cannot be read.
    """
    return json.loads(Path(path_str).read_text())  # type: ignore[return-value]


def read_json_cached(path: Path) -> dict[str, object]:
    """Read + parse JSON at *path*, using the mtime-keyed LRU cache."""
    mtime_ns = path.stat().st_mtime_ns
    return read_json(str(path.resolve()), mtime_ns)


@functools.lru_cache(maxsize=32)
def _validate_model(
    path_str: str,
    mtime_ns: int,
    validator: Callable[[dict[str, object]], _Model],
) -> _Model:
    """Cache a Pydantic-validated report model, mtime-keyed like ``read_json``.

    Raw-JSON caching alone still re-runs the (expensive) Pydantic validation of
    a whole run on every filter/sort/page request. Caching the validated object
    skips that — the payoff on large sim runs. ``validator`` is the model's
    ``model_validate`` classmethod (part of the key so two model types on the
    same path don't collide). Callers MUST treat the returned object as
    read-only; filtering builds new lists and never mutates it.
    """
    return validator(read_json(path_str, mtime_ns))


def load_model_cached(path: Path, validator: Callable[[dict[str, object]], _Model]) -> _Model:
    """Return a cached, validated report model for *path* (see ``_validate_model``)."""
    mtime_ns = path.stat().st_mtime_ns
    return _validate_model(str(path.resolve()), mtime_ns, validator)


def sniff_kind(data: dict[str, object]) -> str | None:
    """Surface from a single required-unique field. sim ('mode') checked first."""
    if 'mode' in data:
        return 'sim'
    if 'pipeline' in data:
        return 'redteam'
    if 'judging' in data:
        return 'pairwise'
    return None


def load_surface(path: Path) -> tuple[str | None, dict[str, object]]:
    """Lenient surface sniff for directory scans: a corrupt/unreadable file is masked
    as an unknown surface (``None``) so one bad file can't break a whole listing."""
    try:
        data = read_json_cached(path)
    except (json.JSONDecodeError, OSError):
        return None, {}
    return sniff_kind(data), data


def load_surface_strict(path: Path) -> tuple[str | None, dict[str, object]]:
    """Like :func:`load_surface`, but lets a corrupt/unreadable file raise instead of
    masking it as an unknown surface. Callers resolving a *specific requested* report
    (not scanning a directory) use this so a syntactically corrupt file is reported as
    corrupt rather than "not found".

    Raises:
        json.JSONDecodeError: the file content is not valid JSON.
        OSError: the file cannot be read.
    """
    data = read_json_cached(path)
    return sniff_kind(data), data


@dataclass(frozen=True)
class ReportCard:
    id: str
    surface: str
    name: str
    created_at: datetime
    headline: str
    # ``None`` for an in-flight (running/error/cancelled) run that has no report
    # on disk yet — such a card renders from the manifest's status/stage alone.
    path: Path | None
    error: str | None = None
    # Lifecycle status/stage from the run manifest. ``None`` for legacy
    # report-only cards (no manifest) so the view keeps its existing behavior.
    status: str | None = None
    stage: str | None = None


def default_roots() -> list[Path]:
    from evaluatorq.pairwise_run import get_pairwise_runs_dir
    from evaluatorq.redteam.runner import get_runs_dir
    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    return [get_runs_dir(), get_sim_runs_dir(), get_pairwise_runs_dir()]


def _iter_report_files(roots: list[Path]) -> Iterator[Path]:
    """Report files across *roots* — the shared per-dir predicate, flattened.

    Yields:
        Each report file, root by root, sorted within a root.
    """
    for root in roots:
        yield from iter_report_files(root)


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
    elif surface == 'pairwise' and 'entries' not in data:
        error = 'missing required field: entries'
    if error:
        headline = ''
    elif surface == 'pairwise':
        entries = data.get('entries')
        count = len(entries) if isinstance(entries, list) else 0
        headline = f'{count} comparisons'
    else:
        headline = f'{data.get("total_results", 0)} {"attacks" if surface == "redteam" else "conversations"}'
    return ReportCard(report_id(path), surface, str(name), created_at, headline, path, error)


def _manifest_card_id(run_id: str) -> str:
    """Stable, URL-safe id for a manifest-backed card that has no report yet.

    Namespaced (``manifest:``) so it never collides with a ``report_id`` (which
    hashes a file path). Used only for in-flight cards; once a run completes and
    gains a ``report_path`` the card switches to ``report_id(path)`` so its
    identity matches the report and detail rendering is unchanged.
    """
    digest = hashlib.sha256(f'manifest:{run_id}'.encode()).digest()[:12]
    return base64.urlsafe_b64encode(digest).decode().rstrip('=')


def _card_from_manifest(m: RunManifest) -> ReportCard:
    """Build a list-row card from a ``RunManifest`` without reading its report.

    The headline for a completed run comes from the manifest's compact
    ``summary`` (never the report). In-flight runs (running/error/cancelled) have
    no summary, so the headline reflects their status/stage instead.
    """
    surface = m.surface.value
    created_at = m.started_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    path = Path(m.report_path) if m.report_path else None
    summary = m.summary or {}
    status = m.status.value
    if status == 'completed' and summary:
        noun = 'attacks' if surface == 'redteam' else 'conversations'
        headline = f'{summary.get("total_results", 0)} {noun}'
    elif status == 'running':
        headline = f'running · {m.stage}' if m.stage else 'running'
    else:  # error / cancelled with no report
        headline = status
    rid = report_id(path) if path is not None else _manifest_card_id(m.run_id)
    return ReportCard(
        rid,
        surface,
        m.run_name,
        created_at,
        headline,
        path,
        error=None,
        status=status,
        stage=m.stage,
    )


def fingerprint(roots: list[Path] | None = None) -> tuple[int, int]:
    """``(file count, newest mtime_ns)`` across every run store — a cheap staleness key.

    Callers cache expensive whole-store aggregates against this: one stat sweep
    instead of re-reading every report. Reports are write-once, so a new run
    always bumps the count; manifests are *not* (an in-flight run rewrites its
    own on every stage), so ``.manifests/`` is swept too and the newest mtime
    catches that stage advance.

    The report sweep goes through :func:`_iter_report_files`, the same predicate
    the aggregate itself reads, so stage artifacts (``01_``/``02_``/``03_``) that
    no aggregate looks at can't invalidate the cache when they are rewritten.

    Unreadable entries are skipped rather than raised on: a fingerprint that
    fails is a dashboard that fails, and a missed file only costs staleness
    until the next change.
    """
    roots = roots or default_roots()
    count = 0
    newest = 0
    for root in roots:
        for p in (*_iter_report_files([root]), *(root / MANIFESTS_DIR_NAME).glob('*.json')):
            count += 1
            try:
                newest = max(newest, p.stat().st_mtime_ns)
            except OSError:  # a stat failure must not sink the whole sweep
                continue
    return (count, newest)


def _backfill_manifest(path: Path, card: ReportCard) -> None:
    """Write a manifest sidecar for a legacy (manifest-less) report.

    Migration-on-read: the first scan pays the full-report read it already pays
    today, then leaves a tiny ``.manifests/`` sidecar behind so every later scan
    (and ``eq runs``) builds this row from the manifest alone. No separate
    migration command to run — the runs dir heals itself as it is browsed.

    The summary comes from the report model's own ``manifest_summary()`` — the
    same builder the live runners use — so a backfilled row is field-identical
    to one written by the run itself.

    Best-effort in every direction: a complete existing sidecar is never
    overwritten (a thin one from an earlier version is), an errored/pairwise
    report is skipped (pairwise has no ``ManifestSurface``), and a failed read or
    write only costs the next scan another full read.
    """
    from evaluatorq.common.run_manifest import ManifestWriter
    from evaluatorq.contracts import ManifestStatus, ManifestSurface, RunManifest
    from evaluatorq.dashboard.surfaces import ADAPTERS

    if card.error or card.surface not in (ManifestSurface.SIM, ManifestSurface.REDTEAM):
        return
    mpath = path.parent / MANIFESTS_DIR_NAME / f'{path.stem}.json'
    if mpath.exists():
        try:
            existing = read_json_cached(mpath)
        except (json.JSONDecodeError, OSError):
            existing = {}
        raw = existing.get('summary')
        if summary_is_complete(raw if isinstance(raw, dict) else None):
            return
    try:
        # Full model validate (mtime-keyed LRU): the price of one migration read.
        summary = ADAPTERS[card.surface].load(path).manifest_summary()
    except Exception as exc:
        # The card already parsed, so a model failure here is a schema mismatch,
        # not routine noise: warn (as list_manifests does) rather than migrate
        # silently-never. The run still lists via the full-report path.
        logger.warning(f'Skipping manifest backfill for {path}: {exc}')
        return
    manifest = RunManifest(
        run_id=path.stem,
        surface=ManifestSurface(card.surface),
        run_name=card.name,
        status=ManifestStatus.COMPLETED,
        started_at=card.created_at,
        updated_at=card.created_at,
        ended_at=card.created_at,
        report_path=str(path),
        summary=summary,
    )
    ManifestWriter(manifest, mpath).flush()


def scan(roots: list[Path] | None = None) -> list[ReportCard]:
    """Discover run cards, manifest-first with a legacy full-report fallback.

    Each ``.manifests/*.json`` sidecar becomes a card built without reading the
    full report (completed runs use their compact ``summary``; in-flight runs use
    status/stage). Report files already covered by a manifest's ``report_path``
    are de-duplicated out; the remaining (legacy, manifest-less) reports fall back
    to the full-report ``_card`` path. A runs dir with only legacy reports lists
    exactly as before.

    Roots are resolved and de-duplicated (order-preserving) before scanning:
    with multiple CLI paths the same directory can arrive twice — a repeated
    argument, relative + absolute forms of one repo, or an argument that sits
    inside another's expansion — and a duplicated root would double-count every
    report in the landing rollups (jobs, spend, costed runs).
    """
    from evaluatorq.common.run_manifest import list_run_records

    roots = roots or default_roots()
    seen_roots: set[Path] = set()
    deduped_roots: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen_roots:
            seen_roots.add(resolved)
            deduped_roots.append(root)
    cards: list[ReportCard] = []
    for root in deduped_roots:
        # list_run_records owns manifest-first ordering, report de-duplication and
        # the demotion of thin-summary sidecars to a legacy row — the CLI run
        # tables read the same function, so the two views can't drift apart.
        for manifest, path in list_run_records(root):
            if manifest is not None:
                cards.append(_card_from_manifest(manifest))
            elif path is not None and (c := _card(path)) is not None:
                cards.append(c)
                _backfill_manifest(path, c)
    return sorted(cards, key=lambda c: c.created_at, reverse=True)


def resolve(rid: str, roots: list[Path] | None = None) -> Path | None:
    roots = roots or default_roots()
    for p in _iter_report_files(roots):
        if report_id(p) == rid:
            return p
    logger.debug('report id not found after rescan: {}', rid)
    return None


def resolve_manifest(rid: str, roots: list[Path] | None = None) -> RunManifest | None:
    """Resolve a report-less card id back to its ``RunManifest``.

    Matches any manifest with no ``report_path``. That is usually an in-flight
    run (running/error/cancelled), but a *completed* run can also lack a
    ``report_path`` when its report save failed — such a run has no openable
    report, so it resolves here (not via :func:`resolve`) and the detail route
    renders its status page (status 'completed', no transcript). Returns the
    ``RunManifest`` so the detail route can render a minimal status page.
    """
    from evaluatorq.common.run_manifest import list_manifests

    roots = roots or default_roots()
    for root in roots:
        for m in list_manifests(root):
            if m.report_path is None and _manifest_card_id(m.run_id) == rid:
                return m
    return None
