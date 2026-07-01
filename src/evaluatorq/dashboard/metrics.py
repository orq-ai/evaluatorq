"""Aggregate metrics for the combined dashboard landing + run lists.

The landing screen needs numbers across *both* run stores (red team + sim).
Rather than fully validating every report (slow, and the landing only needs a
handful of headline fields), these helpers read the cached raw JSON dicts and
pull the aggregate fields defensively.

``run_rows(roots)`` returns one ``RunRow`` per discovered report — id, kind,
name, score, derived status — used by the landing's recent-runs list and the
per-kind run-list screens.  ``landing(roots)`` rolls those up into the stat
band + panel data for the Dashboard screen.

Per the design decision: neither schema tracks dollar cost, so the design's
spend panels are rendered from native metrics (resistance rate, severity,
token usage) instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from evaluatorq.dashboard import library

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

# Severity buckets in display order (matches the report severity scale).
SEVERITY_ORDER = ('critical', 'high', 'medium', 'low')


@dataclass(frozen=True)
class RunRow:
    """One discovered run, with the fields the run lists + landing display."""

    id: str
    surface: str  # 'redteam' | 'sim'
    name: str
    when: str  # preformatted 'YYYY-MM-DD HH:MM'
    headline: str  # e.g. '128 attacks'
    score: float | None  # 0..1; redteam resistance, sim mean scorer avg
    status: str  # 'passed' | 'warning' | 'failed'
    error: bool


@dataclass(frozen=True)
class Landing:
    """Rolled-up data for the Dashboard landing screen."""

    total_runs: int
    redteam_runs: int
    sim_runs: int
    resistance_rate: float | None  # mean across red team runs (0..1)
    total_tokens: int
    by_kind: list[tuple[str, int]]  # [('Red team', n), ('Agent sim', n)]
    severity: list[tuple[str, int]]  # [('critical', n), ...] non-zero only
    tokens_by_kind: list[tuple[str, int]]  # [('Red team', n), ('Agent sim', n)]
    resistant: int  # attacks resisted (for the donut)
    vulnerable: int  # attacks that succeeded (for the donut)
    recent: list[RunRow] = field(default_factory=list)


def _as_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_int(v: object, default: int = 0) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _tokens_total(usage: object) -> int:
    """Pull a total token count from a TokenUsage-shaped dict, defensively."""
    if not isinstance(usage, dict):
        return 0
    for key in ('total_tokens', 'total'):
        if key in usage:
            return _as_int(usage[key])
    # Fall back to summing prompt + completion when no total is present.
    return _as_int(usage.get('prompt_tokens')) + _as_int(usage.get('completion_tokens'))


def _status_from_score(score: float | None, vulnerable: int | None = None) -> str:
    """Map a 0..1 score (and optional vulnerability count) to a status pill."""
    if vulnerable is not None and vulnerable == 0:
        return 'passed'
    if score is None:
        return 'warning'
    if score >= 0.8:
        return 'passed'
    if score >= 0.5:
        return 'warning'
    return 'failed'


def _redteam_row(card: library.ReportCard, data: dict[str, object]) -> RunRow:
    summary = data.get('summary')
    summary = summary if isinstance(summary, dict) else {}
    resistance = _as_float(summary.get('resistance_rate'), default=1.0) if summary else None
    vulns = _as_int(summary.get('vulnerabilities_found')) if summary else None
    status = _status_from_score(resistance, vulnerable=vulns)
    return RunRow(
        id=card.id,
        surface='redteam',
        name=card.name,
        when=card.created_at.strftime('%Y-%m-%d %H:%M'),
        headline=card.headline,
        score=resistance,
        status='failed' if card.error else status,
        error=bool(card.error),
    )


def _sim_row(card: library.ReportCard, data: dict[str, object]) -> RunRow:
    averages = data.get('scorer_averages')
    averages = averages if isinstance(averages, dict) else {}
    vals = [_as_float(v) for v in averages.values()] if averages else []
    score = sum(vals) / len(vals) if vals else None
    return RunRow(
        id=card.id,
        surface='sim',
        name=card.name,
        when=card.created_at.strftime('%Y-%m-%d %H:%M'),
        headline=card.headline,
        score=score,
        status='failed' if card.error else _status_from_score(score),
        error=bool(card.error),
    )


def run_rows(roots: list[Path] | None = None) -> list[RunRow]:
    """Return one RunRow per discovered report, newest-first."""
    rows: list[RunRow] = []
    for card in library.scan(roots):
        try:
            data = library.read_json_cached(card.path)
        except (OSError, ValueError):
            data = {}
        if card.surface == 'redteam':
            rows.append(_redteam_row(card, data))
        elif card.surface == 'sim':
            rows.append(_sim_row(card, data))
    return rows


def landing(roots: list[Path] | None = None) -> Landing:
    """Compute the Dashboard landing aggregates across both run stores."""
    rows = run_rows(roots)
    redteam = [r for r in rows if r.surface == 'redteam']
    sim = [r for r in rows if r.surface == 'sim']

    # Roll up severity counts + token usage + resistant/vulnerable from raw JSON.
    severity_counts: dict[str, int] = {}
    total_tokens = 0
    rt_tokens = 0
    sim_tokens = 0
    resistant = 0
    vulnerable = 0
    for card in library.scan(roots):
        try:
            data = library.read_json_cached(card.path)
        except (OSError, ValueError):
            continue
        if card.surface == 'redteam':
            summary = data.get('summary')
            if isinstance(summary, dict):
                resistant += _as_int(summary.get('evaluated_attacks')) - _as_int(summary.get('vulnerabilities_found'))
                vulnerable += _as_int(summary.get('vulnerabilities_found'))
                tok = _tokens_total(summary.get('token_usage_total'))
                rt_tokens += tok
                total_tokens += tok
                by_sev = summary.get('by_severity')
                if isinstance(by_sev, dict):
                    for sev, entry in by_sev.items():
                        if isinstance(entry, dict):
                            n = _as_int(entry.get('vulnerabilities_found') or entry.get('count'))
                            if n:
                                severity_counts[sev] = severity_counts.get(sev, 0) + n
        elif card.surface == 'sim':
            tok = sum(_as_int(_result_tokens(res)) for res in _results(data))
            sim_tokens += tok
            total_tokens += tok

    severity = [(sev, severity_counts[sev]) for sev in SEVERITY_ORDER if severity_counts.get(sev)]
    by_kind = [('Red team', len(redteam)), ('Agent sim', len(sim))]
    tokens_by_kind = [(k, n) for k, n in (('Red team', rt_tokens), ('Agent sim', sim_tokens)) if n]

    # Attack-weighted resistance, so the headline stat agrees with the donut.
    # The per-run mean is misleading: runs with zero evaluated attacks default
    # resistance_rate to 1.0 and inflate it.  ``None`` when nothing was evaluated.
    evaluated = resistant + vulnerable
    resistance_rate = resistant / evaluated if evaluated else None

    return Landing(
        total_runs=len(rows),
        redteam_runs=len(redteam),
        sim_runs=len(sim),
        resistance_rate=resistance_rate,
        total_tokens=total_tokens,
        by_kind=by_kind,
        severity=severity,
        tokens_by_kind=tokens_by_kind,
        resistant=max(resistant, 0),
        vulnerable=max(vulnerable, 0),
        recent=rows[:5],
    )


def _results(data: dict[str, object]) -> list[dict[str, object]]:
    results = data.get('results')
    return [r for r in results if isinstance(r, dict)] if isinstance(results, list) else []


def _result_tokens(res: dict[str, object]) -> int:
    return _as_int(res.get('total_tokens')) or _tokens_total(res.get('token_usage'))


# ---------------------------------------------------------------------------
# Agent Sim overview (item-level, one row per simulation) — RES-1022
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimItem:
    """One simulation, as the design's item-level 'Recent simulations' table."""

    rid: str  # report id of the run this simulation belongs to (row links here)
    scenario: str
    persona: str
    model: str
    turns: int
    outcome: str  # 'passed' | 'warning' | 'failed'
    error: bool
    score: float  # goal completion score, 0..1


@dataclass(frozen=True)
class SimOverview:
    """Rolled-up Agent Sim surface data: KPI cards + recent-simulations table.

    Per the design decision, dollar cost is not tracked; the 'Avg cost/sim'
    card is rendered as avg tokens/sim (see RES-1038 for the cost-model call).
    """

    simulations_run: int
    goal_completion: float | None  # share of sims that achieved the goal, 0..1
    avg_turns: float | None
    avg_tokens: float | None
    achieved: int  # outcomes donut segments
    not_achieved: int
    errors: int
    recent: list[SimItem] = field(default_factory=list)


def _sim_model(res: dict[str, object], meta: dict[str, object]) -> str:
    for key in ('model', 'target_model'):
        val = meta.get(key) or res.get(key)
        if isinstance(val, str) and val:
            return val
    return 'unknown'


def sim_overview(roots: list[Path] | None = None, *, limit: int = 8) -> SimOverview:
    """Aggregate every simulation across the sim run store into item-level rows
    plus headline KPIs. Rows are newest-run-first, capped at ``limit``."""
    dated: list[tuple[datetime, SimItem]] = []
    turns_total = 0
    tokens_total = 0
    achieved = not_achieved = errors = 0
    total = 0

    for card in library.scan(roots):
        if card.surface != 'sim':
            continue
        try:
            data = library.read_json_cached(card.path)
        except (OSError, ValueError):
            continue
        for res in _results(data):
            total += 1
            meta = res.get('metadata')
            meta = meta if isinstance(meta, dict) else {}
            is_error = str(res.get('terminated_by') or '') == 'error'
            goal = bool(res.get('goal_achieved'))
            score = _as_float(res.get('goal_completion_score'))
            turns = _as_int(res.get('turn_count'))
            turns_total += turns
            tokens_total += _result_tokens(res)
            if is_error:
                errors += 1
            elif goal:
                achieved += 1
            else:
                not_achieved += 1
            outcome = 'failed' if is_error else _status_from_score(score)
            dated.append((
                card.created_at,
                SimItem(
                    rid=card.id,
                    scenario=str(meta.get('scenario', 'unknown')),
                    persona=str(meta.get('persona', 'unknown')),
                    model=_sim_model(res, meta),
                    turns=turns,
                    outcome=outcome,
                    error=is_error,
                    score=score,
                ),
            ))

    dated.sort(key=lambda t: t[0], reverse=True)
    return SimOverview(
        simulations_run=total,
        goal_completion=(achieved / total) if total else None,
        avg_turns=(turns_total / total) if total else None,
        avg_tokens=(tokens_total / total) if total else None,
        achieved=achieved,
        not_achieved=not_achieved,
        errors=errors,
        recent=[item for _, item in dated[:limit]],
    )
