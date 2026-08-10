"""Aggregate metrics for the combined dashboard landing + run lists.

The landing screen needs numbers across all run stores (red team + sim + pairwise).
Rather than fully validating every report (slow, and the landing only needs a
handful of headline fields), these helpers read the cached raw JSON dicts and
pull the aggregate fields defensively.

``run_rows(roots)`` returns one ``RunRow`` per discovered report — id, kind,
name, score, derived status — used by the landing's recent-runs list and the
per-kind run-list screens.  ``landing(roots)`` rolls those up into the stat
band + panel data for the Dashboard screen.

Dollar cost is optional in both schemas: a report only carries cost when the
provider reported it.  The spend panels therefore distinguish "no cost data"
from "$0.00" — see ``_cost_usd`` — and fall back to native metrics (resistance
rate, severity, token usage) when nothing in a store reported cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

from evaluatorq.dashboard import library

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

# Severity buckets in display order (matches the report severity scale).
SEVERITY_ORDER = ('critical', 'high', 'medium', 'low')

# Bucket for a vulnerability whose report does not record a severity. ``severity``
# is a required field, so this only shows up in a truncated or hand-edited
# report — but defaulting those into ``low`` would understate real findings, and
# dropping them would leave the severity bars summing to less than the donut.
# Named and rendered last instead.
UNKNOWN_SEVERITY = 'unknown'


@dataclass(frozen=True)
class RunRow:
    """One discovered run, with the fields the run lists + landing display."""

    id: str
    surface: str  # 'redteam' | 'sim' | 'pairwise'
    name: str
    when: str  # preformatted 'YYYY-MM-DD HH:MM'
    headline: str  # e.g. '128 attacks'
    score: float | None  # 0..1; redteam resistance, sim mean scorer avg
    status: str  # 'passed' | 'warning' | 'failed'
    error: bool
    evaluated: int = 0  # attacks that produced a verdict (red team only)
    attacks: int = 0  # attacks attempted, verdict or not (red team only)
    # The rate recorded in the report itself, set only when ``score`` was
    # re-derived and lands somewhere else. Drives the "recalculated" marker: the
    # dashboard shows its own number but never silently overwrites the record.
    stored_score: float | None = None


@dataclass(frozen=True)
class Landing:
    """Rolled-up data for the Dashboard landing screen."""

    total_runs: int
    redteam_runs: int
    sim_runs: int
    resistance_rate: float | None  # mean across red team runs (0..1)
    total_tokens: int
    by_kind: list[tuple[str, int]]  # [('Red team', n), ('Agent sim', n), ('Pairwise', n)]
    severity: list[tuple[str, int]]  # [('critical', n), ...] non-zero only
    tokens_by_kind: list[tuple[str, int]]  # [('Red team', n), ('Agent sim', n), ('Pairwise', n)]
    resistant: int  # attacks resisted (for the donut)
    vulnerable: int  # attacks that succeeded (for the donut)
    total_cost: float | None = None  # summed cost_usd across all stores; None when nothing records a cost
    costed_runs: int = 0  # runs that actually record a cost — avg cost divides by this
    cost_by_kind: list[tuple[str, float]] = field(default_factory=list)  # non-zero only
    total_input_cost: float | None = None  # summed input_cost across all stores; None when unrecorded
    total_output_cost: float | None = None  # summed output_cost across all stores; None when unrecorded
    recent: list[RunRow] = field(default_factory=list)


def _as_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float_or_none(v: object) -> float | None:
    """Like ``_as_float`` but preserves "unknown" as ``None`` instead of
    collapsing it to a default. Used for cost fields, where ``None`` (not
    recorded) and ``0.0`` (recorded as free) are semantically different and
    must never be conflated."""
    if v is None:
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


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


def _cost_usd(usage: object) -> float | None:
    """Pull the real dollar cost from a TokenUsage-shaped dict.

    Both surfaces already record ``cost_usd`` upstream (sim per-result
    ``token_usage``; red team ``summary.token_usage_total``), so no price model
    is needed — the dashboard just surfaces it.

    Returns ``None`` when the cost is not recorded (missing key, or the key
    present but ``null`` — the ``@model_serializer`` in ``contracts.Usage``
    always emits the ``cost_usd`` key, even when ``total_cost`` is ``None``, so
    a bare ``'cost_usd' in usage`` membership check is NOT sufficient to detect
    "known cost"). Callers must not treat ``None`` as ``0.0``: an unknown cost
    must not be summed into a spend total or counted as a "costed" run.
    """
    if not isinstance(usage, dict):
        return None
    return _as_float_or_none(usage.get('cost_usd'))


def _input_cost(usage: object) -> float | None:
    """Sibling reader for the ``input_cost`` component. Same ``None`` semantics
    as :func:`_cost_usd`."""
    if not isinstance(usage, dict):
        return None
    return _as_float_or_none(usage.get('input_cost'))


def _output_cost(usage: object) -> float | None:
    """Sibling reader for the ``output_cost`` component. Same ``None`` semantics
    as :func:`_cost_usd`."""
    if not isinstance(usage, dict):
        return None
    return _as_float_or_none(usage.get('output_cost'))


def zero_evaluated_attacks(summary: dict[str, object]) -> bool:
    """True when the report explicitly says zero attacks were evaluated.

    The schema default ``resistance_rate`` (1.0) then carries no signal and
    must never render as a perfect score. An absent field (legacy reports
    predating it) is False — those keep their recorded rate.
    """
    return summary.get('evaluated_attacks') is not None and _as_int(summary.get('evaluated_attacks')) == 0


def _lifecycle_status(*, broken: bool, all_errored: bool = False) -> str:
    """Run lifecycle for the Status column: 'error' when the report is broken or
    every case errored, else 'finished'. ('running' isn't derivable — the run
    store only receives a file once a job completes.)"""
    return 'error' if broken or all_errored else 'finished'


def _redteam_row(card: library.ReportCard, data: dict[str, object]) -> RunRow:
    summary = data.get('summary')
    summary = summary if isinstance(summary, dict) else {}
    resistance = _as_float(summary.get('resistance_rate'), default=1.0) if summary else None
    evaluated = _as_int(summary.get('evaluated_attacks')) if summary else 0
    errors = _as_int(summary.get('total_errors')) if summary else 0
    total = _as_int(summary.get('total_attacks')) if summary else 0
    # A run that evaluated nothing has no resistance to report (see
    # zero_evaluated_attacks).
    if zero_evaluated_attacks(summary):
        resistance = None
    stored_score: float | None = None
    attacks = total
    if summary.get('evaluated_attacks') is None:
        # Legacy summary: its stored rate may have been computed over a
        # different denominator than the evaluated-only one the landing donut
        # derives. Use the same classifier, so the row and the donut above it
        # cannot show two different numbers for one run.
        counts = _redteam_counts(data)
        evaluated, attacks = counts.evaluated, counts.attacks
        if counts.evaluated:
            derived = (counts.evaluated - counts.vulnerable) / counts.evaluated
            # Keep the recorded rate visible whenever the derivation moves it,
            # so the row can be reconciled against the run's own exported
            # report instead of quietly contradicting it.
            if resistance is None or abs(derived - resistance) > 0.005:
                stored_score = resistance
            resistance = derived
        # No derivable verdicts: keep whatever rate the report recorded rather
        # than blanking a run that has one. Deriving nothing is not the same as
        # measuring zero.
    return RunRow(
        id=card.id,
        surface='redteam',
        name=card.name,
        when=card.created_at.strftime('%Y-%m-%d %H:%M'),
        headline=card.headline,
        score=resistance,
        status=_lifecycle_status(
            broken=bool(card.error), all_errored=(total > 0 and evaluated == 0 and errors >= total)
        ),
        error=bool(card.error),
        evaluated=evaluated,
        attacks=attacks,
        stored_score=stored_score,
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
        status=_lifecycle_status(broken=bool(card.error)),
        error=bool(card.error),
    )


def _pairwise_row(card: library.ReportCard, data: dict[str, object]) -> RunRow:
    """Score a pairwise run by the share of comparisons the panel decided.

    The win rates say which side led, not whether the run was any good, and
    neither side is "the score". ``mean_agreement`` reads like the natural
    quality number but does not survive the shared ``>= 0.8`` threshold this
    column is rendered against: it is a modal vote share, so with three judges
    it can only be 0.67 or 1.0 and the threshold silently means "unanimous",
    while with five judges 0.8 means "four of five". It is also undefined for a
    single-judge run. The decided share is continuous, independent of panel
    size, defined everywhere, and is the health signal the docs already pair
    with the win rates.
    """
    report = data.get('report')
    report = report if isinstance(report, dict) else {}
    inconclusive = report.get('inconclusive_rate')
    decided = 1.0 - _as_float(inconclusive) if inconclusive is not None else None
    return RunRow(
        id=card.id,
        surface='pairwise',
        name=card.name,
        when=card.created_at.strftime('%Y-%m-%d %H:%M'),
        headline=card.headline,
        score=decided,
        status=_lifecycle_status(broken=bool(card.error)),
        error=bool(card.error),
    )


def _card_status_row(card: library.ReportCard) -> RunRow:
    """Build a RunRow straight from a manifest-backed card (no report read).

    Used for runs that have no openable report: in-flight runs (running/error/
    cancelled) and a completed run whose report save failed. The lifecycle
    status comes from the manifest status, NOT from ``path is None`` — only
    ``cancelled`` and ``error`` are surfaced as non-completed, and only
    ``error`` counts as an error (a cancelled run is a clean terminal state,
    and a completed-but-unsaved run reads as finished).
    """
    status = card.status or 'error'
    lifecycle = {'completed': 'finished'}.get(status, status)
    return RunRow(
        id=card.id,
        surface=card.surface,
        name=card.name,
        when=card.created_at.strftime('%Y-%m-%d %H:%M'),
        headline=card.headline,
        score=None,
        status=lifecycle,
        error=status == 'error',
    )


def run_rows(roots: list[Path] | None = None) -> list[RunRow]:
    """Return one RunRow per discovered run, newest-first (manifest-first)."""
    rows: list[RunRow] = []
    for card in library.scan(roots):
        if card.surface not in ('redteam', 'sim', 'pairwise'):
            continue
        # Derive list state from the manifest status, not ``path is None``: a
        # non-completed run (running/error/cancelled) — and a completed run whose
        # report save failed (status 'completed', no path) — render straight from
        # the card without a report read. ``path is None`` means only "no openable
        # report", never "in-flight". Legacy cards (status=None, always with a
        # path) fall through to the report-read path below.
        if card.status in ('running', 'error', 'cancelled') or card.path is None:
            rows.append(_card_status_row(card))
            continue
        try:
            data = library.read_json_cached(card.path)
        except (OSError, ValueError):
            data = {}
        if card.surface == 'redteam':
            rows.append(_redteam_row(card, data))
        elif card.surface == 'sim':
            rows.append(_sim_row(card, data))
        elif card.surface == 'pairwise':
            rows.append(_pairwise_row(card, data))
    return rows


def landing(roots: list[Path] | None = None) -> Landing:
    """Compute the Dashboard landing aggregates across all run stores."""
    rows = run_rows(roots)
    redteam = [r for r in rows if r.surface == 'redteam']
    sim = [r for r in rows if r.surface == 'sim']
    pairwise = [r for r in rows if r.surface == 'pairwise']

    # Roll up severity counts + token usage + resistant/vulnerable from raw JSON.
    severity_counts: dict[str, int] = {}
    total_tokens = 0
    rt_tokens = 0
    sim_tokens = 0
    rt_cost = 0.0
    sim_cost = 0.0
    costed_runs = 0
    pw_tokens = 0
    pw_cost = 0.0
    input_cost_total = 0.0
    output_cost_total = 0.0
    has_input_cost = False
    has_output_cost = False
    resistant = 0
    vulnerable = 0
    for card in library.scan(roots):
        # In-flight runs have no report on disk — nothing to roll up.
        if card.path is None:
            continue
        try:
            data = library.read_json_cached(card.path)
        except (OSError, ValueError):
            continue
        if card.surface == 'redteam':
            summary = data.get('summary')
            summary = summary if isinstance(summary, dict) else {}
            if summary.get('evaluated_attacks') is None:
                # Legacy report: its summary predates ``evaluated_attacks``, and
                # with it ``by_severity`` and ``token_usage_total``. Every
                # aggregate below is therefore derived from the results list —
                # otherwise the run adds zero to the donut, the severity bars and
                # the cost totals while its own row still shows a rate (RES-1202).
                counts = _redteam_counts(data)
                logger.debug(
                    'landing: derived legacy red-team counts for {} — evaluated={} vulnerable={} tokens={}',
                    card.id,
                    counts.evaluated,
                    counts.vulnerable,
                    counts.tokens,
                )
                resistant += counts.evaluated - counts.vulnerable
                vulnerable += counts.vulnerable
                rt_tokens += counts.tokens
                total_tokens += counts.tokens
                # Derived cost is a known-$0-or-more sum, but a run whose results
                # carried no readable cost must not count as "costed" — mirror
                # the None semantics of _cost_usd for stored summaries.
                if counts.cost:
                    rt_cost += counts.cost
                    costed_runs += 1
                # The input/output split is not recoverable from legacy results.
                by_sev = counts.by_severity
            else:
                resistant += _as_int(summary.get('evaluated_attacks')) - _as_int(summary.get('vulnerabilities_found'))
                vulnerable += _as_int(summary.get('vulnerabilities_found'))
                usage = summary.get('token_usage_total')
                tok = _tokens_total(usage)
                rt_tokens += tok
                total_tokens += tok
                run_cost = _cost_usd(usage)
                if run_cost is not None:
                    rt_cost += run_cost
                    costed_runs += 1
                run_input = _input_cost(usage)
                if run_input is not None:
                    input_cost_total += run_input
                    has_input_cost = True
                run_output = _output_cost(usage)
                if run_output is not None:
                    output_cost_total += run_output
                    has_output_cost = True
                by_sev = _summary_severity(summary)
            for sev, n in by_sev.items():
                severity_counts[sev] = severity_counts.get(sev, 0) + n
        elif card.surface == 'sim':
            results = _results(data)
            tok = sum(_as_int(_result_tokens(res)) for res in results)
            sim_tokens += tok
            total_tokens += tok
            usages = [res.get('token_usage') for res in results]
            costs = [c for c in (_cost_usd(u) for u in usages) if c is not None]
            if costs:
                sim_cost += sum(costs)
                costed_runs += 1
            inputs = [c for c in (_input_cost(u) for u in usages) if c is not None]
            if inputs:
                input_cost_total += sum(inputs)
                has_input_cost = True
            outputs = [c for c in (_output_cost(u) for u in usages) if c is not None]
            if outputs:
                output_cost_total += sum(outputs)
                has_output_cost = True
        elif card.surface == 'pairwise':
            usages = [_comparison_usage(entry) for entry in _entries(data)]
            tok = sum(_tokens_total(u) for u in usages)
            pw_tokens += tok
            total_tokens += tok
            costs = [c for c in (_cost_usd(u) for u in usages) if c is not None]
            if costs:
                pw_cost += sum(costs)
                costed_runs += 1
            inputs = [c for c in (_input_cost(u) for u in usages) if c is not None]
            if inputs:
                input_cost_total += sum(inputs)
                has_input_cost = True
            outputs = [c for c in (_output_cost(u) for u in usages) if c is not None]
            if outputs:
                output_cost_total += sum(outputs)
                has_output_cost = True

    # Unknown sorts last, after the real scale, and only appears when non-zero.
    severity = [(sev, severity_counts[sev]) for sev in (*SEVERITY_ORDER, UNKNOWN_SEVERITY) if severity_counts.get(sev)]
    # Zero-count kinds are dropped, matching tokens_by_kind / cost_by_kind: a
    # workspace with no pairwise runs should not carry an empty 'Pairwise' bar.
    by_kind = [
        (k, n) for k, n in (('Red team', len(redteam)), ('Agent sim', len(sim)), ('Pairwise', len(pairwise))) if n
    ]
    tokens_by_kind = [
        (k, n) for k, n in (('Red team', rt_tokens), ('Agent sim', sim_tokens), ('Pairwise', pw_tokens)) if n
    ]
    cost_by_kind = [(k, c) for k, c in (('Red team', rt_cost), ('Agent sim', sim_cost), ('Pairwise', pw_cost)) if c > 0]

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
        total_cost=(rt_cost + sim_cost + pw_cost) if costed_runs else None,
        costed_runs=costed_runs,
        cost_by_kind=cost_by_kind,
        total_input_cost=input_cost_total if has_input_cost else None,
        total_output_cost=output_cost_total if has_output_cost else None,
        recent=rows[:5],
    )


class _RedteamCounts(NamedTuple):
    """Per-report red-team tallies derived by walking the ``results`` list."""

    attacks: int
    evaluated: int
    vulnerable: int
    errors: int
    tokens: int
    cost: float
    by_severity: dict[str, int]


def _result_outcome(res: dict[str, object]) -> bool | None:
    """``True`` resisted, ``False`` vulnerable, ``None`` never evaluated.

    Mirrors ``_is_evaluated`` / ``_is_vulnerable`` in
    ``redteam.reports.converters``: only a boolean ``evaluation.passed`` counts
    as an evaluation. The top-level ``error`` field carries the *target*
    error only, so a result whose generation succeeded but whose evaluation
    crashed or was skipped has ``error=None, vulnerable=False`` and must not be
    booked as resistant — that would bias a security metric optimistic.
    Reports too old to serialize ``evaluation`` at all fall back to the flags
    they do carry.
    """
    evaluation = res.get('evaluation')
    if isinstance(evaluation, dict):
        passed = evaluation.get('passed')
        return passed if isinstance(passed, bool) else None
    if 'evaluation' in res or res.get('error'):
        return None
    # Oldest schema: no evaluation object at all. A present `vulnerable` flag is
    # the verdict; an absent one means "never evaluated", not "resisted" — the
    # same optimistic bias guarded against above, one schema generation older.
    if 'vulnerable' in res:
        return not res.get('vulnerable')
    return None


def _result_usage(res: dict[str, object]) -> tuple[int, float]:
    """``(tokens, cost)`` for one result, over both the target and judge calls.

    Same two holders ``converters._aggregate_token_usage`` walks.
    """
    tokens = 0
    cost = 0.0
    for holder in ('execution', 'evaluation'):
        parent = res.get(holder)
        usage = parent.get('token_usage') if isinstance(parent, dict) else None
        tokens += _tokens_total(usage)
        holder_cost = _cost_usd(usage)
        if holder_cost is not None:
            cost += holder_cost
    return tokens, cost


def _redteam_counts(data: dict[str, object]) -> _RedteamCounts:
    """Tally a red-team report straight from its ``results`` list.

    The single classifier behind both the red-team overview and the landing
    aggregate's legacy fallback, so the two can never drift apart. A report
    with no results tallies to zeros — there is genuinely nothing to weight in
    an attack-weighted aggregate.
    """
    attacks = evaluated = vulnerable = errors = tokens = 0
    cost = 0.0
    by_severity: dict[str, int] = {}
    for res in _results(data):
        attacks += 1
        if res.get('error'):
            errors += 1
        tok, c = _result_usage(res)
        tokens += tok
        cost += c
        outcome = _result_outcome(res)
        if outcome is None:
            continue
        evaluated += 1
        if not outcome:
            vulnerable += 1
            attack = res.get('attack')
            attack = attack if isinstance(attack, dict) else {}
            severity = str(attack.get('severity') or UNKNOWN_SEVERITY).lower()
            by_severity[severity] = by_severity.get(severity, 0) + 1
    if attacks and not tokens:
        # Distinguishes "shape we couldn't read" from a genuinely costless run —
        # _result_usage returns zeros for any unrecognised usage schema.
        logger.debug('redteam counts: {} results but zero derivable token usage — unrecognised usage shape?', attacks)
    return _RedteamCounts(attacks, evaluated, vulnerable, errors, tokens, cost, by_severity)


def _summary_severity(summary: dict[str, object]) -> dict[str, int]:
    """Vulnerability count per severity as stored in a modern ``by_severity``."""
    by_sev = summary.get('by_severity')
    if not isinstance(by_sev, dict):
        return {}
    out: dict[str, int] = {}
    for sev, entry in by_sev.items():
        if isinstance(entry, dict):
            n = _as_int(entry.get('vulnerabilities_found') or entry.get('count'))
            if n:
                out[sev] = out.get(sev, 0) + n
    return out


def _entries(data: dict[str, object]) -> list[dict[str, object]]:
    entries = data.get('entries')
    return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


def _comparison_usage(entry: dict[str, object]) -> object:
    comparison = entry.get('comparison')
    return comparison.get('token_usage') if isinstance(comparison, dict) else None


def _results(data: dict[str, object]) -> list[dict[str, object]]:
    results = data.get('results')
    return [r for r in results if isinstance(r, dict)] if isinstance(results, list) else []


def _result_tokens(res: dict[str, object]) -> int:
    return _as_int(res.get('total_tokens')) or _tokens_total(res.get('token_usage'))


# ---------------------------------------------------------------------------
# Agent Sim overview (item-level, one row per simulation) — RES-1022
# ---------------------------------------------------------------------------


_TARGET_LABELS: dict[str, str] = {
    'orq_agent': 'Orq agent',
    'orq_deployment': 'Orq deployment',
    'openai_model': 'OpenAI model',
    'vercel': 'Vercel',
    'callback': 'Callback',
}

# Target kind → icon category the view renders ('agent' | 'llm' | 'deployment' | 'other').
_TARGET_KIND_CATEGORY: dict[str, str] = {
    'orq_agent': 'agent',
    'orq_deployment': 'deployment',
    'vercel': 'deployment',
    'openai_model': 'llm',
    'callback': 'other',
}


def _split_target(raw: str) -> tuple[str, str] | None:
    """Parse a prefixed target string into (label, kind).

    Targets are recorded as ``agent:<key>`` / ``deployment:<key>``; a bare
    ``agent`` / ``deployment`` / ``callback`` carries a kind but no concrete
    name.  Returns ``None`` when *raw* is empty.
    """
    raw = raw.strip()
    if not raw:
        return None
    prefix, _, rest = raw.partition(':')
    kind = {'agent': 'agent', 'deployment': 'deployment', 'callback': 'other'}.get(prefix, '')
    if kind:
        return (rest or _TARGET_LABELS.get(f'orq_{prefix}', prefix), kind)
    return (raw, '')  # no known prefix → treat the whole string as a name


def _sim_target(data: dict[str, object]) -> tuple[str, str]:
    """Best available (label, icon-kind) for a sim run.

    Prefers a concrete identifier when the report carries one — an explicit
    ``target``/``model`` field, or a per-result ``metadata.target`` like
    ``agent:<key>`` (``agent:``/``deployment:`` prefixes name the kind) — and
    falls back to a branded ``target_kind`` label.  Old reports store only
    ``target_kind`` + ``run_name``, so they show the kind label.
    """
    kind_cat = _TARGET_KIND_CATEGORY.get(str(data.get('target_kind') or ''), 'other')
    # Preferred: the persisted target name (+ model when the client knew it).
    target = data.get('target')
    target_model = data.get('target_model')
    if isinstance(target, str) and target:
        name, kind = _split_target(target) or (target, '')
        if isinstance(target_model, str) and target_model and target_model != name:
            name = f'{name} · {target_model}'
        return (name, kind or kind_cat)
    # Legacy: a bare ``model`` field (older openai-model runs) → the model id.
    model = data.get('model')
    if isinstance(model, str) and model:
        return (model, 'llm')
    for res in _results(data):
        meta = res.get('metadata')
        tgt = meta.get('target') if isinstance(meta, dict) else None
        if isinstance(tgt, str) and (parsed := _split_target(tgt)):
            return (parsed[0], parsed[1] or kind_cat)
    # No concrete target recorded (old reports). Derive an agent-ish name from
    # run_name rather than the generic 'Orq agent' label: 'sim:<name>:<model>'
    # → '<name>'.  Real fix is persisting the target on SimulationRun upstream.
    run_name = str(data.get('run_name') or '')
    if run_name:
        name = run_name.split(':', 2)[1] if run_name.startswith('sim:') and ':' in run_name[4:] else run_name
        return (name or run_name, kind_cat)
    kind = str(data.get('target_kind') or '')
    return (_TARGET_LABELS.get(kind, kind or 'unknown'), kind_cat)


@dataclass(frozen=True)
class SimRunRow:
    """One full Agent Sim run, as the design's run-level table row."""

    rid: str  # report id (row links here)
    name: str  # run_name / job name
    when: datetime  # created_at (rendered relative in the view)
    targets: list[tuple[str, str]]  # (label, icon-kind) per target under test
    status: str  # 'passed' | 'warning' | 'failed'
    score: float | None  # run score (mean of scorer_averages), 0..1
    cases: int  # number of simulations in the run
    cost: float | None  # summed cost_usd across the run's simulations; None if none recorded
    error: bool


@dataclass(frozen=True)
class SimOverview:
    """Rolled-up Agent Sim surface data: KPI cards + recent-runs table.

    ``avg_cost`` is the real per-sim dollar cost, averaged over only the sims
    that carry a ``token_usage.cost_usd`` (recorded upstream) — a missing cost is
    "unknown", not $0, so cost-less sims don't dilute the mean. ``None`` when no
    sim has cost data, in which case the view falls back to avg tokens/sim.
    ``avg_input_cost`` / ``avg_output_cost`` are the same averaging rule applied
    to the ``input_cost`` / ``output_cost`` components.
    """

    simulations_run: int
    goal_completion: float | None  # share of sims that achieved the goal, 0..1
    avg_turns: float | None
    avg_tokens: float | None
    avg_cost: float | None  # mean cost_usd per sim
    avg_input_cost: float | None  # mean input_cost per sim, averaged over costed sims only
    avg_output_cost: float | None  # mean output_cost per sim, averaged over costed sims only
    achieved: int  # outcomes donut segments
    not_achieved: int
    errors: int
    recent: list[SimRunRow] = field(default_factory=list)  # newest run first, current page
    total_runs: int = 0  # total runs before paging (for the pager)
    page: int = 1
    per_page: int = 8


def _sim_run_score(data: dict[str, object]) -> float | None:
    """Run-level score: mean of the scorer averages (same basis as _sim_row)."""
    averages = data.get('scorer_averages')
    averages = averages if isinstance(averages, dict) else {}
    vals = [_as_float(v) for v in averages.values()] if averages else []
    return sum(vals) / len(vals) if vals else None


def sim_overview(roots: list[Path] | None = None, *, page: int = 1, per_page: int = 8) -> SimOverview:
    """Aggregate the sim run store into headline KPIs plus a run-level table
    (one row per full simulation run). Rows are newest-first, sliced to the
    requested page of ``per_page`` (``total_runs`` carries the full count)."""
    page = max(1, page)
    runs: list[SimRunRow] = []
    turns_total = 0
    tokens_total = 0
    cost_total = 0.0
    costed = 0  # sims that actually carry a known cost_usd — avg_cost divides by this, not total
    input_cost_total = 0.0
    input_costed = 0
    output_cost_total = 0.0
    output_costed = 0
    achieved = not_achieved = errors = 0
    total = 0

    for card in library.scan(roots):
        if card.surface != 'sim' or card.path is None:
            continue
        try:
            data = library.read_json_cached(card.path)
        except (OSError, ValueError):
            continue

        run_costs: list[float] = []
        run_cases = 0
        run_errors = 0
        for res in _results(data):
            total += 1
            run_cases += 1
            is_error = str(res.get('terminated_by') or '') == 'error'
            goal = bool(res.get('goal_achieved'))
            turns_total += _as_int(res.get('turn_count'))
            tokens_total += _result_tokens(res)
            usage = res.get('token_usage')
            cost = _cost_usd(usage)
            if cost is not None:
                run_costs.append(cost)
                costed += 1
            input_cost = _input_cost(usage)
            if input_cost is not None:
                input_cost_total += input_cost
                input_costed += 1
            output_cost = _output_cost(usage)
            if output_cost is not None:
                output_cost_total += output_cost
                output_costed += 1
            # Mirror the donut segments (goal_achieved / error) exactly.
            if is_error:
                errors += 1
                run_errors += 1
            elif goal:
                achieved += 1
            else:
                not_achieved += 1
        run_cost = sum(run_costs) if run_costs else None
        if run_cost is not None:
            cost_total += run_cost

        score = _sim_run_score(data)
        runs.append(
            SimRunRow(
                rid=card.id,
                name=card.name,
                when=card.created_at,
                targets=[_sim_target(data)],
                status=_lifecycle_status(
                    broken=bool(card.error), all_errored=(run_cases > 0 and run_errors == run_cases)
                ),
                score=score,
                cases=run_cases,
                cost=run_cost,
                error=bool(card.error),
            )
        )

    runs.sort(key=lambda r: r.when, reverse=True)
    return SimOverview(
        simulations_run=total,
        goal_completion=(achieved / total) if total else None,
        avg_turns=(turns_total / total) if total else None,
        avg_tokens=(tokens_total / total) if total else None,
        avg_cost=(cost_total / costed) if costed else None,
        avg_input_cost=(input_cost_total / input_costed) if input_costed else None,
        avg_output_cost=(output_cost_total / output_costed) if output_costed else None,
        achieved=achieved,
        not_achieved=not_achieved,
        errors=errors,
        recent=runs[(page - 1) * per_page : page * per_page],
        total_runs=len(runs),
        page=page,
        per_page=per_page,
    )


# ---------------------------------------------------------------------------
# Red Team overview (item-level, one row per attack) — RES-1021
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedTeamRunRow:
    """One full red team run, as the design's run-level table row."""

    rid: str  # report id (row links here)
    name: str  # run_name / description
    when: datetime  # created_at (rendered relative in the view)
    targets: list[tuple[str, str]]  # (label, icon-kind) per tested agent/model
    status: str  # 'passed' (resisted) | 'warning' | 'failed'
    score: float | None  # resistance rate, 0..1
    cases: int  # number of attacks in the run
    cost: float | None  # summed cost_usd for the run; None when not recorded
    error: bool


@dataclass(frozen=True)
class RedTeamOverview:
    """Rolled-up Red Team surface data: KPI cards + recent-runs table."""

    attacks_run: int
    break_rate: float | None  # share of evaluated attacks that were vulnerable
    critical_findings: int
    avg_robustness: float | None  # attack-weighted resistance, 0..1
    total_cost: float | None  # summed cost_usd across red team runs
    total_input_cost: float | None = None  # summed input_cost across red team runs
    total_output_cost: float | None = None  # summed output_cost across red team runs
    recent: list[RedTeamRunRow] = field(default_factory=list)  # newest run first, current page
    total_runs: int = 0  # total runs before paging (for the pager)
    page: int = 1
    per_page: int = 8


def _redteam_targets(data: dict[str, object]) -> list[tuple[str, str]]:
    """One (label, icon-kind) per distinct tested agent/model, so multi-target
    runs render as separate chips instead of a comma-joined blob. Dedupes by
    agent identity (key/model), preferring a display name for the label."""
    seen: dict[str, list[str]] = {}
    order: list[str] = []
    for res in _results(data):
        agent = res.get('agent')
        agent = agent if isinstance(agent, dict) else {}
        ident = agent.get('key') or agent.get('model')
        label = agent.get('display_name') or agent.get('key') or agent.get('model')
        if not isinstance(label, str) or not label:
            continue
        ident = ident if isinstance(ident, str) and ident else label
        kind = 'agent' if agent.get('key') else ('llm' if agent.get('model') else 'other')
        if ident not in seen:
            order.append(ident)
            seen[ident] = [label, kind]
        elif agent.get('display_name'):  # upgrade the label to the display name
            seen[ident][0] = str(agent['display_name'])
    if order:
        return [(seen[i][0], seen[i][1]) for i in order]
    # Fallback: the run's tested_agents strings (strip agent:/deployment: prefix).
    out: list[tuple[str, str]] = []
    tested = data.get('tested_agents')
    if isinstance(tested, list):
        for t in tested:
            if isinstance(t, str) and t:
                name = t.split(':', 1)[1] if t.startswith(('agent:', 'deployment:')) else t
                out.append((name, 'deployment' if t.startswith('deployment:') else 'agent'))
            elif isinstance(t, dict):
                val = t.get('display_name') or t.get('key')
                if isinstance(val, str) and val:
                    out.append((val, 'agent'))
    return out or [('unknown', 'other')]


def redteam_overview(roots: list[Path] | None = None, *, page: int = 1, per_page: int = 8) -> RedTeamOverview:
    """Aggregate the red team run store into headline KPIs plus a run-level table
    (one row per full red team run). Rows are newest-first, sliced to the requested
    page of ``per_page`` (``total_runs`` carries the full count)."""
    page = max(1, page)
    runs: list[RedTeamRunRow] = []
    evaluated = 0
    vulnerable = 0
    critical = 0
    resistant = 0
    cost_total = 0.0
    has_cost = False
    input_cost_total = 0.0
    has_input_cost = False
    output_cost_total = 0.0
    has_output_cost = False
    total_attacks = 0

    for card in library.scan(roots):
        if card.surface != 'redteam' or card.path is None:
            continue
        try:
            data = library.read_json_cached(card.path)
        except (OSError, ValueError):
            continue

        summary = data.get('summary')
        summary = summary if isinstance(summary, dict) else {}
        usage = summary.get('token_usage_total')
        run_cost = _cost_usd(usage)
        if run_cost is not None:
            cost_total += run_cost
            has_cost = True
        run_input = _input_cost(usage)
        if run_input is not None:
            input_cost_total += run_input
            has_input_cost = True
        run_output = _output_cost(usage)
        if run_output is not None:
            output_cost_total += run_output
            has_output_cost = True

        counts = _redteam_counts(data)
        total_attacks += counts.attacks
        evaluated += counts.evaluated
        vulnerable += counts.vulnerable
        resistant += counts.evaluated - counts.vulnerable
        critical += counts.by_severity.get('critical', 0)

        resistance = _as_float(summary.get('resistance_rate')) if 'resistance_rate' in summary else None
        if zero_evaluated_attacks(summary):
            # Same no-score rule as the landing rows: zero evaluated attacks
            # means the rate is only the schema default, never a real score.
            resistance = None
        cases = _as_int(summary.get('total_attacks')) or counts.attacks
        runs.append(
            RedTeamRunRow(
                rid=card.id,
                name=card.name,
                when=card.created_at,
                targets=_redteam_targets(data),
                status=_lifecycle_status(
                    broken=bool(card.error), all_errored=(counts.attacks > 0 and counts.errors == counts.attacks)
                ),
                score=resistance,
                cases=cases,
                cost=run_cost,
                error=bool(card.error),
            )
        )

    runs.sort(key=lambda r: r.when, reverse=True)
    return RedTeamOverview(
        attacks_run=total_attacks,
        break_rate=(vulnerable / evaluated) if evaluated else None,
        critical_findings=critical,
        avg_robustness=(resistant / evaluated) if evaluated else None,
        total_cost=cost_total if has_cost else None,
        total_input_cost=input_cost_total if has_input_cost else None,
        total_output_cost=output_cost_total if has_output_cost else None,
        recent=runs[(page - 1) * per_page : page * per_page],
        total_runs=len(runs),
        page=page,
        per_page=per_page,
    )
