"""HTMX filter sidebar: FilterDef, FILTERS registry, and application logic.

Each ``FilterDef`` encapsulates all filter operations for a surface:

- ``dimensions``          : ordered list of dimension keys
- ``options(obj)``        : compute full option lists from the *full* object
                            (always the complete set — never narrowed to the
                            filtered rows, so a deselected value never vanishes
                            from its own multi-select)
- ``apply(obj, sel)``     : return the filtered result list

Selections are plain ``dict[str, list[str]]`` mapping dimension key to the
list of selected values.  The ``result`` radio dimension uses a
single-element list for the selected value.  The sim ``goal_outcome``
dimension is a two-value multiselect (``Achieved`` / ``Not achieved``):
selecting exactly one value narrows to that outcome; selecting zero or
both values means "all".  Missing or empty selections default to "all
selected" so a partial POST does not silently drop filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from evaluatorq.simulation.metrics import TURN_METRICS

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterDef:
    """All filter operations for a single surface."""

    dimensions: list[str]
    """Ordered list of dimension keys."""

    options: Callable[[Any], dict[str, list[str]]]
    """Return full option lists from the raw report/run object."""

    apply: Callable[[Any, dict[str, list[str]]], list[Any]]
    """Return the filtered result list given the raw object + selections."""

    results: Callable[[Any], list[Any]] = lambda obj: list(obj.results)
    """Return the full, unfiltered result list.

    Defaults to ``obj.results``, which both red team and simulation expose. A
    surface whose runs hold their items under a different name (pairwise keeps
    ``entries``) overrides this so the shared no-selection path does not have to
    know the attribute name.
    """


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sel(selections: dict[str, list[str]], key: str, *, default: list[str]) -> list[str]:
    """Return the selection for *key*, falling back to *default* when absent."""
    v = selections.get(key)
    return v or default


# ---------------------------------------------------------------------------
# Red-team filter
# ---------------------------------------------------------------------------

_REDTEAM_DIMS = [
    'result',
    'severity',
    'min_turns',
    'category',
    'agent',
    'technique',
    'delivery_method',
    'vulnerability',
]


def _rt_options_from_results(results: list[Any]) -> dict[str, list[str]]:
    """Derive option lists from a list of RedTeamResult objects."""
    from evaluatorq.redteam.contracts import Severity as _Severity  # local to avoid top-level circular

    SEVERITY_ORDER = [s.value for s in _Severity]

    all_categories = sorted({r.attack.category for r in results})
    all_severities = [s for s in SEVERITY_ORDER if any(r.attack.severity.value == s for r in results)]
    all_techniques = sorted({r.attack.attack_technique.value for r in results})
    all_delivery = sorted({getattr(dm, 'value', dm) for r in results for dm in (r.attack.delivery_methods or [])})
    all_vulnerabilities = sorted({r.attack.vulnerability for r in results if r.attack.vulnerability})
    all_agents = sorted({r.agent.key or r.agent.display_name or 'unknown' for r in results})

    max_turns = max((r.execution.turns if r.execution else 1 for r in results), default=1)

    return {
        'result': ['Vulnerable', 'Resistant', 'Error'],
        'agent': all_agents,
        'category': all_categories,
        'severity': all_severities,
        'technique': all_techniques,
        'delivery_method': all_delivery,
        'vulnerability': all_vulnerabilities,
        'max_turns': [str(max_turns)],
    }


def _rt_full_options(report: Any) -> dict[str, list[str]]:
    return _rt_options_from_results(report.results)


def _rt_apply(report: Any, selections: dict[str, list[str]]) -> list[Any]:
    """Apply all redteam filter dimensions independently (parity with Streamlit)."""
    results: list[Any] = list(report.results)
    full_opts = _rt_full_options(report)

    # result (3-value multiselect: empty or all-3 => no filter)
    result_sel = selections.get('result', [])
    all_result = {'Vulnerable', 'Resistant', 'Error'}
    if result_sel and set(result_sel) != all_result:

        def _cls(r):
            return 'Error' if r.error else 'Vulnerable' if r.vulnerable else 'Resistant'

        chosen = set(result_sel)
        results = [r for r in results if _cls(r) in chosen]

    # min_turns (slider)
    mt_sel = selections.get('min_turns', ['1'])
    try:
        min_turns = int(mt_sel[0]) if mt_sel else 1
    except (ValueError, TypeError):
        min_turns = 1
    if min_turns > 1:
        results = [r for r in results if (r.execution.turns if r.execution else 1) >= min_turns]

    # category (multiselect)
    all_categories = full_opts['category']
    sel_categories = _sel(selections, 'category', default=all_categories)
    if set(sel_categories) != set(all_categories):
        results = [r for r in results if r.attack.category in sel_categories]

    # severity (multiselect)
    all_severities = full_opts['severity']
    sel_severities = _sel(selections, 'severity', default=all_severities)
    if set(sel_severities) != set(all_severities):
        results = [r for r in results if r.attack.severity.value in sel_severities]

    # technique (multiselect)
    all_techniques = full_opts['technique']
    sel_techniques = _sel(selections, 'technique', default=all_techniques)
    if set(sel_techniques) != set(all_techniques):
        results = [r for r in results if r.attack.attack_technique.value in sel_techniques]

    # delivery_method (multiselect) — only when options exist
    all_delivery = full_opts['delivery_method']
    if all_delivery:
        sel_delivery = _sel(selections, 'delivery_method', default=all_delivery)
        if set(sel_delivery) != set(all_delivery):
            results = [
                r
                for r in results
                if any(getattr(dm, 'value', dm) in sel_delivery for dm in (r.attack.delivery_methods or []))
            ]

    # vulnerability (multiselect) — only when options exist
    all_vulnerabilities = full_opts['vulnerability']
    if all_vulnerabilities:
        sel_vulnerabilities = _sel(selections, 'vulnerability', default=all_vulnerabilities)
        if set(sel_vulnerabilities) != set(all_vulnerabilities):
            results = [r for r in results if r.attack.vulnerability in sel_vulnerabilities]

    # agent (multiselect) — only when >1 agent
    all_agents = full_opts['agent']
    if len(all_agents) > 1:
        sel_agents = _sel(selections, 'agent', default=all_agents)
        if set(sel_agents) != set(all_agents):
            results = [r for r in results if (r.agent.key or r.agent.display_name or 'unknown') in sel_agents]

    return results


# ---------------------------------------------------------------------------
# Simulation filter
# ---------------------------------------------------------------------------


def sim_metric_dim_key(key: str, *, high_is_risky: bool) -> str:
    """Threshold selection key for a turn metric: floor for risky, ceiling otherwise."""
    return f'min_{key}' if high_is_risky else f'max_{key}'


def _sim_metric_dims() -> list[str]:
    return [sim_metric_dim_key(m.key, high_is_risky=m.high_is_risky) for m in TURN_METRICS]


_SIM_DIMS = [
    'persona',
    'scenario',
    'terminated_by',
    'goal_outcome',
    'rule_broken',
    'max_goal_score',
    'min_turns',
    'min_total_tokens',
    *_sim_metric_dims(),
]


def _meta(result: Any, key: str) -> str:
    return str(result.metadata.get(key, 'unknown'))


def _clamp_score(raw: str) -> float | None:
    """Parse *raw* as a float and clamp it to ``0..1``; ``None`` on bad input."""
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    return max(0.0, min(1.0, value))


def _parse_positive_int(raw: str) -> int | None:
    """Parse *raw* as a positive int; ``None`` on bad or non-positive input.

    Count floors (min turns / min total tokens) only ever narrow at values > 0,
    so a zero/negative/garbage value is treated as "no filter" rather than
    applied verbatim."""
    try:
        value = int(raw)
    except (ValueError, TypeError):
        return None
    return value if value > 0 else None


def _sim_multiselect_options(results: list[Any]) -> dict[str, list[str]]:
    """The three "all"-default multiselect option lists (persona / scenario /
    terminated_by) plus the fixed goal-outcome pair. This is the only slice
    ``_sim_apply`` needs, so it is kept separate from the heavier bound/metric
    scan below — the apply path must not pay for the full option build."""
    return {
        'persona': sorted({_meta(r, 'persona') for r in results}),
        'scenario': sorted({_meta(r, 'scenario') for r in results}),
        'terminated_by': sorted({r.terminated_by.value for r in results}),
        'goal_outcome': ['Achieved', 'Not achieved'],
    }


def _sim_options_from_results(results: list[Any]) -> dict[str, list[str]]:
    max_turns = max((r.turn_count for r in results), default=0)
    max_total_tokens = max((r.token_usage.total_tokens for r in results), default=0)
    available_metrics = [
        m.key
        for m in TURN_METRICS
        if any(getattr(tm, m.key, None) is not None for r in results for tm in r.turn_metrics)
    ]
    return {
        **_sim_multiselect_options(results),
        'max_turns': [str(max_turns)],
        'max_total_tokens': [str(max_total_tokens)],
        'metrics': available_metrics,
    }


def _sim_full_options(run: Any) -> dict[str, list[str]]:
    return _sim_options_from_results(run.results)


def _sim_apply(run: Any, selections: dict[str, list[str]]) -> list[Any]:
    """Apply all sim filter dimensions."""
    results: list[Any] = list(run.results)
    # Only the multiselect "all"-defaults are needed here; skip the heavier
    # bound/metric-availability scan that the full option build does.
    full_opts = _sim_multiselect_options(run.results)

    # persona (multiselect)
    all_personas = full_opts['persona']
    sel_personas = _sel(selections, 'persona', default=all_personas)
    if set(sel_personas) != set(all_personas):
        results = [r for r in results if _meta(r, 'persona') in sel_personas]

    # scenario (multiselect)
    all_scenarios = full_opts['scenario']
    sel_scenarios = _sel(selections, 'scenario', default=all_scenarios)
    if set(sel_scenarios) != set(all_scenarios):
        results = [r for r in results if _meta(r, 'scenario') in sel_scenarios]

    # terminated_by (multiselect)
    all_terminated = full_opts['terminated_by']
    sel_terminated = _sel(selections, 'terminated_by', default=all_terminated)
    if set(sel_terminated) != set(all_terminated):
        results = [r for r in results if r.terminated_by.value in sel_terminated]

    # goal_outcome (two-value multiselect: one => that outcome; zero/two => All)
    goal_sel = [v for v in selections.get('goal_outcome', []) if v in {'Achieved', 'Not achieved'}]
    if len(goal_sel) == 1:
        want = goal_sel[0] == 'Achieved'
        results = [r for r in results if bool(r.goal_achieved) == want]

    # rule_broken (opt-in chip): only 'yes' narrows to results with any broken rule.
    if 'yes' in selections.get('rule_broken', []):
        results = [r for r in results if r.rules_broken]

    # max_goal_score (ceiling on goal_completion_score)
    gs_sel = selections.get('max_goal_score', [])
    if gs_sel:
        threshold = _clamp_score(gs_sel[0])
        if threshold is not None:
            results = [r for r in results if r.goal_completion_score <= threshold]

    # min_turns (floor on turn_count, raw integer)
    turns_sel = selections.get('min_turns', [])
    if turns_sel:
        min_turns = _parse_positive_int(turns_sel[0])
        if min_turns is not None:
            results = [r for r in results if r.turn_count >= min_turns]

    # min_total_tokens (floor on token_usage.total_tokens, raw integer)
    tokens_sel = selections.get('min_total_tokens', [])
    if tokens_sel:
        min_tokens = _parse_positive_int(tokens_sel[0])
        if min_tokens is not None:
            results = [r for r in results if r.token_usage.total_tokens >= min_tokens]

    # per-turn quality/risk metric thresholds — worst turn per result, unscored
    # results always stay visible.
    for metric in TURN_METRICS:
        dim_key = sim_metric_dim_key(metric.key, high_is_risky=metric.high_is_risky)
        metric_sel = selections.get(dim_key, [])
        if not metric_sel:
            continue
        threshold = _clamp_score(metric_sel[0])
        if threshold is None:
            continue

        def _keeps(r: Any, metric: Any = metric, threshold: float = threshold) -> bool:
            values = [v for tm in r.turn_metrics for v in [getattr(tm, metric.key, None)] if v is not None]
            if not values:
                return True
            worst = max(values) if metric.high_is_risky else min(values)
            return worst >= threshold if metric.high_is_risky else worst <= threshold

        results = [r for r in results if _keeps(r)]

    return results


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FILTERS: dict[str, FilterDef] = {
    'redteam': FilterDef(
        dimensions=_REDTEAM_DIMS,
        options=_rt_full_options,
        apply=_rt_apply,
    ),
    'sim': FilterDef(
        dimensions=_SIM_DIMS,
        options=_sim_full_options,
        apply=_sim_apply,
    ),
    # Pairwise ships no filter dimensions: a run has no natural facet beyond the
    # consensus winner, so the rail would be a single control. Registered anyway
    # because the report route requires a FilterDef, and so the export paths get
    # the entry list through the same accessor as the other surfaces.
    'pairwise': FilterDef(
        dimensions=[],
        options=lambda _run: {},
        apply=lambda run, _selections: list(run.entries),
        results=lambda run: list(run.entries),
    ),
}


# ---------------------------------------------------------------------------
# Shared application helper (pure — no web-framework imports)
# ---------------------------------------------------------------------------


def apply_or_all(report_obj: Any, surface: str, selections: dict[str, list[str]]) -> list[Any]:
    """Return the filtered result list, defaulting to all results when unfiltered.

    Empty *selections* (no active filter) or an unknown *surface* both map to
    the full ``report_obj.results`` list, so callers never need an explicit
    ``if not selections`` guard.

    Args:
        report_obj: The raw report/run object (must have a ``.results`` attribute).
        surface:    Dashboard surface key (``"redteam"`` or ``"sim"``).
        selections: Dimension → selected-values mapping parsed from the request.
                    An empty dict means "all results".

    Returns:
        Filtered (or full) result list.
    """
    filter_def = FILTERS.get(surface)
    if filter_def is None:
        return list(report_obj.results)
    if not selections:
        return filter_def.results(report_obj)
    return filter_def.apply(report_obj, selections)
