"""HTMX filter sidebar: FilterDef, FILTERS registry, and application logic.

Each ``FilterDef`` encapsulates all filter operations for a surface:

- ``dimensions``          : ordered list of dimension keys
- ``options(obj)``        : compute full option lists from the *full* object
- ``apply(obj, sel)``     : return the filtered result list
- ``recompute_options(filtered)`` : recompute option lists from an
                            *already-filtered* result list so empty options
                            drop out.  The caller is responsible for running
                            ``apply`` first and passing the result in.

Selections are plain ``dict[str, list[str]]`` mapping dimension key to the
list of selected values.  Radio dimensions (``result``, ``goal_outcome``) use
a single-element list for the selected value.  Missing or empty selections
default to "all selected" so a partial POST does not silently drop filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

    recompute_options: Callable[[list[Any]], dict[str, list[str]]]
    """Return option lists recomputed from an *already-filtered* result list.

    The caller must apply ``FilterDef.apply`` first and pass the resulting
    list in.  This avoids running ``apply`` twice per POST.
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
    "result",
    "agent",
    "category",
    "severity",
    "technique",
    "delivery_method",
    "vulnerability",
]


def _rt_options_from_results(results: list[Any]) -> dict[str, list[str]]:
    """Derive option lists from a list of RedTeamResult objects."""
    from evaluatorq.redteam.contracts import Severity as _Severity  # local to avoid top-level circular

    SEVERITY_ORDER = [s.value for s in _Severity]

    all_categories = sorted({r.attack.category for r in results})
    all_severities = [s for s in SEVERITY_ORDER if any(r.attack.severity.value == s for r in results)]
    all_techniques = sorted({r.attack.attack_technique.value for r in results})
    all_delivery = sorted(
        {getattr(dm, "value", dm) for r in results for dm in (r.attack.delivery_methods or [])}
    )
    all_vulnerabilities = sorted({r.attack.vulnerability for r in results if r.attack.vulnerability})
    all_agents = sorted({r.agent.key or r.agent.display_name or "unknown" for r in results})

    return {
        "result": ["All", "Vulnerable", "Resistant", "Error"],
        "agent": all_agents,
        "category": all_categories,
        "severity": all_severities,
        "technique": all_techniques,
        "delivery_method": all_delivery,
        "vulnerability": all_vulnerabilities,
    }


def _rt_full_options(report: Any) -> dict[str, list[str]]:
    return _rt_options_from_results(report.results)


def _rt_apply(report: Any, selections: dict[str, list[str]]) -> list[Any]:
    """Apply all redteam filter dimensions independently (parity with Streamlit)."""
    results: list[Any] = list(report.results)
    full_opts = _rt_full_options(report)

    # result (radio)
    result_sel = selections.get("result", ["All"])
    result_filter = result_sel[0] if result_sel else "All"
    if result_filter == "Vulnerable":
        results = [r for r in results if r.vulnerable]
    elif result_filter == "Resistant":
        results = [r for r in results if not r.vulnerable and not r.error]
    elif result_filter == "Error":
        results = [r for r in results if r.error]

    # category (multiselect)
    all_categories = full_opts["category"]
    sel_categories = _sel(selections, "category", default=all_categories)
    if set(sel_categories) != set(all_categories):
        results = [r for r in results if r.attack.category in sel_categories]

    # severity (multiselect)
    all_severities = full_opts["severity"]
    sel_severities = _sel(selections, "severity", default=all_severities)
    if set(sel_severities) != set(all_severities):
        results = [r for r in results if r.attack.severity.value in sel_severities]

    # technique (multiselect)
    all_techniques = full_opts["technique"]
    sel_techniques = _sel(selections, "technique", default=all_techniques)
    if set(sel_techniques) != set(all_techniques):
        results = [r for r in results if r.attack.attack_technique.value in sel_techniques]

    # delivery_method (multiselect) — only when options exist
    all_delivery = full_opts["delivery_method"]
    if all_delivery:
        sel_delivery = _sel(selections, "delivery_method", default=all_delivery)
        if set(sel_delivery) != set(all_delivery):
            results = [
                r
                for r in results
                if any(getattr(dm, "value", dm) in sel_delivery for dm in (r.attack.delivery_methods or []))
            ]

    # vulnerability (multiselect) — only when options exist
    all_vulnerabilities = full_opts["vulnerability"]
    if all_vulnerabilities:
        sel_vulnerabilities = _sel(selections, "vulnerability", default=all_vulnerabilities)
        if set(sel_vulnerabilities) != set(all_vulnerabilities):
            results = [r for r in results if r.attack.vulnerability in sel_vulnerabilities]

    # agent (multiselect) — only when >1 agent
    all_agents = full_opts["agent"]
    if len(all_agents) > 1:
        sel_agents = _sel(selections, "agent", default=all_agents)
        if set(sel_agents) != set(all_agents):
            results = [
                r
                for r in results
                if (r.agent.key or r.agent.display_name or "unknown") in sel_agents
            ]

    return results


def _rt_recompute_options(filtered: list[Any]) -> dict[str, list[str]]:
    """Recompute option lists from an already-filtered result list.

    The caller is responsible for calling ``_rt_apply`` first and passing
    the result in, so that ``apply`` runs only once per POST.
    """
    opts = _rt_options_from_results(filtered)
    # The result radio always shows all four statuses regardless of filter state.
    opts["result"] = ["All", "Vulnerable", "Resistant", "Error"]
    return opts


# ---------------------------------------------------------------------------
# Simulation filter
# ---------------------------------------------------------------------------

_SIM_DIMS = [
    "persona",
    "scenario",
    "terminated_by",
    "goal_outcome",
]


def _meta(result: Any, key: str) -> str:
    return str(result.metadata.get(key, "unknown"))


def _sim_options_from_results(results: list[Any]) -> dict[str, list[str]]:
    personas = sorted({_meta(r, "persona") for r in results})
    scenarios = sorted({_meta(r, "scenario") for r in results})
    terminated = sorted({r.terminated_by.value for r in results})
    return {
        "persona": personas,
        "scenario": scenarios,
        "terminated_by": terminated,
        "goal_outcome": ["All", "Achieved", "Not achieved"],
    }


def _sim_full_options(run: Any) -> dict[str, list[str]]:
    return _sim_options_from_results(run.results)


def _sim_apply(run: Any, selections: dict[str, list[str]]) -> list[Any]:
    """Apply all sim filter dimensions."""
    results: list[Any] = list(run.results)
    full_opts = _sim_full_options(run)

    # persona (multiselect)
    all_personas = full_opts["persona"]
    sel_personas = _sel(selections, "persona", default=all_personas)
    if set(sel_personas) != set(all_personas):
        results = [r for r in results if _meta(r, "persona") in sel_personas]

    # scenario (multiselect)
    all_scenarios = full_opts["scenario"]
    sel_scenarios = _sel(selections, "scenario", default=all_scenarios)
    if set(sel_scenarios) != set(all_scenarios):
        results = [r for r in results if _meta(r, "scenario") in sel_scenarios]

    # terminated_by (multiselect)
    all_terminated = full_opts["terminated_by"]
    sel_terminated = _sel(selections, "terminated_by", default=all_terminated)
    if set(sel_terminated) != set(all_terminated):
        results = [r for r in results if r.terminated_by.value in sel_terminated]

    # goal_outcome (radio)
    goal_sel = selections.get("goal_outcome", ["All"])
    goal_filter = goal_sel[0] if goal_sel else "All"
    if goal_filter == "Achieved":
        results = [r for r in results if r.goal_achieved]
    elif goal_filter == "Not achieved":
        results = [r for r in results if not r.goal_achieved]

    return results


def _sim_recompute_options(filtered: list[Any]) -> dict[str, list[str]]:
    """Recompute option lists from an already-filtered result list.

    The caller is responsible for calling ``_sim_apply`` first and passing
    the result in, so that ``apply`` runs only once per POST.
    """
    opts = _sim_options_from_results(filtered)
    # The goal_outcome radio always shows all three values.
    opts["goal_outcome"] = ["All", "Achieved", "Not achieved"]
    return opts


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FILTERS: dict[str, FilterDef] = {
    "redteam": FilterDef(
        dimensions=_REDTEAM_DIMS,
        options=_rt_full_options,
        apply=_rt_apply,
        recompute_options=_rt_recompute_options,
    ),
    "sim": FilterDef(
        dimensions=_SIM_DIMS,
        options=_sim_full_options,
        apply=_sim_apply,
        recompute_options=_sim_recompute_options,
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
    if filter_def is None or not selections:
        return list(report_obj.results)
    return filter_def.apply(report_obj, selections)
