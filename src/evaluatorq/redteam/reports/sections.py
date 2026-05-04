"""Renderer-agnostic section data layer for red team reports.

``build_report_sections()`` converts a ``RedTeamReport`` into a list of
``ReportSection`` objects that can be consumed by any renderer (Markdown, HTML,
PDF, …).  Format-specific renderers should NOT query the report directly; they
should consume the sections produced here.

Section kinds:
    - ``summary``                 — aggregate statistics
    - ``severity_definitions``    — severity level reference table
    - ``focus_areas``             — top-5 highest-risk categories
    - ``vulnerability_breakdown`` — per-vulnerability table rows (primary)
    - ``category_breakdown``      — per-category table rows (secondary)
    - ``technique_breakdown``     — per-technique table rows
    - ``delivery_breakdown``      — per-delivery-method ASR breakdown
    - ``error_analysis``          — error counts, types, and detail rows
    - ``attack_heatmap``          — vulnerability x technique attack success rates
    - ``individual_results``      — one entry per ``RedTeamResult``
    - ``agent_comparison``        — multi-agent comparison (>= 2 agents only)
    - ``agent_disagreements``     — per-attack disagreement viewer (>= 2 agents only)
    - ``framework_breakdown``     — per-framework breakdown (> 1 framework only)
    - ``agent_context``           — agent capability cards
    - ``turn_scope_breakdown``    — by-turn-type and by-domain statistics
    - ``turn_depth_analysis``     — multi-turn ASR% by conversation depth
    - ``token_usage``             — token consumption summary
    - ``source_distribution``     — attack source counts
"""

from __future__ import annotations

import operator
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from evaluatorq.redteam.contracts import OWASP_CATEGORY_NAMES, SEVERITY_DEFINITIONS, RedTeamReport, RedTeamResult
from evaluatorq.redteam.reports._utils import extract_prompt, extract_response
from evaluatorq.redteam.reports.guidance import REMEDIATION_GUIDANCE

# ---------------------------------------------------------------------------
# Risk scoring weights
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 4,
    "critical": 8,
}


# ---------------------------------------------------------------------------
# Section data model
# ---------------------------------------------------------------------------


@dataclass
class ReportSection:
    """Renderer-agnostic section of a report.

    Attributes:
        kind:  Machine-readable section identifier (e.g. ``"summary"``).
        title: Human-readable section title.
        data:  Free-form dict of section data consumed by renderers.
    """

    kind: str
    title: str
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_risk_score(vulnerability_rate: float, results: list[RedTeamResult]) -> float:
    """Compute ``risk_score = vulnerability_rate * average_severity_weight``.

    The severity weight is averaged over *vulnerable* results to reflect the
    actual impact of confirmed vulnerabilities.  If there are no vulnerable
    results the score is 0.
    """
    vulnerable = [r for r in results if r.vulnerable]
    if not vulnerable:
        return 0.0
    avg_weight = sum(SEVERITY_WEIGHTS.get(r.attack.severity.value, 1) for r in vulnerable) / len(vulnerable)
    return vulnerability_rate * avg_weight


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_summary_section(report: RedTeamReport) -> ReportSection:
    s = report.summary
    if report.tested_agents:
        targets = sorted(set(report.tested_agents))
    elif report.agent_contexts:
        targets = sorted(report.agent_contexts.keys())
    elif report.agent_context:
        targets = [report.agent_context.display_name or report.agent_context.key or "unknown"]
    else:
        # Last resort: extract from individual results
        agent_names = [r.agent.key or r.agent.display_name or r.agent.model or 'unknown' for r in report.results if r.agent]
        targets = sorted(set(agent_names)) if agent_names else ['unknown']
    critical_sev = s.by_severity.get("critical")
    critical_exposure = critical_sev.vulnerabilities_found if critical_sev else 0

    # Confidence indicator based on sample size and category coverage
    total = s.total_attacks
    num_categories = len(report.categories_tested)
    if total >= 100 and num_categories >= 5:
        confidence = "HIGH"
        confidence_note = f"{total} attacks across {num_categories} categories"
    elif total >= 30 and num_categories >= 3:
        confidence = "MEDIUM"
        confidence_note = f"{total} attacks across {num_categories} categories"
    else:
        confidence = "LOW"
        reasons = []
        if total < 30:
            reasons.append(f"only {total} attack{'s' if total != 1 else ''}")
        if num_categories < 3:
            reasons.append(f"only {num_categories} categor{'ies' if num_categories != 1 else 'y'}")
        confidence_note = ", ".join(reasons)

    return ReportSection(
        kind="summary",
        title="Executive Summary",
        data={
            "target": ", ".join(targets),
            "pipeline": report.pipeline.value,
            "created_at": report.created_at,
            "total_attacks": s.total_attacks,
            "evaluated_attacks": s.evaluated_attacks,
            "vulnerabilities_found": s.vulnerabilities_found,
            "vulnerability_rate": s.vulnerability_rate,
            "resistance_rate": s.resistance_rate,
            "evaluation_coverage": s.evaluation_coverage,
            "total_errors": s.total_errors,
            "duration_seconds": report.duration_seconds,
            "critical_exposure": critical_exposure,
            "by_severity": {k: v.model_dump(mode="json") for k, v in s.by_severity.items()},
            "confidence": confidence,
            "confidence_note": confidence_note,
        },
    )


def _build_focus_areas_section(report: RedTeamReport) -> ReportSection:
    """Build the top-5 highest-risk focus areas section."""
    results_by_category: dict[str, list[RedTeamResult]] = {}
    for r in report.results:
        results_by_category.setdefault(r.attack.category, []).append(r)

    # Build a lookup from vulnerability -> VulnerabilitySummary for enrichment
    vuln_by_category: dict[str, str] = {}
    for vuln_id, vuln_summary in report.summary.by_vulnerability.items():
        for cat_codes in vuln_summary.framework_categories.values():
            for cat_code in cat_codes:
                vuln_by_category.setdefault(cat_code, vuln_id)

    # Build a lookup from category -> recommendation (when available)
    rec_lookup: dict[str, Any] = {}
    if report.focus_area_recommendations:
        for rec in report.focus_area_recommendations:
            rec_lookup[rec.category] = {
                "recommendations": rec.recommendations,
                "patterns_observed": rec.patterns_observed,
                "traces_analyzed": rec.traces_analyzed,
            }

    focus_areas: list[dict[str, Any]] = []
    for cat_code, cat_summary in report.summary.by_category.items():
        if cat_summary.vulnerabilities_found == 0:
            continue
        cat_results = results_by_category.get(cat_code, [])
        risk_score = _compute_risk_score(cat_summary.vulnerability_rate, cat_results)
        vuln_id = vuln_by_category.get(cat_code, "")
        vuln_summary = report.summary.by_vulnerability.get(vuln_id)
        area: dict[str, Any] = {
            "category": cat_code,
            "category_name": OWASP_CATEGORY_NAMES.get(cat_code, cat_code),
            "vulnerability": vuln_id,
            "vulnerability_name": vuln_summary.vulnerability_name if vuln_summary else "",
            "vulnerabilities_found": cat_summary.vulnerabilities_found,
            "vulnerability_rate": cat_summary.vulnerability_rate,
            "risk_score": risk_score,
            "remediation": REMEDIATION_GUIDANCE.get(cat_code, ""),
        }
        # Attach LLM-generated recommendations when available
        if cat_code in rec_lookup:
            area["llm_recommendations"] = rec_lookup[cat_code]
        focus_areas.append(area)

    # Sort by risk score descending, limit to top 5
    focus_areas.sort(key=operator.itemgetter("risk_score"), reverse=True)
    top_areas = focus_areas[:5]

    return ReportSection(
        kind="focus_areas",
        title="Focus Areas (Top Risks)",
        data={"focus_areas": top_areas},
    )


def _build_vulnerability_breakdown_section(report: RedTeamReport) -> ReportSection:
    """Build per-vulnerability breakdown rows sorted by vulnerability rate (worst first)."""
    rows: list[dict[str, Any]] = []
    for vuln_id, vuln_summary in report.summary.by_vulnerability.items():
        total = vuln_summary.total_attacks
        found = vuln_summary.vulnerabilities_found
        vulnerability_rate = 1.0 - vuln_summary.resistance_rate
        rows.append(
            {
                "vulnerability": vuln_id,
                "vulnerability_name": vuln_summary.vulnerability_name,
                "domain": vuln_summary.domain,
                "total_attacks": total,
                "vulnerabilities_found": found,
                "vulnerability_rate": vulnerability_rate,
                "resistance_rate": vuln_summary.resistance_rate,
            }
        )
    # Worst first (highest vulnerability rate)
    rows.sort(key=operator.itemgetter("vulnerability_rate"), reverse=True)
    return ReportSection(
        kind="vulnerability_breakdown",
        title="Per-Vulnerability Breakdown",
        data={"rows": rows},
    )


def _build_category_breakdown_section(report: RedTeamReport) -> ReportSection:
    """Build per-category breakdown rows sorted by vulnerability rate (worst first)."""
    rows: list[dict[str, Any]] = []
    for cat_code, cat_summary in report.summary.by_category.items():
        rows.append(
            {
                "category": cat_code,
                "category_name": OWASP_CATEGORY_NAMES.get(cat_code, cat_code),
                "total_attacks": cat_summary.total_attacks,
                "vulnerabilities_found": cat_summary.vulnerabilities_found,
                "vulnerability_rate": cat_summary.vulnerability_rate,
                "resistance_rate": cat_summary.resistance_rate,
            }
        )
    # Worst first (highest vulnerability rate)
    rows.sort(key=operator.itemgetter("vulnerability_rate"), reverse=True)
    return ReportSection(
        kind="category_breakdown",
        title="Per-Category Breakdown",
        data={"rows": rows},
    )


def _build_technique_breakdown_section(report: RedTeamReport) -> ReportSection:
    """Build per-technique breakdown rows sorted by vulnerability rate (worst first)."""
    rows: list[dict[str, Any]] = []
    for technique, tech_summary in report.summary.by_technique.items():
        rows.append(
            {
                "technique": technique,
                "total_attacks": tech_summary.total_attacks,
                "vulnerabilities_found": tech_summary.vulnerabilities_found,
                "vulnerability_rate": tech_summary.vulnerability_rate,
                "resistance_rate": tech_summary.resistance_rate,
            }
        )
    rows.sort(key=operator.itemgetter("vulnerability_rate"), reverse=True)
    return ReportSection(
        kind="technique_breakdown",
        title="Per-Technique Breakdown",
        data={"rows": rows},
    )


def _build_individual_results_section(report: RedTeamReport) -> ReportSection:
    """Build individual attack result entries."""
    entries: list[dict[str, Any]] = [{
                "id": result.attack.id,
                "vulnerability": result.attack.vulnerability,
                "category": result.attack.category,
                "category_name": OWASP_CATEGORY_NAMES.get(result.attack.category, result.attack.category),
                "technique": result.attack.attack_technique.value,
                "severity": result.attack.severity.value,
                "vulnerable": result.vulnerable,
                "prompt": extract_prompt(result),
                "response": extract_response(result),
                "explanation": result.evaluation.explanation if result.evaluation else "",
                "error": result.error,
            } for result in report.results]
    return ReportSection(
        kind="individual_results",
        title="Individual Attack Results",
        data={"entries": entries},
    )


# ---------------------------------------------------------------------------
# Delivery breakdown section
# ---------------------------------------------------------------------------


def _build_delivery_breakdown_section(report: RedTeamReport) -> ReportSection:
    """Build per-delivery-method ASR breakdown rows sorted by vulnerability rate (worst first)."""
    rows: list[dict[str, Any]] = []
    for method, dm_summary in report.summary.by_delivery_method.items():
        rows.append(
            {
                "delivery_method": method,
                "total_attacks": dm_summary.total_attacks,
                "vulnerabilities_found": dm_summary.vulnerabilities_found,
                "vulnerability_rate": dm_summary.vulnerability_rate,
                "resistance_rate": dm_summary.resistance_rate,
            }
        )
    rows.sort(key=operator.itemgetter("vulnerability_rate"), reverse=True)
    return ReportSection(
        kind="delivery_breakdown",
        title="ASR by Delivery Method",
        data={"rows": rows},
    )


# ---------------------------------------------------------------------------
# Error analysis section
# ---------------------------------------------------------------------------


def _build_error_analysis_section(report: RedTeamReport) -> ReportSection:
    """Build error analysis: metrics, errors by type, and per-result error detail rows."""
    s = report.summary
    total_errors = s.total_errors
    total_attacks = s.total_attacks
    error_rate = total_errors / total_attacks if total_attacks > 0 else 0.0

    # errors_by_type from the summary (dict[str, int])
    errors_by_type: dict[str, int] = dict(s.errors_by_type) if s.errors_by_type else {}

    # Enumerate individual error detail rows from results
    detail_rows: list[dict[str, Any]] = [{
                    "id": result.attack.id,
                    "category": result.attack.category,
                    "technique": result.attack.attack_technique.value,
                    "error_type": result.error_type or "unknown",
                    "stage": result.error_stage or "",
                    "error": result.error,
                } for result in report.results if result.error]

    # If errors_by_type is empty but detail_rows exist, compute it
    if not errors_by_type and detail_rows:
        for row in detail_rows:
            et = row["error_type"]
            errors_by_type[et] = errors_by_type.get(et, 0) + 1

    return ReportSection(
        kind="error_analysis",
        title="Error Analysis",
        data={
            "total_errors": total_errors,
            "error_rate": error_rate,
            "error_types_count": len(errors_by_type),
            "errors_by_type": errors_by_type,
            "detail_rows": detail_rows,
        },
    )


# ---------------------------------------------------------------------------
# Attack heatmap section
# ---------------------------------------------------------------------------


def _build_attack_heatmap_section(report: RedTeamReport) -> ReportSection:
    """Build vulnerability × technique attack success rate data for heatmap rendering."""  # noqa: RUF002
    # Accumulate counts: vulnerability -> technique -> {vuln, total}
    matrix: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"vulnerable": 0, "total": 0})
    )
    vulnerabilities: set[str] = set()
    techniques: set[str] = set()

    for result in report.results:
        vuln = result.attack.vulnerability or result.attack.category
        tech = result.attack.attack_technique.value
        vulnerabilities.add(vuln)
        techniques.add(tech)
        matrix[vuln][tech]["total"] += 1
        if result.vulnerable:
            matrix[vuln][tech]["vulnerable"] += 1

    # Build flat rows for the heatmap
    cells: list[dict[str, Any]] = []
    for vuln in sorted(vulnerabilities):
        for tech in sorted(techniques):
            cell = matrix[vuln].get(tech, {})
            total = cell.get("total", 0)
            found = cell.get("vulnerable", 0)
            asr = found / total if total > 0 else 0.0
            cells.append(
                {
                    "vulnerability": vuln,
                    "technique": tech,
                    "total_attacks": total,
                    "vulnerabilities_found": found,
                    "vulnerability_rate": asr,
                }
            )

    return ReportSection(
        kind="attack_heatmap",
        title="Attack Success Heatmap (Vulnerability × Technique)",  # noqa: RUF001
        data={
            "cells": cells,
            "vulnerabilities": sorted(vulnerabilities),
            "techniques": sorted(techniques),
        },
    )


# ---------------------------------------------------------------------------
# Multi-agent comparison section
# ---------------------------------------------------------------------------


def _build_agent_comparison_section(report: RedTeamReport) -> ReportSection:
    """Build multi-agent comparison data.

    Only called when ``len(report.tested_agents) >= 2``.  Produces:

    - ``agents``: ordered list of agent names.
    - ``agent_metrics``: per-agent summary (total_attacks, vulnerabilities_found, asr).
    - ``vuln_asr_matrix``: list of ``{vulnerability, agents: {agent: {asr, total}}}``
      rows used for the heatmap and grouped bar chart.
    - ``heatmap``: ``{vulnerabilities, agents, z_matrix}`` ready for direct table
      rendering (row-major, rows=vulnerabilities, cols=agents).
    """
    agents = list(report.tested_agents)

    # Group results by agent key
    agent_results: dict[str, list[RedTeamResult]] = {a: [] for a in agents}
    warned_agents: set[str] = set()
    for r in report.results:
        key = r.agent.key or r.agent.display_name or "unknown"
        display = r.agent.display_name or r.agent.key or "unknown"
        if key in agent_results:
            agent_results[key].append(r)
        elif display in agent_results:
            agent_results[display].append(r)
        else:
            if key not in warned_agents:
                logger.warning("Result agent {!r} (display={!r}) not in tested_agents, skipping", key, display)
                warned_agents.add(key)

    # Per-agent top-level metrics
    agent_metrics: list[dict[str, Any]] = []
    for agent_name in agents:
        ar = agent_results[agent_name]
        total = len(ar)
        vulns = sum(1 for r in ar if r.vulnerable)
        asr = vulns / total if total > 0 else 0.0
        agent_metrics.append(
            {
                "agent": agent_name,
                "total_attacks": total,
                "vulnerabilities_found": vulns,
                "asr": asr,
            }
        )

    # Per-vulnerability × agent ASR pivot  # noqa: RUF003
    # pivot[vuln][agent] = {"total": int, "vuln": int}
    pivot: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {a: {"total": 0, "vuln": 0} for a in agents}
    )
    for agent_name in agents:
        for r in agent_results[agent_name]:
            vuln = r.attack.vulnerability or r.attack.category or "unknown"
            pivot[vuln][agent_name]["total"] += 1
            if r.vulnerable:
                pivot[vuln][agent_name]["vuln"] += 1

    # Build flat rows sorted by average ASR descending
    vuln_asr_rows: list[dict[str, Any]] = []
    for vuln, agent_counts in pivot.items():
        per_agent: dict[str, dict[str, Any]] = {}
        total_asr = 0.0
        for agent_name in agents:
            counts = agent_counts.get(agent_name, {"total": 0, "vuln": 0})
            asr = counts["vuln"] / counts["total"] if counts["total"] > 0 else 0.0
            per_agent[agent_name] = {"asr": asr, "total": counts["total"]}
            total_asr += asr
        avg_asr = total_asr / len(agents) if agents else 0.0
        vuln_asr_rows.append({"vulnerability": vuln, "agents": per_agent, "avg_asr": avg_asr})
    vuln_asr_rows.sort(key=operator.itemgetter("avg_asr"), reverse=True)

    # Build heatmap matrix (row=vulnerability, col=agent)
    sorted_vulns = [r["vulnerability"] for r in vuln_asr_rows]
    z_matrix: list[list[float]] = []
    text_matrix: list[list[str]] = []
    for row in vuln_asr_rows:
        z_row: list[float] = []
        t_row: list[str] = []
        for agent_name in agents:
            agent_data = row["agents"].get(agent_name, {"asr": 0.0, "total": 0})
            asr_pct = agent_data["asr"] * 100
            z_row.append(round(asr_pct, 1))
            t_row.append(f"{asr_pct:.0f}% (n={agent_data['total']})")
        z_matrix.append(z_row)
        text_matrix.append(t_row)

    return ReportSection(
        kind="agent_comparison",
        title="Multi-Agent Comparison",
        data={
            "agents": agents,
            "agent_metrics": agent_metrics,
            "vuln_asr_rows": vuln_asr_rows,
            "heatmap": {
                "vulnerabilities": sorted_vulns,
                "agents": agents,
                "z_matrix": z_matrix,
                "text_matrix": text_matrix,
            },
        },
    )


# ---------------------------------------------------------------------------
# Agent disagreements section
# ---------------------------------------------------------------------------


def _build_agent_disagreements_section(report: RedTeamReport) -> ReportSection:
    """Build per-attack disagreement data between agents.

    Only called when ``len(report.tested_agents) >= 2``.  For every attack ID
    where at least two agents produced *different* verdicts, one entry is
    emitted containing the attack metadata and per-agent verdict + response.
    """
    agents = list(report.tested_agents)

    # Map: attack_id -> agent_key -> RedTeamResult
    attack_map: dict[str, dict[str, RedTeamResult]] = defaultdict(dict)
    for r in report.results:
        agent_key = r.agent.key or r.agent.display_name or "unknown"
        attack_map[r.attack.id][agent_key] = r

    disagreements: list[dict[str, Any]] = []
    for attack_id, agent_result_map in attack_map.items():
        # Only consider attacks where at least 2 of the tested agents have results
        agent_keys_present = [a for a in agents if a in agent_result_map]
        if len(agent_keys_present) < 2:
            continue

        # Check for differing verdicts among tested agents present
        verdicts = [agent_result_map[a].vulnerable for a in agent_keys_present]
        if len(set(verdicts)) == 1:
            # All agree — not a disagreement
            continue

        # Use the first available result for attack-level metadata
        first_result = agent_result_map[agent_keys_present[0]]
        per_agent: list[dict[str, Any]] = []
        for a in agent_keys_present:
            r = agent_result_map[a]
            per_agent.append(
                {
                    "agent": a,
                    "vulnerable": r.vulnerable,
                    "response_snippet": extract_response(r)[:500],
                    "explanation": r.evaluation.explanation if r.evaluation else "",
                }
            )

        disagreements.append(
            {
                "attack_id": attack_id,
                "vulnerability": first_result.attack.vulnerability or first_result.attack.category or "unknown",
                "technique": first_result.attack.attack_technique.value,
                "severity": first_result.attack.severity.value,
                "prompt_snippet": extract_prompt(first_result)[:300],
                "per_agent": per_agent,
            }
        )

    # Sort so highest-severity disagreements appear first
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    disagreements.sort(key=lambda d: severity_rank.get(d["severity"], 4))

    return ReportSection(
        kind="agent_disagreements",
        title="Agent Disagreements",
        data={
            "agents": agents,
            "total_disagreements": len(disagreements),
            "disagreements": disagreements,
        },
    )


# ---------------------------------------------------------------------------
# Framework breakdown section
# ---------------------------------------------------------------------------


def _build_framework_breakdown_section(report: RedTeamReport) -> ReportSection:
    """Build per-framework breakdown rows sorted by vulnerability rate (worst first).

    Only called when ``len(report.summary.by_framework) > 1``.
    """
    rows: list[dict[str, Any]] = []
    for framework_name, fw_summary in report.summary.by_framework.items():
        total = fw_summary.total_attacks
        found = fw_summary.vulnerabilities_found
        rows.append(
            {
                "framework": framework_name,
                "total_attacks": total,
                "vulnerabilities_found": found,
                "vulnerability_rate": fw_summary.vulnerability_rate,
                "resistance_rate": fw_summary.resistance_rate,
            }
        )
    rows.sort(key=operator.itemgetter("vulnerability_rate"), reverse=True)
    return ReportSection(
        kind="framework_breakdown",
        title="Framework Breakdown",
        data={"rows": rows},
    )


# ---------------------------------------------------------------------------
# Phase 2 section builders
# ---------------------------------------------------------------------------


def _build_agent_context_section(report: RedTeamReport) -> ReportSection | None:
    """Build agent context capability cards.

    Prefers ``report.agent_contexts`` (dict keyed by agent key) over
    ``report.agent_context`` (single context).  Falls back to building
    minimal stubs from ``report.tested_agents`` when no AgentContext objects
    are available at all.

    Returns ``None`` when there is no agent information to display.
    """
    agents: list[dict[str, Any]] = []

    if report.agent_contexts:
        for agent_key, ctx in report.agent_contexts.items():
            agents.append({
                "key": agent_key,
                "display_name": ctx.display_name or agent_key,
                "model": ctx.model or "",
                "description": ctx.description or "",
                "tools": [t.name for t in ctx.tools],
                "memory_stores": [ms.key or ms.id for ms in ctx.memory_stores],
                "knowledge_bases": [kb.name or kb.key or kb.id for kb in ctx.knowledge_bases],
            })
    elif report.agent_context:
        ctx = report.agent_context
        agents.append({
            "key": ctx.key,
            "display_name": ctx.display_name or ctx.key,
            "model": ctx.model or "",
            "description": ctx.description or "",
            "tools": [t.name for t in ctx.tools],
            "memory_stores": [ms.key or ms.id for ms in ctx.memory_stores],
            "knowledge_bases": [kb.name or kb.key or kb.id for kb in ctx.knowledge_bases],
        })
    elif report.tested_agents:
        # Minimal stubs — no detailed AgentContext available
        agents.extend({
                "key": agent_key,
                "display_name": agent_key,
                "model": "",
                "description": "",
                "tools": [],
                "memory_stores": [],
                "knowledge_bases": [],
            } for agent_key in sorted(set(report.tested_agents)))

    if not agents:
        return None

    return ReportSection(
        kind="agent_context",
        title="Tested Agents",
        data={"agents": agents},
    )


def _build_turn_domain_breakdown_section(report: RedTeamReport) -> ReportSection | None:
    """Build turn-type and vulnerability-domain breakdown statistics.

    Returns ``None`` when neither ``by_turn_type`` nor ``by_domain`` contain
    any data.
    """
    s = report.summary

    def _obj_to_dict(obj: Any) -> dict[str, Any]:
        return {
            "total_attacks": getattr(obj, "total_attacks", 0),
            "vulnerabilities_found": getattr(obj, "vulnerabilities_found", 0),
            "vulnerability_rate": getattr(obj, "vulnerability_rate", 0.0),
            "resistance_rate": getattr(obj, "resistance_rate", 1.0),
        }

    by_turn_type = {k: _obj_to_dict(v) for k, v in s.by_turn_type.items()}
    by_domain = {k: _obj_to_dict(v) for k, v in s.by_domain.items()}

    if not by_turn_type and not by_domain:
        return None

    return ReportSection(
        kind="turn_scope_breakdown",
        title="Turn Type & Domain Breakdown",
        data={
            "by_turn_type": by_turn_type,
            "by_domain": by_domain,
        },
    )


def _build_turn_depth_analysis_section(report: RedTeamReport) -> ReportSection | None:
    """Build multi-turn depth analysis: ASR% grouped by turn count.

    Only includes results where ``result.execution.turns > 1``.
    Returns ``None`` when there are no multi-turn results.
    """
    depth_counts: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "vulnerable": 0})

    for result in report.results:
        if result.execution is None:
            continue
        turns = result.execution.turns
        if turns <= 1:
            continue
        depth_counts[turns]["total"] += 1
        if result.vulnerable:
            depth_counts[turns]["vulnerable"] += 1

    if not depth_counts:
        return None

    rows: list[dict[str, Any]] = []
    for turn_count in sorted(depth_counts):
        entry = depth_counts[turn_count]
        total = entry["total"]
        found = entry["vulnerable"]
        rows.append({
            "turn_count": turn_count,
            "total_attacks": total,
            "vulnerabilities_found": found,
            "vulnerability_rate": found / total if total > 0 else 0.0,
        })

    return ReportSection(
        kind="turn_depth_analysis",
        title="Conversation Depth Analysis (Multi-Turn)",
        data={"rows": rows},
    )


def _build_token_usage_section(report: RedTeamReport) -> ReportSection | None:
    """Build token usage summary: overall totals and per-agent breakdown.

    Returns ``None`` when no token usage data is present in the report.
    """
    # Prefer the canonical summary field; fall back to the legacy field
    overall = report.summary.token_usage_total or report.token_usage_summary

    if overall is None:
        has_any = any(
            r.execution is not None and r.execution.token_usage is not None
            for r in report.results
        )
        if not has_any:
            return None

    overall_dict: dict[str, Any] = {}
    if overall is not None:
        overall_dict = {
            "total_tokens": overall.total_tokens,
            "prompt_tokens": overall.prompt_tokens,
            "completion_tokens": overall.completion_tokens,
            "calls": overall.calls,
        }

    agent_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    )
    for result in report.results:
        if result.execution is None or result.execution.token_usage is None:
            continue
        agent_key = result.agent.key or result.agent.display_name or "unknown"
        tu = result.execution.token_usage
        agent_totals[agent_key]["total_tokens"] += tu.total_tokens
        agent_totals[agent_key]["prompt_tokens"] += tu.prompt_tokens
        agent_totals[agent_key]["completion_tokens"] += tu.completion_tokens
        agent_totals[agent_key]["calls"] += tu.calls

    per_agent = [
        {"agent": agent_key, **totals}
        for agent_key, totals in sorted(
            agent_totals.items(),
            key=lambda kv: kv[1]["total_tokens"],
            reverse=True,
        )
    ]

    return ReportSection(
        kind="token_usage",
        title="Token Usage",
        data={
            "overall": overall_dict,
            "per_agent": per_agent,
        },
    )


def _build_source_distribution_section(report: RedTeamReport) -> ReportSection | None:
    """Build attack source distribution counts.

    Returns ``None`` when there are fewer than two distinct sources (a single
    source gives no distribution to visualise).
    """
    source_counts: Counter[str] = Counter(r.attack.source for r in report.results)

    if len(source_counts) < 2:
        return None

    rows = [
        {"source": source, "count": count}
        for source, count in source_counts.most_common()
    ]

    return ReportSection(
        kind="source_distribution",
        title="Attack Source Distribution",
        data={"rows": rows},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_severity_definitions_section() -> ReportSection:
    """Build the severity level definitions section."""
    return ReportSection(
        kind="severity_definitions",
        title="Severity Levels",
        data={
            "definitions": [
                {"level": level, "description": description}
                for level, description in SEVERITY_DEFINITIONS.items()
            ],
            "weights": dict(SEVERITY_WEIGHTS),
        },
    )


def _build_methodology_section(report: RedTeamReport) -> ReportSection | None:
    """Build methodology disclosure section from report metadata."""
    data: dict[str, Any] = {
        "pipeline": report.pipeline.value,
        "categories_tested": report.categories_tested,
        "scoring_method": "llm-as-judge",
    }

    # Add computed metadata
    data["total_attacks"] = report.summary.total_attacks
    data["evaluation_coverage"] = report.summary.evaluation_coverage
    data["tested_agents"] = report.tested_agents
    data["framework"] = report.framework.value if report.framework else None
    data["duration_seconds"] = report.duration_seconds

    # Compute untested categories
    all_supported = {k for k in OWASP_CATEGORY_NAMES if not k.startswith("OWASP-")}
    tested = set(data.get("categories_tested") or report.categories_tested)
    untested = sorted(all_supported - tested)
    if untested:
        data["untested_categories"] = untested
        data["untested_category_names"] = {cat: OWASP_CATEGORY_NAMES.get(cat, cat) for cat in untested}

    return ReportSection(
        kind="methodology",
        title="Methodology",
        data=data,
    )


def build_report_sections(report: RedTeamReport) -> list[ReportSection]:
    """Convert a ``RedTeamReport`` into renderer-agnostic ``ReportSection`` objects.

    Returns sections in document order optimised for executive readability:
        1.  summary
        2.  methodology
        3.  agent_context            (when agent info is available)
        4.  focus_areas
        5.  agent_comparison         (only when >= 2 agents — key decision info)
        6.  agent_disagreements      (only when >= 2 agents)
        7.  vulnerability_breakdown  (primary breakdown)
        8.  category_breakdown
        9.  attack_heatmap           (vulnerability x technique grid)
        10. technique_breakdown
        11. delivery_breakdown       (when data present)
        12. turn_scope_breakdown     (when turn-type/domain data present)
        13. turn_depth_analysis      (multi-turn results only)
        14. error_analysis           (only when errors exist)
        15. framework_breakdown      (only when > 1 framework)
        16. individual_results
        17. source_distribution      (when >= 2 distinct sources)
        18. token_usage              (when token data present)
        19. severity_definitions     (reference / appendix)
    """
    sections: list[ReportSection] = [
        _build_summary_section(report),
    ]

    # Methodology — how the assessment was conducted
    methodology_section = _build_methodology_section(report)
    if methodology_section is not None:
        sections.append(methodology_section)

    # Agent context near the top — who/what was tested
    agent_ctx_section = _build_agent_context_section(report)
    if agent_ctx_section is not None:
        sections.append(agent_ctx_section)

    # Focus areas — top risks
    sections.append(_build_focus_areas_section(report))

    # Multi-agent comparison early — key decision info for CISO
    if len(report.tested_agents) >= 2:
        sections.extend((_build_agent_comparison_section(report), _build_agent_disagreements_section(report)))

    # Detailed breakdowns
    sections.extend((_build_vulnerability_breakdown_section(report), _build_category_breakdown_section(report)))

    if report.results:
        sections.append(_build_attack_heatmap_section(report))

    sections.append(_build_technique_breakdown_section(report))

    if report.summary.by_delivery_method:
        sections.append(_build_delivery_breakdown_section(report))

    turn_scope_section = _build_turn_domain_breakdown_section(report)
    if turn_scope_section is not None:
        sections.append(turn_scope_section)

    turn_depth_section = _build_turn_depth_analysis_section(report)
    if turn_depth_section is not None:
        sections.append(turn_depth_section)

    # Errors
    if report.summary.total_errors > 0:
        sections.append(_build_error_analysis_section(report))

    if len(report.summary.by_framework) > 1:
        sections.append(_build_framework_breakdown_section(report))

    # Individual results (evidence)
    sections.append(_build_individual_results_section(report))

    # Operational / appendix sections last
    source_dist_section = _build_source_distribution_section(report)
    if source_dist_section is not None:
        sections.append(source_dist_section)

    token_usage_section = _build_token_usage_section(report)
    if token_usage_section is not None:
        sections.append(token_usage_section)

    # Severity definitions as reference/appendix at the end
    sections.append(_build_severity_definitions_section())

    return sections
