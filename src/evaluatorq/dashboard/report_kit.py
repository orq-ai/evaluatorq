"""Shared design-component primitives for the aligned dashboard report pages.

Small pure functions emitting HTML strings, ``esc()`` on all dynamic text, no
state, no vl-convert. Built to serve both the Agent Sim report (this task) and
the Red Team report alignment that follows. Exact styling values live in the
matching spec (docs/superpowers/specs/2026-07-10-agent-sim-report-alignment-design.md)
and in the ``.sim-report`` CSS block in ``styles.py``.
"""

from __future__ import annotations

from operator import itemgetter
from typing import Any

from evaluatorq.common.reports import esc
from evaluatorq.common.reports.html_helpers import pct


def _best_worst_cells(heatmap_data: dict[str, Any]) -> tuple[dict | None, dict | None]:
    """Best/worst persona x scenario cells by success_rate, excluding cells with
    no scored (non-errored) conversations (``n == 0``)."""
    scored = [c for c in heatmap_data.get('cells', []) if c.get('n', 0) > 0]
    if not scored:
        return None, None
    best = max(scored, key=itemgetter('success_rate'))
    worst = min(scored, key=itemgetter('success_rate'))
    return best, worst


def _confidence_pill(confidence: str | None) -> str:
    if not confidence:
        return ''
    tone = {'HIGH': 'green-600', 'MEDIUM': 'amber-600', 'LOW': 'red-600'}.get(confidence.upper(), 'teal-600')
    return (
        f'<span class="es-confidence" style="border-color:var(--{tone});color:var(--{tone})">'
        f'{esc(confidence.upper())} CONFIDENCE</span>'
    )


def exec_summary(*, summary_data: dict[str, Any], heatmap_data: dict[str, Any], confidence: str | None) -> str:
    """Orange-left-bar executive-summary callout (spec §Overview.1)."""
    total = summary_data.get('total_conversations', 0)
    if not total:
        return ''
    personas = heatmap_data.get('personas', [])
    scenarios = heatmap_data.get('scenarios', [])
    success = pct(summary_data.get('success_rate', 0.0))
    avg_score = f'{summary_data.get("avg_goal_completion_score", 0.0):.2f}'

    sentence = (
        f'Across <strong>{total}</strong> conversations '
        f'({len(personas)} personas &times; {len(scenarios)} scenarios), the agent achieved a '
        f'<strong>{success}</strong> goal-completion rate at an average score of <strong>{avg_score}</strong>.'
    )
    best, worst = _best_worst_cells(heatmap_data)
    if best is not None and worst is not None and best is not worst:
        sentence += (
            f' It performed best on <strong>{esc(best["persona"])} &times; {esc(best["scenario"])}</strong> '
            f'({pct(best["success_rate"])}) and weakest on '
            f'<strong>{esc(worst["persona"])} &times; {esc(worst["scenario"])}</strong> ({pct(worst["success_rate"])}).'
        )

    return (
        '<div class="exec-summary">'
        '<div class="es-head">'
        '<span class="es-label">Executive summary</span>'
        f'{_confidence_pill(confidence)}'
        '</div>'
        f'<p class="es-body">{sentence}</p>'
        '</div>'
    )
