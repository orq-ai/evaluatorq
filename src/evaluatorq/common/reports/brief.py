"""Compact HTML briefs derived from saved report data."""

from __future__ import annotations

import re
from operator import itemgetter
from typing import TYPE_CHECKING, Any

from evaluatorq.common.reports.html_helpers import esc
from evaluatorq.common.reports.palette import COLORS, STATUS_COLORS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from evaluatorq.contracts import ReportSection
    from evaluatorq.redteam.contracts import RedTeamReport
    from evaluatorq.simulation.contracts import SimulationRun


def _section(by_kind: Mapping[str, ReportSection], kind: str) -> dict[str, Any]:
    section = by_kind.get(kind)
    return section.data if section is not None else {}


def _table(rows: list[tuple[str, str, str, str, str, str | None]]) -> str:
    if not rows:
        return '<p class="export-reading-note">No confirmed findings are available in the saved report sections.</p>'
    rendered_rows: list[str] = []
    for index, (finding, impact, affected, fix, anchor, full_link) in enumerate(rows, start=1):
        full_link_html = f' <a href="#{esc(full_link)}">Full recommendation</a>' if full_link else ''
        rendered_rows.append(
            '<tr>'
            f'<th scope="row" data-label="Finding" id="export-finding-{index}"><span class="export-finding-value">{index}. {esc(finding)}</span></th>'
            f'<td data-label="Impact"><span class="export-finding-value">{esc(impact)}</span></td>'
            f'<td data-label="Affected"><span class="export-finding-value">{esc(affected)}</span></td>'
            f'<td data-label="Recommended next step"><span class="export-finding-value">{esc(fix)}{full_link_html}</span></td>'
            f'<td data-label="Evidence"><span class="export-finding-value"><a href="{esc(anchor)}">Review evidence</a></span></td>'
            '</tr>'
        )
    body = ''.join(rendered_rows)
    return (
        '<div class="export-findings"><table class="no-stack"><thead><tr><th>Finding</th><th>Impact</th>'
        '<th>Affected</th><th>Recommended next step</th><th>Evidence</th></tr></thead>'
        f'<tbody>{body}</tbody></table></div>'
    )


def _recommendation_excerpt(text: str, *, limit: int = 180) -> str:
    """Keep the brief compact while making a copied recommendation's excerpt clear."""
    first_sentence = re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)[0]
    if first_sentence and len(first_sentence) <= limit:
        return first_sentence
    if len(text) <= limit:
        return text
    return f'{text[:limit].rstrip()}… (excerpt; full recommendation linked below)'


def _finding_rows_sim(
    run: SimulationRun, by_kind: Mapping[str, ReportSection]
) -> list[tuple[str, str, str, str, str, str | None]]:
    entries = _section(by_kind, 'individual_results').get('entries', [])
    recommendations = _section(by_kind, 'recommendations').get('rows', [])
    rec_by_index: dict[int, list[str]] = {
        row.get('index') - 1: row.get('suggestions') or []
        for row in recommendations
        if isinstance(row, dict) and isinstance(row.get('index'), int)
    }
    # Dashboard tabs do not build the recommendations section. The run itself
    # owns these saved suggestions, keyed by the original result position.
    for recommendation in getattr(run, 'recommendations', None) or []:
        rec_by_index[recommendation.result_index] = recommendation.suggestions
    ranked: list[tuple[int, int, tuple[str, str, str, str, str, str | None]]] = []
    for entry in entries:
        index = entry.get('index')
        if not isinstance(index, int):
            continue
        suggestions = rec_by_index.get(index, [])
        failed = [criterion for criterion in entry.get('criteria', []) if criterion.get('state') == 'fail']
        failed.sort(key=lambda criterion: criterion.get('type') != 'must_not_happen')
        execution_error = bool(entry.get('error')) or entry.get('terminated_by') == 'error'
        missed_goal = entry.get('goal_achieved') is False and not execution_error
        if not failed and not suggestions and not missed_goal and not execution_error:
            continue
        criterion = failed[0] if failed else {}
        description = criterion.get('description') or 'Goal was not achieved'
        if failed:
            impact = (
                'A prohibited behavior was confirmed.'
                if criterion.get('type') == 'must_not_happen'
                else 'A criterion was confirmed violated.'
            )
            rank = 0 if criterion.get('type') == 'must_not_happen' else 1
        elif missed_goal:
            impact = 'The conversation did not achieve its goal.'
            rank = 1
        elif suggestions and not execution_error:
            description = 'Recorded improvement suggestion'
            impact = 'The goal was achieved; the saved recommendation suggests a possible improvement.'
            rank = 1
        else:
            description = 'Execution did not complete'
            impact = 'The conversation ended before a complete outcome was available.'
            rank = 2
        recommendation = next((str(s) for s in suggestions if s), None)
        fix = (
            f'Recorded recommendation excerpt: {_recommendation_excerpt(recommendation)}'
            if recommendation
            else 'No recommendation recorded; review the linked evidence.'
        )
        affected = f'{entry.get("persona") or "Persona"} · {entry.get("scenario") or "Scenario"}'
        full_link = (
            'simtab-recommendations'
            if recommendation and (recommendations or getattr(run, 'recommendations', None))
            else None
        )
        ranked.append((rank, index, (str(description), impact, affected, fix, f'#conv-{index + 1}', full_link)))
    ranked.sort(key=itemgetter(0, 1))
    return [row for _, _, row in ranked]


def _finding_rows_redteam(
    report: RedTeamReport, by_kind: Mapping[str, ReportSection]
) -> list[tuple[str, str, str, str, str, str | None]]:
    section = _section(by_kind, 'focus_areas')
    focus = section.get('ranked_focus_areas', section.get('focus_areas', []))
    rows: list[tuple[str, str, str, str, str, str | None]] = []
    for area in focus:
        category = str(area.get('category') or '')
        recommendation = area.get('llm_recommendations') or {}
        saved = recommendation.get('recommendations') or []
        recorded = next((str(item) for item in saved if item), None)
        framework_guidance = area.get('remediation')
        if recorded:
            fix = f'Recorded recommendation excerpt: {_recommendation_excerpt(recorded)}'
            full_link = 'rttab-focus-areas'
        elif framework_guidance:
            fix = f'Framework guidance excerpt: {_recommendation_excerpt(str(framework_guidance))}'
            full_link = 'rttab-focus-areas'
        else:
            fix = 'No recommendation recorded; review the linked evidence.'
            full_link = None
        matching_results = [
            (i, result)
            for i, result in enumerate(report.results, start=1)
            if result.attack.category == category and result.vulnerable is True
        ]
        result_index = next(
            (i for i, result in matching_results),
            None,
        )
        if result_index is None:
            continue
        agents = sorted({
            result.agent.display_name or result.agent.key or 'Unknown agent' for _, result in matching_results
        })
        rows.append((
            str(area.get('category_name') or category),
            f'{area.get("vulnerabilities_found", 0)} confirmed vulnerable attacks in this category.',
            ', '.join(agents) + ' · ' + str(area.get('vulnerability_name') or category),
            fix,
            f'#attack-{result_index}',
            full_link,
        ))
    return rows


def _brief(
    title: str,
    summary: list[str],
    rows: list[tuple[str, str, str, str, str, str | None]],
    total_findings: int,
    chart: str,
    links: list[tuple[str, str]],
) -> str:
    stats = ''.join(f'<li>{esc(item)}</li>' for item in summary)
    nav = ''.join(f'<a href="#{esc(target)}">{esc(label)}</a>' for target, label in links)
    shown = len(rows)
    count = f'Findings shown: {shown} of {total_findings}.'
    return (
        '<section id="export-brief" class="export-brief" aria-labelledby="export-brief-title">'
        f'<h2 id="export-brief-title">{esc(title)}</h2>'
        f'<div class="export-brief-overview"><ul class="export-takeaway">{stats}</ul>{chart}</div>'
        f'<p class="export-reading-note">{esc(count)} Follow the evidence links to inspect the saved conversation or attack.</p>'
        f'{_table(rows)}<nav aria-label="Full report sections">{nav}</nav></section>'
    )


def simulation_brief(run: SimulationRun, by_kind: Mapping[str, ReportSection]) -> str:
    """Render a compact simulation brief from the run and precomputed sections."""
    summary = _section(by_kind, 'summary')
    total = summary.get('total_conversations', run.total_results)
    achieved = summary.get('goals_achieved', 0)
    failed = summary.get('goals_failed', 0)
    errors = summary.get('errors', 0)
    verdicts = max(0, int(total) - int(errors))
    all_findings = _finding_rows_sim(run, by_kind)
    findings = all_findings[:3]
    from evaluatorq.dashboard.report_kit import donut

    chart = ''
    if total:
        chart = (
            '<figure class="chart-card"><figcaption>Conversation outcomes</figcaption>'
            + donut(
                segments=[
                    {'label': 'Goal achieved', 'value': achieved, 'color': COLORS['success_400']},
                    {'label': 'Goal missed', 'value': failed, 'color': COLORS['red_400']},
                    {'label': 'Execution error', 'value': errors, 'color': COLORS['yellow_400']},
                ],
                center_value=str(total),
                center_label='conversations',
            )
            + '</figure>'
        )
    links = []
    if 'individual_results' in by_kind:
        links.append(('simtab-breakdown', 'All conversations'))
    if 'recommendations' in by_kind or getattr(run, 'recommendations', None):
        links.append(('simtab-recommendations', 'All saved recommendations'))
    target = getattr(run, 'target', None) or getattr(run, 'target_kind', 'agent simulation').replace('_', ' ')
    target_model = getattr(run, 'target_model', None)
    model = f' using {target_model}' if target_model else ''
    coverage = f'{verdicts}/{total} conversations have a completed outcome; {errors} ended in execution error.'
    if total == 0:
        takeaway = 'No conversations were tested.'
    elif verdicts == 0:
        takeaway = 'No goal verdict is available because every conversation ended in execution error.'
    else:
        takeaway = f'{achieved}/{verdicts} completed goals achieved; {failed} completed goals missed.'
    return _brief(
        'Simulation report brief',
        [f'Tested: {target}{model}; {total} conversations.', takeaway, coverage],
        findings,
        len(all_findings),
        chart,
        links,
    )


def redteam_brief(report: RedTeamReport, by_kind: Mapping[str, ReportSection]) -> str:
    """Render a compact red-team brief from the report and precomputed sections."""
    summary = _section(by_kind, 'summary')
    total = summary.get('total_attacks', report.total_results)
    evaluated = summary.get('evaluated_attacks', 0)
    vulnerable = summary.get('vulnerabilities_found', 0)
    errors = summary.get('total_errors', 0)
    resistant = max(0, int(evaluated) - int(vulnerable))
    all_findings = _finding_rows_redteam(report, by_kind)
    findings = all_findings[:3]
    from evaluatorq.dashboard.report_kit import donut

    no_verdict = max(0, int(total) - int(evaluated))
    chart = ''
    if total:
        chart = (
            '<figure class="chart-card"><figcaption>Attack outcomes</figcaption>'
            + donut(
                segments=[
                    {'label': 'Resisted', 'value': resistant, 'color': STATUS_COLORS['resistant']},
                    {'label': 'Vulnerable', 'value': vulnerable, 'color': STATUS_COLORS['vulnerable']},
                    {'label': 'No verdict', 'value': no_verdict, 'color': STATUS_COLORS['error']},
                ],
                center_value=str(total),
                center_label='attacks',
            )
            + '</figure>'
        )
    links = []
    if 'focus_areas' in by_kind:
        links.append(('rttab-focus-areas', 'Focus areas'))
    if 'individual_results' in by_kind:
        links.append(('rttab-attacks', 'All attacks'))
    agents = ', '.join(getattr(report, 'tested_agents', [])) or 'agents in saved results'
    category_count = len(report.categories_tested)
    categories = f'{category_count} tested categor{"y" if category_count == 1 else "ies"}'
    if total == 0:
        takeaway = 'No attacks were tested.'
    elif evaluated == 0:
        takeaway = 'No verdict is available for the tested attacks.'
    else:
        takeaway = (
            f'Confirmed vulnerable attacks: {vulnerable}; resistant attacks: {resistant} among evaluated attacks.'
        )
    coverage = f'{evaluated}/{total} attacks received a verdict across {categories}; {no_verdict} have no verdict; errors: {errors}.'
    return _brief(
        'Red-team report brief',
        [f'Tested: {agents} ({getattr(report.pipeline, "value", report.pipeline)} pipeline).', takeaway, coverage],
        findings,
        len(all_findings),
        chart,
        links,
    )
