"""Build the compact ``facts`` string fed to the executive-summary generator.

Pulls headline counts, the per-severity vulnerability breakdown, the single
highest-severity confirmed vulnerability (described by what the agent did), the
dominant attack technique, and the single-vs-multi-turn vulnerability split
(so the narrative's "dominant risk pattern" beat is grounded in real numbers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.reports.executive_summary import truncate_text
from evaluatorq.redteam.contracts import OWASP_CATEGORY_NAMES
from evaluatorq.redteam.reports._utils import extract_prompt, extract_response

if TYPE_CHECKING:
    from evaluatorq.redteam.contracts import RedTeamReport, RedTeamResult

_SEVERITY_RANK = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}


def _sharpest_finding(report: RedTeamReport) -> RedTeamResult | None:
    vulnerable = [r for r in report.results if r.vulnerable]
    if not vulnerable:
        return None
    return max(vulnerable, key=lambda r: _SEVERITY_RANK.get(r.attack.severity.value, 0))


def build_redteam_facts(report: RedTeamReport) -> str:
    """Return a plain-text facts block, or '' when there is nothing to summarize."""
    s = report.summary
    if s.total_attacks == 0:
        return ''

    lines: list[str] = [
        f'Total attacks: {s.total_attacks}',
        f'Categories tested: {len(report.categories_tested)}'
        + (f' ({", ".join(report.categories_tested)})' if report.categories_tested else ''),
        f'Resistance rate: {s.resistance_rate:.0%}',
        f'Vulnerabilities found: {s.vulnerabilities_found}',
    ]

    sev_bits = [
        f'{name}: {summ.vulnerabilities_found}' for name, summ in s.by_severity.items() if summ.vulnerabilities_found
    ]
    if sev_bits:
        lines.append('Vulnerabilities by severity: ' + ', '.join(sev_bits))

    finding = _sharpest_finding(report)
    if finding is not None:
        cat = finding.attack.category
        cat_name = OWASP_CATEGORY_NAMES.get(cat, cat)
        explanation = finding.evaluation.explanation if finding.evaluation else ''
        lines.extend([
            (
                'Most severe confirmed finding — '
                f'severity {finding.attack.severity.value}, category {cat} ({cat_name}), '
                f'technique {finding.attack.attack_technique.value}, '
                f'{finding.execution.turns if finding.execution else 1} turn(s).'
            ),
            f'  Attack prompt: {truncate_text(extract_prompt(finding))}',
            f'  Agent response: {truncate_text(extract_response(finding))}',
        ])
        if explanation:
            lines.append(f'  Evaluator rationale: {truncate_text(explanation)}')

    if s.by_technique:
        dominant = max(
            s.by_technique.items(),
            key=lambda kv: (kv[1].vulnerabilities_found, kv[1].total_attacks),
        )
        lines.append(
            f'Most-exploited technique: {dominant[0]} '
            f'({dominant[1].vulnerabilities_found} vuln / {dominant[1].total_attacks} attacks)'
        )

    # Beat-4 depth signal: single- vs multi-turn vulnerability rates.
    for turn_key in ('single', 'multi'):
        tt = s.by_turn_type.get(turn_key)
        if tt is not None and tt.total_attacks:
            lines.append(
                f'{turn_key.capitalize()}-turn attacks: {tt.total_attacks}, '
                f'vulnerability rate {tt.vulnerability_rate:.0%}'
            )

    return '\n'.join(lines)
