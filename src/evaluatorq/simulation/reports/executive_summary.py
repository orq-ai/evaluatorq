"""Build the compact ``facts`` string for the simulation executive summary.

Summarizes goal-completion rate and the dominant failure mode (most frequently
broken rule) plus one concrete example, for the shared narrative generator.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from evaluatorq.common.reports.executive_summary import truncate_text

if TYPE_CHECKING:
    from evaluatorq.simulation.types import SimulationResult


def build_sim_facts(results: list[SimulationResult]) -> str:
    """Return a plain-text facts block, or '' when there are no results."""
    total = len(results)
    if total == 0:
        return ''

    achieved = sum(1 for r in results if r.goal_achieved)
    rate = achieved / total if total else 0.0

    lines: list[str] = [
        f'Total simulations: {total}',
        f'Goals achieved: {achieved} ({rate:.0%})',
        f'Goals failed: {total - achieved}',
    ]

    broken = Counter(rule for r in results for rule in r.rules_broken)
    if broken:
        top_rule, top_count = broken.most_common(1)[0]
        lines.extend([
            f'Most-broken rule: {top_rule} (broken in {top_count} simulation(s))',
            'Rules broken (by frequency): ' + ', '.join(f'{rule}: {count}' for rule, count in broken.most_common()),
        ])
        example = next(
            (r for r in results if top_rule in r.rules_broken and not r.goal_achieved),
            None,
        )
        if example is not None and example.reason:
            lines.append(f'Example failure: {truncate_text(example.reason)}')

    return '\n'.join(lines)
