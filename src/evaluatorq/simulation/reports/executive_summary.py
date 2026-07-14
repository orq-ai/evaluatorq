"""Build the compact ``facts`` string for the simulation executive summary.

Summarizes goal-completion rate and the dominant failure mode (most frequently
broken rule) plus one concrete example, for the shared narrative generator.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from evaluatorq.common.reports.executive_summary import truncate_text

SIM_EXECUTIVE_SUMMARY_SYSTEM_PROMPT = """\
You are writing the executive summary of an agent simulation report for a
technical but time-poor reader (an engineering lead or product owner). The
report measures whether an agent completes its goal while following its rules.

Write ONE paragraph, 2-4 sentences, following this arc:
  1. Scope: how many simulations were run.
  2. Verdict with tension: the headline goal-completion rate AND the failure
     count and the most-broken rule in the same breath. Use "but"/"however" so
     the failures are not buried.
  3. The single sharpest concrete finding: name the one failure that matters
     most, described as what the agent actually DID ("promised a refund it had
     no authority to issue"), not the rule label.
  4. The dominant failure pattern and its implication, grounded ONLY in the
     provided numbers (e.g. if one rule is broken far more than others, say
     that behaviour is the weak point).

Rules:
- Lead with the numbers you are GIVEN. Never invent a statistic or a trend the
  facts do not state.
- Concrete over abstract: describe behaviour, not rule codes.
- No preamble ("This report..."), no recommendations, no markdown, no headers.
  Prose only.
- If every goal was achieved and no rules were broken, say so plainly in one
  sentence and stop. Do not manufacture problems."""

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
