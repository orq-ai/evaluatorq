from __future__ import annotations

import pytest

from evaluatorq import build_report, llm_jury_pairwise


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pairwise_panel_of_two_judges_picks_the_better_answer() -> None:
    """A 2-judge panel comparing a clearly better answer against a wrong one favours the better one."""
    comparator = llm_jury_pairwise(
        judges=['openai/gpt-5.4-mini', 'anthropic/claude-haiku-4-5'],
    )

    comparison = await comparator.compare(
        question='What is the capital of France?',
        response_a='The capital of France is Paris.',
        response_b='The capital of France is Berlin.',
    )

    assert comparison.winner == 'A'
    assert len(comparison.votes) == 2

    report = build_report([comparison])
    assert report.comparisons == 1
    assert report.a_win_rate == 1.0
