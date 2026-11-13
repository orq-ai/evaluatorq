from __future__ import annotations

import pytest

from evaluatorq import build_report, llm_jury_pairwise


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pairwise_panel_does_not_choose_the_wrong_answer() -> None:
    """A three-judge panel does not prefer a plainly false answer to the correct one.

    A tie is a valid outcome because live judges can disagree. A B win would
    mean panel consensus favored the answer that gets both facts wrong.
    """
    comparator = llm_jury_pairwise(
        judges=['openai/gpt-5.4-mini', 'anthropic/claude-haiku-4-5', 'openai/gpt-4o-mini'],
    )

    comparison = await comparator.compare(
        question='What is the capital of France?',
        response_a='The capital of France is Paris.',
        response_b='The capital of France is Berlin, and Paris is in Germany.',
    )

    assert comparison.winner in {'A', 'tie', 'inconclusive'}
    assert len(comparison.votes) == 3
    assert sum(vote.vote is not None for vote in comparison.votes) >= 2

    report = build_report([comparison])
    assert report.comparisons == 1
    assert report.a_win_rate == (1.0 if comparison.winner == 'A' else None)
    assert report.inconclusive_rate == (1.0 if comparison.winner == 'inconclusive' else 0.0)
