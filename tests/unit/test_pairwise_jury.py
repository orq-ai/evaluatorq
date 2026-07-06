from __future__ import annotations

import sys
from typing import Any

import pytest

from evaluatorq.common.judge import EvaluatorResponsePayload, JudgeOutcome
from evaluatorq.llm_jury import llm_jury_pairwise

# The package rebinds the name `evaluatorq.llm_jury` to the function, shadowing the
# submodule, so reach the real module (whose run_judge global we patch) via sys.modules.
llm_jury_module = sys.modules['evaluatorq.llm_jury']


@pytest.mark.asyncio
async def test_comparator_reconciles_a_consistent_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge that prefers the same actual response in both orderings makes that response win."""

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        first = kwargs['replacements']['response_a']
        value = 'A' if first == 'GOOD' else 'B'
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=value, explanation='x'))

    monkeypatch.setattr(llm_jury_module, 'run_judge', fake_run_judge)

    comparator = llm_jury_pairwise(judges=['judge-1'], client=object())
    result = await comparator.compare(question='Which is better?', response_a='GOOD', response_b='BAD')

    assert result.winner == 'A'
    assert result.votes[0].flipped is False


@pytest.mark.asyncio
async def test_comparator_runs_both_orderings(monkeypatch: pytest.MonkeyPatch) -> None:
    """With swap on, each judge is asked once per ordering (A-first, then B-first)."""
    seen_first: list[str] = []

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        seen_first.append(kwargs['replacements']['response_a'])
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='A', explanation='first slot'))

    monkeypatch.setattr(llm_jury_module, 'run_judge', fake_run_judge)

    comparator = llm_jury_pairwise(judges=['judge-1'], client=object())
    result = await comparator.compare(question='Q?', response_a='GOOD', response_b='BAD')

    assert seen_first == ['GOOD', 'BAD']
    # A pure first-slot judge flips once swapped, so it abstains and the panel is inconclusive.
    assert result.winner == 'inconclusive'


@pytest.mark.asyncio
async def test_swap_off_runs_one_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    """swap=False asks each judge exactly once and keeps its raw pick."""
    calls = 0

    async def fake_run_judge(**kwargs: object) -> JudgeOutcome:
        nonlocal calls
        calls += 1
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='A', explanation='x'))

    monkeypatch.setattr(llm_jury_module, 'run_judge', fake_run_judge)

    comparator = llm_jury_pairwise(judges=['judge-1'], client=object(), swap=False)
    result = await comparator.compare(question='Q?', response_a='GOOD', response_b='BAD')

    assert calls == 1
    assert result.winner == 'A'
