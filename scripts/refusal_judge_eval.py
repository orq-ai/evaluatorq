#!/usr/bin/env python3
"""Score the backtracking refusal judge against the labelled cases.

A case labelled ``refused: true`` is a bare refusal the orchestrator may erase.
Every other case must be kept, so a wrong ``true`` there is the costly error:
it deletes a reply the scorer needed to see.

Usage:
    uv run python scripts/refusal_judge_eval.py --model openai/gpt-6-luna
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

CASES = Path(__file__).resolve().parent.parent / 'tests' / 'redteam' / 'fixtures' / 'refusal_judge_cases.json'


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', required=True)
    parser.add_argument('--cases', type=Path, default=CASES)
    args = parser.parse_args()

    from evaluatorq.common.llm_client import resolve_llm_client
    from evaluatorq.contracts import AgentResponse, TextOutputItem
    from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator

    cases = json.loads(args.cases.read_text())
    resolved = resolve_llm_client()
    orchestrator = MultiTurnOrchestrator(llm_client=resolved.client, model=args.model)

    async def judge(case: dict) -> tuple[dict, bool | None]:
        reply = AgentResponse(output=[TextOutputItem(text=case['reply'], annotations=[])])
        verdict, _ = await orchestrator._is_refusal(case['user'], reply)
        return case, verdict

    results = await asyncio.gather(*(judge(case) for case in cases))
    wrong_erase = [c['id'] for c, v in results if v is True and not c['refused']]
    wrong_keep = [c['id'] for c, v in results if v is not True and c['refused']]
    no_verdict = [c['id'] for c, v in results if v is None]
    kept = sum(1 for c in cases if not c['refused'])

    print(f'model: {args.model}')
    print(f'wrongly erased: {len(wrong_erase)} of {kept} keep cases {wrong_erase}')
    print(f'wrongly kept: {len(wrong_keep)} of {len(cases) - kept} refusal cases {wrong_keep}')
    print(f'no verdict: {len(no_verdict)} {no_verdict}')
    if resolved.owned:
        await resolved.client.close()


if __name__ == '__main__':
    asyncio.run(main())
