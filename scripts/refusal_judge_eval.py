#!/usr/bin/env python3
"""Score the backtracking refusal judge against the labelled cases.

A case labelled ``refused: true`` is a bare refusal the orchestrator may erase.
Every other case must be kept, so a wrong ``true`` there is the costly error:
it deletes a reply the scorer needed to see. A case with no verdict counts in
neither error, and any such case makes the script exit non-zero.

Usage:
    uv run python scripts/refusal_judge_eval.py
    uv run python scripts/refusal_judge_eval.py --model openai/gpt-6-luna --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

CASES = Path(__file__).resolve().parent.parent / 'tests' / 'redteam' / 'fixtures' / 'refusal_judge_cases.json'


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', help='pin the redteam.refusal_judge task to this model')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--cases', type=Path, default=CASES)
    args = parser.parse_args()

    from evaluatorq.common.llm_client import resolve_llm_client
    from evaluatorq.common.model_roles import set_cli_models
    from evaluatorq.contracts import AgentResponse, TextOutputItem
    from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator, refusal_judge_model

    if args.model:
        set_cli_models(overrides={'redteam.refusal_judge': args.model})
    cases = json.loads(args.cases.read_text())
    resolved = resolve_llm_client()
    orchestrator = MultiTurnOrchestrator(llm_client=resolved.client)
    model, use_classify = await refusal_judge_model(resolved.client)

    async def judge(case: dict) -> tuple[dict, bool | None]:
        reply = AgentResponse(output=[TextOutputItem(text=case['reply'], annotations=[])])
        verdict, _ = await orchestrator._is_refusal(case['user'], reply)
        return case, verdict

    print(f'model: {model} ({"classify" if use_classify else "prompt"})')
    failed = False
    for run in range(1, args.runs + 1):
        results = await asyncio.gather(*(judge(case) for case in cases))
        judged_keep = [c for c, v in results if v is not None and not c['refused']]
        judged_refused = [c for c, v in results if v is not None and c['refused']]
        wrong_erase = [c['id'] for c, v in results if v is True and not c['refused']]
        wrong_keep = [c['id'] for c, v in results if v is False and c['refused']]
        no_verdict = [c['id'] for c, v in results if v is None]
        failed = failed or bool(no_verdict)
        print(f'run {run}:')
        print(f'  wrongly erased: {len(wrong_erase)} of {len(judged_keep)} judged keep cases {wrong_erase}')
        print(f'  wrongly kept: {len(wrong_keep)} of {len(judged_refused)} judged refusal cases {wrong_keep}')
        print(f'  no verdict: {len(no_verdict)} {no_verdict}')
    if resolved.owned:
        await resolved.client.close()
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
