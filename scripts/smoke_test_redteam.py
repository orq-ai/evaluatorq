#!/usr/bin/env python3
"""Smoke test for evaluatorq.redteam — calls the library directly for manual testing.

Usage:
    uv run python scripts/smoke_test_redteam.py --target agent:my-agent --mode dynamic --max-turns 3 --max-dynamic-datapoints 5
    uv run python scripts/smoke_test_redteam.py --target agent:a --target agent:b --mode dynamic --max-dynamic-datapoints 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Smoke test for evaluatorq.redteam')
    parser.add_argument('--target', action='append', required=True, help='Target identifier (repeatable for multi-target)')
    parser.add_argument('--mode', default='dynamic', choices=['dynamic', 'static', 'hybrid'])
    parser.add_argument('--max-turns', type=int, default=3)
    parser.add_argument('--categories', nargs='+', default=None)
    parser.add_argument('--parallelism', type=int, default=5)
    parser.add_argument('--max-dynamic-datapoints', type=int, default=None)
    parser.add_argument('--max-static-datapoints', type=int, default=None)
    parser.add_argument('--attack-model', default='azure/gpt-5-mini')
    parser.add_argument('--evaluator-model', default='azure/gpt-5-mini')
    parser.add_argument('--dataset-path', default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    from evaluatorq.redteam import red_team

    target = args.target[0] if len(args.target) == 1 else args.target
    report = await red_team(
        target,
        mode=args.mode,
        max_turns=args.max_turns,
        categories=args.categories,
        parallelism=args.parallelism,
        max_dynamic_datapoints=args.max_dynamic_datapoints,
        max_static_datapoints=args.max_static_datapoints,
        attack_model=args.attack_model,
        evaluator_model=args.evaluator_model,
        dataset=args.dataset_path,
    )
    print(json.dumps(report.summary.model_dump(mode='json'), indent=2))


if __name__ == '__main__':
    asyncio.run(main())
