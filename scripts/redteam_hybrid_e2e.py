#!/usr/bin/env python3
"""Run an end-to-end hybrid red-team test with real API calls.

This script exercises the full `red_team(..., mode="hybrid")` pipeline,
which combines dynamic (strategy-driven multi-turn) and static
(dataset-driven) attack paths in a single evaluatorq run.

Requires ORQ_API_KEY to be set (or passed via --api-key).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from evaluatorq.redteam import OpenAIModelTarget, red_team
from evaluatorq.redteam.contracts import LLMConfig, RedTeamReport


def _validate_report(report: RedTeamReport) -> list[str]:
    errors: list[str] = []

    if report.total_results <= 0:
        errors.append('Expected at least one result in report')
    if report.total_results != len(report.results):
        errors.append('total_results does not match len(results)')

    if report.summary.total_attacks != report.total_results:
        errors.append('summary.total_attacks does not match total_results')

    categories_in_results = {result.attack.category for result in report.results}
    categories_reported = set(report.categories_tested)
    if not categories_in_results:
        errors.append('No categories found in result rows')
    if not categories_in_results.issubset(categories_reported):
        errors.append('categories_tested is missing categories present in result rows')

    for result in report.results:
        if result.error:
            continue
        evaluation = result.evaluation
        if evaluation is None:
            errors.append(f'Result {result.attack.id!r} has no evaluation payload')
            continue
        if evaluation.passed is None:
            errors.append(f'Result {result.attack.id!r} has evaluation.passed=None')

    return errors


async def _run(args: argparse.Namespace) -> int:
    if args.api_key:
        os.environ['ORQ_API_KEY'] = args.api_key
    if args.base_url:
        os.environ['ORQ_BASE_URL'] = args.base_url

    dataset_path: str | None = None
    if args.dataset is not None:
        script_dir = Path(__file__).resolve().parent
        package_root = script_dir.parent

        resolved = Path(args.dataset)
        if not resolved.is_absolute():
            resolved = (package_root / resolved).resolve()

        if not resolved.exists():
            print(f'Dataset path does not exist: {resolved}', file=sys.stderr)
            return 2
        dataset_path = str(resolved)

    resolved_target = _resolve_target(args.target)

    report = await red_team(
        resolved_target,
        mode='hybrid',
        categories=args.categories,
        config=LLMConfig(attack_model=args.attack_model, evaluator_model=args.evaluator_model),
        parallelism=args.parallelism,
        max_turns=args.max_turns,
        max_dynamic_datapoints=args.max_dynamic_datapoints,
        max_static_datapoints=args.max_static_datapoints,
        generate_strategies=not args.no_generate_strategies,
        generated_strategy_count=args.generated_strategy_count,
        dataset=dataset_path,
        description='Hybrid red-team E2E test',
        output_dir=args.output_dir,
    )

    errors = _validate_report(report)
    print(
        f"\nE2E run finished: total_results={report.total_results}, "
        f"vulnerabilities={report.summary.vulnerabilities_found}, "
        f"resistance_rate={report.summary.resistance_rate:.2f}, "
        f"errors={report.summary.total_errors}, "
        f"categories={','.join(sorted(set(report.categories_tested)))}"
    )

    if args.print_json:
        print(report.model_dump_json(indent=2))

    if errors:
        print('\nE2E validation failures:', file=sys.stderr)
        for error in errors:
            print(f'  - {error}', file=sys.stderr)
        return 1

    print('E2E validation checks passed.')
    return 0


def _resolve_target(target: str) -> str | OpenAIModelTarget:
    """Resolve CLI target text to a string ORQ target or OpenAIModelTarget."""
    lower = target.lower()
    if lower.startswith(('agent:', 'deployment:')):
        return target
    if ':' in target:
        prefix, _, value = target.partition(':')
        if prefix.lower() in {'openai', 'llm'}:
            return OpenAIModelTarget(value)
    return OpenAIModelTarget(target)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a hybrid red-team end-to-end test (real API calls)')
    parser.add_argument(
        '--target',
        default='agent:rt-vuln-tools-only',
        help='Target model or ORQ target. Examples: gpt-4o-mini, agent:rt-vuln-tools-only, deployment:my-deployment',
    )
    parser.add_argument(
        '--dataset',
        default=None,
        help='Dataset path relative to packages/evaluatorq-py or absolute path. '
             'When omitted, static datapoints are loaded from the ORQ platform.',
    )
    parser.add_argument('--parallelism', type=int, default=3)
    parser.add_argument('--max-turns', type=int, default=3)
    parser.add_argument('--max-dynamic-datapoints', type=int, default=None)
    parser.add_argument('--max-static-datapoints', type=int, default=None)
    parser.add_argument('--attack-model', default='azure/gpt-5-mini')
    parser.add_argument('--evaluator-model', default='azure/gpt-5-mini')
    parser.add_argument(
        '--categories',
        nargs='*',
        default=['ASI01', 'LLM07'],
        help='Category filters, e.g. ASI01 LLM07',
    )
    parser.add_argument('--no-generate-strategies', action='store_true', default=False, help='Disable LLM strategy generation')
    parser.add_argument('--generated-strategy-count', type=int, default=2)
    parser.add_argument('--api-key', default=None, help='ORQ_API_KEY (or set env var)')
    parser.add_argument('--base-url', default=None, help='ORQ_BASE_URL (or set env var)')
    parser.add_argument('--print-json', action='store_true', help='Print full JSON report')
    parser.add_argument('--output-dir', default=None, help='Directory to save intermediate stage artifacts as numbered JSON files')
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == '__main__':
    raise SystemExit(main())
