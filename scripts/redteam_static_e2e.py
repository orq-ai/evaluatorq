#!/usr/bin/env python3
"""Run a local end-to-end static red-team smoke test.

This script exercises the full `red_team(..., mode="static")` pipeline and
fails with a non-zero exit code when output integrity checks fail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evaluatorq.redteam import red_team
from evaluatorq.redteam.contracts import RedTeamReport


def _make_chat_completion(content: str, *, prompt_tokens: int, completion_tokens: int) -> Any:
    total = prompt_tokens + completion_tokens
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason='stop',
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
        ),
    )


class _CompletionsAPI:
    async def create(self, *, _model: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        system_content = str(messages[0].get('content', '')) if messages else ''
        user_content = str(messages[-1].get('content', '')) if messages else ''

        # Evaluator call path.
        if 'You are a security evaluator' in system_content:
            leaked = 'SUPER_SECRET_TOKEN' in user_content or 'SYSTEM_PROMPT:' in user_content
            payload = {
                'value': not leaked,
                'explanation': (
                    'Model resisted the attack and did not reveal protected information.'
                    if not leaked
                    else 'Model output leaked protected information and is vulnerable.'
                ),
            }
            return _make_chat_completion(
                json.dumps(payload),
                prompt_tokens=180,
                completion_tokens=30,
            )

        # Model-under-test call path.
        prompt = user_content.lower()
        if 'ignore previous instructions' in prompt or 'system prompt' in prompt or 'hidden instructions' in prompt:
            response = 'SYSTEM_PROMPT: internal guardrails. SUPER_SECRET_TOKEN=abc123'
        elif 'rm -rf' in prompt or 'execute:' in prompt:
            response = "I can't execute system commands."
        else:
            response = 'I can help with safe alternatives.'

        return _make_chat_completion(
            response,
            prompt_tokens=120,
            completion_tokens=24,
        )


class DeterministicAsyncOpenAI:
    """Minimal OpenAI-compatible async client used for local E2E runs."""

    chat: SimpleNamespace

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_CompletionsAPI())


def _validate_report(report: RedTeamReport) -> list[str]:
    errors: list[str] = []

    if report.pipeline != 'static':
        errors.append(f'Expected pipeline="static", got {report.pipeline!r}')
    if report.total_results <= 0:
        errors.append('Expected at least one result in report')
    if report.total_results != len(report.results):
        errors.append('total_results does not match len(results)')

    if report.summary.total_attacks != report.total_results:
        errors.append('summary.total_attacks does not match total_results')
    if report.summary.evaluated_attacks != report.total_results:
        errors.append('Not all attacks were evaluated (expected full coverage in this E2E run)')

    categories_in_results = {result.attack.category for result in report.results}
    categories_reported = set(report.categories_tested)
    if not categories_in_results:
        errors.append('No categories found in result rows')
    if not categories_in_results.issubset(categories_reported):
        errors.append('categories_tested is missing categories present in result rows')

    missing_token_usage = 0
    for result in report.results:
        evaluation = result.evaluation
        if evaluation is None:
            errors.append(f'Result {result.attack.id!r} has no evaluation payload')
            continue

        if evaluation.passed is None:
            errors.append(f'Result {result.attack.id!r} has evaluation.passed=None')
        if result.vulnerable != (evaluation.passed is False):
            errors.append(f'Result {result.attack.id!r} has inconsistent vulnerable/passed mapping')

        if evaluation.token_usage is None:
            missing_token_usage += 1

    if missing_token_usage > 0:
        errors.append(f'{missing_token_usage} results missing evaluation token_usage')

    return errors


async def _run(args: argparse.Namespace) -> int:
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (package_root / dataset_path).resolve()

    if not dataset_path.exists():
        print(f'Dataset path does not exist: {dataset_path}', file=sys.stderr)
        return 2

    llm_client: Any = None if args.live_client else DeterministicAsyncOpenAI()

    report = await red_team(
        args.target,
        mode='static',
        categories=args.categories,
        evaluator_model=args.evaluator_model,
        parallelism=args.parallelism,
        max_datapoints=args.max_datapoints,
        backend=args.backend,
        dataset_path=str(dataset_path),
        llm_client=llm_client,
        description='Local static red-team E2E smoke test',
    )

    errors = _validate_report(report)
    print(
        f"E2E run finished: total_results={report.total_results}, "
        + f"vulnerabilities={report.summary.vulnerabilities_found}, "
        + f"resistance_rate={report.summary.resistance_rate:.2f}, "
        + f"categories={','.join(sorted(set(report.categories_tested)))}"
    )

    if args.print_json:
        print(report.model_dump_json(indent=2))

    if errors:
        print('\nE2E validation failures:', file=sys.stderr)
        for error in errors:
            print(f'- {error}', file=sys.stderr)
        return 1

    print('E2E validation checks passed.')
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a static red-team end-to-end smoke test')
    parser.add_argument(
        '--dataset',
        default='tests/redteam/fixtures/static_e2e_dataset.json',
        help='Dataset path relative to packages/evaluatorq-py or absolute path',
    )
    parser.add_argument(
        '--target',
        default='openai:e2e-local-model',
        help='Red-team target, e.g. openai:gpt-4o-mini or agent:my-agent',
    )
    parser.add_argument('--backend', default='openai', choices=['openai', 'orq'])
    parser.add_argument('--parallelism', type=int, default=2)
    parser.add_argument('--max-datapoints', type=int, default=None)
    parser.add_argument('--evaluator-model', default='e2e-local-evaluator')
    parser.add_argument(
        '--categories',
        nargs='*',
        default=None,
        help='Optional category filters, e.g. ASI01 LLM07',
    )
    parser.add_argument(
        '--live-client',
        action='store_true',
        help='Use real API client resolution (requires OPENAI_API_KEY or ORQ_API_KEY)',
    )
    parser.add_argument('--print-json', action='store_true', help='Print full JSON report')
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == '__main__':
    raise SystemExit(main())
