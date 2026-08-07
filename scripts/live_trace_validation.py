#!/usr/bin/env python3
"""Repeatable live trace checks against existing hosted agents.

The defaults intentionally use small, bounded runs:

* agent simulation: 3 personas x 3 scenarios, at most 2 turns;
* red teaming: hybrid mode, up to 3 dynamic + 3 static datapoints, at most 2 turns.

Agent keys are supplied with ``--sim-agent`` / ``--redteam-agent`` or the
``EVALUATORQ_SIM_AGENT_KEY`` / ``EVALUATORQ_REDTEAM_AGENT_KEY`` environment
variables. Credentials are read from ``ORQ_API_KEY`` and ``ORQ_BASE_URL``.

Examples::

    env ORQ_API_KEY=... \
        EVALUATORQ_SIM_AGENT_KEY=... \
        EVALUATORQ_REDTEAM_AGENT_KEY=... \
        uv run python scripts/live_trace_validation.py both

    uv run python scripts/live_trace_validation.py sim --sim-agent my-agent
    uv run python scripts/live_trace_validation.py redteam --redteam-agent my-agent

The script prints the root span name, run-id attribute, and local span count.
Use ``orq traces list`` afterwards to inspect the same runs in the Orq CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from evaluatorq.redteam import red_team
from evaluatorq.redteam.contracts import LLMCallConfig, LLMConfig, SaveMode
from evaluatorq.simulation import generate_and_simulate

SIM_MODEL = 'openai/gpt-5.6-luna'
REDTEAM_ATTACK_MODEL = 'alibaba/deepseek-v4-flash'
REDTEAM_EVALUATOR_MODEL = 'openai/gpt-5.6-luna'
ROOT_RUN_ID_ATTRIBUTE = 'orq.evaluatorq_run_id'


class _TapProcessor:
    """Keep a local copy of ended spans for quick assertions and tree checks."""

    def __init__(self) -> None:
        self.spans: list[Any] = []
        self._seen: set[int] = set()

    def on_start(self, span: Any, parent_context: Any | None = None) -> None:
        return None

    def on_end(self, span: Any) -> None:
        self._record(span)

    def _on_ending(self, span: Any) -> None:
        """Support the OpenTelemetry SDK's current internal processor hook."""
        self._record(span)

    def _record(self, span: Any) -> None:
        span_identity = id(span)
        if span_identity not in self._seen:
            self._seen.add(span_identity)
            self.spans.append(span)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _attach_tap(tap: _TapProcessor) -> None:
    """Attach a local span processor after evaluatorq initializes tracing."""
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, 'add_span_processor'):
        provider.add_span_processor(tap)  # type: ignore[attr-defined]
    else:
        raise RuntimeError('The configured tracer provider cannot accept a span processor')


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _agent_key(env_name: str, argument: str | None) -> str:
    value = os.getenv(env_name, '').strip() or os.getenv('EVALUATORQ_AGENT_KEY', '').strip() or (argument or '')
    if not value:
        raise RuntimeError(f'Supply an agent key with --{env_name.removeprefix("EVALUATORQ_").lower().replace("_key", "")} or {env_name}')
    return value


def _root_spans(tap: _TapProcessor, name: str) -> list[Any]:
    return [span for span in tap.spans if span.name == name and span.parent is None]


def _validate_root(tap: _TapProcessor, name: str) -> Any:
    roots = _root_spans(tap, name)
    if not roots:
        seen = sorted({span.name for span in tap.spans})
        raise RuntimeError(f'Expected root span {name!r}; saw roots/spans: {seen}')
    root = roots[-1]
    attributes = dict(root.attributes or {})
    run_id = attributes.get(ROOT_RUN_ID_ATTRIBUTE)
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError(f'Root span {name!r} has no non-empty {ROOT_RUN_ID_ATTRIBUTE!r}')
    return root


def _print_root(root: Any, result_count: int, span_count: int) -> None:
    attributes = dict(root.attributes or {})
    print(
        f'  root={root.name!r} results={result_count} '
        f'run_id={attributes[ROOT_RUN_ID_ATTRIBUTE]!r} '
        f'root_attributes={len(attributes)} local_spans={span_count}'
    )


async def _run_simulation(args: argparse.Namespace, tap: _TapProcessor) -> None:
    agent = _agent_key('EVALUATORQ_SIM_AGENT_KEY', args.sim_agent)
    model = _env('EVALUATORQ_SIM_MODEL', args.sim_model)
    results = await generate_and_simulate(
        evaluation_name='live-trace-validation-simulation',
        target=f'agent:{agent}',
        agent_description=(
            'A concise helpful assistant that answers questions clearly and '
            'does not take external actions.'
        ),
        num_personas=3,
        num_scenarios=3,
        max_turns=args.sim_max_turns,
        sim_model=model,
        evaluator_names=['goal_achieved', 'criteria_met'],
        parallelism=3,
        upload_results=not args.no_upload,
        save=False,
        executive_summary=False,
        exit_on_failure=False,
    )
    root = _validate_root(tap, 'Orq Agent Simulation')
    if len(results) != 9:
        raise RuntimeError(f'Expected 9 simulation results (3 x 3), got {len(results)}')
    errors = [result for result in results if result.terminated_by.value == 'error']
    if errors:
        raise RuntimeError(f'{len(errors)}/{len(results)} simulation target calls errored')
    _print_root(root, len(results), len(tap.spans))


async def _run_redteam(args: argparse.Namespace, tap: _TapProcessor) -> None:
    agent = _agent_key('EVALUATORQ_REDTEAM_AGENT_KEY', args.redteam_agent)
    attack_model = _env('EVALUATORQ_REDTEAM_ATTACK_MODEL', args.redteam_attack_model)
    evaluator_model = _env('EVALUATORQ_REDTEAM_EVALUATOR_MODEL', args.redteam_evaluator_model)
    report = await red_team(
        target=f'agent:{agent}',
        name='live-trace-validation-redteam',
        description='Small repeatable hybrid red-team trace validation.',
        mode='hybrid',
        categories=args.redteam_categories,
        llm_config=LLMConfig(
            attacker=LLMCallConfig(model=attack_model),
            evaluator=LLMCallConfig(model=evaluator_model),
        ),
        max_turns=args.redteam_max_turns,
        max_dynamic_datapoints=3,
        max_static_datapoints=3,
        parallelism=3,
        generate_strategies=True,
        generated_strategy_count=1,
        generate_executive_summary=False,
        save=SaveMode.NONE,
        cleanup_memory=True,
        verbosity=0,
    )
    root = _validate_root(tap, 'Red Teaming')
    if report.total_results <= 0:
        raise RuntimeError('Expected the hybrid red-team report to contain results')
    breakdown = report.summary.datapoint_breakdown or {}
    dynamic_count = breakdown.get('template_dynamic', 0) + breakdown.get('generated_dynamic', 0)
    if dynamic_count <= 0:
        raise RuntimeError(f'Hybrid report has no dynamic datapoints: {breakdown}')
    if breakdown.get('static', 0) <= 0:
        raise RuntimeError(f'Hybrid report has no static datapoints: {breakdown}')
    context = report.agent_contexts.get(agent)
    if context is None:
        raise RuntimeError(f'Red-team report did not include context for {agent!r}')
    if context.memory_stores:
        raise RuntimeError(
            f'Red-team target {agent!r} has {len(context.memory_stores)} memory store(s); '
            'choose an agent without memory'
        )
    root_run_id = dict(root.attributes or {}).get(ROOT_RUN_ID_ATTRIBUTE)
    if report.run_id != root_run_id:
        raise RuntimeError(f'Red-team report/root run ids differ: {report.run_id!r} != {root_run_id!r}')
    _print_root(root, report.total_results, len(tap.spans))
    print(
        f'  hybrid_results={report.total_results} '
        f'vulnerabilities={report.summary.vulnerabilities_found} '
        f'resistance_rate={report.summary.resistance_rate:.0%}'
    )


async def _run(args: argparse.Namespace) -> int:
    if not os.getenv('ORQ_API_KEY'):
        print('ERROR: ORQ_API_KEY is not set', file=sys.stderr)
        return 2

    from evaluatorq.tracing.setup import init_tracing_if_needed

    try:
        if not await init_tracing_if_needed():
            raise RuntimeError('Tracing could not be initialized; set ORQ_API_KEY and check tracing settings')
        tap = _TapProcessor()
        _attach_tap(tap)
        if args.pipeline in {'sim', 'both'}:
            print('Running agent simulation: 3 personas x 3 scenarios ...')
            await _run_simulation(args, tap)
        if args.pipeline in {'redteam', 'both'}:
            print(
                'Running red teaming: hybrid against agent '
                f'{_agent_key("EVALUATORQ_REDTEAM_AGENT_KEY", args.redteam_agent)!r} ...'
            )
            await _run_redteam(args, tap)
    except Exception as exc:
        print(f'FAIL: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1

    print(f'PASS: {args.pipeline} trace validation completed ({len(tap.spans)} local spans)')
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pipeline', choices=('sim', 'redteam', 'both'), nargs='?', default='both')
    parser.add_argument('--sim-agent')
    parser.add_argument('--redteam-agent')
    parser.add_argument('--sim-model', default=SIM_MODEL)
    parser.add_argument('--redteam-attack-model', default=REDTEAM_ATTACK_MODEL)
    parser.add_argument('--redteam-evaluator-model', default=REDTEAM_EVALUATOR_MODEL)
    parser.add_argument('--sim-max-turns', type=int, default=2)
    parser.add_argument('--redteam-max-turns', type=int, default=2)
    parser.add_argument('--redteam-categories', nargs='+', default=['ASI01'])
    parser.add_argument('--no-upload', action='store_true', help='Skip simulation experiment uploads')
    return parser.parse_args(argv)


if __name__ == '__main__':
    raise SystemExit(asyncio.run(_run(_parse_args())))
