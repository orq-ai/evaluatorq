# ruff: noqa: INP001, ANN002, ANN003, S101, PTH109, C901, PERF203, PERF401, FURB140, FURB171, ISC004, RUF059
"""Compare production trace-finder modes `llm+jev` and `llm` on paired labelled workloads."""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib import metadata
from math import comb
from pathlib import Path
from typing import Any, Literal, NamedTuple, get_args


from evaluatorq.common import structured_output as structured_mod
from evaluatorq.common.judge import ClassifyAnswer, ClassifyOutcome, ClassifyResponse
from evaluatorq.common.model_roles import role_model, role_source
from evaluatorq.common.structured_output import StructuredResult, generate_structured, usage_from_exception
from evaluatorq.contracts import Usage
from evaluatorq.trace_finder import compiler as compiler_mod
from evaluatorq.trace_finder import filter_selector as selector_mod
from evaluatorq.trace_finder.models import FACET_NAMES, FacetCatalogue

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ALL_ARMS = ('llm+jev', 'llm')
NUMERIC_KEYS = ('tokens_min', 'tokens_max', 'duration_ms_min', 'duration_ms_max')
RETENTION_DAYS = 30
TIME_TOLERANCE = timedelta(minutes=60)
# Queries that repeat an example from the *production* prompts (rule, status note, numeric examples); every arm sees
# those examples, so Headline A is also reported without them (audit P2). Matched on query text, so it applies per
# dataset: a supplement whose ids coincide with the main dataset's is not affected.
PROMPT_EXAMPLE_QUERIES = frozenset({
    'failed traces',
    'find traces with tool errors',
    'traces with more than 20k tokens',
    'requests slower than 30 seconds',
    'which model costs the most?',
})
OUTSIDE_KINDS = ('correct', 'correct_with_extra', 'miss', 'wrong_value', 'fallback')
BASELINE_ARM = 'llm+jev'
NON_INFERIORITY_RULE = (
    'Pre-registered: non-inferior iff Headline A (majority vote over repeats, paraphrase rows excluded) >= baseline '
    '- 1 row, AND false-filter events on eligible, non-errored rows (query repeats that emitted any facet, numeric or '
    'time filter where none was expected) <= baseline, AND hallucinated values == 0. Latency/cost are not gated.'
)

# ----------------------------------------------------------------------------------------------
# Usage / latency capture (contextvar-scoped so parallel calls land on the right datapoint+call)
# ----------------------------------------------------------------------------------------------

_CALLS: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar('bench_calls', default=None)
_FAKE_DP: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar('bench_fake_dp', default=None)
# (leg, served model) per provider response seen by the current call; one list per instrumented call.
_LEGS: contextvars.ContextVar[list[tuple[str, str | None]] | None] = contextvars.ContextVar('bench_legs', default=None)


def _usage_fields(usage: Usage | None) -> dict[str, Any]:
    if usage is None:
        return {
            'input_tokens': None,
            'output_tokens': None,
            'reasoning_tokens': None,
            'cached_tokens': None,
            'total_cost': None,
            'priced': False,
            'cost_is_partial': False,
            'cost_state': 'missing',
            'usage_calls': None,
            'priced_calls': None,
            'cost_source': None,
        }
    cost = usage.total_cost
    partial = bool(usage.cost_is_partial)
    return {
        'input_tokens': usage.input_tokens,
        'output_tokens': usage.output_tokens,
        'reasoning_tokens': usage.reasoning_tokens,
        'cached_tokens': usage.cached_tokens,
        'total_cost': cost,
        'priced': cost is not None and not partial,
        'cost_is_partial': partial,
        'cost_state': 'missing' if cost is None else ('partial' if partial else 'reported'),
        'usage_calls': usage.calls,
        'priced_calls': usage.priced_calls,
        'cost_source': 'provider_usage' if cost is not None else None,
    }


def _record(
    call: str,
    model: str,
    started: float,
    usage: Usage | None,
    *,
    served: str | None,
    leg: str | None,
    attempts: int,
    error: str | None,
    structured_output: Any = None,
) -> None:
    sink = _CALLS.get()
    if sink is None:
        return
    attempt = sum(row['call'] == call for row in sink) + 1
    sink.append({
        'call': call,
        'call_attempt': attempt,
        'is_retry': attempt > 1,
        'model_requested': model,
        'model_served': served,
        'leg': leg,
        'provider_responses': attempts,
        'latency_ms': (time.perf_counter() - started) * 1000,
        **_usage_fields(usage),
        **({'structured_output': structured_output} if structured_output is not None else {}),
        'error': error,
    })


def _structured_output(result: Any) -> dict[str, Any] | None:
    parsed = getattr(result, 'parsed', None)
    if parsed is None:
        return None
    if hasattr(parsed, 'model_dump'):
        return parsed.model_dump(mode='json', exclude_none=True)
    return parsed if isinstance(parsed, dict) else None

def instrumented_generate(call: str, inner):
    """Wrap a ``generate_structured``-shaped coroutine function to record usage, latency, served model and leg."""

    async def wrapper(client, *, model, **kwargs):
        legs: list[tuple[str, str | None]] = []
        _LEGS.set(legs)  # each gather child owns its context, so this list belongs to this call alone
        started = time.perf_counter()

        def served_and_leg() -> tuple[str | None, str | None]:
            served = next((m for _, m in reversed(legs) if m), None)
            return served, (legs[-1][0] if legs else None)

        try:
            result = await inner(client, model=model, **kwargs)
        except Exception as exc:
            served, leg = served_and_leg()
            _record(
                call,
                model,
                started,
                usage_from_exception(exc),
                served=served,
                leg=leg,
                attempts=len(legs),
                error=f'{type(exc).__name__}: {exc}',
            )
            raise
        served, leg = served_and_leg()
        _record(
            call,
            model,
            started,
            result.usage,
            served=served,
            leg=leg,
            attempts=len(legs),
            error=None,
            structured_output=_structured_output(result),
        )
        return result

    return wrapper


def instrumented_classify(inner):
    async def wrapper(*, client, model, cfg, request, **kwargs):
        started = time.perf_counter()
        try:
            outcome = await inner(client=client, model=model, cfg=cfg, request=request, **kwargs)
        except Exception as exc:
            _record(
                'jev',
                model,
                started,
                None,
                served=None,
                leg='classify',
                attempts=0,
                error=f'{type(exc).__name__}: {exc}',
            )
            raise
        served = outcome.response.model if outcome.response is not None else None
        error = None
        if outcome.error_kind is not None:
            error = f'{outcome.error_kind}: {outcome.error_message}'
        _record('jev', model, started, outcome.token_usage, served=served, leg='classify', attempts=1, error=error)
        return outcome

    return wrapper


def _leg_wrapper(leg: str, inner):
    async def wrapper(*args, **kwargs):
        response, usage = await inner(*args, **kwargs)
        legs = _LEGS.get()
        if legs is not None:
            legs.append((leg, getattr(response, 'model', None)))
        return response, usage

    return wrapper


def patch_structured_output() -> None:
    """Record which ladder leg answered and which model the router served (``generate_structured`` hides both)."""
    structured_mod.execute_response = _leg_wrapper('responses', structured_mod.execute_response)
    structured_mod.execute_chat_parse = _leg_wrapper('chat_parse', structured_mod.execute_chat_parse)
    structured_mod.execute_chat_completion = _leg_wrapper('chat_completion', structured_mod.execute_chat_completion)


# ----------------------------------------------------------------------------------------------
# Dry-run fakes
# ----------------------------------------------------------------------------------------------


def _fake_usage(input_tokens: int, output_tokens: int, cached: int = 0) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached,
        total_cost=(input_tokens * 1e-7) + (output_tokens * 4e-7),
        calls=1,
        priced_calls=1,
    )


async def fake_generate(client, *, model, messages, response_format, **_: Any) -> StructuredResult:
    expected = (_FAKE_DP.get() or {'expected': {}})['expected']
    await asyncio.sleep(random.uniform(0.005, 0.02))
    legs = _LEGS.get()
    if legs is not None:
        legs.append(('responses', model))
    numeric = {key: expected.get('numeric', {}).get(key) for key in NUMERIC_KEYS}
    time_expected = expected.get('time') or {}
    dimensions: list[dict[str, Any]] = []
    if expected.get('needs_judgment'):
        dimensions.append({
            'name': 'Judged',
            'task': {
                'kind': 'noul',
                'instructions': 'canned',
                'choice_criteria': None,
                'score_criteria': None,
                'noul_threshold': 0.5,
            },
            'selection': {'kind': 'values', 'values': [True]},
        })
    fields = response_format.model_fields
    document: dict[str, Any] = {}
    if 'dimensions' in fields:
        document['dimensions'] = dimensions
    if 'numeric' in fields:
        document['numeric'] = numeric
    if 'unsupported_reason' in fields:
        document['unsupported_reason'] = 'aggregate' if expected.get('aggregate') else None
    if 'time' in fields:
        document['time'] = (
            {'start': time_expected.get('start'), 'end': time_expected.get('end')}
            if time_expected
            else None
        )
    if 'calendar_day' in fields:
        document['calendar_day'] = False
    if 'start' in fields and 'end' in fields:
        document['start'] = time_expected.get('start')
        document['end'] = time_expected.get('end')
    for facet in FACET_NAMES:
        if facet not in fields:
            continue
        field_type = get_args(fields[facet].annotation)
        allowed = get_args(field_type[0]) if field_type else ()
        wanted = expected.get('facets', {}).get(facet, [])
        selected = next((value for value in wanted if value in allowed), None)
        document[facet] = [selected] if selected is not None else []
    parsed = response_format.model_validate(document)
    return StructuredResult(parsed, '', _fake_usage(1200, 150, cached=1000))


async def fake_classify(*, client, model, cfg, request, **_: Any) -> ClassifyOutcome:
    dp = _FAKE_DP.get() or {'expected': {}}
    wanted = dp['expected'].get('facets', {})
    await asyncio.sleep(random.uniform(0.005, 0.02))
    answers: dict[str, ClassifyAnswer] = {}
    for key, question in request.questions.items():
        if question.kind == 'choice':
            criteria = question.criteria or {}
            matching = next(
                (
                    label
                    for label, description in criteria.items()
                    if description and any(repr(value) in description for value in wanted.get(key, []))
                ),
                None,
            )
            chosen = matching or next(
                (label for label in criteria if label.casefold().startswith('none')),
                next(iter(criteria)),
            )
            answers[key] = ClassifyAnswer(type='choice', choice=chosen)
        else:
            hit = any(repr(value) in question.instructions for values in wanted.values() for value in values)
            answers[key] = ClassifyAnswer(type='noul', noul=1.0 if hit else 0.0)
    usage = _fake_usage(3000, 40)
    response = ClassifyResponse(answers=answers, usage={'total_cost': usage.total_cost}, model=model)
    return ClassifyOutcome(response=response, token_usage=usage, raw_content=response.model_dump_json())


def parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') if value is not None else None

# ----------------------------------------------------------------------------------------------
# Shared production planner
# ----------------------------------------------------------------------------------------------


class ArmSpec(NamedTuple):
    label: str
    mode: str


def parse_arm(label: str) -> tuple[str, None]:
    return label.split('@', 1)[0], None


class Ctx:
    def __init__(self, args, client, orq, catalogue: FacetCatalogue, now: datetime, models, specs) -> None:
        self.args = args
        self.client = client
        self.orq = orq
        self.catalogue = catalogue
        self.now = now
        self.models = models
        self.specs = specs

    async def catalogue_snapshot(self, _orq, *, start, end, limit=50):
        return self.catalogue.model_copy(deep=True)


async def run_planner(ctx: Ctx, spec: ArmSpec, query: str, *, live_catalogue: bool = False) -> dict[str, Any]:
    from evaluatorq.trace_finder.models import PopulationRequest
    from evaluatorq.trace_finder.planner import plan_query

    population = PopulationRequest(
        start=ctx.now - timedelta(days=RETENTION_DAYS),
        end=ctx.now,
        time_origin='default',
    )
    planner_args = {
        'client': ctx.client,
        'orq': ctx.orq,
        'compiler_model': ctx.models['compiler'],
        'filter_selection': spec.mode,
        'classifier_model': ctx.models['classifier'],
        'fast_filter_model': ctx.models['fast_filter'],
        'now': ctx.now,
    }
    if not live_catalogue:
        planner_args['catalogue_loader'] = ctx.catalogue_snapshot
    result = await plan_query(query, population, **planner_args)
    generated_time = result.generated_time
    actual = {
        'dimensions': len(result.dimensions),
        'facets': {
            name: sorted(getattr(result.generated_facets, name))
            for name in FACET_NAMES
            if getattr(result.generated_facets, name)
        },
        'numeric': {
            key: value for key, value in result.generated_numeric.model_dump().items() if value is not None
        },
        'numeric_pre': None,
        'time': (
            {'start': _iso(generated_time[0]), 'end': _iso(generated_time[1])}
            if generated_time is not None
            else None
        ),
        'empty_match': result.empty_match,
    }
    errors = {}
    if result.selector_error:
        errors['llm_filter' if spec.mode == 'llm' else 'jev'] = result.selector_error
    return {'actual': actual, 'errors': errors, 'planner_usage': dict(result.usage)}


# Mutable cell so dry runs can swap the production structured-output entry point.
_GENERATE: list[Any] = [generate_structured]

# Failed calls remove only the fields their production branch supplies.
SOURCE_KEYS: dict[str, dict[str, tuple[str, ...]]] = {
    'llm+jev': {
        'compiler': ('numeric', 'numeric_pre', 'dimensions', 'time'),
        'jev': ('facets',),
    },
    'llm': {
        'compiler': ('dimensions', 'numeric', 'numeric_pre', 'time'),
        'llm_filter': ('facets',),
    },
    # Retained only to rescore already-archived historical raw runs.
    'merged': {'merged': ('dimensions', 'facets', 'numeric', 'numeric_pre', 'time')},
}
# ----------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------


def _time_bounds(time_value: dict[str, str | None] | None, now: datetime) -> tuple[datetime, datetime]:
    """Normalise: null end -> now, null start -> retention floor (now-30d), whole value null -> [floor, now]."""
    floor = now - timedelta(days=RETENTION_DAYS)
    start = parse_time((time_value or {}).get('start')) or floor
    end = parse_time((time_value or {}).get('end')) or now
    return start, end


def _within(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> bool:
    return abs(a[0] - b[0]) <= TIME_TOLERANCE and abs(a[1] - b[1]) <= TIME_TOLERANCE





def score(
    dp: dict[str, Any],
    outcome: dict[str, Any],
    catalogue: FacetCatalogue,
    now: datetime,
) -> tuple[dict[str, Any], list[list[str]]]:
    expected = dp['expected']
    actual = outcome['actual']
    facets_ok = 'facets' in actual
    numeric_ok = 'numeric' in actual
    dims_ok = 'dimensions' in actual
    time_ok = 'time' in actual
    actual_facets = actual.get('facets', {})
    expected_facets = {name: sorted(values) for name, values in expected.get('facets', {}).items() if values}
    expected_numeric = {k: v for k, v in expected.get('numeric', {}).items() if v is not None}
    expected_time = expected.get('time')
    floor = now - timedelta(days=RETENTION_DAYS)
    full_window = (floor, now)
    expected_bounds = _time_bounds(expected_time, now)
    actual_bounds = _time_bounds(actual.get('time'), now)
    time_bearing = not _within(expected_bounds, full_window)
    time_emitted = not _within(actual_bounds, full_window)
    time_match = time_ok and _within(expected_bounds, actual_bounds)
    predicted = {(name, value) for name, values in actual_facets.items() for value in values}
    wanted = {(name, value) for name, values in expected_facets.items() for value in values}
    tp, fp, fn = len(predicted & wanted), len(predicted - wanted), len(wanted - predicted)
    hallucinated = [[name, value] for name, value in sorted(predicted) if value not in getattr(catalogue, name, ())]
    facet_exact = facets_ok and predicted == wanted
    numeric_exact = numeric_ok and actual.get('numeric', {}) == expected_numeric
    judgment_agree = dims_ok and (actual['dimensions'] > 0) == bool(expected.get('needs_judgment'))
    emitted = bool(actual_facets) or bool(actual.get('numeric')) or time_emitted
    eligible = not wanted and not expected_numeric and not time_bearing
    filter_failed = not (facets_ok and numeric_ok)

    outside = [
        (facet, value)
        for facet, values in expected_facets.items()
        for value in values
        if value not in getattr(catalogue, facet, ())
    ]
    outside_kind: str | None = None
    if outside:
        outside_facets = {facet for facet, _ in outside}
        if all(pair in predicted for pair in outside):
            outside_kind = 'correct' if facet_exact else 'correct_with_extra'
        elif any(facet in outside_facets and (facet, value) not in wanted for facet, value in predicted):
            outside_kind = 'wrong_value'
        elif any(facet not in expected_facets for facet, _ in predicted):
            outside_kind = 'fallback'
        else:
            outside_kind = 'miss'
    core_exact = facet_exact and numeric_exact
    result = {
        'facet_tp': tp,
        'facet_fp': fp,
        'facet_fn': fn,
        'facet_precision': tp / (tp + fp) if tp + fp else 1.0,
        'facet_recall': tp / (tp + fn) if tp + fn else 1.0,
        'facet_exact': facet_exact,
        'hallucinated': len(hallucinated),
        'numeric_exact': numeric_exact,
        'time_match': time_match,
        'judgment_agree': judgment_agree,
        'false_filter': emitted if eligible and not filter_failed else None,
        'time_bearing': time_bearing,
        'prompt_example': dp['query'].strip().lower() in PROMPT_EXAMPLE_QUERIES,
        'spurious_time': time_emitted if not time_bearing else None,
        'coverage_limited': bool(outside),
        'outside_catalogue': bool(outside),
        'outside_kind': outside_kind,
        'core_exact': core_exact,
        'full_exact': core_exact and time_match,
        'false_filter_errored': eligible and filter_failed,
    }
    return result, hallucinated


# ----------------------------------------------------------------------------------------------
# Run loop
# ----------------------------------------------------------------------------------------------


async def run_one(ctx: Ctx, spec: ArmSpec, dp: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the production planner once; return call rows and a row-level outcome."""
    sink: list[dict[str, Any]] = []
    _CALLS.set(sink)
    _FAKE_DP.set(dp)
    started = time.perf_counter()
    try:
        outcome = await run_planner(ctx, spec, dp['query'])
    except Exception as exc:  # noqa: BLE001 - retain failed calls as scored rows
        outcome = {'actual': {}, 'errors': {'planner': f'{type(exc).__name__}: {exc}'}, 'planner_usage': {}}
    wall_ms = (time.perf_counter() - started) * 1000
    for name, message in outcome['errors'].items():
        matching = [row for row in sink if row['call'] == name]
        if not matching:
            sink.append(_error_row(name, ctx, spec, message))
        elif matching[-1]['error'] is None:
            matching[-1]['error'] = message
    scores, hallucinated = score(dp, outcome, ctx.catalogue, ctx.now)
    actual = dict(outcome['actual'])
    actual.setdefault('facets', {})
    actual.setdefault('numeric', {})
    actual.setdefault('numeric_pre', None)
    actual.setdefault('time', None)
    actual.setdefault('dimensions', None)
    actual['hallucinated'] = hallucinated
    usage = outcome.get('planner_usage') or {}
    query_row = {
        'wall_ms': wall_ms,
        'error': bool(outcome['errors']),
        'errors': [f'{name}: {message}' for name, message in outcome['errors'].items()],
        'actual': actual,
        'expected': dp['expected'],
        'scores': scores,
        'planner_usage': usage,
        'catalogue_cache': usage.get('catalogue_cache'),
    }
    return sink, query_row


def _error_row(name: str, ctx: Ctx, spec: ArmSpec, message: str) -> dict[str, Any]:
    model = {
        'compiler': ctx.models['compiler'],
        'jev': ctx.models['classifier'],
        'llm_filter': ctx.models['fast_filter'],
        'selector': ctx.models['fast_filter'] if spec.mode == 'llm' else ctx.models['classifier'],
    }.get(name)
    return {
        'call': name,
        'model_requested': model,
        'model_served': None,
        'leg': None,
        'provider_responses': 0,
        'latency_ms': 0.0,
        **_usage_fields(None),
        'error': message,
    }


async def cache_scenario(ctx: Ctx, datapoint: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if ctx.args.dry_run:
        return [
            {
                'kind': 'cache_scenario',
                'mode': spec.mode,
                'id': datapoint['id'],
                'cache_state': state,
                'wall_ms': None,
                'catalogue_cache': None,
                'error': 'not measured during dry-run',
            }
            for spec in ctx.specs
            for state in ('cold', 'warm')
        ]
    from evaluatorq.trace_finder.facets import invalidate_facet_catalogue_cache
    for spec in ctx.specs:
        await invalidate_facet_catalogue_cache()
        for state in ('cold', 'warm'):
            calls: list[dict[str, Any]] = []
            _CALLS.set(calls)
            _FAKE_DP.set(datapoint)
            started = time.perf_counter()
            error = None
            usage: dict[str, Any] = {}
            try:
                result = await run_planner(ctx, spec, datapoint['query'], live_catalogue=True)
                usage = result['planner_usage']
                error = '; '.join(result['errors'].values()) or None
            except Exception as exc:  # noqa: BLE001 - retain cache measurement failures
                error = f'{type(exc).__name__}: {exc}'
            wall_ms = (time.perf_counter() - started) * 1000
            identity = {'mode': spec.mode, 'id': datapoint['id'], 'cache_state': state}
            rows.append({
                'kind': 'cache_scenario',
                **identity,
                'wall_ms': wall_ms,
                'catalogue_cache': usage.get('catalogue_cache'),
                'error': error,
            })
            rows.extend({'kind': 'scenario_call', **identity, **call} for call in calls)
    return rows


async def run_benchmark(args, ctx: Ctx, datapoints: list[dict[str, Any]], out: Path) -> None:
    raw_path = out / 'raw.jsonl'
    if raw_path.exists():
        raise SystemExit(f'refusing to overwrite existing raw run: {raw_path}')
    with raw_path.open('w', encoding='utf-8') as raw:

        def emit(row: dict[str, Any]) -> None:
            raw.write(json.dumps(row, ensure_ascii=False) + '\n')
            raw.flush()

        async def measured_call(spec: ArmSpec, dp: dict[str, Any], repeat: int) -> None:
            calls, query_row = await run_one(ctx, spec, dp)
            identity = {'arm': spec.label, 'id': dp['id'], 'category': dp['category'], 'repeat': repeat}
            for call in calls:
                emit({'kind': 'call', **identity, **call})
            emit({'kind': 'query', **identity, **query_row})

        for index in range(args.warmup):
            dp = datapoints[index % len(datapoints)]
            for spec in ctx.specs:
                await run_one(ctx, spec, dp)
        print(f'warmup done ({args.warmup} queries x {[s.label for s in ctx.specs]})', file=sys.stderr)

        for repeat in range(args.repeats):
            order = ctx.specs if repeat % 2 == 0 else tuple(reversed(ctx.specs))
            for dp in datapoints:
                for spec in order:
                    await measured_call(spec, dp, repeat)

        for row in await cache_scenario(ctx, datapoints[0]):
            emit(row)


# ----------------------------------------------------------------------------------------------
# Rescoring (derived scored.jsonl; raw.jsonl is never modified)
# ----------------------------------------------------------------------------------------------


def _resolve_input(path_str: str, expected_sha: str) -> Path | None:
    """meta stores the path as given on the command line (possibly relative); find the file with the recorded sha."""
    given = Path(path_str)
    candidates = [given] if given.is_absolute() else [Path.cwd() / given, REPO_ROOT / given]
    candidates.append(HERE / given.name)
    for candidate in candidates:
        if candidate.is_file() and sha256(candidate) == expected_sha:
            return candidate
    return None


def rescore(rows: list[dict[str, Any]], meta: dict[str, Any] | None) -> tuple[list[dict], dict]:
    """Recompute scores from raw outputs and the recorded workload files; never alter raw.jsonl."""
    info: dict[str, Any] = {'rescored': False, 'rules': 'production-planner'}
    if not meta or 'dataset_path' not in meta or 'catalogue_path' not in meta:
        info['reason'] = 'no meta.json with dataset/catalogue paths; raw scores kept'
        return rows, info
    dataset_path = _resolve_input(meta['dataset_path'], meta['dataset_sha256'])
    catalogue_path = _resolve_input(meta['catalogue_path'], meta['catalogue_sha256'])
    if dataset_path is None or catalogue_path is None:
        info['reason'] = 'dataset or catalogue file missing or its sha256 differs from meta; raw scores kept'
        return rows, info
    datapoints = {
        json.loads(line)['id']: json.loads(line)
        for line in dataset_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    }
    catalogue = FacetCatalogue.model_validate(json.loads(catalogue_path.read_text(encoding='utf-8'))['catalogue'])
    now = parse_time(meta['now'])
    assert now is not None
    rescored: list[dict] = []
    for row in rows:
        if row['kind'] != 'query':
            rescored.append(row)
            continue
        mode = parse_arm(row['arm'])[0]
        actual = {key: value for key, value in row['actual'].items() if key != 'hallucinated'}
        errors: dict[str, str] = {}
        for entry in row.get('errors', []):
            name, _, message = entry.partition(': ')
            errors[name] = message
            for key in SOURCE_KEYS.get(mode, {}).get(name, ()):
                actual.pop(key, None)
        if actual.get('dimensions') is None:
            actual.pop('dimensions', None)
        scores, hallucinated = score(
            datapoints[row['id']],
            {'actual': actual, 'errors': errors},
            catalogue,
            now,
        )
        new_actual = dict(row['actual'])
        new_actual['hallucinated'] = hallucinated
        rescored.append({**row, 'actual': new_actual, 'scores': scores})
    info.update({'rescored': True, 'dataset': str(dataset_path), 'catalogue': str(catalogue_path)})
    return rescored, info


# ----------------------------------------------------------------------------------------------
# Summary (raw/scored rows only)
# ----------------------------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row['scores'].get(key) for row in rows if row['scores'].get(key) is not None]
    return sum(1 for value in values if value) / len(values) if values else None


def _group_by_id(queries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        grouped.setdefault(query['id'], []).append(query)
    return grouped


def _block(rates: dict[str, float], ids: list[str]) -> dict[str, Any]:
    passed = sum(1 for rid in ids if rates[rid] > 0.5)
    return {
        'rows': len(ids),
        'pass_rows': passed,
        'pass_rate_majority': passed / len(ids) if ids else None,
        'mean_pass_rate': _mean([rates[rid] for rid in ids]),
    }


def headline(queries: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """Aggregate repeats within each labelled row before computing rates and pairwise results."""
    rows = _group_by_id([q for q in queries if q['category'] != 'paraphrase'])
    if not rows:
        return None
    rates = {rid: sum(bool(q['scores'].get(key)) for q in qs) / len(qs) for rid, qs in rows.items()}
    coverage_limited = {
        rid for rid, qs in rows.items() if any(q['scores'].get('coverage_limited') for q in qs)
    }
    prompt_example = {rid for rid, qs in rows.items() if any(q['scores'].get('prompt_example') for q in qs)}
    return {
        **_block(rates, list(rates)),
        'coverage_limited_rows': sorted(coverage_limited),
        'without_prompt_examples': (
            {
                'excluded': sorted(prompt_example),
                **_block(rates, [rid for rid in rates if rid not in prompt_example]),
            }
            if prompt_example
            else None
        ),
        'row_rates': rates,
    }



def _cache_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    cache = row.get('catalogue_cache') or {}
    if 'events' in cache:
        events = cache['events']
        return [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []
    return [cache] if any(key in cache for key in ('hit', 'elapsed_seconds', 'source')) else []

def _cache_elapsed_ms(events: list[dict[str, Any]], hit: bool | None = None) -> list[float]:
    return [
        float(event['elapsed_seconds']) * 1000
        for event in events
        if event.get('elapsed_seconds') is not None and (hit is None or event.get('hit') is hit)
    ]


def metrics(queries: list[dict[str, Any]], calls: dict[tuple, list[dict[str, Any]]]) -> dict:
    per_query_calls = [calls.get((query['arm'], query['id'], query['repeat']), []) for query in queries]
    flat = [call for group in per_query_calls for call in group]
    walls = [query['wall_ms'] for query in queries]
    tp = sum(query['scores']['facet_tp'] for query in queries)
    fp = sum(query['scores']['facet_fp'] for query in queries)
    fn = sum(query['scores']['facet_fn'] for query in queries)

    def query_total(group: list[dict[str, Any]], field: str) -> float | None:
        values = [call.get(field) for call in group]
        return sum(values) if values and all(value is not None for value in values) else None

    query_costs = [query_total(group, 'total_cost') for group in per_query_calls]
    query_costs = [
        cost if cost is not None and not any(call.get('cost_is_partial') for call in group) else None
        for cost, group in zip(query_costs, per_query_calls, strict=True)
    ]

    def per_query_tokens(field: str) -> float | None:
        totals = [query_total(group, field) for group in per_query_calls]
        return _mean([total for total in totals if total is not None])

    by_call: dict[str, dict[str, Any]] = {}
    for name in sorted({call['call'] for call in flat}):
        named = [call for call in flat if call['call'] == name]

        def mean_of(field: str, named=named) -> float | None:
            return _mean([call[field] for call in named if call[field] is not None])

        by_call[name] = {
            'n': len(named),
            'input': mean_of('input_tokens'),
            'output': mean_of('output_tokens'),
            'reasoning': mean_of('reasoning_tokens'),
            'cached': mean_of('cached_tokens'),
            'latency_ms_p50': percentile([call['latency_ms'] for call in named], 0.5),
            'cost_usd_mean': mean_of('total_cost'),
            'unpriced_calls': sum(call.get('total_cost') is None for call in named),
            'partial_cost_calls': sum(bool(call.get('cost_is_partial')) for call in named),
        }

    hallucinated = [q['scores']['hallucinated'] for q in queries]
    time_rows = [q for q in queries if q['scores'].get('time_bearing')]
    other_rows = [q for q in queries if not q['scores'].get('time_bearing')]
    paraphrase = [q for q in queries if q['category'] == 'paraphrase']
    para_tp = sum(q['scores']['facet_tp'] for q in paraphrase)
    para_fp = sum(q['scores']['facet_fp'] for q in paraphrase)
    para_fn = sum(q['scores']['facet_fn'] for q in paraphrase)
    outside = [q for q in queries if q['scores'].get('outside_kind') is not None]
    eligible = [q for q in queries if q['scores'].get('false_filter') is not None]
    ff_by_row = _group_by_id(eligible)
    catalogue_events = [event for query in queries for event in _cache_events(query)]
    cache_elapsed_ms = _cache_elapsed_ms(catalogue_events)
    cache_hit_elapsed_ms = _cache_elapsed_ms(catalogue_events, True)
    cache_miss_elapsed_ms = _cache_elapsed_ms(catalogue_events, False)
    cache_sources = {
        source: sum(event.get('source') == source for event in catalogue_events)
        for source in sorted({event.get('source') for event in catalogue_events if event.get('source') is not None})
    }
    return {
        'n': len(queries),
        'n_errors': sum(bool(q['error']) for q in queries),
        'calls': len(flat),
        'unpriced_calls': sum(call.get('total_cost') is None for call in flat),
        'partial_cost_calls': sum(bool(call.get('cost_is_partial')) for call in flat),
        'catalogue_cache_events': len(catalogue_events),
        'catalogue_cache_hit_events': sum(event.get('hit') is True for event in catalogue_events),
        'catalogue_cache_miss_events': sum(event.get('hit') is False for event in catalogue_events),
        'catalogue_cache_unknown_events': sum(event.get('hit') is None for event in catalogue_events),
        'catalogue_cache_elapsed_ms_p50': percentile(cache_elapsed_ms, 0.5),
        'catalogue_cache_elapsed_ms_p95': percentile(cache_elapsed_ms, 0.95),
        'catalogue_cache_sources': cache_sources,
        'catalogue_cache_hit_elapsed_ms_p50': percentile(cache_hit_elapsed_ms, 0.5),
        'catalogue_cache_hit_elapsed_ms_p95': percentile(cache_hit_elapsed_ms, 0.95),
        'catalogue_cache_miss_elapsed_ms_p50': percentile(cache_miss_elapsed_ms, 0.5),
        'catalogue_cache_miss_elapsed_ms_p95': percentile(cache_miss_elapsed_ms, 0.95),
        'latency_ms_p50': percentile(walls, 0.5),
        'latency_ms_p95': percentile(walls, 0.95),
        'mean_tokens_by_call': by_call,
        'mean_tokens_per_query': {
            'input': per_query_tokens('input_tokens'),
            'output': per_query_tokens('output_tokens'),
            'cached': per_query_tokens('cached_tokens'),
        },
        'cost_per_query_known_queries': sum(value is not None for value in query_costs),
        'cost_per_query_missing_queries': sum(
            not group or all(call.get('total_cost') is None for call in group) for group in per_query_calls
        ),
        'cost_per_query_partial_queries': sum(
            any(call.get('cost_is_partial') for call in group)
            or (
                any(call.get('total_cost') is None for call in group)
                and any(call.get('total_cost') is not None for call in group)
            )
            for group in per_query_calls
        ),
        'cost_per_query_reported_usd': _mean([value for value in query_costs if value is not None]),
        'headline_a': headline(queries, 'core_exact'),
        'headline_b': headline(queries, 'full_exact'),
        'time_accuracy_on_time_rows': _rate(time_rows, 'time_match'),
        'spurious_time_rate': _rate(other_rows, 'spurious_time'),
        'flexibility': {
            'n': len(paraphrase),
            'facet_precision': para_tp / (para_tp + para_fp) if para_tp + para_fp else None,
            'facet_recall': para_tp / (para_tp + para_fn) if para_tp + para_fn else None,
            'facet_exact_rate': _rate(paraphrase, 'facet_exact'),
        },
        'outside_catalogue': {
            'n': len(outside),
            **{kind: sum(q['scores']['outside_kind'] == kind for q in outside) for kind in OUTSIDE_KINDS},
        },
        'false_filter_eligible': len(eligible),
        'false_filter_count': sum(bool(q['scores']['false_filter']) for q in eligible),
        'false_filter_errored': sum(bool(q['scores'].get('false_filter_errored')) for q in queries),
        'false_filter_rows_majority': sum(
            sum(bool(q['scores']['false_filter']) for q in group) / len(group) > 0.5
            for group in ff_by_row.values()
        ),
        'facet_precision': tp / (tp + fp) if tp + fp else None,
        'facet_recall': tp / (tp + fn) if tp + fn else None,
        'facet_exact_rate': _rate(queries, 'facet_exact'),
        'hallucinated_total': sum(hallucinated),
        'hallucinated_per_query': _mean(hallucinated),
        'hallucinated_query_rate': _mean([1.0 if count else 0.0 for count in hallucinated]),
        'numeric_exact_rate': _rate(queries, 'numeric_exact'),
        'time_match_rate': _rate(queries, 'time_match'),
        'judgment_agree_rate': _rate(queries, 'judgment_agree'),
        'false_filter_rate': _rate(queries, 'false_filter'),
        'full_exact_rate': _rate(queries, 'full_exact'),
    }


def mcnemar_exact(baseline: dict[str, float], arm: dict[str, float]) -> dict[str, Any]:
    """Exact two-sided McNemar on majority-vote pass (> half of repeats) over the rows both arms scored."""
    ids = sorted(set(baseline) & set(arm))
    base_only = [rid for rid in ids if baseline[rid] > 0.5 and not arm[rid] > 0.5]
    arm_only = [rid for rid in ids if arm[rid] > 0.5 and not baseline[rid] > 0.5]
    discordant = len(base_only) + len(arm_only)
    k = min(len(base_only), len(arm_only))
    p_value = min(1.0, 2 * sum(comb(discordant, i) for i in range(k + 1)) / 2**discordant) if discordant else 1.0
    return {
        'n_rows': len(ids),
        'baseline_only_pass': len(base_only),
        'arm_only_pass': len(arm_only),
        'both_pass': sum(1 for rid in ids if baseline[rid] > 0.5 and arm[rid] > 0.5),
        'both_fail': sum(1 for rid in ids if not baseline[rid] > 0.5 and not arm[rid] > 0.5),
        'p_value': p_value,
        'discordant_rows': {'baseline_only': base_only, 'arm_only': arm_only},
    }


def compare_to_baseline(arms: dict[str, Any]) -> dict[str, Any]:
    baseline = arms.get(BASELINE_ARM)
    if baseline is None or baseline['headline_a'] is None:
        return {}
    result: dict[str, Any] = {}
    for label, arm in arms.items():
        if label == BASELINE_ARM or arm['headline_a'] is None:
            continue
        mc = mcnemar_exact(baseline['headline_a']['row_rates'], arm['headline_a']['row_rates'])
        base_ff = baseline['false_filter_count']
        arm_ff = arm['false_filter_count']
        a_ok = arm['headline_a']['pass_rows'] >= baseline['headline_a']['pass_rows'] - 1
        ff_ok = arm_ff <= base_ff
        h_ok = arm['hallucinated_total'] == 0
        sens_arm = arm['headline_a']['without_prompt_examples']
        sens_base = baseline['headline_a']['without_prompt_examples']
        result[label] = {
            'mcnemar_headline_a': mc,
            'non_inferiority': {
                'rule': NON_INFERIORITY_RULE,
                'non_inferior': a_ok and ff_ok and h_ok,
                'headline_a': {
                    'arm_pass_rows': arm['headline_a']['pass_rows'],
                    'baseline_pass_rows': baseline['headline_a']['pass_rows'],
                    'rows': arm['headline_a']['rows'],
                    'ok': a_ok,
                },
                'false_filter': {
                    'arm_events': arm_ff,
                    'baseline_events': base_ff,
                    'arm_eligible': arm['false_filter_eligible'],
                    'baseline_eligible': baseline['false_filter_eligible'],
                    'arm_rate': arm['false_filter_rate'],
                    'baseline_rate': baseline['false_filter_rate'],
                    'arm_majority_rows': arm['false_filter_rows_majority'],
                    'baseline_majority_rows': baseline['false_filter_rows_majority'],
                    'arm_errored': arm['false_filter_errored'],
                    'ok': ff_ok,
                },
                'hallucinated': {'arm_total': arm['hallucinated_total'], 'ok': h_ok},
            },
            'sensitivity_without_prompt_examples': (
                {
                    'excluded': sens_arm['excluded'],
                    'rows': sens_arm['rows'],
                    'arm_pass_rows': sens_arm['pass_rows'],
                    'baseline_pass_rows': sens_base['pass_rows'],
                    'a_ok': sens_arm['pass_rows'] >= sens_base['pass_rows'] - 1,
                }
                if sens_arm and sens_base
                else None
            ),
        }
    return result


def noise_floor(queries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """llm+jev and llm run the identical compile_query, so needs_judgment disagreement is run-to-run noise."""
    labels = list(dict.fromkeys(q['arm'] for q in queries))
    llm = 'llm' if 'llm' in labels else next((label for label in labels if parse_arm(label)[0] == 'llm'), None)
    if BASELINE_ARM not in labels or llm is None:
        return None
    dims = {(q['arm'], q['id'], q['repeat']): q['actual'].get('dimensions') for q in queries}
    pairs = disagree = 0
    for (arm, rid, repeat), baseline_dims in dims.items():
        other = dims.get((llm, rid, repeat))
        if arm != BASELINE_ARM or baseline_dims is None or other is None:
            continue
        pairs += 1
        disagree += (baseline_dims > 0) != (other > 0)
    return {
        'arms': [BASELINE_ARM, llm],
        'pairs': pairs,
        'disagreements': disagree,
        'rate': disagree / pairs if pairs else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    queries = [row for row in rows if row['kind'] == 'query']
    calls: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        if row['kind'] == 'call':
            calls.setdefault((row['arm'], row['id'], row['repeat']), []).append(row)
    arms: dict[str, Any] = {}
    for arm in dict.fromkeys(query['arm'] for query in queries):
        arm_queries = [query for query in queries if query['arm'] == arm]
        result = metrics(arm_queries, calls)
        result['by_category'] = {
            category: metrics(
                [query for query in arm_queries if query['category'] == category],
                calls,
            )
            for category in sorted({query['category'] for query in arm_queries})
        }
        arms[arm] = result

    cache_runs = [row for row in rows if row['kind'] == 'cache_scenario']
    scenario_calls = [row for row in rows if row['kind'] == 'scenario_call']
    cache_scenarios: dict[str, Any] = {}
    for mode in dict.fromkeys(row['mode'] for row in cache_runs):
        states: dict[str, Any] = {}
        for state in ('cold', 'warm'):
            samples = [row for row in cache_runs if row['mode'] == mode and row['cache_state'] == state]
            calls_for_state = [
                call for call in scenario_calls if call['mode'] == mode and call['cache_state'] == state
            ]
            cache_events = [event for row in samples for event in _cache_events(row)]
            cache_elapsed = _cache_elapsed_ms(cache_events)
            cache_hit_elapsed = _cache_elapsed_ms(cache_events, True)
            cache_miss_elapsed = _cache_elapsed_ms(cache_events, False)
            wall_values = [row['wall_ms'] for row in samples if row.get('wall_ms') is not None]
            costs = [call.get('total_cost') for call in calls_for_state]
            partial_cost = any(call.get('cost_is_partial') for call in calls_for_state)
            states[state] = {
                'n': len(samples),
                'wall_ms_p50': percentile(wall_values, 0.5),
                'wall_ms_p95': percentile(wall_values, 0.95),
                'catalogue_cache_hit': cache_events[0].get('hit') if len(cache_events) == 1 else None,
                'catalogue_cache_events': len(cache_events),
                'catalogue_cache_hit_events': sum(event.get('hit') is True for event in cache_events),
                'catalogue_cache_miss_events': sum(event.get('hit') is False for event in cache_events),
                'catalogue_cache_elapsed_ms_p50': percentile(cache_elapsed, 0.5),
                'catalogue_cache_elapsed_ms_p95': percentile(cache_elapsed, 0.95),
                'catalogue_cache_hit_elapsed_ms_p50': percentile(cache_hit_elapsed, 0.5),
                'catalogue_cache_hit_elapsed_ms_p95': percentile(cache_hit_elapsed, 0.95),
                'catalogue_cache_miss_elapsed_ms_p50': percentile(cache_miss_elapsed, 0.5),
                'catalogue_cache_miss_elapsed_ms_p95': percentile(cache_miss_elapsed, 0.95),
                'calls': len(calls_for_state),
                'input_tokens': sum(call['input_tokens'] or 0 for call in calls_for_state)
                if calls_for_state and all(call.get('input_tokens') is not None for call in calls_for_state)
                else None,
                'output_tokens': sum(call['output_tokens'] or 0 for call in calls_for_state)
                if calls_for_state and all(call.get('output_tokens') is not None for call in calls_for_state)
                else None,
                'cost_usd': sum(costs)
                if costs and all(cost is not None for cost in costs) and not partial_cost
                else None,
                'unpriced_calls': sum(call.get('total_cost') is None for call in calls_for_state),
                'partial_cost_calls': sum(bool(call.get('cost_is_partial')) for call in calls_for_state),
                'errors': [row['error'] for row in samples if row.get('error')],
            }
        cache_scenarios[mode] = states
    return {
        'n_queries': len(queries),
        'cache_scenarios': cache_scenarios,
        'baseline_arm': BASELINE_ARM if BASELINE_ARM in arms else None,
        'arms': arms,
        'comparisons': compare_to_baseline(arms),
        'compiler_noise_floor': noise_floor(queries),
    }


def _fmt(value: float | None, *, pct: bool = False, digits: int = 0, money: bool = False) -> str:
    if value is None:
        return 'n/a'
    if money:
        return f'${value:.5f}'
    return f'{value * 100:.0f}%' if pct else f'{value:.{digits}f}'


def render_md(summary: dict[str, Any], meta: dict[str, Any] | None) -> str:
    lines = ['# Filter-selection benchmark summary', '']
    if meta:
        lines += [
            f'- modes: {", ".join(meta.get("arms", []))}; repeats/query {meta.get("repeats")}; warmups discarded {meta.get("warmup")}; concurrency {meta.get("arm_concurrency")}',
            f'- models: compiler `{meta.get("compiler_model")}`, classifier `{meta.get("classifier_model")}`, fast filter `{meta.get("fast_filter_model")}`',
            f'- model sources: `{json.dumps(meta.get("model_sources", {}), sort_keys=True)}`',
            f'- settings snapshot: `{meta.get("settings_path")}` sha256 `{meta.get("settings_sha256")}`',
            f'- dataset/catalogue sha256: `{meta.get("dataset_sha256")}` / `{meta.get("catalogue_sha256")}`',
            f'- dry run: {meta.get("dry_run")}; other benchmark processes at start: {meta.get("other_bench_processes_at_start")}',
            '',
        ]
    lines += [
        '## Per mode',
        '',
        '| mode | query runs | errors | p50 ms | p95 ms | input tokens/query | output tokens/query | cost/query | known / partial / missing cost queries | A majority | A mean | B majority | time accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for mode, metrics_result in summary['arms'].items():
        ha, hb = metrics_result['headline_a'], metrics_result['headline_b']
        lines.append(
            f'| {mode} | {metrics_result["n"]} | {metrics_result["n_errors"]} | '
            f'{_fmt(metrics_result["latency_ms_p50"])} | {_fmt(metrics_result["latency_ms_p95"])} | '
            f'{_fmt(metrics_result["mean_tokens_per_query"]["input"])} | {_fmt(metrics_result["mean_tokens_per_query"]["output"])} | '
            f'{_fmt(metrics_result["cost_per_query_reported_usd"], money=True)} | '
            f'{metrics_result["cost_per_query_known_queries"]} / {metrics_result["cost_per_query_partial_queries"]} / {metrics_result["cost_per_query_missing_queries"]} | '
            f'{ha["pass_rows"]}/{ha["rows"]} | {_fmt(ha["mean_pass_rate"], pct=True)} | '
            f'{hb["pass_rows"]}/{hb["rows"]} | {_fmt(metrics_result["time_match_rate"], pct=True)} |'
        )

    lines += [
        '',
        '## Production cache events for scored planner calls',
        '',
        '| mode | events | hits | misses | elapsed p50/p95 ms | hit p50/p95 ms | miss p50/p95 ms | sources |',
        '|---|---:|---:|---:|---|---|---|---|',
    ]
    for mode, metrics_result in summary['arms'].items():
        lines.append(
            f'| {mode} | {metrics_result["catalogue_cache_events"]} | '
            f'{metrics_result["catalogue_cache_hit_events"]} | {metrics_result["catalogue_cache_miss_events"]} | '
            f'{_fmt(metrics_result["catalogue_cache_elapsed_ms_p50"])} / '
            f'{_fmt(metrics_result["catalogue_cache_elapsed_ms_p95"])} | '
            f'{_fmt(metrics_result["catalogue_cache_hit_elapsed_ms_p50"])} / '
            f'{_fmt(metrics_result["catalogue_cache_hit_elapsed_ms_p95"])} | '
            f'{_fmt(metrics_result["catalogue_cache_miss_elapsed_ms_p50"])} / '
            f'{_fmt(metrics_result["catalogue_cache_miss_elapsed_ms_p95"])} | '
            f'`{json.dumps(metrics_result["catalogue_cache_sources"], sort_keys=True)}` |'
        )
    lines += [
        '',
        'Headline A is facets+numeric exact; Headline B also requires datetime exact in both modes. Repeat outcomes are first aggregated within each labelled query; paired comparisons use query rows, never repeats as independent samples.',
        '',
        '## Measured live-cache end-to-end scenarios',
        '',
        '| mode | state | e2e p50/p95 ms | cache events/hits/misses | elapsed p50/p95 ms | hit p50/p95 ms | miss p50/p95 ms | tokens in/out | cost | unpriced / partial calls | errors |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for mode, states in summary.get('cache_scenarios', {}).items():
        for state, values in states.items():
            lines.append(
                f'| {mode} | {state} | {_fmt(values.get("wall_ms_p50"))} / {_fmt(values.get("wall_ms_p95"))} | '
                f'{values.get("catalogue_cache_events")} / {values.get("catalogue_cache_hit_events")} / {values.get("catalogue_cache_miss_events")} | '
                f'{_fmt(values.get("catalogue_cache_elapsed_ms_p50"))} / {_fmt(values.get("catalogue_cache_elapsed_ms_p95"))} | '
                f'{_fmt(values.get("catalogue_cache_hit_elapsed_ms_p50"))} / {_fmt(values.get("catalogue_cache_hit_elapsed_ms_p95"))} | '
                f'{_fmt(values.get("catalogue_cache_miss_elapsed_ms_p50"))} / {_fmt(values.get("catalogue_cache_miss_elapsed_ms_p95"))} | '
                f'{_fmt(values.get("input_tokens"))} / {_fmt(values.get("output_tokens"))} | '
                f'{_fmt(values.get("cost_usd"), money=True)} | {values.get("unpriced_calls")} / {values.get("partial_cost_calls")} | '
                f'{"; ".join(values.get("errors", [])) or "none"} |'
            )
    lines += ['', '## Per-call usage and cost', '', '| mode | call | n | input | output | reasoning | cached | p50 ms | mean cost | unpriced | partial |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode, mode_metrics in summary['arms'].items():
        for call_name, call_metrics in mode_metrics['mean_tokens_by_call'].items():
            lines.append(
                f'| {mode} | {call_name} | {call_metrics["n"]} | {_fmt(call_metrics["input"])} | {_fmt(call_metrics["output"])} | '
                f'{_fmt(call_metrics["reasoning"])} | {_fmt(call_metrics["cached"])} | {_fmt(call_metrics["latency_ms_p50"])} | '
                f'{_fmt(call_metrics["cost_usd_mean"], money=True)} | {call_metrics["unpriced_calls"]} | {call_metrics["partial_cost_calls"]} |'
            )
    lines += ['', '## Offered catalogue coverage and flexibility', '', '| mode | paraphrase rows | facet precision | facet recall | facet exact | expected values outside offered catalogue |', '|---|---:|---:|---:|---:|---:|']
    for mode, mode_metrics in summary['arms'].items():
        flexibility = mode_metrics['flexibility']
        outside = mode_metrics['outside_catalogue']
        lines.append(
            f'| {mode} | {flexibility["n"]} | {_fmt(flexibility["facet_precision"], pct=True)} | '
            f'{_fmt(flexibility["facet_recall"], pct=True)} | {_fmt(flexibility["facet_exact_rate"], pct=True)} | {outside["n"]} |'
        )
    if summary.get('comparisons'):
        lines += ['', f'## Paired comparison vs `{summary["baseline_arm"]}`', '', '| mode | baseline-only pass rows | mode-only pass rows | both pass | both fail | exact McNemar p |', '|---|---:|---:|---:|---:|---:|']
        for mode, comparison in summary['comparisons'].items():
            mc = comparison['mcnemar_headline_a']
            lines.append(
                f'| {mode} | {mc["baseline_only_pass"]} | {mc["arm_only_pass"]} | {mc["both_pass"]} | '
                f'{mc["both_fail"]} | {mc["p_value"]:.3f} |'
            )
    lines += ['', '## Per mode and category', '', '| mode / category | query rows | p50 ms | cost/query | Headline A | Headline B |', '|---|---:|---:|---:|---:|---:|']
    for mode, mode_metrics in summary['arms'].items():
        for category, category_metrics in mode_metrics['by_category'].items():
            a, b = category_metrics['headline_a'], category_metrics['headline_b']
            a_text = f'{a["pass_rows"]}/{a["rows"]}' if a else 'n/a'
            b_text = f'{b["pass_rows"]}/{b["rows"]}' if b else 'n/a'
            lines.append(
                f'| {mode} / {category} | {category_metrics["n"]} | {_fmt(category_metrics["latency_ms_p50"])} | '
                f'{_fmt(category_metrics["cost_per_query_reported_usd"], money=True)} | {a_text} | {b_text} |'
            )
    lines.append('')
    return '\n'.join(lines)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_summary(out: Path) -> dict[str, Any]:
    """Regenerate derived scores and summaries while leaving raw.jsonl unchanged."""
    rows = read_rows(out / 'raw.jsonl')
    meta_path = out / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else None
    scored, info = rescore(rows, meta)
    if info['rescored']:
        with (out / 'scored.jsonl').open('w', encoding='utf-8') as handle:
            for row in scored:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = summarize(scored)
    summary['rescore'] = info
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (out / 'summary.md').write_text(render_md(summary, meta), encoding='utf-8')
    return summary


# ----------------------------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def other_bench_processes() -> list[int]:
    """PIDs of other running copies of this harness (their LLM traffic overlaps ours); Linux /proc only."""
    mine = {os.getpid(), os.getppid()}
    found = []
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if 'filter_selection/bench.py' in command and int(entry.name) not in mine and ' --summarize' not in command:
            found.append(int(entry.name))
    return found


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--arms', default=','.join(ALL_ARMS), help='comma-separated production modes: llm+jev,llm')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2, help='discarded queries per mode before measuring')
    parser.add_argument('--limit-datapoints', type=int, default=None)
    parser.add_argument('--ids', default=None, help='comma list of datapoint ids, e.g. q01,q02')
    parser.add_argument('--compiler-model', default=None, help='explicit override for finder.compiler')
    parser.add_argument('--classifier-model', default=None, help='explicit override for finder.classifier')
    parser.add_argument('--fast-filter-model', default=None, help='explicit override for finder.filters')
    parser.add_argument('--arm-concurrency', type=int, default=1, help='latency requires exactly one')
    parser.add_argument('--overlap-note', default=None)
    parser.add_argument('--profile', default='research-bauke')
    parser.add_argument('--dataset', default=str(HERE / 'dataset.jsonl'))
    parser.add_argument('--catalogue', default=str(HERE / 'catalogue.json'))
    parser.add_argument('--out', default=None)
    parser.add_argument('--dry-run', action='store_true', help='fake LLM outputs, no network')
    parser.add_argument('--summarize', metavar='DIR', default=None, help='regenerate scored/summary files from DIR/raw.jsonl')
    args = parser.parse_args(argv)
    if args.summarize:
        return args
    args.arms = [arm.strip() for arm in args.arms.split(',') if arm.strip()]
    if not args.arms or len(set(args.arms)) != len(args.arms):
        parser.error('--arms must be a non-empty list without duplicates')
    unknown = set(args.arms) - set(ALL_ARMS)
    if unknown:
        parser.error(f'unknown arm(s) {sorted(unknown)!r}; supported modes: {", ".join(ALL_ARMS)}')
    if args.arm_concurrency != 1:
        parser.error('latency-grade benchmark runs require --arm-concurrency 1')
    if args.repeats < 1:
        parser.error('--repeats must be >= 1')
    if args.warmup < 0 or (args.warmup < 1 and not args.dry_run):
        parser.error('--warmup must be >= 1 for live runs')
    return args


def resolve_models(args) -> tuple[dict[str, str], dict[str, str], dict[str, Any], dict[str, Any]]:
    from evaluatorq.trace_finder.settings import settings_path

    roles = (
        ('compiler', args.compiler_model, 'finder.compiler', 'fast'),
        ('classifier', args.classifier_model, 'finder.classifier', 'classifier'),
        ('fast_filter', args.fast_filter_model, 'finder.filters', 'fast'),
    )
    models: dict[str, str] = {}
    sources: dict[str, str] = {}
    role_report: dict[str, Any] = {}
    for key, override, task, role in roles:
        try:
            configured = role_model(role, task=task)
            source = role_source(role, task)
        except Exception as exc:  # noqa: BLE001 - surface settings resolution failures
            raise SystemExit(f'cannot resolve configured model for {task}: {type(exc).__name__}: {exc}') from exc
        models[key] = override or configured
        sources[key] = 'flag' if override else f'role:{source}'
        role_report[task] = {'role_model': configured, 'source': source}
    configured_path = settings_path().expanduser()
    settings_snapshot = {
        'path': str(configured_path.resolve()),
        'sha256': sha256(configured_path) if configured_path.is_file() else None,
    }
    return models, sources, role_report, settings_snapshot


def load_inputs(args) -> tuple[list[dict[str, Any]], FacetCatalogue, datetime, dict[str, Any]]:
    dataset_path, catalogue_path = Path(args.dataset), Path(args.catalogue)
    snapshot = json.loads(catalogue_path.read_text(encoding='utf-8'))
    catalogue = FacetCatalogue.model_validate(snapshot['catalogue'])
    datapoints = [json.loads(line) for line in dataset_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not datapoints:
        raise SystemExit(f'{dataset_path} has no datapoints')
    for dp in datapoints:
        for facet, values in dp['expected'].get('facets', {}).items():
            if facet not in FACET_NAMES:
                raise SystemExit(f'{dp["id"]}: unknown facet {facet!r}')
            for value in values:
                if value not in getattr(catalogue, facet):
                    raise SystemExit(f'{dp["id"]}: {facet}={value!r} is not in the catalogue')
    nows = {dp['now'] for dp in datapoints}
    if len(nows) != 1:
        raise SystemExit(f'datapoints disagree on now: {nows}')
    now = parse_time(nows.pop())
    if now is None:
        raise SystemExit('unparseable now')
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(',') if i.strip()]
        by_id = {dp['id']: dp for dp in datapoints}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f'unknown datapoint ids {missing}')
        datapoints = [by_id[i] for i in wanted]
    if args.limit_datapoints:
        datapoints = datapoints[: args.limit_datapoints]
    meta = {
        'dataset_path': str(dataset_path),
        'dataset_sha256': sha256(dataset_path),
        'catalogue_path': str(catalogue_path),
        'catalogue_sha256': sha256(catalogue_path),
        'n_datapoints': len(datapoints),
    }
    return datapoints, catalogue, now, meta


def build_clients(args):
    """Return (LLM client, Orq client, resolved); dry-run uses a local SDK fixture."""
    if args.dry_run:
        from evaluatorq.common.orq_client import resolve_orq_client

        return None, resolve_orq_client('filter-selection-dry-run', base_url='http://127.0.0.1'), None
    from evaluatorq.common.llm_client import resolve_llm_client
    from evaluatorq.common.orq_client import list_orq_profiles, resolve_orq_client

    profile = next((p for p in list_orq_profiles() if p.name == args.profile), None)
    if profile is None:
        raise SystemExit(f'orq profile {args.profile!r} not found (or has no readable key)')
    resolved = resolve_llm_client(
        require_orq=True, max_retries=0, extra_api_key=profile.api_key, orq_host=profile.server
    )
    orq = resolve_orq_client(profile.api_key, base_url=profile.server)
    return resolved.client, orq, resolved






def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize:
        out = Path(args.summarize)
        summary = write_summary(out)
        print(f'wrote {out / "summary.json"} and {out / "summary.md"} ({summary["n_queries"]} queries)')
        return 0

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    arm_slug = '+'.join(re.sub(r'[^\w.]+', '-', label) for label in args.arms)
    out = Path(args.out) if args.out else HERE / 'results' / f'{stamp}_{arm_slug}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'raw.jsonl').exists() or (out / 'meta.json').exists():
        raise SystemExit(f'refusing to overwrite existing benchmark artifacts in {out}')

    datapoints, catalogue, now, input_meta = load_inputs(args)
    models, model_sources, role_report, settings_snapshot = resolve_models(args)
    specs = [ArmSpec(mode, mode) for mode in args.arms]
    client, orq, resolved = build_clients(args)
    real_classify = fake_classify if args.dry_run else selector_mod.run_classify
    _GENERATE[0] = fake_generate if args.dry_run else generate_structured
    compiler_mod.generate_structured = instrumented_generate('compiler', _GENERATE[0])
    selector_mod.generate_structured = instrumented_generate('llm_filter', _GENERATE[0])
    selector_mod.run_classify = instrumented_classify(real_classify)
    patch_structured_output()

    offered_counts = {name: len(getattr(catalogue, name)) for name in FACET_NAMES}
    overlapping = other_bench_processes()
    meta = {
        'arms': args.arms,
        'arm_models': {spec.label: models['fast_filter'] if spec.mode == 'llm' else models['classifier'] for spec in specs},
        'arm_concurrency': 1,
        'repeats': args.repeats,
        'warmup': args.warmup,
        'ids': args.ids,
        'limit_datapoints': args.limit_datapoints,
        'compiler_model': models['compiler'],
        'classifier_model': models['classifier'],
        'fast_filter_model': models['fast_filter'],
        'model_sources': model_sources,
        'role_models': role_report,
        'settings_path': settings_snapshot['path'],
        'settings_sha256': settings_snapshot['sha256'],
        'cwd': os.getcwd(),
        'catalogue_values_offered': sum(offered_counts.values()),
        'catalogue_values_by_facet': offered_counts,
        'catalogue_limit_per_facet': 50,
        'catalogue_loader': 'production planner seam; fixed snapshot injected for scored workload',
        'cache_scenario': 'live production planner/cache, invalidated cold then immediately repeated warm',
        'profile': None if args.dry_run else args.profile,
        'dry_run': args.dry_run,
        'now': _iso(now),
        'started_at': datetime.now(timezone.utc).isoformat(),
        'git_sha': git_sha(),
        'bench_sha256': sha256(Path(__file__)),
        **input_meta,
        'other_bench_processes_at_start': len(overlapping),
        'overlap_note': args.overlap_note,
        'python': platform.python_version(),
        'packages': {name: _version(name) for name in ('evaluatorq', 'openai', 'pydantic')},
        'out_dir': str(out),
    }
    meta_path = out / 'meta.json'
    meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    ctx = Ctx(args, client, orq, catalogue, now, models, specs)

    async def guarded() -> None:
        try:
            await run_benchmark(args, ctx, datapoints, out)
        finally:
            if resolved is not None and resolved.owned:
                await resolved.client.close()
            if orq is not None:
                from evaluatorq.common.orq_client import close_orq_client

                await close_orq_client(orq)

    asyncio.run(guarded())
    meta['finished_at'] = datetime.now(timezone.utc).isoformat()
    meta['other_bench_processes_at_end'] = len(other_bench_processes())
    meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    summary = write_summary(out)
    print(f'results: {out}')
    for mode, mode_metrics in summary['arms'].items():
        headline_a = mode_metrics['headline_a']
        print(
            f'{mode:8s} n={mode_metrics["n"]} err={mode_metrics["n_errors"]} '
            f'A={headline_a["pass_rows"]}/{headline_a["rows"]} '
            f'p50={_fmt(mode_metrics["latency_ms_p50"])}ms '
            f'$ / query={_fmt(mode_metrics["cost_per_query_reported_usd"], money=True)} '
            f'full_exact={_fmt(mode_metrics["full_exact_rate"], pct=True)}'
        )
    return 0


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


if __name__ == '__main__':
    raise SystemExit(main())
