"""Group B signals: tool errors, retries, loops, schemas and result sizes."""

from __future__ import annotations

import json
import re
import shlex
from collections import defaultdict
from typing import Any, Literal

from evaluatorq.formats._shared import RAW_ARGUMENTS_KEY
from evaluatorq.signals import preconditions as pre
from evaluatorq.signals.models import Evidence, SignalFn, SignalResult, result
from evaluatorq.signals.walk import CallRecord, SignalContext, result_text, tool_schemas

_CLIS = frozenset({'git', 'orq', 'gh', 'uv', 'docker', 'kubectl', 'npm', 'pnpm', 'yarn', 'bun', 'cargo', 'go'})
_ASSIGNMENT = re.compile(r'[A-Za-z_][A-Za-z_0-9]*=.*')


def _call_ev(record: CallRecord, reason: str = '') -> Evidence:
    return Evidence(
        step_id=record.step.step.step_id,
        agent_path=record.step.agent_path,
        call_id=record.call.tool_call_id,
        reason=reason,
    )


def _result_ev(record: CallRecord, reason: str = '') -> Evidence:
    return Evidence(
        step_id=record.step.step.step_id,
        agent_path=record.step.agent_path,
        call_id=record.call.tool_call_id,
        reason=reason,
    )


def _related(record: CallRecord) -> list[tuple[tuple[str, ...], int]]:
    return [(record.step.agent_path, record.step.step.step_id)]


def _error_reason(record: CallRecord, ctx: SignalContext) -> str | None:
    if record.result is None:
        return None
    if record.is_error:
        return 'status'
    if ctx.config.error_detection == 'status_and_content':
        text = result_text(record.result)
        for pattern in ctx.config.error_patterns:
            if pattern in text:
                return f'matched {pattern!r}'
    return None


def _error_evidence(failed: list[tuple[CallRecord, str]]) -> list[Evidence]:
    return [
        Evidence(
            step_id=record.step.step.step_id,
            agent_path=record.step.agent_path,
            call_id=record.call.tool_call_id,
            related=_related(record),
            related_call_ids=[record.call.tool_call_id],
            reason='' if why == 'status' else why,
        )
        for record, why in failed
    ]


def _failed(ctx: SignalContext) -> list[tuple[CallRecord, str]]:
    return [(record, why) for record in ctx.calls if (why := _error_reason(record, ctx))]


def tool_error_count(ctx: SignalContext) -> SignalResult:
    failed = _failed(ctx)
    return result(
        'tool_error_count',
        'B',
        len(failed),
        _error_evidence(failed),
        [pre.explicit_error_status(ctx), pre.results_matched(ctx)],
    )


def tool_error_rate(ctx: SignalContext) -> SignalResult:
    failed = _failed(ctx)
    pcs = [pre.has_tool_calls(ctx), pre.explicit_error_status(ctx), pre.results_matched(ctx)]
    value = round(len(failed) / len(ctx.calls), 4) if ctx.calls else None
    return result('tool_error_rate', 'B', value, _error_evidence(failed), pcs)


def _arguments(record: CallRecord) -> dict[str, Any]:
    args = record.call.arguments
    return args if isinstance(args, dict) and RAW_ARGUMENTS_KEY not in args else {}


def _canonical(record: CallRecord, ctx: SignalContext) -> str:
    args = record.call.arguments
    try:
        return json.dumps(args, sort_keys=ctx.config.canonicalisation == 'sorted_keys', ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def _fingerprint(record: CallRecord, ctx: SignalContext) -> tuple[str, str]:
    return record.call.function_name, _canonical(record, ctx)


def _repeat_evidence(ctx: SignalContext) -> list[Evidence]:
    first: dict[tuple[str, str], CallRecord] = {}
    evidence = []
    for record in ctx.calls:
        key = _fingerprint(record, ctx)
        if key in first:
            original = first[key]
            evidence.append(
                _call_ev(record).model_copy(
                    update={'related': _related(original), 'related_call_ids': [original.call.tool_call_id]}
                )
            )
        else:
            first[key] = record
    return evidence


def duplicate_tool_call_count(ctx: SignalContext) -> SignalResult:
    evidence = _repeat_evidence(ctx)
    return result('duplicate_tool_call_count', 'B', len(evidence), evidence, [pre.args_parseable(ctx)])


def _retry_links(ctx: SignalContext) -> dict[int, CallRecord]:
    streams: dict[tuple[str, ...], list[CallRecord]] = defaultdict(list)
    for record in ctx.calls:
        streams[record.step.agent_path].append(record)
    links: dict[int, CallRecord] = {}
    for stream in streams.values():
        for position, record in enumerate(stream):
            if ctx.config.retry_definition == 'consecutive_same_tool':
                window = stream[max(0, position - 1) : position]
                matches = [prior for prior in window if prior.call.function_name == record.call.function_name]
            else:
                window = stream[max(0, position - ctx.config.retry_window) : position]
                key = _fingerprint(record, ctx)
                matches = [prior for prior in window if _fingerprint(prior, ctx) == key]
            failed = [prior for prior in matches if _error_reason(prior, ctx)]
            if failed:
                links[record.seq] = failed[-1]
    return links


def _changed_args(before: CallRecord, after: CallRecord) -> list[str]:
    left, right = _arguments(before), _arguments(after)
    if not left and not right:
        return [] if before.call.arguments == after.call.arguments else ['<arguments>']
    return sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))


def _retry_evidence(record: CallRecord, failed: CallRecord, ctx: SignalContext) -> Evidence:
    reason = ''
    if ctx.config.retry_definition == 'consecutive_same_tool':
        changed = _changed_args(failed, record)
        reason = f'changed args: {", ".join(changed)}' if changed else 'same args'
    return _call_ev(record, reason).model_copy(
        update={'related': _related(failed), 'related_call_ids': [failed.call.tool_call_id]}
    )


def tool_retry_count(ctx: SignalContext) -> SignalResult:
    links = _retry_links(ctx)
    evidence = [_retry_evidence(r, links[r.seq], ctx) for r in ctx.calls if r.seq in links]
    return result(
        'tool_retry_count', 'B', len(evidence), evidence, [pre.explicit_error_status(ctx), pre.results_matched(ctx)]
    )


def tool_succeeded_after_retry_count(ctx: SignalContext) -> SignalResult:
    links = _retry_links(ctx)
    evidence = []
    for record in ctx.calls:
        if record.seq not in links or record.result is None or _error_reason(record, ctx):
            continue
        chain = []
        cursor = links[record.seq]
        while True:
            chain.append(cursor)
            if cursor.seq not in links:
                break
            cursor = links[cursor.seq]
        chain.reverse()
        evidence.append(
            _call_ev(record).model_copy(
                update={
                    'related': sorted({_related(item)[0] for item in chain}),
                    'related_call_ids': [item.call.tool_call_id for item in chain],
                }
            )
        )
    return result(
        'tool_succeeded_after_retry_count',
        'B',
        len(evidence),
        evidence,
        [pre.explicit_error_status(ctx), pre.results_matched(ctx)],
    )


def _schema_violation(schema: dict[str, Any], arguments: Any) -> str | None:
    """Return the first validation error, or None when valid/unusable; jsonschema stays optional."""
    try:
        import jsonschema
        from jsonschema.validators import validator_for
    except ImportError:
        return 'jsonschema unavailable'
    try:
        validator = validator_for(schema, default=jsonschema.Draft202012Validator)
        validator.check_schema(schema)
        errors = sorted(validator(schema).iter_errors(arguments), key=lambda error: list(error.absolute_path))
    except (jsonschema.SchemaError, TypeError, ValueError):
        return None
    if not errors:
        return None
    error = errors[0]
    path = '/'.join(str(part) for part in error.absolute_path) or '$'
    text = re.sub(r'\s+', ' ', f'{path}: {error.message}').strip()
    return text if len(text) <= 160 else text[:159] + '…'


def invalid_schema_tool_call_count(ctx: SignalContext) -> SignalResult:
    pcs = pre.tool_schemas_coverage(ctx)
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        from evaluatorq.signals.models import SignalResult

        return SignalResult(
            name='invalid_schema_tool_call_count',
            group='B',
            no_basis='jsonschema is unavailable',
            preconditions=pcs,
        )
    evidence = []
    root_schemas = tool_schemas(ctx.trajectory)
    for record in ctx.calls:
        schemas = tool_schemas(record.step.trajectory)
        definition = schemas.get(record.call.function_name) or root_schemas.get(record.call.function_name)
        schema = definition.get('parameters') if definition else None
        if not isinstance(schema, dict):
            continue
        args = record.call.arguments
        violation = 'arguments are not valid JSON' if RAW_ARGUMENTS_KEY in args else _schema_violation(schema, args)
        if violation and violation != 'jsonschema unavailable':
            evidence.append(_call_ev(record, f'{record.call.function_name}: {violation}'))
    return result('invalid_schema_tool_call_count', 'B', len(evidence), evidence, pcs)


def _streams(ctx: SignalContext) -> list[list[CallRecord]]:
    streams: dict[tuple[str, ...], list[CallRecord]] = defaultdict(list)
    for record in ctx.calls:
        streams[record.step.agent_path].append(record)
    return list(streams.values())


def _shell_family(record: CallRecord, ctx: SignalContext) -> str | None:
    if ctx.config.tool_role(record.call.function_name) != 'bash':
        return None
    command = _arguments(record).get('command', _arguments(record).get('cmd'))
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|')
        lexer.whitespace_split = True
        lexer.commenters = '#'
        tokens = list(lexer)
    except ValueError:
        return None
    while tokens and _ASSIGNMENT.fullmatch(tokens[0]):
        tokens.pop(0)
    if len(tokens) >= 3 and tokens[0] == 'cd' and tokens[2] == '&&':
        tokens = tokens[3:]
    if not tokens or tokens[0] in {';', '&&', '||', '|'}:
        return None
    executable = tokens[0].rsplit('/', 1)[-1]
    if (
        executable in _CLIS
        and len(tokens) > 1
        and tokens[1] not in {';', '&&', '||', '|'}
        and not tokens[1].startswith('-')
    ):
        return f'{executable} {tokens[1]}'
    return executable


def consecutive_same_tool_max(ctx: SignalContext) -> SignalResult:
    runs = []
    for stream in _streams(ctx):
        run: list[CallRecord] = []
        for record in stream:
            if run and run[-1].call.function_name != record.call.function_name:
                runs.append(run)
                run = []
            run.append(record)
        if run:
            runs.append(run)
    longest = max((len(run) for run in runs), default=0)
    evidence = [
        _call_ev(run[0], f'{len(run)}x {run[0].call.function_name}').model_copy(
            update={
                'related': [_related(item)[0] for item in run[1:]],
                'related_call_ids': [item.call.tool_call_id for item in run[1:]],
            }
        )
        for run in runs
        if len(run) == longest and longest > 1
    ]
    return result('consecutive_same_tool_max', 'B', longest, evidence)


def consecutive_command_family_max(ctx: SignalContext) -> SignalResult:
    runs: list[tuple[str, list[CallRecord]]] = []
    for stream in _streams(ctx):
        family = None
        run: list[CallRecord] = []
        for record in stream:
            current = _shell_family(record, ctx)
            if run and current != family:
                runs.append((family or '', run))
                run = []
            if current:
                run.append(record)
            family = current
        if run and family:
            runs.append((family, run))
    longest = max((len(run) for _, run in runs), default=0)
    evidence = [
        _call_ev(run[0], f'{len(run)}x {family}').model_copy(
            update={
                'related': [_related(item)[0] for item in run[1:]],
                'related_call_ids': [item.call.tool_call_id for item in run[1:]],
            }
        )
        for family, run in runs
        if len(run) > 1
    ]
    return result('consecutive_command_family_max', 'B', longest, evidence)


def _episodes(ctx: SignalContext) -> list[tuple[str, list[CallRecord]]]:
    episodes = []
    for stream in _streams(ctx):
        keys = [_fingerprint(record, ctx) for record in stream]
        start = 0
        while start < len(stream):
            end = start + 1
            while end < len(stream) and keys[end] == keys[start]:
                end += 1
            if end - start >= 3:
                episodes.append(('identical retry', stream[start:end]))
            start = end
        for start in range(len(stream) - 3):
            if keys[start] == keys[start + 1] or keys[start] != keys[start + 2] or keys[start + 1] != keys[start + 3]:
                continue
            if start and keys[start - 1] == keys[start + 1]:
                continue
            end = start + 4
            while end < len(stream) and keys[end] == keys[end - 2]:
                end += 1
            episodes.append(('oscillation', stream[start:end]))
    return sorted(episodes, key=lambda episode: episode[1][0].seq)


def _episode_evidence(kind: str, run: list[CallRecord], ctx: SignalContext) -> Evidence:
    first = run[0]
    first_label = _shell_family(first, ctx) or first.call.function_name
    if kind == 'identical retry':
        reason = f'{len(run)}x identical {first_label}'
    else:
        second_label = _shell_family(run[1], ctx) or run[1].call.function_name
        pair = (
            f'{first_label} ↔ {second_label}' if first_label != second_label else f'{first_label} (two argument sets)'
        )
        reason = f'{len(run)}-call oscillation: {pair}'
    return _call_ev(first, reason).model_copy(
        update={
            'related': [_related(record)[0] for record in run[1:]],
            'related_call_ids': [record.call.tool_call_id for record in run[1:]],
        }
    )


def identical_tool_call_run_count(ctx: SignalContext) -> SignalResult:
    episodes = [(kind, run) for kind, run in _episodes(ctx) if kind == 'identical retry']
    return result(
        'identical_tool_call_run_count', 'B', len(episodes), [_episode_evidence(k, r, ctx) for k, r in episodes]
    )


def tool_oscillation_count(ctx: SignalContext) -> SignalResult:
    episodes = [(kind, run) for kind, run in _episodes(ctx) if kind == 'oscillation']
    return result('tool_oscillation_count', 'B', len(episodes), [_episode_evidence(k, r, ctx) for k, r in episodes])


def tool_loop_count(ctx: SignalContext) -> SignalResult:
    episodes = _episodes(ctx)
    return result('tool_loop_count', 'B', len(episodes), [_episode_evidence(k, r, ctx) for k, r in episodes])


def distinct_tool_arg_ratio(ctx: SignalContext) -> SignalResult:
    distinct = len({_fingerprint(record, ctx) for record in ctx.calls})
    value = round(distinct / len(ctx.calls), 4) if ctx.calls else None
    return result(
        'distinct_tool_arg_ratio', 'B', value, _repeat_evidence(ctx), [pre.has_tool_calls(ctx), pre.args_parseable(ctx)]
    )


def _empty_kind(record: CallRecord, ctx: SignalContext) -> str | None:
    if record.result is None:
        return None
    text = result_text(record.result).strip()
    if text in ctx.config.empty_literals:
        return text
    if text == '' and '' in ctx.config.empty_values:
        return ''
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return None
    kinds = {'[]': decoded == [], '{}': decoded == {}, 'null': decoded is None, '': decoded == ''}
    return next((kind for kind, match in kinds.items() if match and kind in ctx.config.empty_values), None)


def empty_tool_result_count(ctx: SignalContext) -> SignalResult:
    evidence = []
    for record in ctx.calls:
        kind = _empty_kind(record, ctx)
        if kind is not None:
            evidence.append(_result_ev(record, f'empty: {kind!r}').model_copy(update={'related': _related(record)}))
    return result('empty_tool_result_count', 'B', len(evidence), evidence, [pre.result_content_available(ctx)])


def _sized(ctx: SignalContext) -> list[tuple[CallRecord, int]]:
    return [
        (record, len(result_text(record.result).encode('utf-8'))) for record in ctx.calls if record.result is not None
    ]


def max_tool_result_bytes(ctx: SignalContext) -> SignalResult:
    sized = _sized(ctx)
    largest = max((size for _, size in sized), default=None)
    evidence = [
        _result_ev(record, f'{size} bytes').model_copy(update={'related': _related(record)})
        for record, size in sized
        if size == largest
    ]
    return result(
        'max_tool_result_bytes', 'B', largest, evidence, [pre.has_tool_results(ctx), pre.result_content_available(ctx)]
    )


def total_tool_result_bytes(ctx: SignalContext) -> SignalResult:
    sized = _sized(ctx)
    evidence = [
        _result_ev(record, f'{size} bytes').model_copy(update={'related': _related(record)}) for record, size in sized
    ]
    return result(
        'total_tool_result_bytes', 'B', sum(size for _, size in sized), evidence, [pre.result_content_available(ctx)]
    )


SIGNALS: dict[str, tuple[Literal['B'], SignalFn]] = {
    'tool_error_count': ('B', tool_error_count),
    'tool_error_rate': ('B', tool_error_rate),
    'duplicate_tool_call_count': ('B', duplicate_tool_call_count),
    'tool_retry_count': ('B', tool_retry_count),
    'tool_succeeded_after_retry_count': ('B', tool_succeeded_after_retry_count),
    'invalid_schema_tool_call_count': ('B', invalid_schema_tool_call_count),
    'consecutive_same_tool_max': ('B', consecutive_same_tool_max),
    'consecutive_command_family_max': ('B', consecutive_command_family_max),
    'identical_tool_call_run_count': ('B', identical_tool_call_run_count),
    'tool_oscillation_count': ('B', tool_oscillation_count),
    'tool_loop_count': ('B', tool_loop_count),
    'distinct_tool_arg_ratio': ('B', distinct_tool_arg_ratio),
    'empty_tool_result_count': ('B', empty_tool_result_count),
    'max_tool_result_bytes': ('B', max_tool_result_bytes),
    'total_tool_result_bytes': ('B', total_tool_result_bytes),
}
