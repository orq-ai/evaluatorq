"""Preconditions over the walked view."""

from __future__ import annotations

from evaluatorq.formats.atif import AtifMetrics
from evaluatorq.signals import preconditions as pre
from evaluatorq.signals.config import SignalsConfig
from evaluatorq.signals.walk import calls, walk

from .conftest import agent, bare, call, failed, ok, spawned, traj, user

USAGE = AtifMetrics(prompt_tokens=10, completion_tokens=2)


def test_call_and_result_checks() -> None:
    empty = calls(walk(traj([user()])))
    assert pre.has_tool_calls(empty).met is False
    assert pre.results_matched(empty).met is True
    recs = calls(walk(traj([agent(calls=[call('A', call_id='a'), call('B', call_id='b')], results=[ok(call_id='a')])])))
    assert pre.has_tool_calls(recs).met is True
    assert pre.has_tool_results(recs).met is True
    matched = pre.results_matched(recs)
    assert matched.met == 'partial'
    assert not matched.required
    assert 'B @step 1' in matched.detail


def test_explicit_error_status_is_required_only_for_status_detection() -> None:
    recs = calls(walk(traj([agent(calls=[call('A')], results=[bare()])])))
    assert pre.explicit_error_status(recs, SignalsConfig()).met is False
    assert pre.explicit_error_status(recs, SignalsConfig()).required
    sniff = pre.explicit_error_status(recs, SignalsConfig(error_detection='status_and_content'))
    assert (sniff.required, 'content sniffing only' in sniff.detail) == (False, True)


def test_args_parseable_flags_raw_arguments() -> None:
    recs = calls(walk(traj([agent(calls=[call('A', {'_raw': '{oops'}, 'a'), call('B', {'x': 1}, 'b')])])))
    got = pre.args_parseable(recs)
    assert got.met == 'partial'
    assert 'A @step 1' in got.detail


def test_result_content_available_reads_the_truncation_marker() -> None:
    recs = calls(walk(traj([agent(calls=[call('A')], results=[ok(content_truncated=True)])])))
    assert pre.result_content_available(recs).met is False


def test_tool_schemas_coverage() -> None:
    defs = [
        {'type': 'function', 'function': {'name': 'A', 'parameters': {}}},
        {'name': 'B', 'parameters': {}},
    ]
    recs = calls(walk(traj([agent(calls=[call('A', call_id='a'), call('C', call_id='c')])], tool_definitions=defs)))
    root = traj([agent()], tool_definitions=defs)
    present, cover = pre.tool_schemas_coverage(root, recs)
    assert present.met is True
    assert cover.met == 'partial'
    assert 'C' in cover.detail
    bare_recs = calls(walk(traj([agent(calls=[call('A', call_id='a')])])))
    missing = pre.tool_schemas_coverage(traj([agent()]), bare_recs)
    assert [(p.name, p.met) for p in missing] == [('tool schemas present', False)]


def test_llm_step_checks() -> None:
    inv = {'invocation': {'finish_reasons': ['stop'], 'start_timestamp': 1.0, 'end_timestamp': 2.0}}
    full = walk(traj([user(), agent(model='openai/gpt-x', metrics=USAGE, extra=inv)]))
    assert pre.model_present(full).met is True
    assert pre.token_usage_complete(full).met is True
    assert pre.finish_reason_present(full).met is True
    assert pre.provider_derivable(full).met is True
    assert pre.timestamps(full).met is True
    mixed = walk(traj([agent(model='claude-x', metrics=USAGE), agent(model='claude-x')]))
    assert pre.token_usage_complete(mixed).met is False
    provider = pre.provider_derivable(mixed)
    assert provider.met == 'partial' and 'inferred' in provider.detail
    assert pre.finish_reason_present(mixed).met is False
    none = walk(traj([user()]))
    assert pre.model_present(none).met is False
    assert pre.token_usage_complete(none).met is False


def test_timestamps_fall_back_to_iso_as_partial() -> None:
    iso = walk(traj([agent(model='m', timestamp='2026-01-01T00:00:00Z')]))
    assert pre.timestamps(iso).met == 'partial'
    assert pre.timestamps(walk(traj([agent(model='m')]))).met is False


def test_subagent_checks() -> None:
    child = traj([agent('c')], trajectory_id='child')
    stray = traj([agent('s')], trajectory_id='stray')
    linked = traj([agent(calls=[call('Task')], results=[spawned('child')])], subagents=[child])
    walked = walk(linked)
    assert pre.subagent_linkage(walked, calls(walked), SignalsConfig()).met is True
    assert pre.has_subagents(walked).met is True
    unlinked = traj([agent(calls=[call('Task')], results=[ok()])], subagents=[stray])
    walked = walk(unlinked)
    got = pre.subagent_linkage(walked, calls(walked), SignalsConfig())
    assert got.met is False
    assert 'Task @step 1' in got.detail and 'stray' in got.detail
    solo = walk(traj([agent(calls=[call('Read')], results=[failed()])]))
    assert pre.has_subagents(solo).met is False
    assert pre.subagent_linkage(solo, calls(solo), SignalsConfig()).met is True
