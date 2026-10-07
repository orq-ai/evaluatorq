"""Preconditions over the shared `SignalContext`."""

from __future__ import annotations

from evaluatorq.formats.atif import AtifMetrics
from evaluatorq.signals import preconditions as pre
from evaluatorq.signals.config import SignalsConfig

from .conftest import context, agent, bare, call, failed, ok, spawned, traj, user

USAGE = AtifMetrics(prompt_tokens=10, completion_tokens=2)


def test_call_and_result_checks() -> None:
    empty = context(traj([user()]))
    assert pre.has_tool_calls(empty).met is False
    assert pre.results_matched(empty).met is True
    recs = context(traj([agent(calls=[call('A', call_id='a'), call('B', call_id='b')], results=[ok(call_id='a')])]))
    assert pre.has_tool_calls(recs).met is True
    assert pre.has_tool_results(recs).met is True
    matched = pre.results_matched(recs)
    assert matched.met == 'partial'
    assert not matched.required
    assert 'B @step 1' in matched.detail


def test_source_tool_coverage_finds_calls_preserved_outside_atif_tool_calls() -> None:
    extra = {'evaluatorq.responses_output_items': [
        {'type': 'custom_tool_call', 'call_id': 'c', 'name': 'lookup'},
        {'type': 'mcp_call', 'name': 'remote'},
    ]}
    recs = context(traj([agent(extra=extra)]))
    coverage = pre.source_tool_coverage(recs)
    assert coverage.met is False
    assert 'lookup' in coverage.detail and 'remote' in coverage.detail


def test_source_tool_coverage_matches_function_calls_by_source_id() -> None:
    extra = {'evaluatorq.responses_output_items': [
        {'type': 'function_call', 'call_id': 'c', 'name': 'lookup', 'arguments': '{}'},
    ]}
    recs = context(traj([agent(calls=[call('lookup', call_id='c')], extra=extra)]))
    coverage = pre.source_tool_coverage(recs)
    assert coverage.met is True


def test_source_tool_coverage_finds_unsupported_output_without_raw_call_item() -> None:
    extra = {'evaluatorq.responses_output_items': [
        {'type': 'custom_tool_call_output', 'call_id': 'c', 'output': 'done'},
        {'type': 'function_call_output', 'call_id': 'standard', 'output': 'mapped'},
    ]}
    coverage = pre.source_tool_coverage(context(traj([agent(extra=extra)])))
    assert coverage.met is False
    assert 'custom_tool_call_output @step 1' in coverage.detail
    assert 'function_call_output' not in coverage.detail


def test_explicit_error_status_is_required_only_for_status_detection() -> None:
    recs = context(traj([agent(calls=[call('A')], results=[bare()])]))
    assert pre.explicit_error_status(context(recs.trajectory)).met is False
    assert pre.explicit_error_status(context(recs.trajectory)).required
    sniff = pre.explicit_error_status(context(recs.trajectory, SignalsConfig(error_detection='status_and_content')))
    assert (sniff.required, 'content sniffing only' in sniff.detail) == (False, True)


def test_args_parseable_flags_raw_arguments() -> None:
    raw = call('A', {}, 'a').model_copy(update={'extra': {'evaluatorq.raw_arguments': '{oops'}})
    recs = context(traj([agent(calls=[raw, call('B', {'_raw': '{valid JSON key}'}, 'b')])]))
    got = pre.args_parseable(recs)
    assert got.met == 'partial'
    assert 'A @step 1' in got.detail


def test_literal_raw_key_is_parseable() -> None:
    recs = context(traj([agent(calls=[call('A', {'_raw': 'literal'}, 'a')])]))
    assert pre.args_parseable(recs).met is True


def test_result_content_available_reads_the_truncation_marker() -> None:
    recs = context(traj([agent(calls=[call('A')], results=[ok(content_truncated=True)])]))
    assert pre.result_content_available(recs).met is False


def test_tool_schemas_coverage() -> None:
    defs = [
        {'type': 'function', 'function': {'name': 'A', 'parameters': {}}},
        {'name': 'B', 'parameters': {}},
    ]
    recs = context(traj([agent(calls=[call('A', call_id='a'), call('C', call_id='c')])], tool_definitions=defs))
    present, cover = pre.tool_schemas_coverage(recs)
    assert present.met is True
    assert cover.met == 'partial'
    assert 'C' in cover.detail
    bare_recs = context(traj([agent(calls=[call('A', call_id='a')])]))
    missing = pre.tool_schemas_coverage(bare_recs)
    assert [(p.name, p.met) for p in missing] == [('tool schemas present', False)]


def test_llm_step_checks() -> None:
    inv = {'invocation': {'finish_reasons': ['stop'], 'start_timestamp': 1.0, 'end_timestamp': 2.0}}
    full = context(traj([user(), agent(model='openai/gpt-x', metrics=USAGE, extra=inv)]))
    assert pre.model_present(full).met is True
    assert pre.token_usage_complete(full).met is True
    assert pre.finish_reason_present(full).met is True
    assert pre.provider_derivable(full).met is True
    assert pre.timestamps(full).met is True
    mixed = context(traj([agent(model='claude-x', metrics=USAGE), agent(model='claude-x')]))
    assert pre.token_usage_complete(mixed).met is False
    provider = pre.provider_derivable(mixed)
    assert provider.met == 'partial' and 'inferred' in provider.detail
    assert pre.finish_reason_present(mixed).met is False
    none = context(traj([user()]))
    assert pre.model_present(none).met is False
    assert pre.token_usage_complete(none).met is False


def test_timestamps_fall_back_to_iso_as_partial() -> None:
    iso = context(traj([agent(model='m', timestamp='2026-01-01T00:00:00Z')]))
    assert pre.timestamps(iso).met == 'partial'
    assert pre.timestamps(context(traj([agent(model='m')]))).met is False


def test_timestamps_only_count_steps_the_walker_identifies_as_llm_calls() -> None:
    no_llm = context(traj([agent('no call', llm_call_count=0)]))
    assert pre.timestamps(no_llm).met is True
    assert pre.timestamps(no_llm).detail == 'no LLM steps'

    model_fallback = context(traj([agent('known model', model='openai/gpt-x')]))
    assert pre.timestamps(model_fallback).met is False


def test_subagent_checks() -> None:
    child = traj([agent('c')], trajectory_id='child')
    stray = traj([agent('s')], trajectory_id='stray')
    linked = traj([agent(calls=[call('Task')], results=[spawned('child')])], subagents=[child])
    walked = context(linked)
    assert pre.subagent_linkage(walked).met is True
    assert pre.has_subagents(1).met is True
    unlinked = traj([agent(calls=[call('Task')], results=[ok()])], subagents=[stray])
    got = pre.subagent_linkage(context(unlinked))
    assert got.met is False
    assert 'Task @step 1' in got.detail and 'stray' in got.detail
    solo = context(traj([agent(calls=[call('Read')], results=[failed()])]))
    assert pre.has_subagents(0).met is False
    assert pre.subagent_linkage(solo).met is True
