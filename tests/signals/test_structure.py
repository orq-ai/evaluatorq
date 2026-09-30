"""ATIF-backed cases for the Group A structure and token signals."""

from __future__ import annotations

from evaluatorq.formats.atif import AtifMetrics
from evaluatorq.signals import compute_signals
from evaluatorq.signals.config import SignalsConfig

from .conftest import agent, call, ok, spawned, traj, user


def test_message_turn_and_llm_counts_skip_bookkeeping() -> None:
    report = compute_signals(traj([user(), agent(llm_call_count=2), user(), agent(llm_call_count=0)]))
    assert report.values()['user_message_count'] == 2
    assert report.values()['assistant_message_count'] == 2
    assert report.values()['turn_count'] == 2
    assert report.values()['llm_call_count'] == 2


def test_token_totals_remove_cached_tokens_and_compute_peak_prompt() -> None:
    report = compute_signals(traj([
        agent(model='openai/gpt-4.1', metrics=AtifMetrics(prompt_tokens=100, cached_tokens=30, completion_tokens=10)),
        agent(model='anthropic/claude-4', metrics=AtifMetrics(prompt_tokens=200, cached_tokens=50, completion_tokens=20)),
    ]))
    assert report.values()['total_input_tokens'] == 220
    assert report.values()['total_output_tokens'] == 30
    assert report.values()['total_tokens'] == 250
    assert report.values()['cache_read_token_share'] == 80 / 300
    assert report.values()['peak_context_tokens'] == 200
    assert report.values()['model_count'] == 2
    assert report.values()['provider_count'] == 2


def test_tool_value_signals_parse_command_hosts_and_skills() -> None:
    calls = [call('Bash', {'command': 'FOO=bar cd /tmp && git status --short'}, 'b1'),
             call('WebFetch', {'url': 'https://Example.COM/path'}, 'w1'),
             call('Skill', {'skill': '  clarify  '}, 's1'), call('Read', {}, 'r1')]
    root = traj([agent(calls=calls, results=[ok(call_id=c.tool_call_id) for c in calls])])
    report = compute_signals(root)
    values = report.values()
    assert values['tool_call_count'] == 4
    assert values['unique_tools_used'] == 4
    assert values['tool_call_value_count'] == {'Bash': 1, 'WebFetch': 1, 'Skill': 1, 'Read': 1}
    assert values['bash_command_value_count'] == {'git status': 1}
    assert values['webview_value_count'] == {'example.com': 1}
    assert values['loaded_skill_value_count'] == {'clarify': 1}


def test_subagent_depth_invocations_and_messages_follow_linkage() -> None:
    child = traj([user('delegate'), agent('done')], trajectory_id='child')
    root = traj([user(), agent(calls=[call('Agent', {}, 'spawn')], results=[spawned('child', call_id='spawn')])],
                trajectory_id='root', subagents=[child])
    values = compute_signals(root).values()
    assert values['max_depth'] == 1
    assert values['subagent_invocation_count'] == 1
    assert values['total_subagent_messages'] == 2
    assert values['avg_messages_per_subagent_invocation'] == 2


def test_token_signals_have_no_basis_when_usage_is_incomplete() -> None:
    report = compute_signals(traj([agent(model='openai/gpt-4.1')]))
    assert report.results['total_input_tokens'].no_basis == 'token usage complete: 0 of 1 agent steps carry token usage'


def test_tool_roles_come_from_config() -> None:
    root = traj([agent(calls=[call('run', {'cmd': 'npm test'}, 'x')])])
    config = SignalsConfig(tool_roles={'run': 'bash'})
    assert compute_signals(root, config, only=['bash_command_value_count']).values() == {'bash_command_value_count': {'npm test': 1}}
