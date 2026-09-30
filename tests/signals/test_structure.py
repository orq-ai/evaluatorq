"""ATIF-backed cases for the Group A structure and token signals."""

from __future__ import annotations

from evaluatorq.formats.atif import AtifMetrics
from evaluatorq.signals import compute_signals
from evaluatorq.signals.config import SignalsConfig

from .conftest import agent, call, compaction, copied, ok, spawned, traj, user


def test_message_turn_and_llm_counts_skip_bookkeeping() -> None:
    report = compute_signals(
        traj([user(), copied(), compaction(), agent(llm_call_count=2), user(), agent(llm_call_count=0)])
    )
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


def test_subagent_evidence_anchors_child_step_and_relates_spawning_call() -> None:
    child = traj([user('delegate'), agent('done')], trajectory_id='child')
    root = traj(
        [user(), agent(calls=[call('Agent', {}, 'spawn')], results=[spawned('child', call_id='spawn')])],
        trajectory_id='root',
        subagents=[child],
    )
    result = compute_signals(root, only=['subagent_invocation_count']).results['subagent_invocation_count']
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert (evidence.agent_path, evidence.step_id) == (('child',), 1)
    assert evidence.related == [((), 2)]
    assert evidence.related_call_ids == ['spawn']


def test_empty_visible_linked_subagent_counts_and_uses_spawn_evidence() -> None:
    child = traj([compaction()], trajectory_id='child')
    root = traj(
        [agent(calls=[call('Agent', {}, 'spawn')], results=[spawned('child', call_id='spawn')])],
        trajectory_id='root',
        subagents=[child],
    )

    report = compute_signals(
        root,
        only=[
            'subagent_invocation_count',
            'total_subagent_messages',
            'avg_messages_per_subagent_invocation',
        ],
    )

    assert report.values() == {
        'subagent_invocation_count': 1,
        'total_subagent_messages': 0,
        'avg_messages_per_subagent_invocation': 0,
    }
    evidence = report.results['subagent_invocation_count'].evidence
    assert len(evidence) == 1
    assert (evidence[0].agent_path, evidence[0].step_id, evidence[0].call_id) == ((), 1, 'spawn')
    assert evidence[0].reason == 'subagent has no visible steps'


def test_max_depth_is_zero_at_root_and_counts_nested_subagents() -> None:
    root_only = compute_signals(traj([user(), agent()], trajectory_id='root'), only=['max_depth'])
    assert root_only.values()['max_depth'] == 0

    grandchild = traj([agent('deep')], trajectory_id='grandchild')
    child = traj(
        [agent(calls=[call('Agent', {}, 'spawn-grandchild')], results=[spawned('grandchild', call_id='spawn-grandchild')])],
        trajectory_id='child',
        subagents=[grandchild],
    )
    root = traj(
        [agent(calls=[call('Agent', {}, 'spawn-child')], results=[spawned('child', call_id='spawn-child')])],
        trajectory_id='root',
        subagents=[child],
    )
    assert compute_signals(root, only=['max_depth']).values()['max_depth'] == 2


def test_max_depth_counts_a_nested_unlinked_subtree_at_its_embedded_depth() -> None:
    grandchild = traj([agent('deep')], trajectory_id='grandchild')
    child = traj([agent('middle')], trajectory_id='child', subagents=[grandchild])
    root = traj([agent('root')], trajectory_id='root', subagents=[child])

    result = compute_signals(root, only=['max_depth']).results['max_depth']

    assert result.value == 2
    assert [(item.agent_path, item.step_id) for item in result.evidence] == [(('child', 'grandchild'), 1)]


def test_finish_reason_length_count_reads_invocation_finish_reasons() -> None:
    report = compute_signals(
        traj(
            [
                agent('cut off', model='openai/gpt-4.1', extra={'invocation': {'finish_reasons': ['length']}}),
                agent('complete', model='openai/gpt-4.1', extra={'invocation': {'finish_reasons': ['stop']}}),
            ]
        ),
        only=['finish_reason_length_count'],
    )
    assert report.values()['finish_reason_length_count'] == 1
    assert [(item.step_id, item.agent_path) for item in report.results['finish_reason_length_count'].evidence] == [(1, ())]


def test_token_signals_have_no_basis_when_usage_is_incomplete() -> None:
    report = compute_signals(traj([agent(model='openai/gpt-4.1')]))
    assert report.results['total_input_tokens'].no_basis == 'token usage complete: 0 of 1 agent steps carry token usage'


def test_tool_roles_come_from_config() -> None:
    root = traj([agent(calls=[call('run', {'cmd': 'npm test'}, 'x')])])
    config = SignalsConfig(tool_roles={'run': 'bash'})
    assert compute_signals(root, config, only=['bash_command_value_count']).values() == {'bash_command_value_count': {'npm test': 1}}
