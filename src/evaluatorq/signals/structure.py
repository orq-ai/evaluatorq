"""Group A signals: trajectory structure, message volume, tools, and token usage."""

from __future__ import annotations

from collections import Counter
from itertools import starmap
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from evaluatorq.signals import preconditions as pre
from evaluatorq.signals.models import Evidence, SignalFn, SignalResult, result
from evaluatorq.signals.shell import shell_command_family
from evaluatorq.signals.walk import SignalContext, WalkedStep, finish_reasons, infer_provider, is_llm_step, step_model

if TYPE_CHECKING:
    from evaluatorq.formats.atif import AtifTrajectory


def _ev(entry: WalkedStep, reason: str = '') -> Evidence:
    return Evidence(step_id=entry.step.step_id, agent_path=entry.agent_path, reason=reason)


def _llm(ctx: SignalContext) -> list[WalkedStep]:
    return [entry for entry in ctx.walked if is_llm_step(entry)]


def _token_input(entry: WalkedStep) -> int:
    metrics = entry.step.metrics
    return (
        max(0, metrics.prompt_tokens - (metrics.cached_tokens or 0))
        if metrics and metrics.prompt_tokens is not None
        else 0
    )


def _count(name: str, value: Any, evidence: list[Evidence]) -> SignalResult:
    return result(name, 'A', value, evidence)


def max_depth(ctx: SignalContext) -> SignalResult:
    """Deepest embedded subagent nesting level; root is zero, even for unlinked subtrees."""
    levels: list[tuple[int, tuple[str, ...], AtifTrajectory]] = []

    def visit(trajectory: AtifTrajectory, depth: int, path: tuple[str, ...]) -> None:
        levels.append((depth, path, trajectory))
        for subagent in trajectory.subagent_trajectories or []:
            child_path = (*path, subagent.trajectory_id) if subagent.trajectory_id else path
            visit(subagent, depth + 1, child_path)

    visit(ctx.trajectory, 0, ())
    depth = max(level for level, _, _ in levels)
    evidence = []
    for level, path, trajectory in levels:
        if level != depth:
            continue
        first = next((entry for entry in ctx.walked if entry.trajectory is trajectory), None)
        if first is not None:
            evidence.append(Evidence(step_id=first.step.step_id, agent_path=path))
    return _count('max_depth', depth, evidence)


def llm_call_count(ctx: SignalContext) -> SignalResult:
    """Sum explicit LLM call counts, falling back to one for each agent step."""
    entries = [w for w in ctx.walked if w.step.source == 'agent']
    value = sum(w.step.llm_call_count if w.step.llm_call_count is not None else 1 for w in entries)
    return _count(
        'llm_call_count',
        value,
        [_ev(w, f'{w.step.llm_call_count if w.step.llm_call_count is not None else 1} LLM calls') for w in entries],
    )


def user_message_count(ctx: SignalContext) -> SignalResult:
    entries = [w for w in ctx.walked if w.step.source == 'user']
    return _count('user_message_count', len(entries), [_ev(w) for w in entries])


def assistant_message_count(ctx: SignalContext) -> SignalResult:
    entries = [w for w in ctx.walked if w.step.source == 'agent']
    return _count('assistant_message_count', len(entries), [_ev(w) for w in entries])


def turn_count(ctx: SignalContext) -> SignalResult:
    evidence: list[Evidence] = []
    for index, entry in enumerate(ctx.walked):
        if entry.step.source != 'user':
            continue
        following = next(
            (
                w
                for w in ctx.walked[index + 1 :]
                if w.agent_path == entry.agent_path and w.step.source in {'user', 'agent'}
            ),
            None,
        )
        if following is not None and following.step.source == 'agent':
            evidence.append(
                Evidence(
                    step_id=entry.step.step_id,
                    agent_path=entry.agent_path,
                    related=[(following.agent_path, following.step.step_id)],
                )
            )
    return _count('turn_count', len(evidence), evidence)


def _token_signal(ctx: SignalContext, name: str, field: str) -> SignalResult:
    entries = _llm(ctx)
    pcs = [pre.token_usage_complete(ctx)]
    if field == 'input':
        value_of = _token_input
    elif field == 'output':

        def value_of(entry: WalkedStep) -> int:
            return (
                entry.step.metrics.completion_tokens
                if entry.step.metrics and entry.step.metrics.completion_tokens is not None
                else 0
            )

    else:

        def value_of(entry: WalkedStep) -> int:
            completion = (
                entry.step.metrics.completion_tokens
                if entry.step.metrics and entry.step.metrics.completion_tokens is not None
                else 0
            )
            return _token_input(entry) + completion

    evidence = [_ev(w, f'{value_of(w)} {field} tokens') for w in entries if w.step.metrics is not None]
    return result(name, 'A', sum(value_of(w) for w in entries), evidence, pcs)


def total_input_tokens(ctx: SignalContext) -> SignalResult:
    """Input tokens excluding cached tokens, requiring complete usage."""
    return _token_signal(ctx, 'total_input_tokens', 'input')


def total_output_tokens(ctx: SignalContext) -> SignalResult:
    """Output tokens across LLM steps, requiring complete usage."""
    return _token_signal(ctx, 'total_output_tokens', 'output')


def total_tokens(ctx: SignalContext) -> SignalResult:
    """Input excluding cached tokens plus output tokens, requiring complete usage."""
    return _token_signal(ctx, 'total_tokens', 'total')


def cache_read_token_share(ctx: SignalContext) -> SignalResult:
    """Cached tokens divided by prompt tokens, requiring complete nonzero prompt usage."""
    entries = _llm(ctx)
    prompt = sum(
        w.step.metrics.prompt_tokens for w in entries if w.step.metrics and w.step.metrics.prompt_tokens is not None
    )
    cached = sum(w.step.metrics.cached_tokens or 0 for w in entries if w.step.metrics)
    pcs = [
        pre.token_usage_complete(ctx),
        pre.Precondition(name='has input tokens', met=prompt > 0, detail=f'{prompt} prompt tokens'),
    ]
    evidence = [
        _ev(w, f'{w.step.metrics.cached_tokens or 0} cached of {w.step.metrics.prompt_tokens} prompt tokens')
        for w in entries
        if w.step.metrics
    ]
    return result('cache_read_token_share', 'A', cached / prompt if prompt else None, evidence, pcs)


def peak_context_tokens(ctx: SignalContext) -> SignalResult:
    """Largest raw prompt context in one LLM request."""
    entries = _llm(ctx)
    peak = max((w.step.metrics.prompt_tokens or 0 for w in entries if w.step.metrics), default=0)
    pcs = [pre.token_usage_complete(ctx)]
    evidence = [
        _ev(w, f'{peak} prompt tokens') for w in entries if w.step.metrics and w.step.metrics.prompt_tokens == peak
    ]
    return result('peak_context_tokens', 'A', peak, evidence, pcs)


def _call_ev(record, reason: str = '') -> Evidence:
    return Evidence(
        step_id=record.step.step.step_id,
        agent_path=record.step.agent_path,
        call_id=record.call.tool_call_id,
        reason=reason,
    )


def tool_call_count(ctx: SignalContext) -> SignalResult:
    return _count('tool_call_count', len(ctx.calls), [_call_ev(c) for c in ctx.calls])


def unique_tools_used(ctx: SignalContext) -> SignalResult:
    first = {}
    for record in ctx.calls:
        first.setdefault(record.call.function_name, record)
    return _count('unique_tools_used', len(first), [_call_ev(record, name) for name, record in first.items()])


def tool_call_value_count(ctx: SignalContext) -> SignalResult:
    counts = dict(Counter(c.call.function_name for c in ctx.calls).most_common())
    return _count('tool_call_value_count', counts, [_call_ev(c, c.call.function_name) for c in ctx.calls])


def _arguments(call: Any) -> dict[str, Any]:
    return call.arguments if isinstance(call.arguments, dict) and '_raw' not in call.arguments else {}


def _role_calls(ctx: SignalContext, role: str):
    return [c for c in ctx.calls if ctx.config.tool_role(c.call.function_name) == role]


def bash_command_value_count(ctx: SignalContext) -> SignalResult:
    calls = _role_calls(ctx, 'bash')
    pairs = [
        (c, shell_command_family(_arguments(c.call).get('command', _arguments(c.call).get('cmd'))) or 'unknown command')
        for c in calls
    ]
    return _count(
        'bash_command_value_count', dict(Counter(v for _, v in pairs).most_common()), list(starmap(_call_ev, pairs))
    )


def webview_value_count(ctx: SignalContext) -> SignalResult:
    calls = _role_calls(ctx, 'webview')
    pairs = []
    for call in calls:
        url_text = _arguments(call.call).get('url')
        try:
            url = urlsplit(url_text) if isinstance(url_text, str) else None
            host = (
                url.hostname.rstrip('.').lower() if url and url.scheme in {'http', 'https'} and url.hostname else None
            )
        except ValueError:
            host = None
        pairs.append((call, host or 'unknown domain'))
    return _count(
        'webview_value_count', dict(Counter(v for _, v in pairs).most_common()), list(starmap(_call_ev, pairs))
    )


def loaded_skill_value_count(ctx: SignalContext) -> SignalResult:
    calls = _role_calls(ctx, 'skill')
    pairs = []
    for call in calls:
        name = _arguments(call.call).get('skill', _arguments(call.call).get('name'))
        pairs.append((call, name.strip() if isinstance(name, str) and name.strip() else 'unknown skill'))
    return _count(
        'loaded_skill_value_count', dict(Counter(v for _, v in pairs).most_common()), list(starmap(_call_ev, pairs))
    )


def _subagents(ctx: SignalContext) -> list[tuple[tuple[str, ...], AtifTrajectory]]:
    """Return every embedded child trajectory and its path, including children with no walked steps."""
    found: list[tuple[tuple[str, ...], AtifTrajectory]] = []
    seen: set[int] = set()

    def visit(trajectory: AtifTrajectory, path: tuple[str, ...]) -> None:
        for child in trajectory.subagent_trajectories or []:
            if id(child) in seen:
                continue
            seen.add(id(child))
            child_path = (*path, child.trajectory_id) if child.trajectory_id is not None else path
            found.append((child_path, child))
            visit(child, child_path)

    visit(ctx.trajectory, ())
    return found


def subagent_invocation_count(ctx: SignalContext) -> SignalResult:
    subagents = _subagents(ctx)
    pc = pre.subagent_linkage(ctx)
    first_by_trajectory: dict[int, WalkedStep] = {}
    for entry in ctx.walked:
        if entry.agent_path:
            first_by_trajectory.setdefault(id(entry.trajectory), entry)
    subagent_by_path = {path: trajectory for path, trajectory in subagents}

    evidence: list[Evidence] = []
    linked_trajectories: set[int] = set()
    for call in ctx.calls:
        if call.result is None:
            continue
        for ref in call.result.subagent_trajectory_ref or []:
            if ref.trajectory_id is None:
                continue
            child_path = (*call.step.agent_path, ref.trajectory_id)
            trajectory = subagent_by_path.get(child_path)
            if trajectory is None:
                continue
            linked_trajectories.add(id(trajectory))
            first = first_by_trajectory.get(id(trajectory))
            if first is not None:
                evidence.append(
                    Evidence(
                        step_id=first.step.step_id,
                        agent_path=first.agent_path,
                        related=[(call.step.agent_path, call.step.step.step_id)],
                        related_call_ids=[call.call.tool_call_id],
                    )
                )
            else:
                evidence.append(_call_ev(call, 'subagent has no visible steps'))
    for _, trajectory in subagents:
        if id(trajectory) in linked_trajectories:
            continue
        first = first_by_trajectory.get(id(trajectory))
        if first is not None:
            evidence.append(_ev(first, 'unlinked subagent'))
    return result('subagent_invocation_count', 'A', len(subagents), evidence, [pc])


def total_subagent_messages(ctx: SignalContext) -> SignalResult:
    entries = [w for w in ctx.walked if w.agent_path]
    return result('total_subagent_messages', 'A', len(entries), [_ev(w) for w in entries], [pre.subagent_linkage(ctx)])


def avg_messages_per_subagent_invocation(ctx: SignalContext) -> SignalResult:
    subagents = _subagents(ctx)
    entries = [w for w in ctx.walked if w.agent_path]
    pcs = [
        pre.has_subagents(len(subagents)),
        pre.subagent_linkage(ctx),
    ]
    return result(
        'avg_messages_per_subagent_invocation',
        'A',
        round(len(entries) / len(subagents), 3) if subagents else None,
        [],
        pcs,
    )


def model_count(ctx: SignalContext) -> SignalResult:
    first = {}
    for entry in _llm(ctx):
        model = step_model(entry)
        if model:
            first.setdefault(model, entry)
    return result(
        'model_count', 'A', len(first), [_ev(entry, model) for model, entry in first.items()], [pre.model_present(ctx)]
    )


def provider_count(ctx: SignalContext) -> SignalResult:
    first = {}
    for entry in _llm(ctx):
        provider = infer_provider(step_model(entry))
        if provider:
            first.setdefault(provider, entry)
    return result(
        'provider_count',
        'A',
        len(first),
        [_ev(entry, provider) for provider, entry in first.items()],
        [pre.provider_derivable(ctx)],
    )


def finish_reason_length_count(ctx: SignalContext) -> SignalResult:
    entries = [w for w in _llm(ctx) if 'length' in finish_reasons(w.step)]
    return result(
        'finish_reason_length_count', 'A', len(entries), [_ev(w) for w in entries], [pre.finish_reason_present(ctx)]
    )


SIGNALS: dict[str, tuple[Literal['A'], SignalFn]] = {
    'max_depth': ('A', max_depth),
    'llm_call_count': ('A', llm_call_count),
    'user_message_count': ('A', user_message_count),
    'assistant_message_count': ('A', assistant_message_count),
    'turn_count': ('A', turn_count),
    'total_input_tokens': ('A', total_input_tokens),
    'total_output_tokens': ('A', total_output_tokens),
    'total_tokens': ('A', total_tokens),
    'cache_read_token_share': ('A', cache_read_token_share),
    'peak_context_tokens': ('A', peak_context_tokens),
    'tool_call_count': ('A', tool_call_count),
    'unique_tools_used': ('A', unique_tools_used),
    'tool_call_value_count': ('A', tool_call_value_count),
    'bash_command_value_count': ('A', bash_command_value_count),
    'webview_value_count': ('A', webview_value_count),
    'loaded_skill_value_count': ('A', loaded_skill_value_count),
    'subagent_invocation_count': ('A', subagent_invocation_count),
    'total_subagent_messages': ('A', total_subagent_messages),
    'avg_messages_per_subagent_invocation': ('A', avg_messages_per_subagent_invocation),
    'model_count': ('A', model_count),
    'provider_count': ('A', provider_count),
    'finish_reason_length_count': ('A', finish_reason_length_count),
}
