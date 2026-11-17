"""API-independent checks for OpenAI-compatible provider configuration."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from examples.agent_simulation.failure_and_recovery.live_refund_evaluatorq import LiveRefundTarget, resolve_base_url
from examples.agent_simulation.failure_and_recovery.refund_recovery import MockRefundGateway

# Ruff's credential heuristics mistake API parameter names for secrets here.
# ruff: noqa: INP001, S101, S105, S106


@pytest.mark.parametrize(
    ('provider', 'supplied', 'env', 'expected'),
    [
        ('orq', None, None, 'https://api.orq.ai/v3/router'),
        ('openai', None, None, None),
        ('custom', 'https://example.test/v1', None, 'https://example.test/v1'),
        ('custom', None, 'https://example.test/api/v1/', 'https://example.test/api/v1'),
        ('custom', 'http://localhost:1234/v1', None, 'http://localhost:1234/v1'),
    ],
)
def test_valid_provider_endpoints(provider, supplied, env, expected):
    assert resolve_base_url(provider, supplied, env) == expected


@pytest.mark.parametrize(
    'url',
    [
        'http://public.example/v1',
        'ftp://example.test/v1',
        'https://token@host.test/v1',
        'https://host.test/v1?token=a',
        'https://host.test/v1#secret',
    ],
)
def test_reject_unsafe_custom_urls(url):
    with pytest.raises(ValueError):
        resolve_base_url('custom', url, None)


def test_reject_missing_url():
    with pytest.raises(ValueError):
        resolve_base_url('custom', None, None)


class RecordingCompletions:
    def __init__(self):
        self.kwargs = []

    async def create(self, **kwargs: Any):
        self.kwargs.append(kwargs)
        tc = SimpleNamespace(
            id='f1',
            function=SimpleNamespace(
                name='submit_final_decision',
                arguments=(
                    '{"outcome":"unknown","evidence":"none","order_id":"ORD-999",'
                    '"idempotency_key":"refund:ORD-999:v1","customer_message":"Unknown."}'
                ),
            ),
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))], usage=None
        )


@pytest.mark.asyncio
async def test_custom_model_parameters_passed_to_openai_sdk():
    comps = RecordingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=comps))
    target = LiveRefundTarget(
        MockRefundGateway(), client, 'vendor/model-id', tool_choice='auto', token_parameter='max_completion_tokens'
    )
    assert await target([]) == 'Unknown.'
    assert len(comps.kwargs) == 1
    assert comps.kwargs[0]['tool_choice'] == 'auto'
    assert comps.kwargs[0]['max_completion_tokens'] == 600
    assert 'max_tokens' not in comps.kwargs[0]


@pytest.mark.asyncio
async def test_openai_gpt6_luna_chat_completion_parameters():
    comps = RecordingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=comps))
    target = LiveRefundTarget(
        MockRefundGateway(), client, 'gpt-6-luna', token_parameter='max_completion_tokens', reasoning_effort='none'
    )
    assert await target([]) == 'Unknown.'
    assert comps.kwargs[0]['model'] == 'gpt-6-luna'
    assert comps.kwargs[0]['reasoning_effort'] == 'none'
    assert comps.kwargs[0]['max_completion_tokens'] == 600
    assert comps.kwargs[0]['tool_choice'] == 'required'


@pytest.mark.asyncio
async def test_suite_forwards_openai_reasoning_setting(monkeypatch):
    import argparse

    import examples.agent_simulation.failure_and_recovery.live_refund_evaluatorq as runner

    observed = []

    async def fake_run_one(client, model, case, policy, max_model_calls, **kwargs: Any):
        await asyncio.sleep(0)
        observed.append(kwargs)
        return {'passed': True, 'issues': [], 'model_calls': 1, 'gateway_ledger_rows': 1}

    monkeypatch.setattr(runner, 'run_one', fake_run_one)
    args = argparse.Namespace(
        model='gpt-6-luna',
        case='post_commit',
        policy='hardened',
        max_model_calls=4,
        tool_choice='required',
        token_parameter='max_completion_tokens',
        reasoning_effort='none',
        runs=1,
    )
    out = await runner.run_suite(args, client=object())
    assert len(out) == 1
    assert observed[0]['reasoning_effort'] == 'none'
    assert observed[0]['token_parameter'] == 'max_completion_tokens'
