"""Run-scoped registries (RES-1922)."""

import asyncio

import pytest

from evaluatorq.redteam.adaptive.strategy_registry import get_strategies_for_vulnerability
from evaluatorq.redteam.contracts import Vulnerability, is_known_open_value
from evaluatorq.redteam.custom import register_vulnerability
from evaluatorq.redteam.frameworks.owasp.evaluators import get_evaluator_for_vulnerability
from evaluatorq.redteam.registry_scope import run_scope
from evaluatorq.redteam.vulnerability_registry import get_vulnerability_def


def test_scope_sees_global_registrations_and_keeps_its_own() -> None:
    register_vulnerability(id='global_rule', description='Never insult the customer.')
    with run_scope():
        assert is_known_open_value(Vulnerability, 'global_rule')
        vuln = register_vulnerability(id='run_rule', description='Never promise a refund.', attacks=['Refund?'])
        assert get_vulnerability_def(vuln) is not None
        assert get_evaluator_for_vulnerability(vuln, model_id='m') is not None
        assert [s.prompt_template for s in get_strategies_for_vulnerability(vuln)] == ['Refund?']
    assert not is_known_open_value(Vulnerability, 'run_rule')
    assert is_known_open_value(Vulnerability, 'global_rule')


def test_same_id_with_different_definitions_in_two_scopes() -> None:
    with run_scope():
        register_vulnerability(id='refund_promises', description='Never promise a refund.')
    with run_scope():
        register_vulnerability(id='refund_promises', description='Never promise a discount.')


@pytest.mark.asyncio
async def test_concurrent_runs_do_not_see_each_other() -> None:
    started = asyncio.Event()
    seen: dict[str, bool] = {}

    async def run(own: str, other: str) -> None:
        with run_scope():
            register_vulnerability(id=own, description=f'Rule {own}.')
            if started.is_set():
                await asyncio.sleep(0)
            else:
                started.set()
                await asyncio.sleep(0.01)
            seen[own] = is_known_open_value(Vulnerability, other)

    await asyncio.gather(run('rule_a', 'rule_b'), run('rule_b', 'rule_a'))
    assert seen == {'rule_a': False, 'rule_b': False}
    assert not is_known_open_value(Vulnerability, 'rule_a')


@pytest.mark.asyncio
async def test_scope_follows_to_thread() -> None:
    with run_scope():
        register_vulnerability(id='thread_rule', description='Never promise a refund.')
        assert await asyncio.to_thread(is_known_open_value, Vulnerability, 'thread_rule')
