"""Run-scoped registries (RES-1922)."""

import asyncio

import pytest

from evaluatorq.redteam.adaptive.strategy_registry import get_strategies_for_vulnerability
from evaluatorq.redteam.contracts import RegisteredValue, Vulnerability, is_known_open_value
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
    for rule in ('Never promise a refund.', 'Never promise a discount.'):
        with run_scope():
            vuln = register_vulnerability(id='refund_promises', description=rule)
            vdef = get_vulnerability_def(vuln)
            assert vdef is not None
            assert vdef.description == rule


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


@pytest.mark.asyncio
async def test_red_team_runs_use_their_own_definition_of_the_same_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any
    from unittest.mock import patch

    from evaluatorq.redteam import red_team
    from evaluatorq.redteam.contracts import SaveMode
    from evaluatorq.redteam.custom import CustomVulnerability
    from evaluatorq.redteam.hooks import DefaultHooks

    from .test_manifest_lifecycle import _fake_report

    class QuietHooks(DefaultHooks):
        async def on_complete(self, report: Any, **_kwargs: Any) -> None:
            return None

    seen: list[str] = []

    async def fake_pipeline(**_kwargs: Any) -> Any:
        vdef = get_vulnerability_def(RegisteredValue('refund_promises'))
        assert vdef is not None
        seen.append(vdef.description)
        report = _fake_report()
        report.tested_agents = []
        return report

    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    with patch('evaluatorq.redteam.runner._run_dynamic_or_hybrid', side_effect=fake_pipeline):
        for rule in ('Never promise a refund.', 'Never promise a discount.'):
            await red_team(
                'agent:x',
                mode='dynamic',
                vulnerabilities=['refund_promises'],
                custom=[CustomVulnerability(id='refund_promises', description=rule)],
                save=SaveMode.NONE,
                hooks=QuietHooks(),
                generate_executive_summary=False,
            )
    assert seen == ['Never promise a refund.', 'Never promise a discount.']
    assert not is_known_open_value(Vulnerability, 'refund_promises')
