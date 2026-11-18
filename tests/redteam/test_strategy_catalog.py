"""The curated strategy catalog helpers and the docs hook that renders them into the red teaming guide."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities
from evaluatorq.redteam.adaptive.strategy_planner import full_sweep_attack_ceiling, planned_vulnerabilities
from evaluatorq.redteam.adaptive.strategy_registry import (
    STRATEGY_REGISTRY,
    _filter_applicable_strategies,
    curated_strategies,
    default_vulnerabilities,
    strategy_needs,
)
from evaluatorq.redteam.contracts import AgentCapability, AgentContext, AttackStrategy, ToolInfo
from evaluatorq.redteam.vulnerability_registry import (
    CATEGORY_TO_VULNERABILITY,
    FRAMEWORK_TO_VULNERABILITIES,
)

_HOOKS = Path(__file__).resolve().parents[2] / 'docs' / 'hooks.py'


def _strategy(**overrides: object) -> AttackStrategy:
    base = next(iter(curated_strategies()))
    return base.model_copy(update=overrides)


def test_curated_strategies_lists_each_strategy_once_without_aliases() -> None:
    names = [s.name for s in curated_strategies()]
    assert len(names) == len(set(names))
    assert len(names) == sum(len(g) for c, g in STRATEGY_REGISTRY.items() if not c.startswith('OWASP-'))


@pytest.mark.parametrize(
    ('requires_tools', 'capabilities', 'expected'),
    [
        (False, [], ''),
        (True, [], 'tools'),
        (False, [AgentCapability.MEMORY_READ], 'memory_read'),
        (True, [AgentCapability.WEB_REQUEST, AgentCapability.USER_DATA], 'tools + web_request or user_data'),
    ],
)
def test_strategy_needs(requires_tools: bool, capabilities: list[AgentCapability], expected: str) -> None:
    strategy = _strategy(requires_tools=requires_tools, required_capabilities=capabilities)
    assert strategy_needs(strategy) == expected


def _runs(strategy: AttackStrategy, *, tools: bool, capabilities: list[AgentCapability]) -> bool:
    context = AgentContext(key='agent', tools=[ToolInfo(name='tool')] if tools else [])
    classified = AgentCapabilities(capabilities={'tool': capabilities})
    return bool(_filter_applicable_strategies([strategy], 'test', context, classified))


@pytest.mark.parametrize('strategy', curated_strategies(), ids=lambda s: s.name)
def test_needs_is_exactly_what_the_filter_requires(strategy: AttackStrategy) -> None:
    """A target with what the Needs column says runs the strategy; one missing any part of it does not."""
    unrelated = [c for c in AgentCapability if c not in strategy.required_capabilities][:1]
    for needed in [[c] for c in strategy.required_capabilities] or [unrelated]:  # 'or': any one alone suffices
        assert _runs(strategy, tools=True, capabilities=needed)
        if strategy.requires_tools:
            assert not _runs(strategy, tools=False, capabilities=needed)
    if strategy.required_capabilities:
        assert not _runs(strategy, tools=True, capabilities=unrelated)


def test_sweep_and_coverage_table_cover_the_same_vulnerabilities() -> None:
    """The ceiling sweeps `default_vulnerabilities`; the coverage table lists every framework code (LLM03 too)."""
    swept = set(default_vulnerabilities())
    listed = {CATEGORY_TO_VULNERABILITY[code] for codes in FRAMEWORK_TO_VULNERABILITIES.values() for code in codes}
    assert swept == listed


def test_full_sweep_ceiling_adds_the_multi_agent_only_vulnerabilities() -> None:
    generated = 3
    single = full_sweep_attack_ceiling(multi_agent=False, generated_per_vulnerability=generated)
    multi = full_sweep_attack_ceiling(multi_agent=True, generated_per_vulnerability=generated)
    # Every curated strategy is planned for some vulnerability: none is lost to a vulnerability two categories share.
    assert full_sweep_attack_ceiling(multi_agent=True, generated_per_vulnerability=0) == len(curated_strategies())
    # The multi-agent-only vulnerabilities have no curated strategies, only generated ones.
    every = default_vulnerabilities()
    multi_agent_only = len(every) - len(planned_vulnerabilities(every, multi_agent=False))
    assert multi_agent_only > 0
    assert multi - single == generated * multi_agent_only


def test_a_run_with_no_selection_covers_default_vulnerabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner and the full-sweep ceiling read the same default selection."""
    from evaluatorq.redteam import runner

    sentinel = default_vulnerabilities()[:1]
    monkeypatch.setattr(runner, 'default_vulnerabilities', lambda: sentinel)
    vulnerabilities, _ = runner._resolve_vulns_and_categories(replay=None, vulnerabilities=None, categories=None)
    assert vulnerabilities == sentinel


@pytest.fixture(scope='module')
def hooks():  # noqa: ANN201
    pytest.importorskip('mkdocs')
    spec = importlib.util.spec_from_file_location('docs_hooks', _HOOKS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(hooks, markdown: str) -> str:  # noqa: ANN001
    page = SimpleNamespace(file=SimpleNamespace(src_uri='guides/red-teaming.md'))
    return hooks.on_page_markdown(markdown, page=page, config=None, files=None)


def test_hook_renders_every_generated_block(hooks) -> None:  # noqa: ANN001
    rendered = _render(hooks, '\n\n'.join(hooks._GENERATED))
    assert '<!--' not in rendered
    assert '| `tool_output_hijack` |' in rendered
    assert f'{len(curated_strategies())} curated strategies' in rendered
    single = hooks._full_sweep(multi_agent=False)
    assert f'{single} ({hooks._full_sweep(multi_agent=True)} multi-agent)' in rendered
    assert '| `ASI03` | Identity & Privilege Abuse (`identity_privilege_abuse`) | generated |' in rendered
    assert rendered.endswith(str(hooks._generated_per_vulnerability()))


@pytest.mark.parametrize('missing', ['coverage-table', 'curated-strategies', 'full-sweep-ceiling', 'full-sweep-total', 'generated-per-vulnerability'])
def test_hook_fails_the_build_when_a_marker_is_gone(hooks, missing: str) -> None:  # noqa: ANN001
    from mkdocs.exceptions import PluginError

    page = ' '.join(marker for marker in hooks._GENERATED if missing not in marker)
    with pytest.raises(PluginError, match=missing):
        _render(hooks, page)
