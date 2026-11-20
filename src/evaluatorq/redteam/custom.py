"""Register custom vulnerabilities, strategies, judges and open-set values (RES-1922).

Everything registered here is process-local and lives until `clear_custom_registrations`.
"""

from __future__ import annotations

from evaluatorq.redteam.adaptive.strategy_registry import (
    clear_custom_strategies,
    refresh_strategy_registries,
    register_strategy,
)
from evaluatorq.redteam.contracts import (
    AgentCapability,
    AttackTechnique,
    Framework,
    RegisteredValue,
    VulnerabilityDef,
    VulnerabilityDomain,
    clear_registered_values,
    register_open_value,
)
from evaluatorq.redteam.frameworks.owasp.evaluators import clear_custom_judges, register_judge
from evaluatorq.redteam.vulnerability_registry import clear_custom_vulnerabilities
from evaluatorq.redteam.vulnerability_registry import register_vulnerability_def as _register_def

__all__ = [
    'clear_custom_registrations',
    'register_capability',
    'register_domain',
    'register_framework',
    'register_judge',
    'register_strategy',
    'register_technique',
    'register_vulnerability_def',
]


def register_technique(name: str) -> AttackTechnique | RegisteredValue:
    """Make *name* a valid attack technique; a built-in name returns its member."""
    return register_open_value(AttackTechnique, name)


def register_capability(name: str) -> AgentCapability | RegisteredValue:
    """Make *name* a valid agent capability tag; a built-in name returns its member."""
    return register_open_value(AgentCapability, name)


def register_domain(name: str) -> VulnerabilityDomain | RegisteredValue:
    """Make *name* a valid vulnerability domain; a built-in name returns its member."""
    return register_open_value(VulnerabilityDomain, name)


def register_framework(name: str) -> Framework | RegisteredValue:
    """Make *name* a valid framework; a built-in name returns its member."""
    return register_open_value(Framework, name)


def register_vulnerability_def(vdef: VulnerabilityDef) -> None:
    """Register a full vulnerability definition; see `vulnerability_registry.register_vulnerability_def`."""
    _register_def(vdef)
    refresh_strategy_registries()


def clear_custom_registrations() -> None:
    """Forget every custom registration in every registry (test isolation)."""
    clear_custom_judges()
    clear_custom_strategies()
    clear_custom_vulnerabilities()
    clear_registered_values()
