"""Tests for the delivery-method registry (RES-966).

Covers the coerce-known + passthrough-unknown policy, registration of custom
methods, and the identity property that lets registry-backed objects flow
through the pipeline unchanged (no enum -> string -> enum hop).
"""

from __future__ import annotations

import pytest

from evaluatorq.redteam import delivery_method_registry as reg
from evaluatorq.redteam.contracts import DeliveryMethod, RedTeamInput, Severity, TurnType, VulnerabilityDomain


@pytest.fixture(autouse=True)
def _clean_custom_registry():
    """Each test starts and ends with an empty custom registry (module global)."""
    reg._CUSTOM_DELIVERY_METHODS.clear()
    yield
    reg._CUSTOM_DELIVERY_METHODS.clear()


# ---------------------------------------------------------------------------
# resolve: coerce-known, passthrough-unknown, identity
# ---------------------------------------------------------------------------


def test_known_string_coerces_to_enum() -> None:
    resolved = reg.resolve_delivery_method('crescendo')
    assert resolved is DeliveryMethod.CRESCENDO
    assert isinstance(resolved, DeliveryMethod)


def test_unknown_string_passes_through() -> None:
    resolved = reg.resolve_delivery_method('my-custom-jailbreak')
    assert resolved == 'my-custom-jailbreak'
    assert not isinstance(resolved, DeliveryMethod)


def test_enum_member_is_returned_as_is() -> None:
    # The identity that lets objects flow through the pipeline without conversion.
    assert reg.resolve_delivery_method(DeliveryMethod.DAN) is DeliveryMethod.DAN


def test_resolve_list_dedupes_preserving_order_and_coerces() -> None:
    out = reg.resolve_delivery_methods(['DAN', 'custom', 'DAN', DeliveryMethod.DAN, 'crescendo'])
    assert out == [DeliveryMethod.DAN, 'custom', DeliveryMethod.CRESCENDO]
    # a str 'DAN' and DeliveryMethod.DAN dedupe to one (StrEnum equality)
    assert out.count(DeliveryMethod.DAN) == 1


# ---------------------------------------------------------------------------
# is_known / register / list
# ---------------------------------------------------------------------------


def test_is_known_enum_member() -> None:
    assert reg.is_known_delivery_method('crescendo')
    assert reg.is_known_delivery_method('DAN')


def test_is_known_false_for_unregistered() -> None:
    assert not reg.is_known_delivery_method('totally-unknown')


def test_register_makes_custom_known_and_resolvable_as_string() -> None:
    assert not reg.is_known_delivery_method('honeypot')
    reg.register_delivery_method('honeypot', category='experimental')
    assert reg.is_known_delivery_method('honeypot')
    # Registered customs stay strings (open set): no enum member exists for them.
    assert reg.resolve_delivery_method('honeypot') == 'honeypot'
    assert 'honeypot' in {str(m) for m in reg.list_available_delivery_methods()}


def test_register_existing_enum_value_is_noop() -> None:
    reg.register_delivery_method('crescendo')
    assert 'crescendo' not in reg._CUSTOM_DELIVERY_METHODS  # enum is already canonical


def test_register_empty_raises() -> None:
    with pytest.raises(ValueError, match='empty'):
        reg.register_delivery_method('')


def test_list_available_includes_enum_and_customs() -> None:
    reg.register_delivery_method('honeypot')
    values = {str(m) for m in reg.list_available_delivery_methods()}
    assert 'crescendo' in values  # enum
    assert 'honeypot' in values  # custom
    assert len(reg.list_available_delivery_methods()) == len(DeliveryMethod) + 1


# ---------------------------------------------------------------------------
# StrEnum identity that makes the seam removal safe
# ---------------------------------------------------------------------------


def test_delivery_method_set_membership_matches_string() -> None:
    # The loader filters `dp.delivery_method in selected`. This is why passing
    # DeliveryMethod objects works against the dataset's string values.
    selected: set[DeliveryMethod | str] = {DeliveryMethod.DAN, 'custom'}
    assert 'DAN' in selected
    assert DeliveryMethod.DAN in selected
    assert 'custom' in selected


# ---------------------------------------------------------------------------
# RedTeamInput boundary coercion routes through the registry
# ---------------------------------------------------------------------------


def _input(delivery_method: str) -> RedTeamInput:
    return RedTeamInput(
        id='x',
        vulnerability='goal_hijacking',
        delivery_method=delivery_method,
        severity=Severity.LOW,
        vulnerability_domain=VulnerabilityDomain.AGENT,
        turn_type=TurnType.SINGLE,
        source='test',
    )


def test_contract_boundary_coerces_known_delivery_method() -> None:
    dp = _input('crescendo')
    assert dp.delivery_method is DeliveryMethod.CRESCENDO


def test_contract_boundary_passes_unknown_through() -> None:
    dp = _input('bespoke-technique')
    assert dp.delivery_method == 'bespoke-technique'
    assert not isinstance(dp.delivery_method, DeliveryMethod)


def test_contract_boundary_coerces_registered_custom_to_string() -> None:
    reg.register_delivery_method('honeypot')
    dp = _input('honeypot')
    # A registered custom has no enum member, so it stays a string; the point is
    # it validates as "known" (no warning) while remaining open-set.
    assert dp.delivery_method == 'honeypot'
    assert reg.is_known_delivery_method(dp.delivery_method)
