"""Tests for the delivery-method registry (RES-966).

Covers the coerce-known + passthrough-unknown policy, registration of custom
methods, and the identity property that lets registry-backed objects flow
through the pipeline unchanged (no enum -> string -> enum hop).
"""

from __future__ import annotations

import pytest

from evaluatorq.redteam import delivery_method_registry as reg
from evaluatorq.redteam.contracts import DeliveryMethod, RedTeamInput, Severity, TurnType, VulnerabilityDomain


# Isolation comes from the autouse _clean_custom_delivery_registry fixture in
# tests/redteam/conftest.py — it covers every red-team test, not just this module,
# since registration is process-global.


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
    assert 'honeypot' in {reg.delivery_method_str(m) for m in reg.list_available_delivery_methods()}


def test_register_existing_enum_value_is_noop() -> None:
    reg.register_delivery_method('crescendo')
    assert 'crescendo' not in reg._CUSTOM_DELIVERY_METHODS  # enum is already canonical


@pytest.mark.parametrize('member', list(DeliveryMethod), ids=lambda m: m.name)
def test_register_any_enum_value_is_noop(member: DeliveryMethod) -> None:
    """Every member's value must register as a no-op, not just the name != value ones.

    DeliveryMethod.DAN is spelled 'DAN' as both name and value, so a name check
    ordered before the value lookup rejects the canonical value and suggests the
    identical string back. Only a sweep over all members catches that.
    """
    assert reg.register_delivery_method(reg.delivery_method_str(member)) == reg.delivery_method_str(member)
    assert reg.delivery_method_str(member) not in reg._CUSTOM_DELIVERY_METHODS


def test_register_enum_member_name_is_rejected() -> None:
    """Registering the member NAME instead of its value must not create a shadow.

    'ROLE_PLAY' is not a DeliveryMethod value ('role-play' is), so registering it
    would land in the custom set: reported known, CLI warning suppressed, and
    filtering against a spelling no dataset row carries.
    """
    with pytest.raises(ValueError, match='role-play'):
        reg.register_delivery_method('ROLE_PLAY')
    assert 'ROLE_PLAY' not in reg._CUSTOM_DELIVERY_METHODS


def test_register_empty_raises() -> None:
    with pytest.raises(ValueError, match='empty'):
        reg.register_delivery_method('')


def test_list_available_includes_enum_and_customs() -> None:
    reg.register_delivery_method('honeypot')
    values = {reg.delivery_method_str(m) for m in reg.list_available_delivery_methods()}
    assert 'crescendo' in values  # enum
    assert 'honeypot' in values  # custom
    assert len(reg.list_available_delivery_methods()) == len(DeliveryMethod) + 1


# ---------------------------------------------------------------------------
# StrEnum identity that makes the seam removal safe
# ---------------------------------------------------------------------------


def test_delivery_method_set_membership_matches_string() -> None:
    # The loader filters `dp.delivery_method in selected`. This is why passing
    # DeliveryMethod objects works against the dataset's string values.
    # CRESCENDO deliberately: its name ('CRESCENDO') differs from its value
    # ('crescendo'), so this fails if membership ever keyed on the name. A member
    # like DAN, where name == value, cannot tell the two apart.
    selected: set[DeliveryMethod | str] = {DeliveryMethod.CRESCENDO, 'custom'}
    assert 'crescendo' in selected
    assert DeliveryMethod.CRESCENDO in selected
    assert 'CRESCENDO' not in selected  # the member name is not the match key
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


# ---------------------------------------------------------------------------
# delivery_method_str: version-independent value stringification
# ---------------------------------------------------------------------------


def test_delivery_method_str_returns_value_for_member() -> None:
    # str(member) is the repr on the 3.10 StrEnum polyfill; delivery_method_str
    # must always return the value so display/keying is version-independent.
    assert reg.delivery_method_str(DeliveryMethod.DAN) == 'DAN'
    assert reg.delivery_method_str(DeliveryMethod.CRESCENDO) == 'crescendo'


def test_delivery_method_str_passes_strings_through() -> None:
    assert reg.delivery_method_str('my-custom') == 'my-custom'


# ---------------------------------------------------------------------------
# _check_filter_results diagnostics use canonical values (regression: the seam
# removal must not report a matched method as unmatched, nor leak enum reprs)
# ---------------------------------------------------------------------------


def test_check_filter_results_matched_enum_method_no_unmatched_warning(caplog) -> None:
    import logging

    from evaluatorq.redteam.runner import _check_filter_results
    from evaluatorq.types import DataPoint

    # A datapoint whose static delivery_method matches the requested member.
    # CRESCENDO (name != value) so a name/value confusion in the comparison shows
    # up here as a spurious "unmatched" warning.
    dp = DataPoint(inputs={'delivery_method': 'crescendo', 'category': 'ASI01'})
    with caplog.at_level(logging.WARNING):
        _check_filter_results([dp], None, {DeliveryMethod.CRESCENDO}, names_apply=False)
    assert 'Unmatched delivery method' not in caplog.text


def test_check_filter_results_empty_run_error_reports_values_not_reprs() -> None:
    from evaluatorq.redteam.exceptions import RedTeamError
    from evaluatorq.redteam.runner import _check_filter_results

    with pytest.raises(RedTeamError) as exc:
        _check_filter_results([], None, {DeliveryMethod.CRESCENDO, 'custom'}, names_apply=False)
    text = str(exc.value)
    assert 'crescendo' in text
    assert 'custom' in text
    # never the enum repr, which the 3.10 polyfill would produce via str()
    assert 'DeliveryMethod.CRESCENDO' not in text


def test_every_enum_member_has_a_category() -> None:
    """Completeness: DELIVERY_METHOD_CATEGORY must map every DeliveryMethod
    member, so a newly added method can never ship without a technique family."""
    missing = [m for m in DeliveryMethod if m not in reg.DELIVERY_METHOD_CATEGORY]
    assert not missing, f'delivery methods missing from DELIVERY_METHOD_CATEGORY: {missing}'
