"""Regression tests for enum delivery-method display rendering."""

from __future__ import annotations

from evaluatorq.redteam.contracts import DeliveryMethod


def test_chart_dimension_displays_enum_value(rt_result_safe) -> None:
    from evaluatorq.dashboard.redteam_charts import _dim_value

    result = rt_result_safe.model_copy(
        update={'attack': rt_result_safe.attack.model_copy(update={'delivery_methods': [DeliveryMethod.CRESCENDO]})}
    )
    rendered = _dim_value(result, 'delivery_method')
    assert 'crescendo' in rendered
    # Version-honest leak check: str(member) yields 'DeliveryMethod.CRESCENDO' on the
    # 3.10 polyfill; a .name mistake yields 'CRESCENDO' on any version. The member
    # NAME must never appear — only the value 'crescendo' should.
    assert 'CRESCENDO' not in rendered


def test_attack_fragment_displays_enum_value(rt_result_safe) -> None:
    from evaluatorq.dashboard.redteam_transcripts import render_attack_fragment

    result = rt_result_safe.model_copy(
        update={'attack': rt_result_safe.attack.model_copy(update={'delivery_methods': [DeliveryMethod.CRESCENDO]})}
    )
    rendered = render_attack_fragment(result)
    assert 'crescendo' in rendered
    # See test_chart_dimension_displays_enum_value: NAME leak is caught on any version.
    assert 'CRESCENDO' not in rendered
