"""Cost-breakdown coverage for simulation token-usage rendering (SDD task 4).

Covers the two paths that matter for the ``None`` vs ``0.0`` cost distinction:
1. Cost known — input/output/total cost rows appear with correct values.
2. Cost unknown (``None``, e.g. an old saved report) — cost rows are absent
   entirely, and nothing renders a fabricated ``$0.00``.
"""

from __future__ import annotations

import pytest

from evaluatorq.contracts import Message, TokenUsage
from evaluatorq.simulation.reports.sections import _build_token_usage_section
from evaluatorq.simulation.reports.token_usage import build_token_usage_rows
from evaluatorq.simulation.types import SimulationResult, TerminatedBy


def _make_result(*, token_usage: TokenUsage) -> SimulationResult:
    return SimulationResult(
        messages=[Message(role='user', content='hi')],
        terminated_by=TerminatedBy.judge,
        reason='done',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=token_usage,
        turn_metrics=[],
        metadata={'persona': 'P', 'scenario': 'S'},
    )


# ---------------------------------------------------------------------------
# _build_token_usage_section
# ---------------------------------------------------------------------------


def test_build_token_usage_section_carries_known_cost():
    results = [
        _make_result(
            token_usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cache_creation_tokens=2,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
            )
        ),
        _make_result(
            token_usage=TokenUsage(
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
                cache_creation_tokens=1,
                input_cost=0.004,
                output_cost=0.006,
                total_cost=0.01,
            )
        ),
    ]
    section = _build_token_usage_section(results)
    assert section.data['input_cost'] == pytest.approx(0.005)
    assert section.data['output_cost'] == pytest.approx(0.008)
    assert section.data['total_cost'] == pytest.approx(0.013)
    assert section.data['cache_creation_tokens'] == 3


def test_build_token_usage_section_cost_none_when_unreported():
    results = [
        _make_result(token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
        _make_result(token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)),
    ]
    section = _build_token_usage_section(results)
    assert section.data['input_cost'] is None
    assert section.data['output_cost'] is None
    assert section.data['total_cost'] is None


# ---------------------------------------------------------------------------
# build_token_usage_rows (shared HTML + Markdown row builder)
# ---------------------------------------------------------------------------


def test_build_token_usage_rows_includes_cost_when_known():
    data = {
        'input_tokens': 10,
        'output_tokens': 5,
        'total_tokens': 15,
        'avg_total_per_conversation': 15.0,
        'avg_input_per_conversation': 10.0,
        'avg_output_per_conversation': 5.0,
        'cache_creation_tokens': 4,
        'input_cost': 0.0012,
        'output_cost': 0.0034,
        'total_cost': 0.0046,
    }
    rows = build_token_usage_rows(data)
    rows_dict = dict(rows)
    assert rows_dict['Cache-Write Tokens'] == '4'
    assert rows_dict['Input Cost'] == '$0.0012'
    assert rows_dict['Output Cost'] == '$0.0034'
    assert rows_dict['Total Cost'] == '$0.0046'


def test_build_token_usage_rows_omits_cost_when_none():
    """Old saved reports predate the cost breakdown entirely: no cost keys at
    all in `data`. Rendering must degrade to no cost rows, never `$0.00`."""
    data = {
        'input_tokens': 10,
        'output_tokens': 5,
        'total_tokens': 15,
        'avg_total_per_conversation': 15.0,
        'avg_input_per_conversation': 10.0,
        'avg_output_per_conversation': 5.0,
    }
    rows = build_token_usage_rows(data)
    joined = ' '.join(f'{k}:{v}' for k, v in rows)
    assert 'Cost' not in joined
    assert '$0.00' not in joined
    assert '$' not in joined


def test_build_token_usage_rows_omits_cost_when_explicitly_null():
    """Same as above but with explicit `None` values (as they'd appear once
    Usage.model_dump() round-trips through JSON): must not render $0.00."""
    data = {
        'input_tokens': 10,
        'output_tokens': 5,
        'total_tokens': 15,
        'avg_total_per_conversation': 15.0,
        'avg_input_per_conversation': 10.0,
        'avg_output_per_conversation': 5.0,
        'input_cost': None,
        'output_cost': None,
        'total_cost': None,
    }
    rows = build_token_usage_rows(data)
    joined = ' '.join(f'{k}:{v}' for k, v in rows)
    assert 'Cost' not in joined
    assert '$' not in joined
