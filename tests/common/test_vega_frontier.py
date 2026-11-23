from __future__ import annotations

from typing import Any

from evaluatorq.common.reports.palette import COLORS
from evaluatorq.common.reports.vega import vl_frontier_scatter

_ROWS = [
    {'x': 1.0, 'score': 40.0, 'status': 'Frontier', 'estimated': 'measured', 'label': 'a'},
    {'x': 3.0, 'score': 30.0, 'status': 'Dominated', 'estimated': 'measured', 'label': ''},
]


def _spec(guides: list[dict[str, Any]] | None) -> dict[str, Any]:
    return vl_frontier_scatter(
        rows=_ROWS, x_title='Price', y_title='Score', lower_is_better=True, tooltip=[], guides=guides
    )


def test_guides_layer_only_when_given() -> None:
    guides = [{'x': 1.0, 'score': 20.0, 'k': '10'}, {'x': 2.0, 'score': 30.0, 'k': '10'}]
    plain, guided = _spec(None), _spec(guides)
    assert len(_spec([])['layer']) == len(plain['layer'])
    assert len(guided['layer']) == len(plain['layer']) + 1
    # Drawn first so it sits behind the dots, with no tooltip and no legend entry.
    layer = guided['layer'][0]
    assert layer['data'] == {'values': guides}
    line, label = layer['layer']
    assert line['mark']['strokeDash'] == [4, 3]
    assert line['mark']['color'] == COLORS['ink_300']
    assert line['mark']['opacity'] == 0.5
    assert line['encoding']['detail'] == {'field': 'k', 'type': 'nominal'}
    for inner in (line, label):
        assert inner['mark']['tooltip'] is None
        assert 'color' not in inner['encoding'] and 'tooltip' not in inner['encoding']
    assert any("' per $'" in t.get('as', '') or "' per $'" in t.get('calculate', '') for t in label['transform'])
