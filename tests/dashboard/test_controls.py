"""The shared filter controls in ``dashboard/controls.py``."""

from __future__ import annotations

from evaluatorq.dashboard.controls import dropdown, range_control, range_engaged, rich_pick


def test_dropdown_searchable_adds_the_search_box() -> None:
    plain = dropdown('persona', 'Persona', ['a', 'b'], [])
    searchable = dropdown('persona', 'Persona', ['a', 'b'], [], searchable=True)

    assert 'class="facet-search"' not in plain
    assert 'class="facet-values"' not in plain
    assert 'class="facet-search"' in searchable
    assert 'class="facet-values"' in searchable
    assert 'facet-no-results' in searchable


def test_rich_pick_keeps_its_hidden_input() -> None:
    html = rich_pick('x', 'price', [('price', '<b>Price</b>')])

    assert '<input type="hidden" name="x" value="price">' in html


def test_range_default_moves_the_no_op_bound_only_when_given() -> None:
    # Without `default` a floor is a no-op at its minimum, as the report rails expect.
    assert range_engaged(['10'], min_val=0, max_val=100, floor=True)
    assert not range_engaged(['0'], min_val=0, max_val=100, floor=True)
    # A page whose own default is 10 reads 10 as no filter and 0 as one.
    assert not range_engaged(['10'], min_val=0, max_val=100, floor=True, default=10)
    assert range_engaged(['0'], min_val=0, max_val=100, floor=True, default=10)
    html = range_control('s', 'S', min_val=0, max_val=100, step='1', sel=[], floor=True, default=10)
    assert 'value="10"' in html and 'data-default="10"' in html
    assert '<span class="filter-slider-readout">' in html
    assert 'data-default="0"' in range_control('s', 'S', min_val=0, max_val=100, step='1', sel=[], floor=True)
