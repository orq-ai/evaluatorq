"""The shared filter controls in ``dashboard/controls.py``."""

from __future__ import annotations

from evaluatorq.dashboard.controls import dropdown, rich_pick


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
