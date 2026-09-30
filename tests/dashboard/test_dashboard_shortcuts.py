from pathlib import Path


def test_traces_keyboard_shortcuts_have_accessible_guide_and_guarded_actions() -> None:
    js = Path('src/evaluatorq/dashboard/static/dashboard.js').read_text()

    assert "window.location.pathname !== '/traces'" in js
    assert "evt.key === '/'" in js
    assert "evt.key === '?'" in js
    assert "evt.key === 'o'" in js
    assert "evt.key === 'c'" in js
    assert "guide.setAttribute('aria-labelledby', 'finder-shortcut-title')" in js
    assert 'guide.showModal()' in js
    assert "evt.key === 'Escape' && guide?.open" in js
    assert "target.closest?.('details, summary')" in js
    assert 'evt.ctrlKey || evt.altKey || evt.metaKey' in js
    assert "finderEditable(target) || finderEditable(active)" in js
    assert "evt.key === 'j' || evt.key === 'J' || evt.key === 'ArrowDown'" in js
    assert "evt.key === 'k' || evt.key === 'K' || evt.key === 'ArrowUp'" in js
