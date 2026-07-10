from evaluatorq.dashboard import report_kit


def _heatmap(cells):
    personas = sorted({c['persona'] for c in cells})
    scenarios = sorted({c['scenario'] for c in cells})
    return {'personas': personas, 'scenarios': scenarios, 'cells': cells}


def test_exec_summary_includes_stats_and_best_worst():
    summary = {'total_conversations': 10, 'success_rate': 0.7, 'avg_goal_completion_score': 0.62}
    heatmap = _heatmap([
        {'persona': 'Alice', 'scenario': 'Refund', 'success_rate': 0.9, 'n': 5},
        {'persona': 'Bob', 'scenario': 'Refund', 'success_rate': 0.2, 'n': 5},
    ])
    html = report_kit.exec_summary(summary_data=summary, heatmap_data=heatmap, confidence='HIGH')
    assert '70%' in html
    assert 'Alice' in html and 'Bob' in html
    assert 'HIGH CONFIDENCE' in html


def test_exec_summary_drops_best_worst_when_all_cells_errored():
    summary = {'total_conversations': 4, 'success_rate': 0.0, 'avg_goal_completion_score': 0.0}
    heatmap = _heatmap([
        {'persona': 'Alice', 'scenario': 'Refund', 'success_rate': 0.0, 'n': 0},
    ])
    html = report_kit.exec_summary(summary_data=summary, heatmap_data=heatmap, confidence=None)
    assert 'Alice' not in html  # error-only cell excluded
    assert 'CONFIDENCE' not in html  # no pill when confidence is None


def test_exec_summary_empty_run_returns_empty():
    html = report_kit.exec_summary(summary_data={'total_conversations': 0}, heatmap_data=_heatmap([]), confidence=None)
    assert html == ''


def test_exec_summary_escapes_persona_names():
    summary = {'total_conversations': 2, 'success_rate': 0.5, 'avg_goal_completion_score': 0.5}
    heatmap = _heatmap([
        {'persona': '<script>', 'scenario': 'X', 'success_rate': 0.9, 'n': 1},
        {'persona': 'safe', 'scenario': 'X', 'success_rate': 0.1, 'n': 1},
    ])
    html = report_kit.exec_summary(summary_data=summary, heatmap_data=heatmap, confidence='LOW')
    assert '<script>' not in html
    assert '&lt;script&gt;' in html


def test_interp_color_stops():
    assert report_kit._interp_color(0.0).lower() == '#df5325'
    assert report_kit._interp_color(0.5).lower() == '#ff8f34'
    assert report_kit._interp_color(1.0).lower() == '#299d8f'


def test_heatmap_text_ink_threshold():
    # v <= 0.55 => ink-900 text; v > 0.55 => white text
    cells = [{'persona': 'P', 'scenario': 'S', 'success_rate': 0.5, 'n': 3}]
    low = report_kit.heatmap(['P'], ['S'], cells)
    assert 'var(--ink-900)' in low
    cells_hi = [{'persona': 'P', 'scenario': 'S', 'success_rate': 0.9, 'n': 3}]
    hi = report_kit.heatmap(['P'], ['S'], cells_hi)
    assert '#fff' in hi.lower() or 'white' in hi.lower()


def test_heatmap_empty_cell_dash():
    html = report_kit.heatmap(['P'], ['S'], [])  # no cell for P×S
    assert '—' in html


def test_meta_grid_skips_null_values():
    html = report_kit.meta_grid([('Target', 'agent-x'), ('Model', None), ('Mode', '')])
    assert 'agent-x' in html
    assert 'Model' not in html and 'Mode' not in html


def test_line_chart_empty_series_returns_empty():
    assert report_kit.line_chart([0, 1], {}) == ''


def test_line_chart_excludes_all_none_series_from_legend():
    html = report_kit.line_chart([0, 1, 2], {'real': [0.1, 0.2, 0.3], 'empty': [None, None, None]})
    assert 'real' in html
    assert 'empty' not in html  # all-None series draws nothing -> not in legend


def test_bar_rows_escapes_labels():
    html = report_kit.bar_rows([('<b>x</b>', 3.0)], width=200, label_w=60, color='var(--chart-2)')
    assert '<b>x</b>' not in html
    assert '&lt;b&gt;' in html
