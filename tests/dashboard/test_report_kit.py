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
