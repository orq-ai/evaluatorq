from evaluatorq.common.reports import palette


def test_palette_exports_core_and_semantic():
    assert palette.COLORS['orange_300'] == '#ff8f34'
    assert palette.SEVERITY_COLORS['critical'] == palette.COLORS['red_400']
    assert palette.SEVERITY_ORDER == ['critical', 'high', 'medium', 'low']
    assert len(palette.QUALITATIVE) >= 6
    assert palette.ORQ_SCALE_GOOD_BAD[0][1] == palette.COLORS['success_400']
    assert palette.ORQ_SCALE_GOOD_BAD[-1][1] == palette.COLORS['red_400']


from evaluatorq.common.reports.palette import SEVERITY_ORDER, SEVERITY_RANK, SEVERITY_WEIGHTS


def test_severity_weights_keys_match_order():
    assert set(SEVERITY_WEIGHTS.keys()) == set(SEVERITY_ORDER)


def test_severity_rank_keys_match_order():
    assert set(SEVERITY_RANK.keys()) == set(SEVERITY_ORDER)


def test_severity_rank_monotonic():
    """Higher severity has higher rank value."""
    assert SEVERITY_RANK['critical'] > SEVERITY_RANK['high'] > SEVERITY_RANK['medium'] > SEVERITY_RANK['low']
