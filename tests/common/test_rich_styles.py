from evaluatorq.common.reports import rate_style


def test_higher_is_better():
    assert rate_style(0.9) == "green"
    assert rate_style(0.6) == "yellow"
    assert rate_style(0.2) == "red"


def test_lower_is_better_inverts():
    assert rate_style(0.1, higher_is_better=False) == "green"
    assert rate_style(0.4, higher_is_better=False) == "yellow"
    assert rate_style(0.8, higher_is_better=False) == "red"


def test_matches_legacy_redteam_thresholds():
    # parity with the old _rate_style / _asr_style this replaces
    assert rate_style(0.8) == "green" and rate_style(0.5) == "yellow"
    assert rate_style(0.2, higher_is_better=False) == "green"
    assert rate_style(0.5, higher_is_better=False) == "yellow"
