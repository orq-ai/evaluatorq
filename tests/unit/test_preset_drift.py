"""The drift report against hand-built gardens (RES-1346). No network."""

from datetime import datetime, timezone

import pytest

from evaluatorq.common.model_frontier import ScoredModel
from evaluatorq.jury_presets import MAX_SEAT_AGE_DAYS, PRESETS, all_router_ids
from evaluatorq.preset_drift import DriftReport, drift_report, earliest_listing

UTC = timezone.utc
DAY_MS = 86_400_000
NOW = datetime(2026, 11, 12, tzinfo=UTC)


def _card(router_id: str, index: float = 40.0, input_price: float = 1.0, output_price: float = 4.0) -> ScoredModel:
    return ScoredModel(
        router_id=router_id,
        intelligence_index=index,
        input_price=input_price,
        output_price=output_price,
        headline_effort=None,
    )


def _garden() -> dict[str, ScoredModel]:
    return {router_id: _card(router_id) for router_id in all_router_ids()}


def _everything_listed(age_days: int = 10) -> dict[str, int | None]:
    stamp = int(NOW.timestamp() * 1000) - age_days * DAY_MS
    return {router_id: stamp for router_id in all_router_ids()}


def test_a_retired_seat_is_reported_with_its_reserve_as_the_option():
    listed = _everything_listed()
    seat = PRESETS['balanced-trio'].judges[0]
    del listed[seat]

    report = drift_report(_garden(), listed, now=NOW)

    finding = next(f for f in report.findings if f.kind == 'not_served' and f.subject == seat)
    assert 'balanced-trio' in finding.held_by
    assert any(PRESETS['balanced-trio'].reserve_judges[0] in option for option in finding.options)
    assert not finding.needs_human


def test_a_retired_seat_with_no_reserve_needs_a_human(monkeypatch):
    listed = _everything_listed()
    seat = PRESETS['eu-region'].judges[0]
    del listed[seat]
    monkeypatch.setattr(PRESETS['eu-region'], 'reserve_judges', ())

    report = drift_report(_garden(), listed, now=NOW)

    finding = next(f for f in report.findings if f.kind == 'not_served' and f.subject == seat)
    assert finding.needs_human
    assert report.needs_human


def test_an_old_seat_without_a_waiver_is_reported_with_the_waiver_as_an_option():
    listed = _everything_listed(age_days=MAX_SEAT_AGE_DAYS + 30)

    report = drift_report(_garden(), listed, pools={}, now=NOW)

    aged = [f for f in report.findings if f.kind == 'aged_seat']
    assert aged
    assert all(any('STALE_SEAT_WAIVERS' in option for option in f.options) for f in aged)


def test_young_seats_raise_no_age_finding():
    report = drift_report(_garden(), _everything_listed(age_days=10), now=NOW)

    assert not [f for f in report.findings if f.kind == 'aged_seat']


def test_the_report_serialises_to_json_an_agent_can_read():
    report = drift_report(_garden(), _everything_listed(), now=NOW)

    assert DriftReport.model_validate_json(report.model_dump_json()) == report


def test_an_undated_listing_does_not_crash():
    listed = _everything_listed()
    listed[PRESETS['strong-jury'].judges[0]] = None

    drift_report(_garden(), listed, now=NOW)


@pytest.mark.parametrize(('stamp', 'expected'), [(1_700_000_000, 1_700_000_000_000), (1_700_000_000_000, 1_700_000_000_000)])
def test_seconds_and_milliseconds_stamps_date_the_same(stamp, expected):
    assert earliest_listing({'openai/gpt-6-luna': stamp}) == {'gpt-6-luna': expected}
