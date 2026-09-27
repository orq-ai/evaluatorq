from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from evaluatorq.dashboard.trace_finder import routes as finder_routes


def test_parse_range_applies_browser_offset() -> None:
    start, end = finder_routes.parse_range('2026-09-27T10:00:05', '2026-09-27T12:00:00', '-120')
    assert start == datetime(2026, 9, 27, 8, 0, 5, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 27, 10, 0, 0, tzinfo=timezone.utc)


def test_parse_range_accepts_minutes_only() -> None:
    start, _ = finder_routes.parse_range('2026-09-27T10:00', '2026-09-27T11:00', '0')
    assert start == datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc)


def test_parse_range_rejects_inverted() -> None:
    with pytest.raises(ValueError, match='From must be before To'):
        finder_routes.parse_range('2026-09-27T12:00', '2026-09-27T11:00', '0')


def test_parse_range_rejects_over_30_days() -> None:
    with pytest.raises(ValueError, match='30 days'):
        finder_routes.parse_range('2026-08-01T00:00', '2026-09-27T00:00', '0')


def test_parse_range_treats_bad_offset_as_utc() -> None:
    start, _ = finder_routes.parse_range('2026-09-27T10:00', '2026-09-27T11:00', 'abc')
    assert start.tzinfo == timezone.utc
