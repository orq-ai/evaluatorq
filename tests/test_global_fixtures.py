from __future__ import annotations

import logging

import pytest
from loguru import logger

from tests.conftest import _run_store_path


@pytest.fixture
def indirect_caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Resolve caplog through another fixture, as larger test helpers do."""
    return caplog


def test_run_store_path_is_deterministic_unique_and_lazy(tmp_path_factory: pytest.TempPathFactory) -> None:
    base = tmp_path_factory.getbasetemp()

    first = _run_store_path(base, 'tests/a.py::test_same[value]')
    repeated = _run_store_path(base, 'tests/a.py::test_same[value]')
    other = _run_store_path(base, 'tests/a.py::test_other[value]')

    assert first == repeated
    assert first != other
    assert first.is_relative_to(base)
    assert first.parent == base / 'evaluatorq-run-stores'
    assert len(first.name) == 64 and all(character in '0123456789abcdef' for character in first.name)
    assert not first.exists()
    assert not other.exists()


def test_loguru_bridge_is_not_installed_without_caplog() -> None:
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Capture()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        logger.warning('loguru-only sentinel')
    finally:
        root.removeHandler(handler)

    assert records == []


def test_loguru_bridge_supports_indirect_caplog_resolution(indirect_caplog: pytest.LogCaptureFixture) -> None:
    logger.warning('indirect caplog sentinel')

    assert 'indirect caplog sentinel' in indirect_caplog.text
