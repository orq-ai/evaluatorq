from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger


@pytest.fixture
def indirect_caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Resolve caplog through another fixture, as larger test helpers do."""
    return caplog


@pytest.mark.parametrize('case', ['first', 'second'])
def test_running_fixture_sets_a_deterministic_unique_lazy_run_store(
    case: str, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> None:
    base = tmp_path_factory.getbasetemp()
    actual = Path(os.environ['EVALUATORQ_DIR'])
    expected_digest = hashlib.sha256(request.node.nodeid.encode()).hexdigest()
    sibling_case = 'second' if case == 'first' else 'first'
    sibling_nodeid = f'{request.node.nodeid.rsplit("[", 1)[0]}[{sibling_case}]'
    sibling_digest = hashlib.sha256(sibling_nodeid.encode()).hexdigest()

    assert actual == base / 'evaluatorq-run-stores' / expected_digest
    assert actual != base / 'evaluatorq-run-stores' / sibling_digest
    assert actual.is_relative_to(base)
    assert len(actual.name) == 64 and all(character in '0123456789abcdef' for character in actual.name)
    assert not actual.exists()


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


def test_loguru_bridge_is_removed_before_the_next_test(tmp_path: Path) -> None:
    nested_test = tmp_path / 'test_loguru_lifecycle.py'
    nested_test.write_text(
        """\
import logging

import pytest
from loguru import logger


@pytest.fixture
def indirect_caplog(caplog):
    return caplog


def test_bridge_is_installed_for_indirect_caplog(indirect_caplog):
    logger.warning("nested captured sentinel")
    assert "nested captured sentinel" in indirect_caplog.text


def test_bridge_did_not_leak_into_no_caplog_test():
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        logger.warning("nested cleanup sentinel")
    finally:
        root.removeHandler(handler)
    assert records == []
"""
    )
    runner = """\
import sys

import pytest
from tests import conftest

raise SystemExit(pytest.main([sys.argv[1], "-q", "-p", "no:cacheprovider"], plugins=[conftest]))
"""
    completed = subprocess.run(
        [sys.executable, '-c', runner, str(nested_test)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, f'nested pytest failed:\n{completed.stdout}\n{completed.stderr}'
    assert '2 passed' in completed.stdout
