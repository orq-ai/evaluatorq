import logging

import pytest

from evaluatorq.common.llm_call import reset_reasoning_rejectors


@pytest.fixture(autouse=True)
def _reset_reasoning_rejectors():
    """The per-model rejection memo is process-lifetime; isolate each test."""
    reset_reasoning_rejectors()


@pytest.fixture(autouse=True)
def _isolate_run_stores(tmp_path, monkeypatch):
    """Point the redteam/sim run stores at a tmp dir so tests never write
    runs into the repo's real ``.evaluatorq/`` store."""
    monkeypatch.setenv("EVALUATORQ_DIR", str(tmp_path / ".evaluatorq"))
    yield


@pytest.fixture(autouse=True)
def _disable_real_span_export(monkeypatch):
    """Keep the suite from installing a live OTLP exporter.

    ``init_tracing_if_needed()`` installs a real BatchSpanProcessor whenever
    ORQ_API_KEY is set, and it latches for the process — so on a developer
    machine with a real key the first test that reaches it wins, and every span
    the rest of the suite produces is queued for export to my.orq.ai and flushed
    at interpreter shutdown (a 401, or worse, a successful upload of test spans).
    Which test gets there first depends on file order, which is why this shows
    up in some batches and not others. Tests that exercise setup itself opt back
    in by deleting this var.
    """
    monkeypatch.setenv("ORQ_DISABLE_TRACING", "1")
    yield


@pytest.fixture(autouse=True)
def _clear_span_max_text_chars_cache():
    """Clear lru_cache between tests so EVALUATORQ_SPAN_MAX_TEXT_CHARS env changes propagate."""
    from evaluatorq.common.tracing import _default_span_max_text_chars
    _default_span_max_text_chars.cache_clear()
    yield
    _default_span_max_text_chars.cache_clear()


class _LoguruPropagateHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


@pytest.fixture(autouse=True)
def _propagate_loguru_to_stdlib():
    """Bridge loguru records into stdlib logging so pytest's caplog can capture them."""
    from loguru import logger

    handler_id = logger.add(_LoguruPropagateHandler(), format="{message}", level="DEBUG")
    yield
    try:
        logger.remove(handler_id)
    except ValueError:
        pass  # cli.py calls logger.remove() globally; handler may already be gone
