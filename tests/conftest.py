import logging

import pytest


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
