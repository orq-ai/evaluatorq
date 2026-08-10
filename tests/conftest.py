import logging
import socket

import pytest

from evaluatorq.common.llm_call import reset_reasoning_rejectors


class LeakedNetworkCall(AssertionError):
    """A unit test tried to open a connection to a real host."""


def _is_loopback(address: object) -> bool:
    """True for localhost targets, which tests legitimately use for fake servers."""
    if not isinstance(address, tuple) or not address:
        return False
    host = str(address[0])
    return host in {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


@pytest.fixture(autouse=True)
def _hide_ambient_credentials(request, monkeypatch):
    """Unset real API keys so unit tests cannot reach a live service.

    Several code paths gate purely on "is a key set" — ``evaluate()`` uploads its
    results whenever ``ORQ_API_KEY`` exists, for instance — so on a developer
    machine with a real key exported, plain unit tests were POSTing to the
    platform. Tests that need a key set one explicitly via ``monkeypatch.setenv``,
    which still wins because this runs first.
    """
    if request.node.get_closest_marker("integration"):
        yield
        return
    for var in ("ORQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture(autouse=True)
def _block_outbound_network(request, monkeypatch):
    """Fail any non-integration test that opens a connection to a real host.

    Unit tests were reaching the live Orq router with whatever ORQ_API_KEY the
    developer had exported — 127 requests per suite run, showing up as real
    (billed, 404-ing) traces in the workspace. Mocks are the fix; this fixture
    is what stops a new one from slipping in silently.

    Loopback is allowed so tests may still spin up a local fake server. Mark a
    test ``@pytest.mark.allow_network`` to opt out.
    """
    if request.node.get_closest_marker("allow_network") or request.node.get_closest_marker("integration"):
        yield
        return

    real_connect = socket.socket.connect
    leaked: list[str] = []

    def guarded_connect(self, address, *args, **kwargs):
        if _is_loopback(address):
            return real_connect(self, address, *args, **kwargs)
        # Recorded as well as raised: the calling code is usually wrapped in a
        # broad ``except Exception``, which swallows this and lets the test pass
        # green while the connection attempt really happened. The teardown
        # assertion below is what actually surfaces it.
        leaked.append(str(address))
        raise LeakedNetworkCall(
            f"Test opened a network connection to {address!r}. Unit tests must not "
            f"call real services — mock the client. Use @pytest.mark.allow_network "
            f"only if the connection is genuinely required."
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect)
    yield
    if leaked:
        raise LeakedNetworkCall(
            f"Test attempted {len(leaked)} outbound connection(s) to real hosts: "
            f"{sorted(set(leaked))}. Mock the client so the suite never touches a live service."
        )


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
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
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
