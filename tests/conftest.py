import logging
import socket
import threading
import traceback

import pytest

from evaluatorq.common.judge import reset_responses_rejectors
from evaluatorq.common.llm_call import reset_reasoning_rejectors
from evaluatorq.common.model_catalogue import clear_model_overrides, reset_catalogue_cache


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

    The patch is process-wide for the duration of one test, so a connection from
    a thread an EARLIER test left running is attributed to whichever test happens
    to be executing — the report then names an innocent test (a pure-string SVG
    helper, in the case that prompted this) and reproduces on neither a rerun of
    that test nor of the file. Each leak therefore records its thread and the
    stack that reached the socket, and the failure prints them: the culprit is
    identifiable from one CI log instead of a bisect.
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
        thread = threading.current_thread()
        origin = "".join(traceback.format_stack()[:-1][-6:])
        leaked.append(f"{address!r} from thread {thread.name!r}\n{origin}")
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
            f"Test attempted {len(leaked)} outbound connection(s) to real hosts. Mock the "
            f"client so the suite never touches a live service. If the stack below is not "
            f"this test's own code, an earlier test leaked a background thread — fix that "
            f"test, not this one.\n\n" + "\n".join(leaked)
        )


@pytest.fixture(autouse=True)
def _reset_reasoning_rejectors():
    """The per-model rejection memo is process-lifetime; isolate each test."""
    reset_reasoning_rejectors()
    reset_responses_rejectors()


class _OfflineCatalogues(dict):  # pyright: ignore[reportMissingTypeArgument]
    """A catalogue cache that reports every host as already-fetched-and-empty."""

    def get(self, key, default=None):
        return super().get(key, {})


@pytest.fixture(autouse=True)
def _offline_model_catalogue(request, monkeypatch):
    """Make the model catalogue a cache hit for every host, so no test fetches it.

    ``price_usage`` now sits on every chat-completion path and lazily GETs
    ``/v2/models``. With a credential in the environment that is a live, billed
    request — the network guard above catches it, but only after the call has
    gone out. Reporting every host as cached means the fetch is never attempted,
    while leaving ``_load_catalogue`` itself real so its own tests can exercise
    it (they swap this cache back for a plain dict).
    """
    from evaluatorq.common import model_catalogue

    reset_catalogue_cache()
    # `reset_catalogue_cache` deliberately spares `register_model` overrides —
    # they are caller intent, not cached state. That makes them process-lifetime
    # here, so a test that registers one silently repriced every later test in
    # the same process. Clear them alongside the cache: inside the suite there is
    # no caller intent worth carrying across tests.
    clear_model_overrides()
    if not request.node.get_closest_marker("integration"):
        monkeypatch.setattr(model_catalogue, "_catalogues", _OfflineCatalogues())
    yield
    reset_catalogue_cache()
    clear_model_overrides()


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
