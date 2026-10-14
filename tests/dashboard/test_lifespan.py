"""The app must start and stop cleanly: TestClient as a context manager runs the lifespan."""

from __future__ import annotations

from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app


def test_lifespan_starts_and_stops() -> None:
    # fasthtml 0.14 iterates the lifespan with ``async for``; a context-manager lifespan crashed startup.
    with TestClient(build_app()) as client:
        assert client.get('/auth/status').status_code == 200
