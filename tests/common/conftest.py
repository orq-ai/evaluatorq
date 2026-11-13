"""Shared fixtures for the model-catalogue tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from evaluatorq.common import model_catalogue

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@pytest.fixture
def fake_catalogue(monkeypatch: pytest.MonkeyPatch) -> Callable[[Mapping[str, model_catalogue.ModelInfo]], None]:
    """Factory: ``fake_catalogue({'gpt-5-mini': ModelInfo(...)})`` serves that table as the live catalogue.

    Replaces ``_load_catalogue`` for the test and clears the catalogue cache before it (and, via the
    suite-wide autouse fixture, after it), so a table installed mid-test is never shadowed by a
    cached one. Call it again to swap the table.
    """

    def install(table: Mapping[str, model_catalogue.ModelInfo]) -> None:
        async def fake_load(client: object = None) -> Mapping[str, model_catalogue.ModelInfo]:  # noqa: ARG001
            return table

        monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
        model_catalogue.reset_catalogue_cache()

    return install
