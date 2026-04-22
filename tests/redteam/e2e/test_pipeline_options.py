"""E2E tests for pipeline options: hooks, output artifacts, error handling."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from openai import AsyncOpenAI

from evaluatorq.redteam import red_team
from evaluatorq.redteam.backends.base import BackendBundle
from evaluatorq.redteam.contracts import SaveMode
from evaluatorq.redteam.exceptions import CancelledError
from evaluatorq.redteam.hooks import ConfirmPayload, DefaultHooks

from .conftest import DeterministicAsyncOpenAI


class _CancellingHooks(DefaultHooks):
    """Hooks implementation that captures the confirm payload and cancels."""

    def __init__(self) -> None:
        self.confirm_payload: ConfirmPayload | None = None

    def on_confirm(self, payload: ConfirmPayload) -> bool:
        self.confirm_payload = payload
        return False


@contextmanager
def _dynamic_patches(mock_backend_bundle: BackendBundle):
    with (
        patch("evaluatorq.redteam.runner.resolve_backend", return_value=mock_backend_bundle),
        patch("evaluatorq.redteam.runner.init_tracing_if_needed", return_value=None),
        patch("evaluatorq.redteam.runner.capture_parent_context", return_value=None),
    ):
        yield


# ---------------------------------------------------------------------------
# Hooks — on_confirm cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hooks_on_confirm_cancels_dynamic(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: BackendBundle,
) -> None:
    """on_confirm returning False should raise RuntimeError."""
    hooks = _CancellingHooks()
    with _dynamic_patches(mock_backend_bundle):
        with pytest.raises(CancelledError, match="cancelled"):
            await red_team(
                "agent:e2e-test-agent",
                mode="dynamic",
                categories=["ASI01"],
                generate_strategies=False,
                attack_model="e2e-attack-model",
                evaluator_model="e2e-evaluator",
                parallelism=2,
                backend="openai",
                llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
                hooks=hooks,
            )

    assert hooks.confirm_payload is not None
    assert "num_datapoints" in hooks.confirm_payload
    assert "categories" in hooks.confirm_payload


@pytest.mark.asyncio
async def test_hooks_on_confirm_cancels_hybrid(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: BackendBundle,
    static_dataset_path: Path,
) -> None:
    """on_confirm returning False in hybrid mode should raise RuntimeError."""
    hooks = _CancellingHooks()
    with _dynamic_patches(mock_backend_bundle):
        with pytest.raises(CancelledError, match="cancelled"):
            await red_team(
                "agent:e2e-test-agent",
                mode="hybrid",
                categories=["ASI01"],
                generate_strategies=False,
                attack_model="e2e-attack-model",
                evaluator_model="e2e-evaluator",
                parallelism=2,
                backend="openai",
                llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
                dataset=str(static_dataset_path),
                hooks=hooks,
            )

    assert hooks.confirm_payload is not None
    assert "num_dynamic" in hooks.confirm_payload
    assert "num_static" in hooks.confirm_payload


# ---------------------------------------------------------------------------
# Output artifacts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_static_output_artifacts(
    mock_llm_client: DeterministicAsyncOpenAI,
    static_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Static mode should write numbered artifact files."""
    output_dir = tmp_path / "static_artifacts"
    await red_team(
        "openai:e2e-static-model",
        mode="static",
        evaluator_model="e2e-evaluator",
        parallelism=2,
        backend="openai",
        dataset=str(static_dataset_path),
        llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
        output_dir=output_dir,
        save=SaveMode.DETAIL,
    )

    assert (output_dir / "01_datapoints.json").exists()
    assert (output_dir / "02_attack_results.json").exists()
    assert (output_dir / "03_summary_report.json").exists()

    # Verify envelope structure on stage files
    import json

    datapoints = json.loads((output_dir / "01_datapoints.json").read_text())
    assert "saved_at" in datapoints
    assert "data" in datapoints


@pytest.mark.asyncio
async def test_dynamic_output_artifacts(
    mock_llm_client: DeterministicAsyncOpenAI,
    mock_backend_bundle: BackendBundle,
    tmp_path: Path,
) -> None:
    """Dynamic mode should write artifact files via the unified pipeline."""
    output_dir = tmp_path / "dynamic_artifacts"
    with _dynamic_patches(mock_backend_bundle):
        await red_team(
            "agent:e2e-test-agent",
            mode="dynamic",
            categories=["ASI01"],
            generate_strategies=False,
            attack_model="e2e-attack-model",
            evaluator_model="e2e-evaluator",
            parallelism=2,
            backend="openai",
            llm_client=cast(AsyncOpenAI, cast(object, mock_llm_client)),
            output_dir=output_dir,
            save=SaveMode.DETAIL,
        )

    assert (output_dir / "01_all_datapoints.json").exists()
    assert (output_dir / "02_attack_results.json").exists()
    assert (output_dir / "03_summary_report.json").exists()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_mode_raises() -> None:
    """Invalid mode should raise ValueError."""
    with pytest.raises(ValueError, match="is not a valid Pipeline"):
        await red_team(
            "openai:some-model",
            mode="nonexistent",
            backend="openai",
        )


@pytest.mark.asyncio
async def test_empty_targets_raises() -> None:
    """Empty target list should raise ValueError."""
    with pytest.raises(ValueError, match="at least one target"):
        await red_team(
            [],
            mode="static",
            backend="openai",
        )
