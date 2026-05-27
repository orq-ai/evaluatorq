"""Regression test for spec §8 / Risk 1 — pre-existing bug at runner.py:833.

Before PR1, the dynamic-job wiring hardcoded ``DefaultErrorMapper()`` which
masked the backend's provider-specific mapping. After PR1 the orchestrator
holds a ``Backend`` and routes target exceptions through ``backend.map_error``,
so an ORQ HTTP error surfaces as ``orq.http.<status>`` instead of the generic
``target_error``.

The first test pins the unit-level mapping; the second pins the *wiring* — it
drives a real ``MultiTurnOrchestrator.run_attack`` with a backend and asserts
the mapped code reaches ``OrchestratorResult.error_code``. The second test is
the one that actually guards the regression: it fails if the orchestrator ever
goes back to a hardcoded default mapper.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.redteam.backends.base import Backend


class _OrqHTTPError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"orq returned {status_code}")
        self.status_code = status_code


_PATCH_REDTEAM_SPAN = "evaluatorq.redteam.adaptive.orchestrator.with_redteam_span"
_PATCH_LLM_SPAN = "evaluatorq.redteam.adaptive.orchestrator.with_llm_span"
_PATCH_RECORD_LLM = "evaluatorq.redteam.adaptive.orchestrator.record_llm_response"


@asynccontextmanager
async def _noop_span_ctx(*args, **kwargs):
    yield None


def _make_completion(content: str):
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    response.model = "test-model"
    response.id = "resp-123"
    return response


def _make_context():
    from evaluatorq.redteam.contracts import AgentContext

    return AgentContext(key="victim-agent", display_name="Victim", description="d")


def _make_strategy():
    from evaluatorq.redteam.contracts import (
        AttackStrategy,
        AttackTechnique,
        DeliveryMethod,
        TurnType,
    )

    return AttackStrategy(
        category="ASI01",
        name="test-strategy",
        description="Test strategy description",
        attack_technique=AttackTechnique.INDIRECT_INJECTION,
        delivery_methods=[DeliveryMethod.CRESCENDO],
        turn_type=TurnType.MULTI,
        objective_template="Convince {agent_name} to follow instructions",
    )


def test_orq_http_exception_maps_to_orq_http_status_code():
    """Unit: ORQ rate-limit exception → ``orq.http.429``, not ``target_error``.

    ``_OrqHTTPError`` has a ``status_code`` attribute, so ``extract_status_code``
    picks it up structurally → ``orq.http.429``.
    """
    from evaluatorq.redteam.backends.orq import ORQBackend

    backend = ORQBackend(orq_client=MagicMock(), timeout_ms=1000)
    code, msg = backend.map_error(_OrqHTTPError(429))
    assert code == "orq.http.429", f"expected orq.http.429, got {code!r}"
    assert "orq returned 429" in msg


def test_orq_text_only_rate_limit_maps_to_orq_rate_limit():
    """Unit: rate-limit indicated only in text (no ``status_code`` attr) → ``orq.rate_limit``.

    Documents the distinction vs. the structured-attribute path above: without a
    ``status_code`` attribute, the mapper falls through to text heuristics and
    emits the generic ``orq.rate_limit`` code.
    """
    from evaluatorq.redteam.backends.orq import ORQBackend

    class _RateLimitedError(Exception):
        pass

    backend = ORQBackend(orq_client=MagicMock(), timeout_ms=1000)
    code, _msg = backend.map_error(_RateLimitedError("upstream returned 429"))
    assert code == "orq.rate_limit", f"expected orq.rate_limit, got {code!r}"


@pytest.mark.asyncio
@patch(_PATCH_RECORD_LLM)
@patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
@patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
async def test_orchestrator_uses_backend_map_error_for_error_code(_rs, _ls, _rl):
    """Wiring: orchestrator routes target exceptions through ``backend.map_error``.

    This is the actual regression guard. If the orchestrator reverts to a
    hardcoded ``DefaultErrorMapper``, ``error_code`` becomes ``target_error``
    and this fails.
    """
    from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator

    class _MapErrorBackend(Backend):
        def __init__(self):
            super().__init__(name="orq")

        def create_target(self, agent_key):
            raise NotImplementedError("not used in this test")

        async def cleanup_memory(self, ctx, entity_ids):
            return None

        def map_error(self, exc):
            return ("orq.http.429", f"{type(exc).__name__}: {exc}")

    mock_llm = MagicMock()
    mock_llm.chat = MagicMock()
    mock_llm.chat.completions = MagicMock()
    mock_llm.chat.completions.create = AsyncMock(
        side_effect=[_make_completion("Attack turn 1"), _make_completion("Attack turn 2")]
    )

    orchestrator = MultiTurnOrchestrator(
        llm_client=mock_llm,
        model="test-model",
        backend=_MapErrorBackend(),
    )

    target = MagicMock()
    target.respond = AsyncMock(side_effect=[_OrqHTTPError(429), _OrqHTTPError(429)])
    target.new = MagicMock()

    result = await orchestrator.run_attack(
        target=target,
        strategy=_make_strategy(),
        objective="Test",
        agent_context=_make_context(),
        max_turns=5,
    )

    assert result.error_code == "orq.http.429", (
        f"expected backend-mapped orq.http.429, got {result.error_code!r} — "
        "orchestrator may be using a hardcoded default mapper"
    )
    assert result.error_type == "target_error"
