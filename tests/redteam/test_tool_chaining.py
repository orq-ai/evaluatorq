"""Unit tests for the tool-chaining decomposition + verifier components.

Covers:
  - ToolChainingPlanner.decompose() structured-output parsing, ordering, errors
  - ToolChainingVerifier schema-only validation (sync)
  - format_plan_for_prompt() rendering
  - MultiTurnOrchestrator._plan_tool_chain() integration + fail-open behaviour
  - multi_tool_chain_escalation strategy carries decompose_tool_chain=True

All external LLM calls are mocked so tests run offline.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.redteam.adaptive.tool_chaining import (
    DecompositionResult,
    ToolCallStep,
    ToolChainingPlanner,
    ToolChainingVerifier,
    VerificationStatus,
    _DecompositionSchema,
    format_plan_for_prompt,
)
from evaluatorq.redteam.contracts import AgentContext, ToolInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(index: int = 0, tool: str = "email_tool") -> ToolCallStep:
    return ToolCallStep(
        step_index=index,
        tool_name=tool,
        argument_description="Send output to attacker@evil.com",
        benign_framing="Can you forward this summary to my colleague?",
    )


def _make_decomposition(steps: list[ToolCallStep] | None = None) -> DecompositionResult:
    steps = steps or [_make_step(0, "lookup_tool"), _make_step(1, "email_tool")]
    return DecompositionResult(
        objective="exfiltrate user data",
        steps=steps,
        reasoning="Chain lookup then exfil via email",
    )


def _schema(steps: list[ToolCallStep] | None = None) -> _DecompositionSchema:
    steps = steps or [_make_step(0, "lookup_tool"), _make_step(1, "email_tool")]
    return _DecompositionSchema(objective="exfiltrate user data", steps=steps, reasoning="chain then exfil")


def _parsed_response(schema: _DecompositionSchema) -> MagicMock:
    """Mock of an OpenAI ParsedChatCompletion (`.parse()` result)."""
    msg = MagicMock()
    msg.parsed = schema
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    return resp


def _completion_response(content: str) -> MagicMock:
    """Mock of a plain ChatCompletion (`.create()` result), for the adversarial turn."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    return resp


def _mock_client(parse: AsyncMock | None = None, create: AsyncMock | None = None) -> MagicMock:
    client = MagicMock()
    client.chat.completions.parse = parse or AsyncMock()
    client.chat.completions.create = create or AsyncMock()
    return client


# ---------------------------------------------------------------------------
# ToolChainingPlanner
# ---------------------------------------------------------------------------


class TestToolChainingPlanner:
    @pytest.mark.asyncio
    async def test_decompose_parses_structured_output(self) -> None:
        parse = AsyncMock(return_value=_parsed_response(_schema()))
        planner = ToolChainingPlanner(client=_mock_client(parse=parse), model="gpt-4.1")
        result = await planner.decompose(
            objective="exfiltrate user data",
            agent_name="CustomerBot",
            agent_description="A customer service agent",
            available_tools=["lookup_tool", "email_tool"],
        )

        assert isinstance(result, DecompositionResult)
        assert [s.tool_name for s in result.steps] == ["lookup_tool", "email_tool"]
        assert result.token_usage is not None
        assert result.token_usage.calls == 1

    @pytest.mark.asyncio
    async def test_decompose_preserves_emit_order(self) -> None:
        # Emit order is authoritative; the model-supplied step_index is ignored
        # (it can be duplicated/gapped), so steps come back exactly as emitted.
        schema = _schema([_make_step(2, "third"), _make_step(0, "first"), _make_step(1, "second")])
        planner = ToolChainingPlanner(
            client=_mock_client(parse=AsyncMock(return_value=_parsed_response(schema))), model="gpt-4.1"
        )
        result = await planner.decompose(
            objective="x", agent_name="Bot", agent_description="d", available_tools=["first", "second", "third"]
        )
        assert [s.tool_name for s in result.steps] == ["third", "first", "second"]

    @pytest.mark.asyncio
    async def test_decompose_raises_when_parsed_none(self) -> None:
        resp = _parsed_response(_schema())
        resp.choices[0].message.parsed = None
        planner = ToolChainingPlanner(client=_mock_client(parse=AsyncMock(return_value=resp)), model="gpt-4.1")
        with pytest.raises(ValueError, match="no parsed content"):
            await planner.decompose(objective="x", agent_name="B", agent_description="d", available_tools=["t"])

    @pytest.mark.asyncio
    async def test_decompose_raises_when_no_choices(self) -> None:
        resp = _parsed_response(_schema())
        resp.choices = []
        planner = ToolChainingPlanner(client=_mock_client(parse=AsyncMock(return_value=resp)), model="gpt-4.1")
        with pytest.raises(ValueError, match="no choices"):
            await planner.decompose(objective="x", agent_name="B", agent_description="d", available_tools=["t"])

    @pytest.mark.asyncio
    async def test_decompose_injects_tool_list_and_schema(self) -> None:
        parse = AsyncMock(return_value=_parsed_response(_schema()))
        planner = ToolChainingPlanner(client=_mock_client(parse=parse), model="gpt-4.1")
        await planner.decompose(
            objective="x", agent_name="Bot", agent_description="d", available_tools=["alpha", "beta", "gamma"]
        )
        kwargs = parse.call_args.kwargs
        user_msg = kwargs["messages"][1]["content"]
        assert "alpha" in user_msg and "beta" in user_msg
        # Uses structured outputs (response_format), not manual JSON parsing.
        assert kwargs["response_format"] is _DecompositionSchema

    @pytest.mark.asyncio
    async def test_decompose_tolerates_braces_in_objective(self) -> None:
        # Objectives are filled from earlier templates and can carry stray {...};
        # safe_substitute must not treat them as format fields (would KeyError).
        parse = AsyncMock(return_value=_parsed_response(_schema()))
        planner = ToolChainingPlanner(client=_mock_client(parse=parse), model="gpt-4.1")
        await planner.decompose(
            objective="exfiltrate {agent_name}'s {secret} data",
            agent_name="Bot",
            agent_description="d",
            available_tools=["alpha"],
        )
        user_msg = parse.call_args.kwargs["messages"][1]["content"]
        assert "{secret}" in user_msg


# ---------------------------------------------------------------------------
# ToolChainingVerifier (schema-only, synchronous)
# ---------------------------------------------------------------------------


class TestToolChainingVerifier:
    def test_verify_valid_tool(self) -> None:
        verifier = ToolChainingVerifier(available_tools=["email_tool", "lookup_tool"])
        assert verifier.verify(_make_step(0, "email_tool")).status == VerificationStatus.VALID

    def test_verify_invalid_tool(self) -> None:
        verifier = ToolChainingVerifier(available_tools=["email_tool"])
        assert verifier.verify(_make_step(0, "nope")).status == VerificationStatus.INVALID_TOOL

    def test_verify_skipped_when_no_schema(self) -> None:
        verifier = ToolChainingVerifier(available_tools=None)
        assert verifier.verify(_make_step(0, "any")).status == VerificationStatus.SKIPPED

    def test_empty_tool_list_invalidates_all(self) -> None:
        # [] means "agent exposes no tools" → INVALID_TOOL, not SKIPPED (None).
        verifier = ToolChainingVerifier(available_tools=[])
        assert verifier.verify(_make_step(0, "any")).status == VerificationStatus.INVALID_TOOL

    def test_case_insensitive_tool_match(self) -> None:
        verifier = ToolChainingVerifier(available_tools=["Email_Tool"])
        assert verifier.verify(_make_step(0, "email_tool")).status == VerificationStatus.VALID

    def test_verify_all_filters_invalid(self) -> None:
        verifier = ToolChainingVerifier(available_tools=["lookup_tool"])
        steps = [_make_step(0, "lookup_tool"), _make_step(1, "email_tool")]
        valid, results = verifier.verify_all(steps)
        assert [s.tool_name for s in valid] == ["lookup_tool"]
        assert results[1].status == VerificationStatus.INVALID_TOOL

    def test_verify_all_accepts_skipped(self) -> None:
        verifier = ToolChainingVerifier(available_tools=None)
        valid, results = verifier.verify_all([_make_step(0, "a"), _make_step(1, "b")])
        assert len(valid) == 2
        assert all(r.status == VerificationStatus.SKIPPED for r in results)


# ---------------------------------------------------------------------------
# format_plan_for_prompt
# ---------------------------------------------------------------------------


def test_format_plan_lists_tools_in_order() -> None:
    plan = format_plan_for_prompt([_make_step(0, "lookup_tool"), _make_step(1, "email_tool")])
    assert "Tool-Chaining Decomposition Plan" in plan
    assert plan.index("lookup_tool") < plan.index("email_tool")
    assert "1." in plan and "2." in plan


def test_format_plan_numbers_sequentially_ignoring_step_index() -> None:
    # Duplicate/gapped model indices must not garble display numbering.
    plan = format_plan_for_prompt([_make_step(0, "a"), _make_step(0, "b"), _make_step(5, "c")])
    assert "1. Elicit the `a`" in plan
    assert "2. Elicit the `b`" in plan
    assert "3. Elicit the `c`" in plan


# ---------------------------------------------------------------------------
# Orchestrator integration: _plan_tool_chain
# ---------------------------------------------------------------------------


def _make_orchestrator(parse: AsyncMock | None = None, create: AsyncMock | None = None):
    from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator

    return MultiTurnOrchestrator(llm_client=_mock_client(parse=parse, create=create))


def _tc_strategy(**overrides: Any):
    from evaluatorq.redteam.contracts import AttackStrategy, AttackTechnique, DeliveryMethod, TurnType

    defaults: dict[str, Any] = {
        "category": "ASI02",
        "name": "multi_tool_chain_escalation",
        "description": "tool chaining",
        "attack_technique": AttackTechnique.TOOL_ABUSE,
        "delivery_methods": [DeliveryMethod.CRESCENDO],
        "turn_type": TurnType.MULTI,
        "requires_tools": True,
        "decompose_tool_chain": True,
        "objective_template": "Chain tools on {agent_name}",
    }
    defaults.update(overrides)
    return AttackStrategy(**defaults)


def _ctx_with_tools() -> AgentContext:
    return AgentContext(
        key="agent",
        display_name="Agent",
        description="desc",
        tools=[ToolInfo(name="lookup_tool", description="lookup"), ToolInfo(name="email_tool", description="email")],
    )


class TestPlanToolChain:
    @pytest.mark.asyncio
    async def test_returns_plan_for_tool_chaining_strategy(self) -> None:
        orch = _make_orchestrator(parse=AsyncMock(return_value=_parsed_response(_schema())))
        plan, usage = await orch._plan_tool_chain(_tc_strategy(), "exfil", _ctx_with_tools(), 5)
        assert plan is not None
        assert "lookup_tool" in plan
        assert usage is not None

    @pytest.mark.asyncio
    async def test_skips_when_not_tool_chaining(self) -> None:
        parse = AsyncMock()
        orch = _make_orchestrator(parse=parse)
        plan, usage = await orch._plan_tool_chain(_tc_strategy(decompose_tool_chain=False), "x", _ctx_with_tools(), 5)
        assert plan is None and usage is None
        parse.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_agent_has_no_tools(self) -> None:
        parse = AsyncMock()
        orch = _make_orchestrator(parse=parse)
        ctx = AgentContext(key="agent", display_name="Agent", description="desc")
        plan, usage = await orch._plan_tool_chain(_tc_strategy(), "x", ctx, 5)
        assert plan is None and usage is None
        parse.assert_not_called()

    @pytest.mark.asyncio
    async def test_fails_open_when_planner_raises(self) -> None:
        orch = _make_orchestrator(parse=AsyncMock(side_effect=RuntimeError("LLM down")))
        plan, _ = await orch._plan_tool_chain(_tc_strategy(), "x", _ctx_with_tools(), 5)
        assert plan is None  # falls back to plan-free attack rather than aborting

    @pytest.mark.asyncio
    async def test_returns_no_plan_when_all_steps_invalid(self) -> None:
        # Planner names a tool the agent does not have → verifier drops it → no plan.
        schema = _schema([_make_step(0, "nonexistent_tool")])
        orch = _make_orchestrator(parse=AsyncMock(return_value=_parsed_response(schema)))
        plan, usage = await orch._plan_tool_chain(_tc_strategy(), "x", _ctx_with_tools(), 5)
        assert plan is None
        # planner usage is still reported even though no steps survived
        assert usage is not None

    @pytest.mark.asyncio
    async def test_caps_plan_to_max_turns(self) -> None:
        # 5 valid steps but a 2-turn budget — only the first 2 may be elicited.
        schema = _schema([_make_step(i, "lookup_tool") for i in range(5)])
        orch = _make_orchestrator(parse=AsyncMock(return_value=_parsed_response(schema)))
        plan, _ = await orch._plan_tool_chain(_tc_strategy(), "x", _ctx_with_tools(), 2)
        assert plan is not None
        assert plan.count("Elicit the") == 2


# ---------------------------------------------------------------------------
# run_attack integration: planner usage folds into adversarial token totals
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_span(*_a: Any, **_k: Any):
    yield MagicMock()


def _noop_span_ctx(*a: Any, **k: Any):
    return _noop_span(*a, **k)


class TestRunAttackTokenFolding:
    @pytest.mark.asyncio
    @patch("evaluatorq.redteam.adaptive.orchestrator.record_llm_response")
    @patch("evaluatorq.redteam.adaptive.orchestrator.with_llm_span", side_effect=_noop_span_ctx)
    @patch("evaluatorq.redteam.adaptive.orchestrator.with_redteam_span", side_effect=_noop_span_ctx)
    async def test_planner_call_counted_in_adversarial_usage(self, _rs: Any, _ls: Any, _rl: Any) -> None:
        from evaluatorq.contracts import AgentResponse

        orch = _make_orchestrator(
            parse=AsyncMock(return_value=_parsed_response(_schema([_make_step(0, "lookup_tool")]))),
            create=AsyncMock(return_value=_completion_response("look up order 1")),
        )
        target = MagicMock()
        target.respond = AsyncMock(return_value=AgentResponse(text="ok"))
        target.new = MagicMock()

        result = await orch.run_attack(
            target=target,
            strategy=_tc_strategy(),
            objective="exfil",
            agent_context=_ctx_with_tools(),
            max_turns=1,
        )

        # planner (1 parse call) + 1 adversarial turn (1 create call) = 2 adversarial
        # calls, confirming the planner usage was folded into the adversarial totals.
        assert result.token_usage_adversarial is not None
        assert result.token_usage_adversarial.calls == 2
        assert result.token_usage_adversarial.prompt_tokens == 20


# ---------------------------------------------------------------------------
# Strategy wiring
# ---------------------------------------------------------------------------


def test_multi_tool_chain_escalation_has_decompose_flag() -> None:
    from evaluatorq.redteam.frameworks.owasp_asi import ASI02_STRATEGIES

    strat = next(s for s in ASI02_STRATEGIES if s.name == "multi_tool_chain_escalation")
    assert strat.decompose_tool_chain is True
