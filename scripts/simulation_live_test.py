"""Live integration test for the simulation module.

Usage:
    cd packages/evaluatorq-py
    uv run python scripts/simulation_live_test.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

errors: list[str] = []


def check(condition: bool, msg: str) -> None:
    if condition:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        errors.append(msg)


async def main() -> None:
    from evaluatorq.simulation import (
        CommunicationStyle,
        Criterion,
        Persona,
        Scenario,
        SimulationResult,
        TerminatedBy,
        simulate,
        to_open_responses,
    )

    api_key = os.environ.get("ORQ_API_KEY")
    if not api_key:
        print("ERROR: ORQ_API_KEY not set")
        sys.exit(1)

    print("=== Simulation Live Test ===\n")

    persona = Persona(
        name="Frustrated Customer",
        patience=0.2,
        assertiveness=0.8,
        politeness=0.4,
        technical_level=0.3,
        communication_style=CommunicationStyle.casual,
        background="A customer who has been waiting for help",
    )

    scenario = Scenario(
        name="Product inquiry",
        goal="Get information about available products or services",
        context="The customer wants to know what the company offers",
        criteria=[
            Criterion(description="Agent provides a helpful response", type="must_happen"),
            Criterion(description="Agent is rude or dismissive", type="must_not_happen"),
        ],
    )

    # ── 1. Run simulation ────────────────────────────────────────────────
    print("[1] Running simulation against 'deployment_chat_reply'...\n")
    results = await simulate(
        evaluation_name="live-test",
        agent_key="deployment_chat_reply",
        personas=[persona],
        scenarios=[scenario],
        max_turns=5,
        model="azure/gpt-4o-mini",
        evaluator_names=["goal_achieved", "criteria_met", "turn_efficiency", "conversation_quality"],
    )

    # ── 2. Validate SimulationResult ─────────────────────────────────────
    print("[2] Validating SimulationResult...\n")
    check(len(results) == 1, "exactly 1 result for 1 persona x 1 scenario")

    r = results[0]
    check(isinstance(r, SimulationResult), "result is SimulationResult instance")
    check(r.terminated_by in list(TerminatedBy), f"terminated_by is valid enum: {r.terminated_by.value}")
    check(isinstance(r.reason, str) and len(r.reason) > 0, "reason is non-empty string")
    check(isinstance(r.goal_achieved, bool), "goal_achieved is bool")
    check(0.0 <= r.goal_completion_score <= 1.0, f"goal_completion_score in [0,1]: {r.goal_completion_score}")
    check(isinstance(r.rules_broken, list), "rules_broken is list")
    check(r.turn_count >= 1, f"turn_count >= 1: {r.turn_count}")
    check(len(r.messages) >= 2, f"at least 2 messages (user + assistant): got {len(r.messages)}")

    # Token usage
    check(r.token_usage.total_tokens > 0, f"total_tokens > 0: {r.token_usage.total_tokens}")
    check(r.token_usage.prompt_tokens > 0, f"prompt_tokens > 0: {r.token_usage.prompt_tokens}")
    check(r.token_usage.completion_tokens > 0, f"completion_tokens > 0: {r.token_usage.completion_tokens}")

    # Messages have correct roles
    roles = {m.role for m in r.messages}
    check("user" in roles, "messages contain user role")
    check("assistant" in roles, "messages contain assistant role")
    for m in r.messages:
        check(m.role in ("user", "assistant", "system"), f"message role is valid: {m.role}")
        check(len(m.content) > 0, f"message content is non-empty ({m.role})")

    # Evaluator scores in metadata
    scores = r.metadata.get("evaluator_scores", {})
    check(isinstance(scores, dict), "evaluator_scores in metadata")
    for name in ["goal_achieved", "criteria_met", "turn_efficiency", "conversation_quality"]:
        check(name in scores, f"score '{name}' present")
        check(0.0 <= scores.get(name, -1) <= 1.0, f"score '{name}' in [0,1]: {scores.get(name)}")

    check(r.metadata.get("evaluation_name") == "live-test", "evaluation_name in metadata")

    # Criteria results (judge should have evaluated them)
    if r.terminated_by != TerminatedBy.error:
        check(r.criteria_results is not None, "criteria_results populated by judge")
        if r.criteria_results:
            check(len(r.criteria_results) == 2, f"2 criteria results: got {len(r.criteria_results)}")
            for key, val in r.criteria_results.items():
                check(isinstance(val, bool), f"criteria '{key}' is bool: {val}")

    print()

    # ── 3. Validate OpenResponses mapping ────────────────────────────────
    print("[3] Validating OpenResponses mapping...\n")
    resp = to_open_responses(r)

    # Shape
    check(resp["object"] == "response", "object == 'response'")
    check(isinstance(resp["id"], str) and resp["id"].startswith("resp_"), f"id starts with resp_: {resp['id']}")
    check(isinstance(resp["created_at"], int), "created_at is int timestamp")
    check(resp["model"] == "simulation", "default model == 'simulation'")

    # Status mapping
    expected_status = {
        TerminatedBy.judge: "completed",
        TerminatedBy.error: "failed",
        TerminatedBy.max_turns: "incomplete",
        TerminatedBy.timeout: "incomplete",
    }
    check(
        resp["status"] == expected_status[r.terminated_by],
        f"status maps {r.terminated_by.value} -> {resp['status']}",
    )

    # Error field
    if r.terminated_by == TerminatedBy.error:
        check(resp["error"] is not None, "error populated for failed status")
        check(resp["error"]["message"] == r.reason, "error message matches reason")
    else:
        check(resp["error"] is None, "error is None for non-error status")

    # Input/output message split
    user_msgs = [m for m in r.messages if m.role in ("user", "system")]
    asst_msgs = [m for m in r.messages if m.role == "assistant"]
    check(len(resp["input"]) == len(user_msgs), f"input count matches user+system msgs: {len(resp['input'])}")
    check(len(resp["output"]) == len(asst_msgs), f"output count matches assistant msgs: {len(resp['output'])}")

    # Input message structure
    for i, inp in enumerate(resp["input"]):
        check(inp["role"] in ("user", "system"), f"input[{i}].role is user/system: {inp['role']}")
        check(inp["content"][0]["type"] == "input_text", f"input[{i}] content type is input_text")
        check(len(inp["content"][0]["text"]) > 0, f"input[{i}] has text content")
        check(isinstance(inp["id"], str) and inp["id"].startswith("msg_"), f"input[{i}].id starts with msg_")

    # Output message structure
    for i, out in enumerate(resp["output"]):
        check(out["role"] == "assistant", f"output[{i}].role is assistant")
        check(out["content"][0]["type"] == "output_text", f"output[{i}] content type is output_text")
        check(len(out["content"][0]["text"]) > 0, f"output[{i}] has text content")
        check(isinstance(out["id"], str) and out["id"].startswith("msg_"), f"output[{i}].id starts with msg_")

    # Message content matches original
    input_texts = [inp["content"][0]["text"] for inp in resp["input"]]
    output_texts = [out["content"][0]["text"] for out in resp["output"]]
    orig_user_texts = [m.content for m in r.messages if m.role in ("user", "system")]
    orig_asst_texts = [m.content for m in r.messages if m.role == "assistant"]
    check(input_texts == orig_user_texts, "input texts match original user messages")
    check(output_texts == orig_asst_texts, "output texts match original assistant messages")

    # Unique IDs
    all_ids = [m["id"] for m in resp["input"]] + [m["id"] for m in resp["output"]]
    check(len(all_ids) == len(set(all_ids)), "all message IDs are unique")

    # Usage mapping
    if r.token_usage.total_tokens > 0:
        check(resp["usage"] is not None, "usage present when tokens > 0")
        check(resp["usage"]["input_tokens"] == r.token_usage.prompt_tokens, "input_tokens == prompt_tokens")
        check(resp["usage"]["output_tokens"] == r.token_usage.completion_tokens, "output_tokens == completion_tokens")
        check(resp["usage"]["total_tokens"] == r.token_usage.total_tokens, "total_tokens matches")

    # Metadata mapping
    meta = resp["metadata"]
    check(meta["framework"] == "simulation", "metadata.framework == 'simulation'")
    check(meta["goal_achieved"] == r.goal_achieved, "metadata.goal_achieved matches")
    check(meta["goal_completion_score"] == r.goal_completion_score, "metadata.goal_completion_score matches")
    check(meta["terminated_by"] == r.terminated_by.value, "metadata.terminated_by matches enum value")
    check(meta["reason"] == r.reason, "metadata.reason matches")
    check(meta["turn_count"] == r.turn_count, "metadata.turn_count matches")

    if r.criteria_results is not None:
        check(meta["criteria_results"] == r.criteria_results, "metadata.criteria_results matches")
    else:
        check("criteria_results" not in meta, "criteria_results omitted when None")

    # Custom model parameter
    resp_custom = to_open_responses(r, model="gpt-4o")
    check(resp_custom["model"] == "gpt-4o", "custom model parameter works")

    print()

    # ── Summary ──────────────────────────────────────────────────────────
    if errors:
        print(f"=== FAILED: {len(errors)} error(s) ===")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
