"""Manual validation: LangGraphTarget — exhaustive checks.

Tests multi-turn memory, reset effectiveness, clone isolation,
parallel safety, config passthrough, and error handling.

Run with:
    cd packages/evaluatorq-py
    PYTHONPATH=src uv run python scripts/manual_tests/test_langgraph.py
"""

import asyncio
import os
import warnings

warnings.filterwarnings("ignore", message=".*create_react_agent.*")

from dotenv import load_dotenv  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402

from evaluatorq.integrations.langgraph_integration import LangGraphTarget

load_dotenv()

# Route OpenAI calls through the orq router
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("ORQ_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", "https://api.orq.ai/v2/router")

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} — {detail}")


def make_graph() -> CompiledStateGraph:  # pyright: ignore[reportMissingTypeArgument]
    api_key = os.environ["ORQ_API_KEY"]
    model = ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://api.orq.ai/v2/router",
        api_key=api_key,  # pyright: ignore[reportArgumentType]
    )
    return create_react_agent(model, tools=[], checkpointer=MemorySaver())


async def test_basic_response() -> None:
    """Agent returns a non-empty string response."""
    print("\n--- Basic response ---")
    target = LangGraphTarget(make_graph())
    r = await target.send_prompt("Say exactly: PONG")
    check("returns string", isinstance(r, str))
    check("non-empty", len(r) > 0, f"got empty string")
    check("contains expected content", "PONG" in r.upper(), f"got: {r!r}")


async def test_multi_turn_memory() -> None:
    """Agent remembers previous messages within the same thread."""
    print("\n--- Multi-turn memory ---")
    target = LangGraphTarget(make_graph())

    await target.send_prompt(
        "My favorite color is lavender and my pet's name is Mochi. "
        "Please confirm you understand."
    )
    r2 = await target.send_prompt("What is my pet's name?")
    check(
        "agent remembers from previous turn",
        "mochi" in r2.lower(),
        f"agent forgot — response: {r2!r}",
    )


async def test_reset_clears_memory() -> None:
    """After reset, the agent should NOT remember previous conversation."""
    print("\n--- Reset clears memory ---")
    target = LangGraphTarget(make_graph())

    await target.send_prompt(
        "My favorite fruit is persimmon. Please confirm."
    )
    target.reset_conversation()

    r = await target.send_prompt(
        "What is my favorite fruit? "
        "If you don't know, reply: I don't know"
    )
    check(
        "agent does NOT remember after reset",
        "persimmon" not in r.lower(),
        f"agent still remembers — response: {r!r}",
    )


async def test_clone_isolation() -> None:
    """Cloned targets have independent conversation state."""
    print("\n--- Clone isolation ---")
    target = LangGraphTarget(make_graph())

    await target.send_prompt("My favorite city is Reykjavik. Please confirm.")

    cloned = target.clone()
    r = await cloned.send_prompt(
        "What is my favorite city? If you don't know, reply: unknown"
    )
    check(
        "clone does NOT inherit conversation",
        "reykjavik" not in r.lower(),
        f"clone leaked state — response: {r!r}",
    )


async def test_parallel_clones() -> None:
    """Multiple clones can run concurrently without interference."""
    print("\n--- Parallel clones ---")
    graph = make_graph()
    targets = [LangGraphTarget(graph) for _ in range(5)]

    async def run_target(target: LangGraphTarget, word: str) -> str:
        await target.send_prompt(f"My favorite tree is {word}. Confirm.")
        return await target.send_prompt(f"What is my favorite tree?")

    secrets = ["maple", "cedar", "birch", "willow", "aspen"]
    results = await asyncio.gather(
        *[run_target(t, s) for t, s in zip(targets, secrets)]
    )

    check("all 5 returned", len(results) == 5)
    check("all are strings", all(isinstance(r, str) for r in results))

    # Each should remember its own secret (not another target's)
    correct = sum(1 for s, r in zip(secrets, results) if s in r.lower())
    check(
        f"at least 3/5 remembered their own secret",
        correct >= 3,
        f"only {correct}/5 correct. Results: {list(zip(secrets, results))}",
    )


async def test_config_passthrough() -> None:
    """Extra config keys (like recursion_limit) are passed through."""
    print("\n--- Config passthrough ---")
    graph = make_graph()
    # recursion_limit=5 is a valid config key — should not crash
    target = LangGraphTarget(graph, config={"recursion_limit": 5})
    r = await target.send_prompt("Say hello in one word.")
    check("works with extra config", isinstance(r, str) and len(r) > 0, f"got: {r!r}")


async def test_config_with_configurable() -> None:
    """User-provided configurable keys don't crash (no duplicate key error)."""
    print("\n--- Config with configurable ---")
    graph = make_graph()
    target = LangGraphTarget(
        graph,
        config={"configurable": {"custom_key": "value"}, "recursion_limit": 10},
    )
    r = await target.send_prompt("Say hello.")
    check("no crash with user configurable", isinstance(r, str) and len(r) > 0, f"got: {r!r}")


async def main() -> None:
    api_key = os.environ.get("ORQ_API_KEY")
    if not api_key:
        print("✗ ORQ_API_KEY not set. Add it to your .env file.")
        return

    print("=" * 50)
    print("LangGraphTarget — Manual Validation")
    print("=" * 50)

    await test_basic_response()
    await test_multi_turn_memory()
    await test_reset_clears_memory()
    await test_clone_isolation()
    await test_parallel_clones()
    await test_config_passthrough()
    await test_config_with_configurable()

    print("\n" + "=" * 50)
    if failed == 0:
        print(f"✓ ALL PASSED ({passed}/{passed})")
    else:
        print(f"✗ {failed} FAILED, {passed} passed")
    print("=" * 50)


asyncio.run(main())
