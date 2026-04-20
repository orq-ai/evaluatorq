"""Manual validation: OpenAIAgentTarget — exhaustive checks.

Tests multi-turn memory, reset effectiveness, clone isolation,
parallel safety, run_kwargs passthrough, and final_output handling.

Run with:
    cd packages/evaluatorq-py
    PYTHONPATH=src uv run python scripts/manual_tests/test_openai_agents.py
"""

import asyncio
import os

from agents import Agent, OpenAIChatCompletionsModel
from dotenv import load_dotenv
from openai import AsyncOpenAI

from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget

load_dotenv()

# Route OpenAI calls through the orq router
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("ORQ_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", "https://api.orq.ai/v2/router")

# Disable OpenAI Agents SDK tracing (orq key is not valid for OpenAI's trace endpoint)
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

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


def make_agent() -> Agent:
    api_key = os.environ["ORQ_API_KEY"]
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.orq.ai/v2/router",
    )
    return Agent(
        name="test-agent",
        instructions="You are a helpful assistant. Keep replies to one sentence.",
        model=OpenAIChatCompletionsModel(model="openai/gpt-4o-mini", openai_client=client),
    )


async def test_basic_response() -> None:
    """Agent returns a non-empty string response."""
    print("\n--- Basic response ---")
    target = OpenAIAgentTarget(make_agent())
    r = await target.send_prompt("Say exactly: PONG")
    check("returns string", isinstance(r, str))
    check("non-empty", len(r) > 0)
    check("contains expected content", "PONG" in r.upper(), f"got: {r!r}")


async def test_multi_turn_memory() -> None:
    """Agent remembers previous messages via history management."""
    print("\n--- Multi-turn memory ---")
    target = OpenAIAgentTarget(make_agent())

    await target.send_prompt(
        "My favorite color is turquoise and my pet's name is Biscuit. Please confirm."
    )
    r2 = await target.send_prompt("What is my pet's name?")
    check(
        "agent remembers from previous turn",
        "biscuit" in r2.lower(),
        f"agent forgot — response: {r2!r}",
    )

    # Verify history grew
    check(
        "history has accumulated",
        len(target._history) > 2,
        f"history length: {len(target._history)}",
    )


async def test_reset_clears_memory() -> None:
    """After reset, the agent has no knowledge of previous turns."""
    print("\n--- Reset clears memory ---")
    target = OpenAIAgentTarget(make_agent())

    await target.send_prompt("My favorite fruit is dragonfruit. Please confirm.")
    target.reset_conversation()

    check("history is empty after reset", len(target._history) == 0)

    r = await target.send_prompt(
        "What is my favorite fruit? "
        "If you don't know, reply exactly: I don't know"
    )
    check(
        "agent does NOT remember after reset",
        "dragonfruit" not in r.lower(),
        f"agent still remembers — response: {r!r}",
    )


async def test_clone_isolation() -> None:
    """Cloned targets start with empty history."""
    print("\n--- Clone isolation ---")
    target = OpenAIAgentTarget(make_agent())

    await target.send_prompt("My name is Juniper. Please confirm.")
    check("original has history", len(target._history) > 0)

    cloned = target.clone()
    check("clone starts with empty history", len(cloned._history) == 0)

    r = await cloned.send_prompt(
        "What is my name? If you don't know, reply exactly: unknown"
    )
    check(
        "clone does NOT know original's conversation",
        "juniper" not in r.lower(),
        f"clone leaked state — response: {r!r}",
    )


async def test_parallel_clones() -> None:
    """Multiple independent targets can run concurrently."""
    print("\n--- Parallel targets ---")
    agent = make_agent()

    words = ["maple", "cedar", "birch", "willow", "aspen"]

    async def run_one(word: str) -> str:
        target = OpenAIAgentTarget(agent)
        await target.send_prompt(f"My favorite tree is {word}. Confirm.")
        return await target.send_prompt("What is my favorite tree?")

    results = await asyncio.gather(*[run_one(w) for w in words])

    check("all 5 returned", len(results) == 5)
    check("all are strings", all(isinstance(r, str) for r in results))

    correct = sum(1 for w, r in zip(words, results) if w in r.lower())
    check(
        f"at least 3/5 remembered their own secret",
        correct >= 3,
        f"only {correct}/5. Results: {list(zip(words, results))}",
    )


async def test_three_turn_conversation() -> None:
    """A longer conversation maintains coherent context."""
    print("\n--- Three-turn conversation ---")
    target = OpenAIAgentTarget(make_agent())

    await target.send_prompt("I'm going to tell you three colors.")
    await target.send_prompt("The colors are: red, blue, green.")
    r3 = await target.send_prompt("List the three colors I told you.")

    has_colors = all(c in r3.lower() for c in ["red", "blue", "green"])
    check(
        "remembers all three colors after 3 turns",
        has_colors,
        f"response: {r3!r}",
    )


async def test_run_kwargs_passthrough() -> None:
    """Extra run_kwargs are accepted without crashing."""
    print("\n--- run_kwargs passthrough ---")
    target = OpenAIAgentTarget(make_agent(), run_kwargs={"max_turns": 3})
    r = await target.send_prompt("Say hello.")
    check("works with run_kwargs", isinstance(r, str) and len(r) > 0, f"got: {r!r}")


async def main() -> None:
    api_key = os.environ.get("ORQ_API_KEY")
    if not api_key:
        print("✗ ORQ_API_KEY not set. Add it to your .env file.")
        return

    print("=" * 50)
    print("OpenAIAgentTarget — Manual Validation")
    print("=" * 50)

    await test_basic_response()
    await test_multi_turn_memory()
    await test_reset_clears_memory()
    await test_clone_isolation()
    await test_parallel_clones()
    await test_three_turn_conversation()
    await test_run_kwargs_passthrough()

    print("\n" + "=" * 50)
    if failed == 0:
        print(f"✓ ALL PASSED ({passed}/{passed})")
    else:
        print(f"✗ {failed} FAILED, {passed} passed")
    print("=" * 50)


asyncio.run(main())
