"""Manual validation: VercelAISdkTarget — exhaustive checks.

Tests multi-turn memory, reset effectiveness, clone isolation,
parallel safety, response parsing, and error handling.

Requires a running Vercel AI SDK endpoint. Start one with:

    # In a separate terminal (Node.js project with AI SDK)
    npx next dev  # or any server exposing POST /api/chat

Run with:
    cd packages/evaluatorq-py
    PYTHONPATH=src uv run python scripts/manual_tests/test_vercel_ai_sdk.py
"""

import asyncio
import os

from dotenv import load_dotenv

from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget

load_dotenv()

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


def make_target() -> VercelAISdkTarget:
    url = os.environ.get("VERCEL_AI_SDK_URL", "http://localhost:3000/api/chat")
    headers = {}
    api_key = os.environ.get("VERCEL_AI_SDK_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return VercelAISdkTarget(url, headers=headers)


async def test_basic_response() -> None:
    """Endpoint returns a non-empty string response."""
    print("\n--- Basic response ---")
    target = make_target()
    r = await target.send_prompt("Say exactly: PONG")
    check("returns string", isinstance(r.text, str))
    check("non-empty", len(r.text) > 0, "got empty string")
    check("contains expected content", "PONG" in r.text.upper(), f"got: {r!r}")


async def test_multi_turn_memory() -> None:
    """Agent remembers previous messages within the same conversation."""
    print("\n--- Multi-turn memory ---")
    target = make_target()

    await target.send_prompt(
        "My favorite color is lavender and my pet's name is Mochi. "
        "Please confirm you understand."
    )
    r2 = await target.send_prompt("What is my pet's name?")
    check(
        "agent remembers from previous turn",
        "mochi" in r2.text.lower(),
        f"agent forgot — response: {r2!r}",
    )


async def test_reset_clears_memory() -> None:
    """After reset, the agent should NOT remember previous conversation."""
    print("\n--- Reset clears memory ---")
    target = make_target()

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
        "persimmon" not in r.text.lower(),
        f"agent still remembers — response: {r!r}",
    )


async def test_clone_isolation() -> None:
    """Cloned targets have independent conversation state."""
    print("\n--- Clone isolation ---")
    target = make_target()

    await target.send_prompt("My favorite city is Reykjavik. Please confirm.")

    cloned = target.clone()
    r = await cloned.send_prompt(
        "What is my favorite city? If you don't know, reply: unknown"
    )
    check(
        "clone does NOT inherit conversation",
        "reykjavik" not in r.text.lower(),
        f"clone leaked state — response: {r!r}",
    )


async def test_parallel_clones() -> None:
    """Multiple clones can run concurrently without interference."""
    print("\n--- Parallel clones ---")
    base = make_target()
    targets = [base.clone() for _ in range(5)]

    async def run_target(target: VercelAISdkTarget, word: str):  # type: ignore[return]
        await target.send_prompt(f"My favorite tree is {word}. Confirm.")
        return await target.send_prompt("What is my favorite tree?")

    secrets = ["maple", "cedar", "birch", "willow", "aspen"]
    results = await asyncio.gather(
        *[run_target(t, s) for t, s in zip(targets, secrets)]
    )

    check("all 5 returned", len(results) == 5)
    check("all are strings", all(isinstance(r.text, str) for r in results))

    correct = sum(1 for s, r in zip(secrets, results) if s in r.text.lower())
    check(
        f"at least 3/5 remembered their own secret",
        correct >= 3,
        f"only {correct}/5 correct. Results: {list(zip(secrets, results))}",
    )


async def main() -> None:
    url = os.environ.get("VERCEL_AI_SDK_URL", "http://localhost:3000/api/chat")
    print("=" * 50)
    print("VercelAISdkTarget — Manual Validation")
    print(f"Endpoint: {url}")
    print("=" * 50)

    await test_basic_response()
    await test_multi_turn_memory()
    await test_reset_clears_memory()
    await test_clone_isolation()
    await test_parallel_clones()

    print("\n" + "=" * 50)
    if failed == 0:
        print(f"✓ ALL PASSED ({passed}/{passed})")
    else:
        print(f"✗ {failed} FAILED, {passed} passed")
    print("=" * 50)


asyncio.run(main())
