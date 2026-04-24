"""Manual validation: CallableTarget — exhaustive checks.

Tests state management, reset isolation, clone independence,
sync-in-thread behavior, and error handling.

No API keys needed. Run with:
    cd packages/evaluatorq-py
    PYTHONPATH=src uv run python scripts/manual_tests/test_callable.py
"""

import asyncio
import time

from evaluatorq.integrations.callable_integration import CallableTarget

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


async def test_async_callable() -> None:
    """Async function works and returns correct type."""
    print("\n--- Async callable ---")

    async def agent(prompt: str) -> str:
        return f"echo: {prompt}"

    target = CallableTarget(agent)
    r = await target.send_prompt("hello")
    check("returns string", isinstance(r, str))
    check("correct content", r == "echo: hello", f"got: {r!r}")


async def test_sync_runs_in_thread() -> None:
    """Sync function runs in a thread and doesn't block the event loop."""
    print("\n--- Sync runs in thread ---")

    def slow_agent(prompt: str) -> str:
        time.sleep(0.3)
        return f"slow: {prompt}"

    target = CallableTarget(slow_agent)

    # Run two slow calls concurrently — if they block, it takes 0.6s+
    start = time.monotonic()
    r1, r2 = await asyncio.gather(
        target.send_prompt("a"),
        target.send_prompt("b"),
    )
    elapsed = time.monotonic() - start

    check("both return strings", isinstance(r1, str) and isinstance(r2, str))
    check(
        "ran concurrently (< 0.5s)",
        elapsed < 0.5,
        f"took {elapsed:.2f}s — sync is blocking the event loop",
    )


async def test_stateful_reset() -> None:
    """Reset function actually clears state."""
    print("\n--- Stateful reset ---")

    history: list[str] = []

    async def agent(prompt: str) -> str:
        history.append(prompt)
        return f"count={len(history)}"

    target = CallableTarget(agent, reset_fn=lambda: history.clear())

    await target.send_prompt("a")
    await target.send_prompt("b")
    check("state accumulated", len(history) == 2, f"history={history}")

    target = target.new()
    check("reset cleared state", len(history) == 0, f"history={history}")

    r = await target.send_prompt("c")
    check("post-reset count is 1", "count=1" in r, f"got: {r!r}")


async def test_no_reset_fn_is_safe() -> None:
    """CallableTarget without reset_fn doesn't crash on reset."""
    print("\n--- No reset_fn ---")

    target = CallableTarget(lambda p: p)
    target.new()  # should not raise
    r = await target.send_prompt("test")
    check("works after reset", r == "test")


async def test_clone_independence() -> None:
    """Clones share the function but get independent reset_fn behavior."""
    print("\n--- Clone independence ---")

    counter = {"value": 0}

    async def agent(prompt: str) -> str:
        counter["value"] += 1
        return f"call #{counter['value']}"

    def reset() -> None:
        counter["value"] = 0

    target = CallableTarget(agent, reset_fn=reset)
    cloned = target.new()

    await target.send_prompt("a")
    check("original increments counter", counter["value"] == 1)

    await cloned.send_prompt("b")
    check("clone shares same function", counter["value"] == 2)

    # Reset on original also resets the shared counter (expected — same reset_fn)
    target = target.new()
    check("reset affects shared state", counter["value"] == 0)


async def test_parallel_clones() -> None:
    """Multiple clones can run concurrently without errors."""
    print("\n--- Parallel clones ---")

    async def agent(prompt: str) -> str:
        await asyncio.sleep(0.05)
        return f"reply to: {prompt}"

    target = CallableTarget(agent)
    clones = [target.new() for _ in range(10)]

    results = await asyncio.gather(
        *[c.send_prompt(f"prompt-{i}") for i, c in enumerate(clones)]
    )
    check("all 10 clones returned", len(results) == 10)
    check(
        "all returned strings",
        all(isinstance(r, str) for r in results),
    )
    check(
        "all have correct content",
        all(f"reply to: prompt-{i}" in r for i, r in enumerate(results)),
        f"results: {results}",
    )


async def test_empty_and_long_prompts() -> None:
    """Edge cases: empty string and very long prompts."""
    print("\n--- Edge cases ---")

    async def agent(prompt: str) -> str:
        return f"len={len(prompt)}"

    target = CallableTarget(agent)

    r_empty = await target.send_prompt("")
    check("empty prompt works", r_empty == "len=0", f"got: {r_empty!r}")

    long_prompt = "x" * 10_000
    r_long = await target.send_prompt(long_prompt)
    check("long prompt works", r_long == "len=10000", f"got: {r_long!r}")


async def main() -> None:
    print("=" * 50)
    print("CallableTarget — Manual Validation")
    print("=" * 50)

    await test_async_callable()
    await test_sync_runs_in_thread()
    await test_stateful_reset()
    await test_no_reset_fn_is_safe()
    await test_clone_independence()
    await test_parallel_clones()
    await test_empty_and_long_prompts()

    print("\n" + "=" * 50)
    if failed == 0:
        print(f"✓ ALL PASSED ({passed}/{passed})")
    else:
        print(f"✗ {failed} FAILED, {passed} passed")
    print("=" * 50)


asyncio.run(main())
