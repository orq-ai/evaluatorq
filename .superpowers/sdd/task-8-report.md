# Task 8 report: replay seed context in the dynamic orchestrator

Base SHA: `16aef00b8ca859d46cd5c1389e2bc057adeabfea`.

## RED

Added first-user bootstrap, target-owned last-assistant rejection, and caller-owned prefix replay tests before implementation. The focused replay command failed because `MultiTurnOrchestrator.run_attack()` did not yet accept `seed_messages`.

## GREEN

Implemented optional trace seed context in the dynamic orchestrator. `first_user` performs one target bootstrap call, records the live assistant response as seed context, leaves `max_turns` unchanged, and keeps bootstrap usage separate while including it in aggregate usage. `last_assistant` requires caller-owned history and rejects target-owned history before any attacker call. Every attack target request now includes the seed prefix followed by prior attack turns and the next attack. Seeded attacker prompts explicitly name vulnerability, technique, delivery methods, and a `delimit()`-protected transcript. Task 7 seed metadata is parsed and forwarded by the dynamic pipeline, while unseeded calls retain their original argument path. The fixed-template single-turn path now shares the same replay helper, preserves structured bootstrap tool messages, and carries replay fields through report conversion without counting bootstrap usage twice.

## Verification

Focused replay tests: `44 passed`.

Dynamic job tests: `27 passed`.

Report converter tests: `73 passed`.

Trace-seed tests: `17 passed`.

`uv run ruff check src`: passed.

`uv run ruff format --check src`: passed.

`uv run basedpyright`: passed with 0 errors, 0 warnings, and 0 notes.

## Concerns

The bootstrap exchange uses the shared target-call retry wrapper and returns a run-level error if bootstrap cannot complete; no production trace thread, task, or memory identifier is reused.
