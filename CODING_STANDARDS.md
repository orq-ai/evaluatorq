# Coding standards

Use these rules when writing and reviewing changes to `src/evaluatorq` and `tests`. A review finding names the rule and points to the changed code. Keep fixed, mechanically checkable rules in Ruff, `tests/test_reuse_guardrails.py`, or CI; keep commands and PR procedure in `CLAUDE.md`.

## Architecture and reuse

- Put behavior shared by two or more surfaces in `common/`. Keep surface modules focused on their own vocabulary and delegate the shared work.
- Extend an existing helper when its interface is too narrow. Route calls through the shared helper so its validation, retry, tracing, and rendering still apply. Use the shared machinery map in `CLAUDE.md` to find the owner.
- Preserve the existing data model: a vulnerability is the atomic red-team concept; framework categories are derived mappings. Shared contracts belong in `contracts.py`, with surface-specific contracts in their own packages.
- Keep changes local. Avoid reformatting unrelated files and adding speculative abstractions. A helper must have a caller; otherwise, remove it.

## Results and failure behavior

- Preserve result meaning across every path. In red teaming, `passed=True` means resistant and `passed=False` means vulnerable. When a result cannot be determined, represent it as unknown or inconclusive where the contract allows; an unreadable result must never become success.
- Make degradation visible. A fallback, skipped operation, dropped setting, or unknown provider shape needs a warning that says what happened. Unknown usage is unpriced usage, not zero cost.
- Keep target-call errors separate from job failures. A target-calling job emits the nested output `error` key, including `None` on success; a job that absorbs a failure also reports its top-level failure. Flatten target errors with `TargetCallResult.error_payload()` before report serialization.
- Preserve caller intent. Caller-supplied values win merges. If a call uses only part of an `LLMCallConfig`, warn about the fields it did not use; an explicit call argument takes precedence over a configured value.
- Render messages and tool results through the shared adapters. Preserve turn boundaries, tool calls, and structured data rather than turning them into Python representations or silently dropping content.

## Provider calls and concurrency

- Give each provider call one retry owner: either SDK retries or `with_retry`. Record which one owns retries in the call path's docstring.
- Account for every provider request, including fallback attempts and failed structured-output attempts. Test usage and cost handling against provider SDK models when the provider's response shape matters.
- Give each concurrent datapoint its own stateful target through `new()`. Associate results with dataset rows, never completion order, and do not reuse an `asyncio` primitive across event loops.
- Place cache breakpoints only on a stable prefix of a freshly rendered request. Account for the volatile tail in the units that API uses, and never store the marked render back in a growing transcript.

## Tests and user-facing changes

- Test the failure behavior a docstring promises. A success-only fake does not establish that a fallback, warning, or inconclusive result works.
- Test observable contracts and meaningful edge cases. When concurrency or provider response shape caused the risk, make the test exercise that risk.
- Every filtered UI section needs a useful empty state.
- Update documentation in the same change when a public entry point, option, default, or behavior changes.

## Enforced rules and reference

- `pyproject.toml` defines Python support, Ruff, formatting, and type checking.
- `tests/test_reuse_guardrails.py` enforces shared call sites and other exact reuse rules.
- `.github/workflows/ci.yml` defines the checks run on pull requests.
- `CLAUDE.md` contains the shared machinery map and the detailed provider, cache, reporting, and documentation procedures.
