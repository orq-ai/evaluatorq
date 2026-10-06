# Coding standards

Use these rules when writing and reviewing changes to `src/evaluatorq` and `tests`. Each rule was distilled from a review finding that recurred, and each cost a review round. A review finding cites the rule by its bold name and points to the changed code.

Rules under **Hard rules** are violations when broken. Rules under **Judgement calls** need weighing against the change, and a finding against one says so. Anything Ruff, `tests/test_reuse_guardrails.py`, or CI already enforces is out of scope for review. Commands, PR procedure, and the shared machinery map (which `common/` helper owns which job) live in [`CLAUDE.md`](CLAUDE.md).

## Hard rules

### Shared machinery

- **Never bypass a wrapper to get at its inner call.** If a helper's signature is in your way, change the helper. Going around it drops its validation, retry, tracing, and rendering, which is how `_RESERVED_COMPLETION_KEYS` got skipped. The machinery map in `CLAUDE.md` names the owner of each job.
- **A helper with no caller is a bug.** Either call it or delete it; do not recompute its body inline.
- **Caller-supplied values win merges.** `{**defaults, **caller}`, and the docstring states it.
- **A call site that reads part of an `LLMCallConfig` says which part.** Sizing your own budget or picking your own endpoint is fine, but dropping the caller's `max_tokens` without a word makes a config that did nothing look like a config that worked. Call `common.structured_output.warn_unread_config_fields(config, <fields you read>, caller=...)`, or route the call through `generate_structured` with `config=` set, which warns on your behalf for that call only; it warns about nothing when the config never reaches it. A field an explicit keyword beats was dropped, not read: take it back out of the set you warn against, or the warning claims a value applied that never did.
- **A new registry copies `vulnerability_registry.py`.** Assert `set(Enum) == set(registry)` at import time and freeze with `MappingProxyType`, because a plain mutable dict drifts silently as the enum grows. The same goes for any hand-maintained mirror of another type's fields (`simulation/agents/base.py`'s `_MIRRORED_FIELDS`): assert it against `model_fields` at import time, or the next field added there is dropped from the request in silence.
- **Messages go through the shared adapters.** Turn content, tool results, and transcripts into text with `contracts.content_to_text`, `tool_result_to_text`, and `common.messages.messages_to_text`. `str()` on content renders a Python repr that a judge then scores, and joining `content` glues turns together and drops `tool_calls`.

### Results and failure behavior

- **`passed` means resistant.** In red teaming, `passed=True` means RESISTANT and `passed=False` means VULNERABLE, on every path.
- **No optimistic defaults on unknown shapes.** A result whose schema you cannot read must log and count as unknown, never as passed, resisted, or $0. Silence reads as a clean run.
- **Signal inputs keep their source scope.** A tool schema belongs to the response that supplied it; do not apply a later schema to earlier calls. Preserve unsupported tool activity as a coverage failure for dependent signals, and retain missing-data reasons and evidence when saving reports.
- **Saved coverage excludes capture payloads.** Persist Insights source coverage through `TraceDocument.source_coverage`, which selects known provenance and coverage fields. Never copy unrestricted `capture_metadata` into a saved review result; source adapters can carry conversation payloads there.
- **Names are labels, not identities.** Dataset and experiment rows can contain personas or scenarios with the same name but different content. Deduplicate seeds by full model content in first-seen order; test both exact repeats and distinct same-named variants.
- **A degraded path announces itself.** Falling back, skipping, dropping a setting, or returning a literal gets a `logger.warning` naming the cause. Two branches next to each other must not differ in whether they log.
- **Target-call jobs always emit the nested `error` key.** Set it to `None` on success, because a missing key lets the judge score an `[ERROR: ...]` marker as a real reply and a dead target comes back RESISTANT. Flatten target errors with `TargetCallResult.error_payload()`, since an `AgentResponseError` object fails report validation for the whole run. A job that absorbs a failure also reports the top-level `JobReturn['error']`.
- **Never ask a judge for a verdict that inverts between types.** `must_happen` and `must_not_happen` mean opposite things by the same `passed` flag, and models get it backwards: gpt-5.4-mini marked a satisfied `must_happen` as unmet while its own `reason` said the opposite. Ask for the one factual thing (*did it occur?*) and map occurrence to pass/fail in code.

### Provider calls and concurrency

- **One retry layer.** SDK `max_retries` and `with_retry` compose multiplicatively. Pick one per call path, and say which in the docstring.
- **Provider usage and cost shapes are not interchangeable.** Anthropic reports cache reads top-level where Orq and OpenAI nest them. Build the test fixture from the provider SDK's own models so a schema move fails the test instead of confirming the guess.
- **Nothing stateful is shared across concurrent work.** `evaluate()`, `simulate()`, and `red_team()` run datapoints concurrently. Give each task its own target via `new()` (a shared `ORQAgentTarget` races on `_task_id`), and key per-item assignment on the dataset row, never on an arrival-order cursor. An `asyncio` primitive binds to the loop that first blocks on it, so don't reuse one across loops.
- **Only write a cache breakpoint where the next turn will still have that prefix.** A write costs 1.25x and is read back only by a request repeating the marked prefix byte-for-byte, so marking a message the caller rebuilds each turn is a pure loss; the judge's per-turn instruction once cost the whole transcript, every turn. `volatile_tail` is a required keyword for that reason: say how many trailing messages you rebuild (`0` when the whole list persists). On the Responses path the count is `volatile_items`, not messages, because one tool-calling `Message` renders to several `input` items; convert with `responses_volatile_items`. Never set `ttl`: the 5m default is right, and `1h` costs more and only Anthropic honours it. Do not use the Responses top-level `cache_control` body field, which marks the end of the whole input and so cannot be kept off a rebuilt trailing item (measured: 0 reads).
- **Mark a render, never a store.** `apply_cache_breakpoints` and `mark_responses_input` return a copy and never mutate. Feed them the freshly rendered `list[dict]` and let the result die with the request. Assigning the marked copy back onto a transcript you keep appending to exceeds Anthropic's 4-breakpoint limit several billed turns in. Annotate the transcript with its real type (`list[ChatCompletionMessageParam]`) so the type checker rejects the assignment.

### Tests, UI, and docs

- **Test the failure branch you documented.** If the docstring promises degradation to inconclusive, a test exercises it. All-success fakes prove nothing.
- **Every filtered UI section renders an empty state.** A section that disappears on zero matches is indistinguishable from a bug.
- **Docs ship in the same change.** A change to a public entry point, option, default, env var, or registry member updates the docs in the same PR, because docs deferred to a follow-up are docs that drift.
- **No drive-by reformatting.** Quote-style and signature re-wrapping in an unrelated file hides the behaviour change and collides with parallel sessions.

## Judgement calls

- **Shared behaviour lives in `common/`.** Behaviour used by two or more surfaces belongs in `common/`; surface modules keep their own vocabulary and delegate the shared work. Extend an existing helper before adding a sibling, because every module there exists to end a drift between two copies.
- **Preserve the data model.** A vulnerability is the atomic red-team concept and framework categories are derived mappings. Shared contracts belong in `contracts.py`, and surface-specific contracts in their own packages.
- **No speculative abstraction.** Add a seam, parameter, or hook when a second caller needs it, not before.
- **Account for every provider request.** Fallback attempts and failed structured-output attempts cost tokens too, so they reach usage and cost tracking.
- **Stable text goes before varying text.** Text stuck behind a placeholder is uncacheable however stable it is, because a breakpoint is per-message and cannot split one. The OWASP judge rubrics are the standing example: ~1500 stable tokens sit around the transcript placeholders and none of them can be marked.
- **Test the risk that motivated the change.** Test observable contracts, not internals. When concurrency or a provider response shape caused the bug, the test exercises that concurrency or that shape.
- **Pass arguments by keyword in new code.** Call `f(model=model, max_tokens=512)`, not `f(model, 512)`, because a positional call silently binds the wrong value when a signature gains or reorders a parameter. Positional-only parameters (`len(x)`, `str(x)`, `isinstance(x, T)`) are exempt. Existing positional calls are not a finding unless the change touches them.

## Where this repo overrides common smells

A review that also checks for generic code smells (such as Fowler's *Refactoring* list) drops these:

- **Surface tracing wrappers are not Middle Man.** `redteam/tracing.py` and `simulation/tracing.py` delegate to `common.tracing` on purpose: they map surface vocabulary (`orq.redteam.llm_purpose`, `orq.simulation.llm_purpose`) onto neutral keys so `common/` never learns surface names.
- **Heavy use of a `common/` helper is not Feature Envy.** A surface module that calls mostly into `common/` is following the shared-machinery rule, not reaching into another module's data.

## Enforced elsewhere

- `pyproject.toml` defines Python support, Ruff, formatting, and type checking.
- `tests/test_reuse_guardrails.py` enforces shared call sites and other exact reuse rules. A failure there names the canonical helper: use it, don't extend the allowlist.
- `.github/workflows/ci.yml` defines the checks run on pull requests.
