# CLAUDE.md — evaluatorq-py

This file provides guidance to Claude Code when working in `packages/evaluatorq-py`.

## Surfacing something important

When you flag something that changes what the user should do — a bug you found, a decision only they can make, a thing you could not finish, a risk in what they just asked for — **explain it like they are eight years old.** Not the tone, the *clarity*: say what broke, what happens because of it, and what you want from them, in words that carry no assumed context. Name the thing before the acronym for it. An explanation the reader has to already understand in order to follow is not a warning, it is a receipt.

This is about the surfacing, not the work. Code, commits, and PR bodies stay normal.

## Parallel sessions

Parallel agent sessions typically run in their own git worktree, so uncommitted changes you did not make may appear in the working tree from concurrent work. **Never run `git stash` (any subcommand) or `git reset`.** Not `stash` to clean the tree, and not `stash pop`/`apply` either: the stash holds other sessions' autostash entries, and popping one drops a merge into your tree and consumes the entry. `git checkout <path>` and `git checkout -- .` are equally destructive to uncommitted work you did not write. When committing, stage only the exact files your task changed.

The same applies to every subagent you dispatch — say it in the dispatch prompt. A reviewer that "just needed a clean tree for a moment" has already popped another session's autostash once.

To read a file as it is on HEAD without touching the tree, use `git show HEAD:<path>`. To see only your own changes on a shared dirty tree, diff the paths you touched: `git diff -- <your paths>`.

Processes are shared too. **Never `pkill -f` a command name** (`mkdocs serve`, `uvicorn`, `pytest`) — every worktree runs the same ones, so the pattern kills a sibling session's server. Kill the port you own: `lsof -nP -iTCP:<port> -sTCP:LISTEN`, then the PID.

## Quick Reference

```bash
# Install dependencies (dev group + all optional extras)
uv sync --all-extras --all-groups

# Run unit tests (excludes integration tests)
uv run pytest -m 'not integration'

# Run a specific test file
uv run pytest tests/redteam/test_vulnerability_first.py -v

# Run integration tests (requires ORQ_API_KEY in .env)
uv run pytest -m integration

# Lint
uv run ruff check src

# Format
uv run ruff format src

# Type check — the whole repo. Never scope it to a path (see "Before pushing").
uv run basedpyright

# Build
uv build

# Serve the docs site locally. Pick a FREE port (81xx) and check it is yours —
# parallel worktrees serve the same path, so the default binds to whichever
# session got there first and you end up reviewing THEIR build. Live-reload does
# not fire in a Conductor worktree: the server keeps showing the build it
# started with, so restart it after an edit.
lsof -nP -iTCP:8125 -sTCP:LISTEN            # empty = free
uv run --group docs mkdocs serve -a 127.0.0.1:8125

# Build the docs site (strict — fails on warnings, as CI does)
uv run --group docs mkdocs build --strict

# Validate mermaid diagrams render in strict renderers (GitHub/VS Code) — runs in CI.
# strict build does NOT catch mermaid label defects; this does.
uv run python scripts/validate_mermaid.py
```

When the user says **“do a test run”**, run the live trace validation for both pipelines using the configured agent key:

```bash
ORQ_API_KEY=... EVALUATORQ_AGENT_KEY=... \
  uv run python scripts/live_trace_validation.py both
```

This runs 3 personas × 3 scenarios for agent simulation and a small hybrid red-team check, then validates the root spans and run metadata. Use `orq traces list` to inspect the resulting traces.

## Opening a PR

**Assign the PR to its author.** `gh pr create` leaves the assignee empty, so open one with `--assignee @me` — an unassigned PR has no one the board can point at when it stalls.

Reviewers need no flag: `.github/CODEOWNERS` requests them on every PR and skips the author. Change that file, not the `gh` invocation, to change who reviews.

## Before pushing to a PR

Run the same checks CI runs, **verbatim**, before every push:

```bash
uv run ruff check src
uv run ruff format --check src
uv run basedpyright                 # whole repo — NOT a path
uv run pytest -m 'not integration'
```

**Do not scope `basedpyright` to a path.** CI runs it bare, which covers `tests/` as well as `src/`. Running `uv run basedpyright src/` passes clean while CI fails on type errors in test files — parametrized args annotated `str` where the signature wants a `Literal`, raw dicts passed where a pydantic model is expected. That exact mistake left PR #119 red across all four Python versions for three commits without any local signal.

Note the asymmetry: **ruff** is scoped to `src` (tests are deliberately not ruff-formatted, so `ruff format --check tests/` reports the whole tree as unformatted — don't "fix" that). **basedpyright** is not scoped. Match CI, not intuition.

CI does not run integration tests. Real-API coverage runs weekly via `.github/workflows/examples-weekly.yml`, which opens an issue on failure rather than blocking a PR.

## Package Map

```
src/evaluatorq/
├── evaluatorq.py, evaluators.py, pairwise*.py  # Core evaluation + pairwise entry points
├── contracts.py, types.py   # Cross-subpackage data models (RunManifest, LLMConfig, …)
├── cli.py                   # CLI entry point (evaluatorq / eq)
├── common/                  # SHARED MACHINERY — read the table below before writing anything here or near it
├── redteam/                 # eq redteam: adaptive/ (pipeline), backends/, frameworks/, reports/
├── simulation/              # eq simulate: runner/, agents/, generators/, reports/
├── dashboard/               # FastHTML dashboard (eq dashboard)
├── openresponses/           # OpenAI Responses API integration
├── tracing/                 # OTel setup + evaluation/run/job spans
└── integrations/            # LangChain, LangGraph, CrewAI, pydantic-ai, openai-agents
```

Read the directory itself for the file list — it is always current, this file is not.

## Need X? Use Y. Do not reinvent.

`common/` is the shared layer. Every module there exists because two surfaces drifted apart and a review consolidated them. Adding a third copy is the failure mode this table exists to prevent.

| Doing | Use | Never |
|---|---|---|
| Any chat completion | `common.llm_call.execute_chat_completion` / `execute_chat_parse` | `client.chat.completions.create(...)` at a new call site |
| Structured / schema output | `common.structured_output` (parse + `json_object` fallback) | hand-rolled `response_format` + `json.loads` |
| Parsing JSON out of model text | `common.extract_json.extract_json_from_response` | bespoke fence-stripping regex |
| Building call params | `LLMCallConfig.request_params(...)` (`contracts.py`) | hand-built `extra_kwargs` dict — it skips the reserved-key guard |
| Resolving an LLM client | `common.llm_client.resolve_llm_client` | `AsyncOpenAI(...)` anywhere but that module |
| Retry / backoff | `common.retry.with_retry`, or the SDK's own `max_retries` — **exactly one of the two** | a second retry layer on a client that already retries (they multiply) |
| Calling the target under test | `common.target_call.call_target_with_retry` | ad-hoc `respond()` + try/except |
| LLM-as-judge | `common.judge.run_judge`; multi-judge via `common.jury` | new judge prompt + parse loop |
| OTel spans, token usage, cost | `common.tracing` (`with_llm_span`, `record_token_usage`, `record_llm_response`) | `get_tracer` / `start_as_current_span` outside a `tracing.py` module |
| Surface-specific span naming | `redteam/tracing.py`, `simulation/tracing.py` — thin wrappers that delegate | new span vocabulary |
| Prompt caching on a replayed conversation | `common.prompt_cache` (`apply_cache_breakpoints` for chat, `mark_responses_input` for Responses), gated on `caching_applies(client, model)` | hand-placed `cache_control`, a `prompt_cache_key` (OpenAI caches automatically; it needs none), or a bare `client_routes_through_orq` gate |
| Prompt templating | `common.template_engine.render_template` | f-string prompt assembly |
| Untrusted text into a prompt | `common.sanitize.delimit` | raw interpolation |
| Run lifecycle state | `common.run_manifest` (`start_manifest`, `list_manifests`) | new status dict / sidecar file |
| Applying recommendations to an agent | `common.apply.apply_recommendations` | surface-local merge logic |
| Console / HTML / MD report output | `common/reports/` (`console`, `render`, `vega`, `palette`) | new renderer or colour set |
| CLI output, errors, JSON, width | `common/cli_*.py` | bespoke `typer.echo` formatting |
| Normalising agent output shapes | `common.output_adapters`, `common.messages` | per-surface `isinstance` ladders |
| Turning message content or a tool result into text | `contracts.content_to_text` / `tool_result_to_text` | `str()` on a `str \| list[ContentPart]` — it renders a Python repr that a judge then scores |
| Building an Orq SDK client | `common.orq_client.resolve_orq_client` | `Orq(...)` anywhere but that module |
| Rendering a transcript as Responses `input` | `openresponses.input_items.messages_to_responses_input` | a hand-built `{'role', 'content'}` list — an assistant turn needs `output_text` parts or the Orq router **silently drops it** |


## House rules

Distilled from review findings that recurred. Each cost a review round.

- **One retry layer.** SDK `max_retries` and `with_retry` compose multiplicatively. Pick one per call path, and say which in the docstring.
- **No optimistic defaults on unknown shapes.** A result whose schema you cannot read must log and count as unknown — never as passed, resisted, or $0. Silence reads as a clean run.
- **A degraded path announces itself.** Falling back, skipping, or returning a literal gets a `logger.warning` naming the cause. Two branches next to each other must not differ in whether they log.
- **Never bypass a wrapper to get at its inner call.** If a helper's signature is in your way, change the helper. Going around it drops its validation — that is how `_RESERVED_COMPLETION_KEYS` got skipped.
- **Caller-supplied values win merges.** `{**defaults, **caller}`, and the docstring states it.
- **No drive-by reformatting.** Quote-style and signature re-wrapping in an unrelated file hides the behaviour change and collides with parallel sessions.
- **Test the failure branch you documented.** If the docstring promises degradation to inconclusive, a test exercises it. All-success fakes prove nothing.
- **A helper with no caller is a bug.** Either call it or delete it; do not recompute its body inline.
- **A call site that reads part of an `LLMCallConfig` says which part.** Sizing your own budget or picking your own endpoint is fine; dropping the caller's `max_tokens` without a word makes a config that did nothing look like a config that worked. Call `common.structured_output.warn_unread_config_fields(config, <fields you read>, caller=...)`, or route the call through `generate_structured` with `config=` set, which warns on your behalf for that call only — it warns about nothing when the config never reaches it. A field an explicit keyword beats was dropped, not read: take it back out of the set you warn against, or the warning claims a value applied that never did.
- **Nothing stateful is shared across concurrent work.** `evaluate()` / `simulate()` / `red_team()` run datapoints concurrently: give each task its own target via `new()` (a shared `ORQAgentTarget` races on `_task_id`), and key per-item assignment on the dataset row, never on an arrival-order cursor. An `asyncio` primitive binds to the loop that first blocks on it — don't reuse one across loops.
- **A new registry copies `vulnerability_registry.py`.** Assert `set(Enum) == set(registry)` at import time and freeze with `MappingProxyType`; a plain mutable dict drifts silently as the enum grows. The same goes for any hand-maintained mirror of another type's fields (`simulation/agents/base.py`'s `_MIRRORED_FIELDS`): assert it against `model_fields` at import time, or the next field added there is dropped from the request in silence.
- **Every filtered UI section renders an empty state.** A section that disappears on zero matches is indistinguishable from a bug.
- **Never ask a judge for a verdict that inverts between types.** `must_happen` and `must_not_happen` mean opposite things by the same `passed` flag, and models get it backwards — gpt-5.4-mini marked a satisfied `must_happen` as unmet while its own `reason` said the opposite. Ask for the one factual thing (*did it occur?*) and map occurrence to pass/fail in code.
- **Provider usage/cost shapes are not interchangeable.** Anthropic reports cache reads top-level where Orq/OpenAI nest them. Build the test fixture from the provider SDK's own models so a schema move fails the test instead of confirming the guess.
- **Only write a cache breakpoint where the next turn will still have that prefix.** A write costs 1.25x and is read back only by a request repeating the marked prefix byte-for-byte, so marking a message the caller rebuilds each turn is a pure loss — the judge's per-turn instruction cost the whole transcript, every turn. `volatile_tail` is a **required** keyword for that reason: say how many trailing messages you rebuild (`0` when the whole list persists). On the Responses path the count is `volatile_items`, **not** messages — one tool-calling `Message` renders to several `input` items — so convert with `responses_volatile_items` and never pass a message count through. Never set `ttl` — the 5m default is right, `1h` costs more and only Anthropic honours it. Both APIs take a **positioned, per-item** marker, so this holds on either; do not use the Responses *top-level* `cache_control` body field, which marks the end of the whole input and so cannot be kept off a rebuilt trailing item (measured: 0 reads).
- **Stable text goes before varying text.** Text stuck behind a placeholder is uncacheable however stable it is, because a breakpoint is per-message and cannot split one. The OWASP judge rubrics are the standing example: ~1500 stable tokens sit around the transcript placeholders and none of them can be marked.
- **Mark a render, never a store.** `apply_cache_breakpoints` / `mark_responses_input` return a copy and never mutate; feed them the freshly-rendered `list[dict]` and let the result die with the request. Assigning the marked copy back onto the transcript you keep appending to is the one way to exceed Anthropic's 4-breakpoint limit — the old markers stay, two more are added each turn, and the API rejects the request several billed turns in. There is no runtime guard for this by design: annotate the transcript with its real type (`list[ChatCompletionMessageParam]`) and basedpyright refuses the assignment.

Guardrails for the mechanical parts live in `tests/test_reuse_guardrails.py`. A failure there names the canonical helper — use it, don't extend the allowlist.

## Keeping this file true

This file only works if it absorbs what review teaches. When a review comment, CI failure, or bug traces back to a convention that was not written down:

1. Add it **in the same PR** — one table row or one house rule, not a paragraph.
2. Add the mechanical check too, if one is possible (`tests/test_reuse_guardrails.py`, a ruff rule).
3. Delete something stale while you are here. Above ~200 lines this file gets skimmed, and skimmed is the same as absent.

Do not add a directory tree, a file inventory, or anything else the filesystem already answers.

## Key Patterns

### Data Model

- **Vulnerability is the atomic primitive** — strategies, evaluators, and datapoints all bind to `Vulnerability` enum values
- Framework categories (ASI01, LLM01) are a derived mapping layer via `VulnerabilityDef.framework_mappings`
- `passed=True` means RESISTANT (attack failed), `passed=False` means VULNERABLE (attack succeeded)

### Target calls and error payloads

**Every target call goes through `call_target_with_retry`** (`common/target_call.py`). Calling `target.respond()` directly skips retry, the per-call timeout, and backend error mapping — and, worse, tends to skip the error payload with it.

**Never hand an `AgentResponseError` object to the report layer.** `JobOutputPayload.error` and `AttackOutput.error` are `str`; the object fails validation and takes down report generation for the **entire run**, after every attack has been executed, judged and billed. Flatten with `TargetCallResult.error_payload()` — it is the single source for the six `error*` fields, so the format can't drift between the static, hybrid, pipeline and orchestrator paths (it did, three ways, before it was consolidated).

The reverse failure is quieter and worse: a job that returns **no** `error` key at all makes `output_error_text` return `None`, so the judge scores the literal `[ERROR: ...]` marker as a genuine agent reply and a dead target comes back RESISTANT. Both static legs now emit the key unconditionally (`None` on success). A new job that calls a target must do the same.

**Two different `error` keys, one word.** The one above is *inside the output payload* of a target or LLM invocation, and it decides what a judge reads. `JobReturn['error']` is at the **top level of a job's return value**, and it decides whether the row counts as failed in `Failed Jobs` and under `check_pass_failures(treat_errors_as_failure=True)`. An invocation lives inside a job, so a job can carry both, one, or neither. The rule above is about the nested one; a job that swallows a failure to keep the batch alive — `simulate()`'s and `wrap_simulation_agent()`'s do — also emits the top-level one, `None` on success. `@job()` nests everything it is handed under `output`, so a decorated job reports failures by raising; that is the contract, not a gap.

### Adding New Features

- New vulnerabilities: see `docs/custom-evaluators-and-frameworks.md`
- New evaluators: create a function returning `LlmEvaluatorEntity`, register in `VULNERABILITY_EVALUATOR_REGISTRY`
- New strategies: create `AttackStrategy` objects, register in `strategy_registry.py`
- New backends: implement the `AgentTarget` ABC from `evaluatorq.contracts` (subclass `Backend` from `backends/base.py` for full target lifecycle)

### Testing Conventions

- Unit tests in `tests/unit/`, integration tests in `tests/integration/`
- Red team tests in `tests/redteam/`
- Mark integration tests with `@pytest.mark.integration`
- Default pytest timeout is 120s (configured in `pyproject.toml`)
- Use `pytest-asyncio` for async tests

Prompt-cache behavior is covered by `tests/unit/test_prompt_cache.py`, `tests/openresponses/test_prompt_cache.py`, `tests/redteam/test_orchestrator_prompt_cache.py`, and `tests/simulation/test_agent_prompt_cache.py`. Run those together with `tests/test_reuse_guardrails.py` when changing cache placement, router/model gating, or transcript rendering: `uv run pytest tests/unit/test_prompt_cache.py tests/openresponses/test_prompt_cache.py tests/redteam/test_orchestrator_prompt_cache.py tests/simulation/test_agent_prompt_cache.py tests/test_reuse_guardrails.py`.

These tests use fakes and do not require API credentials. The repository does not include a live provider probe; live cache measurements require a separately maintained investigation against the configured provider and model.

### Dependencies

- `uv` (not pip), `hatchling` build. Runtime deps and extras are listed in `pyproject.toml` — read it there.
- Only `pydantic`/`httpx`/`rich`/`loguru`/`typer`/`openai` are always installed. Everything else is behind an extra: a module that imports one at module scope must itself only be imported behind that extra (that is why `redteam/ui/`, `dashboard/` and the integrations can import `streamlit`/`fasthtml`/`langchain` at the top). Anywhere else, import inside the function.
- Adding a new dependency needs a reason a few lines of stdlib cannot cover.

### Environment Variables

- `ORQ_API_KEY` — ORQ platform authentication
- `ORQ_BASE_URL` — ORQ API base URL (optional override; default `https://my.orq.ai`)
- `OPENAI_API_KEY` — for direct OpenAI backend or pipeline LLM calls
- `ORQ_WORKSPACE` (or `ORQ_WORKSPACE_SLUG`) — workspace slug for dashboard→Orq trace deep-links; buttons hidden when unset
- `ORQ_UI_BASE_URL` — optional Orq UI base for deep-links (defaults to `ORQ_BASE_URL` or `https://my.orq.ai`)
- `EVALUATORQ_PROPAGATE_TRACE_CONTEXT` — `false`/`0` stops W3C `traceparent` injection on outgoing LLM/target calls (default on)
- `EVALUATORQ_APPLY_MODEL` — model for the dashboard's apply-recommendations merge (default `openai/gpt-5.6-luna`, the shared `DEFAULT_PIPELINE_MODEL`)

### Code Style

- Python 3.10+ compatible (use `from __future__ import annotations` for newer typing syntax)
- `StrEnum` polyfill for Python 3.10 (native in 3.11+)
- Linting: ruff
- Type checking: basedpyright (lenient config — many rules disabled)
- Logging: `loguru` everywhere (core runtime dependency since 1.3)

### Releases

Releases are **tag-driven**. The package version comes from the latest git tag via `hatch-vcs` (`[tool.hatch.version] source = "vcs"`) — there is **no `version` field in `pyproject.toml`** and nothing is committed back to `main` on release. The release workflow (`.github/workflows/release.yml`, on `workflow_run` of **CI** concluding `success` for a push to `main`) only pushes a **tag**, which the `Protect-main` branch ruleset does not govern, so a plain `GITHUB_TOKEN` is enough (no deploy key, no bypass). **You do not bump the version or tag by hand on the normal path** — commit messages drive it.

- **Commits MUST follow [Conventional Commits](https://www.conventionalcommits.org).** `.github/workflows/pr-title.yml` gates the PR title, and — because squash uses `COMMIT_OR_PR_TITLE` — the subject of a single-commit PR too. python-semantic-release (used only as a *version calculator* — `semantic-release version --print`) maps the commit types since the last tag to the next version:
  - `feat:` → minor; `fix:`/`perf:` → patch; `feat!:`/`fix!:`/`BREAKING CHANGE:` → major; `docs:`/`chore:`/`ci:`/`test:`/`refactor:`/`style:`/`build:`/`revert:` → no release.
- The workflow releases the **exact commit CI validated** (`workflow_run.head_sha`), never whatever `main` points at when the job starts. A manual `workflow_dispatch` off any ref other than `main` is refused before checkout — tags are not governed by the branch ruleset, so nothing else would stop a feature branch from being tagged and published.
- On a release-worthy push the workflow: computes the next version, **pushes the tag `vX.Y.Z`**, builds the wheel/sdist (version derived from the tag), **publishes to PyPI via token auth** (the `PYPI_TOKEN` repo secret, passed to `pypa/gh-action-pypi-publish`), then creates a GitHub Release with PR-based auto-notes (`.github/release.yml` controls the categories — merged PRs + contributor attributions).
- **Never write a breaking commit without explicit approval.** Do not use `feat!:`/`fix!:` or a `BREAKING CHANGE:` footer unless the user has explicitly approved a major release for that change. A single one halts **all** releases (see next bullet) until a human forces a bump — in July 2026 three `!` commits froze PyPI on `v1.10.1` for 24 days and 317 commits, and the client bug that surfaced it was already fixed in an unreleased commit. If a change is genuinely breaking, ask first; otherwise land it as `feat:`/`fix:`/`refactor:` and describe the break in the PR body.
- **Accidental majors are refused.** A computed `major` bump is skipped unless you re-run via **workflow_dispatch** with `force_level=major`. Use `force_level=minor`/`patch` to override the computed level (e.g. to ship breaking changes as a minor deliberately). The refusal **fails the run** (`::error::` + `exit 1`) so a blocked release is visible; a genuine no-op (a `docs:`-only push, or the tag already pointing at the commit being released) still exits 0 and stays green. A tag that already exists but points at a **different** commit fails the run with both SHAs rather than skipping quietly. The `!` commits stay in range until someone releases, so the block is permanent, not transient — a red Release run means act, not retry.
- The GitHub Release notes are generated from merged PRs and are created last, non-blocking — a notes failure never blocks the PyPI publish. The committed `CHANGELOG.md` is hand-written and separate from them: a behaviour change to a public default belongs under its `### Notable defaults` section in the same PR, not only in a docstring.
- PyPI publishing uses the **`PYPI_TOKEN`** repo secret (an API token). To switch to OIDC trusted publishing later, configure a **Trusted Publisher** on PyPI for this repo + `release.yml` (PyPI → project → Publishing) and delete the `with: password:` block in the publish step — `id-token: write` is already granted.

### Docs

**Docs ship in the same PR as the feature.** If a change touches public surface (`__all__`, CLI flags or defaults, env vars, enum/registry members) or adds a feature, update the docs before opening the PR:

1. `docs-drift` skill, scoped to the diff — finds claims the change made untrue.
2. `docs-coverage` skill — if the change opens a new usage path (a new mode, backend, surface, entry point), write the prose now rather than deferring it.

The interaction surface those skills reason about — the axes, the tier rules, and the impossible-combination list — is [`.claude/skills/docs-coverage/axes.md`](.claude/skills/docs-coverage/axes.md). Read it when adding anything users choose between; **a new dimension means editing that file in the same PR**, or coverage checking silently stops seeing it.

Rules live in the skills, not here. Both are under `.claude/skills/`.

**Docs examples name a current model.** A sample that says `gpt-4o-mini` dates the page the moment a reader sees it, and the reader copies it. The catalog is the source of truth for what is current: `orq models list --json` returns a `created` timestamp per entry, so `model_developer == 'openai' and model_type == 'chat'`, sorted by `created` descending, gives the latest OpenAI family in one command. As of 2026-08-28 that is **gpt-5.6** — `gpt-5.6-luna` (cheap, and the value of `DEFAULT_PIPELINE_MODEL`), `gpt-5.6-terra` (mid), `gpt-5.6-sol` (premium). Do not write `gpt-4o*`, `gpt-4.1*` or `gpt-5-*` into a new example; prefer `gpt-5.6-luna` unless the sample is specifically about a bigger model. Re-run the query rather than trusting this line — it is a date-stamped fact in a file that does not know today's date.

**A new docs page goes in `mkdocs.yml` twice.** Once under `nav:`, once under `plugins.llmstxt.sections` — the llmstxt plugin silently drops any nav page it does not list from `llms.txt` and `llms-full.txt` without failing the build, so `docs/hooks.py` fails `--strict` on the mismatch instead. The check is one-directional (nav → sections) and accepts globs, which is why `reference/evaluatorq/*.md` covers the generated API pages with one entry.

**Never cut a sentence off.** Every sentence you write — in a docs page, a table cell, a docstring, a commit message, a PR title or body — ends. No trailing `…`, no `[...]`, no clause abandoned mid-thought because the line got long. Truncation belongs to *captured output* (a log excerpt, a receipt's first few lines), never to prose you authored. Too long is a signal to write less, not to chop the tail off. This binds humans, agents, and the docs-autofill routine equally.

**Do not hard-wrap prose you are writing.** One line per paragraph or list item; let the reader's editor wrap. A sentence broken across source lines is the mechanism by which the rule above gets violated by accident, and it makes every later edit a multi-line diff. This covers every prose surface: any `.md` you touch, commit message bodies, PR titles and descriptions, issue bodies, and review comments — GitHub renders a hard-wrapped paragraph as one line anyway, so the breaks buy nothing and survive into every later quote of the text. Every tracked `.md` in this repo is unwrapped, so a hard-wrapped paragraph is a defect to fix in the file you are already editing, not a style you match. Reflowing a file you are **not** otherwise editing is still drive-by reformatting — raise it instead of doing it.

**The single exception is inside code**: source lines and the comments and docstrings attached to them wrap at the file's normal width, because that is what the formatter and the surrounding code do. Prose *about* code follows the no-wrap rule; prose *inside* a `.py` file follows the code.

**Fence every code sample in a docstring, with a language.** An indented block or an RST `Example::` literal reaches Pygments with no lexer and renders as grey text; nothing warns, and `mkdocs build --strict` stays green. Keep the body under `Example:` / `Usage:` **un-indented** — griffe only opens a Google section when the body is indented, and an indented fence inside a `cleandoc`-ed docstring becomes a literal code block instead. In `examples/*.py`, a fence inside the module docstring is embedded in the generated page's own fence, so it must stay 3 backticks — `write_example_pages` widens the outer one to compensate. `docs/hooks.py` fails the build on any unhighlighted block, on **every** page; the two exemptions there are prose diagrams, not an escape hatch.
