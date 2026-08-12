# CLAUDE.md — evaluatorq-py

This file provides guidance to Claude Code when working in `packages/evaluatorq-py`.

## Parallel sessions

Parallel agent sessions typically run in their own git worktree, so uncommitted
changes you did not make may appear in the working tree from concurrent work.
**Never `git stash` or `git reset`** to clean the tree — you would destroy another
session's work. When committing, stage only the exact files your task changed.

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

# Serve the docs site locally (live-reload at http://127.0.0.1:8000/evaluatorq/)
uv run --group docs mkdocs serve

# Build the docs site (strict — fails on warnings, as CI does)
uv run --group docs mkdocs build --strict

# Validate mermaid diagrams render in strict renderers (GitHub/VS Code) — runs in CI.
# strict build does NOT catch mermaid label defects; this does.
uv run python scripts/validate_mermaid.py
```

When the user says **“do a test run”**, run the live trace validation for both
pipelines using the configured agent key:

```bash
ORQ_API_KEY=... EVALUATORQ_AGENT_KEY=... \
  uv run python scripts/live_trace_validation.py both
```

This runs 3 personas × 3 scenarios for agent simulation and a small hybrid red-team
check, then validates the root spans and run metadata. Use `orq traces list` to
inspect the resulting traces.

## Opening a PR

**Assign the PR to its author.** `gh pr create` leaves the assignee empty, so
open one with `--assignee @me` — an unassigned PR has no one the board can point
at when it stalls.

Reviewers need no flag: `.github/CODEOWNERS` requests them on every PR and skips
the author. Change that file, not the `gh` invocation, to change who reviews.

## Before pushing to a PR

Run the same checks CI runs, **verbatim**, before every push:

```bash
uv run ruff check src
uv run ruff format --check src
uv run basedpyright                 # whole repo — NOT a path
uv run pytest -m 'not integration'
```

**Do not scope `basedpyright` to a path.** CI runs it bare, which covers `tests/`
as well as `src/`. Running `uv run basedpyright src/` passes clean while CI fails
on type errors in test files — parametrized args annotated `str` where the
signature wants a `Literal`, raw dicts passed where a pydantic model is expected.
That exact mistake left PR #119 red across all four Python versions for three
commits without any local signal.

Note the asymmetry: **ruff** is scoped to `src` (tests are deliberately not
ruff-formatted, so `ruff format --check tests/` reports the whole tree as
unformatted — don't "fix" that). **basedpyright** is not scoped. Match CI, not
intuition.

CI does not run integration tests. Real-API coverage runs weekly via
`.github/workflows/examples-weekly.yml`, which opens an issue on failure rather
than blocking a PR.

## Package Structure

```
src/evaluatorq/
├── __init__.py              # Public API: evaluate(), DataPoint, EvaluationResult
├── cli.py                   # CLI entry point (evaluatorq / eq commands)
├── evaluatorq.py            # Core evaluation runner
├── evaluators.py            # Built-in evaluator definitions
├── types.py                 # Shared types (ScorerParameter, etc.)
├── contracts.py             # Cross-subpackage shared data models (RunManifest, StageRecord, ManifestStatus, ManifestSurface, etc.)
├── deployment.py            # ORQ deployment integration
├── fetch_data.py            # Dataset fetching
├── pairwise.py              # Pairwise comparison evaluation entry point
├── pairwise_run.py          # Pairwise run orchestration
├── pairwise_reports/        # Pairwise HTML report generation
│   ├── export_html.py       # HTML report renderer
│   └── sections.py          # Report section builders
├── common/                  # Cross-surface shared utilities (redteam + simulation)
│   ├── run_manifest.py      # Run-lifecycle manifest behavior (create/update/finalize RunManifest)
│   ├── judge.py             # LLM-as-judge helper shared across surfaces
│   ├── jury.py              # Multi-judge aggregation
│   ├── llm_call.py          # Shared LLM invocation wrapper
│   ├── llm_client.py        # LLM client construction
│   ├── retry.py             # Retry/backoff helpers
│   ├── tracing.py           # OTel span helpers shared across surfaces
│   ├── template_engine.py   # Prompt templating
│   ├── reports/             # Shared report rendering (console, HTML, vega charts)
│   │   ├── console.py       # Rich console report rendering
│   │   ├── executive_summary.py # Executive summary section builder
│   │   ├── render.py        # HTML/MD render orchestration
│   │   └── vega.py          # Vega-Lite chart spec builders
│   └── ui/                  # Shared Streamlit dashboard launch helper
│       └── launch.py        # Streamlit app launch helper
├── integrations/            # Third-party integrations (LangChain, etc.)
├── tracing/                 # OpenTelemetry tracing
├── openresponses/           # OpenAI Responses API integration
├── simulation/              # Agent simulation subpackage (eq simulate) — persona/scenario-driven multi-turn agent testing
│   ├── api.py               # Public simulate() entry point
│   ├── cli.py                # Typer CLI for simulation
│   ├── types.py               # Simulation data models
│   ├── traces.py              # Trace capture/reconstruction
│   ├── wrap_agent.py          # Agent wrapping/instrumentation helper
│   ├── runner/                 # Simulation execution loop
│   │   └── simulation.py       # Persona x scenario run orchestration
│   ├── agents/                  # Simulated user + judge agents
│   │   ├── judge.py             # Judging agent
│   │   └── user_simulator.py    # Simulated user agent
│   ├── generators/                # LLM-driven persona/scenario/datapoint generation
│   ├── reports/                    # Report generation (console, HTML, MD, exec summary)
│   ├── quality/                     # Robustness checks
│   │   └── message_perturbation.py  # Message perturbation testing
│   └── ui/                           # Streamlit dashboard for simulation results
│       └── dashboard.py
├── dashboard/               # FastHTML web dashboard (eq dashboard — preview, in dev; ui commands still serve the Streamlit dashboards)
│   ├── app.py               # build_app(roots) — ASGI app factory + all routes
│   ├── _compat.py           # Starlette 1.3.x / FastHTML 0.12.x compat shim (applied on import)
│   ├── shell.py             # page() — full HTML page shell with head assets
│   ├── view.py              # HTML fragment helpers (index, filter form, downloads)
│   ├── library.py           # File discovery, sniff_kind(), report_id(), scan(), read_json_cached()
│   ├── surfaces.py          # SurfaceAdapter registry (redteam + sim adapters)
│   ├── filters.py           # FilterDef registry (redteam 7-dim, sim 4-dim)
│   ├── filter_request.py    # parse_selections() — query-string filter parser
│   ├── styles.py            # Shared CSS constants / class-name helpers
│   ├── theme.py             # Theme tokens / light-dark styling
│   ├── metrics.py           # Aggregate metric computation for dashboard views
│   ├── report_kit.py        # Shared report-card component helpers
│   ├── report_tabs.py       # Report tab navigation
│   ├── orq_links.py         # Orq UI deep-link builders
│   ├── orq_workspace.py     # Orq workspace slug resolution
│   ├── trace_links.py       # Trace deep-link builders
│   ├── sim_compare.py       # Simulation run comparison view
│   ├── redteam_views.py     # HTMX fragment routes for 4 interactive redteam views
│   ├── redteam_charts.py    # Interactive breakdown chart + agent heatmap fragments
│   ├── redteam_transcripts.py # Conversation viewer + disagreement viewer fragments
│   ├── sim_views.py         # HTMX fragment routes: sim row list, transcript viewer, filter plumbing
│   ├── launch.py            # CLI launch helper (uvicorn entry point)
│   └── static/              # Vendored JS: htmx, vega trio, dashboard.js
└── redteam/                 # Red teaming subpackage
    ├── contracts.py         # Red-team-specific data models, enums, Pydantic schemas (shared cross-subpackage models live in top-level contracts.py)
    ├── vulnerability_registry.py  # Single source of truth for vulnerabilities
    ├── delivery_method_registry.py # Canonical + custom delivery methods (register/resolve/is_known)
    ├── runner.py            # Unified red_team() entry point
    ├── cli.py               # Typer CLI for red teaming
    ├── hooks.py             # Pipeline lifecycle hooks (DefaultHooks, RichHooks)
    ├── tracing.py           # OTel span helpers
    ├── exceptions.py        # Custom exceptions
    ├── judge.py             # LLM judge invocation for red-team evaluators
    ├── replay.py            # Replay a prior red-team run from stored artifacts
    ├── utils.py             # Misc red-team helpers
    ├── adaptive/            # Dynamic pipeline components
    │   ├── pipeline.py      # Datapoint generation pipeline
    │   ├── orchestrator.py  # Attack execution orchestrator
    │   ├── evaluator.py     # OWASPEvaluator wrapper
    │   ├── strategy_planner.py    # Strategy selection + LLM generation
    │   ├── strategy_registry.py   # Strategy lookup by vulnerability/category
    │   ├── attack_generator.py    # Adversarial prompt generation
    │   ├── objective_generator.py # Attack objective generation
    │   ├── capability_classifier.py # LLM-based agent capability classification
    │   ├── agent_context.py # Agent context retrieval
    │   └── tool_chaining.py # Multi-tool attack chaining strategy support
    ├── backends/            # Target backends (ORQ agents, OpenAI models)
    │   ├── base.py          # AgentTarget protocol
    │   ├── orq.py           # ORQ agent backend
    │   ├── openai.py        # Direct OpenAI backend
    │   ├── openresponses.py # OpenAI Responses API backend
    │   ├── registry.py      # Backend/client factory
    │   └── _errors.py       # Backend error normalization
    ├── frameworks/          # Framework-specific strategies and evaluators
    │   ├── owasp_asi.py     # OWASP ASI attack strategies
    │   ├── owasp_llm.py     # OWASP LLM Top 10 attack strategies
    │   └── owasp/           # OWASP evaluators
    │       ├── evaluators.py       # Evaluator registry
    │       ├── agent_evaluators.py # ASI evaluator prompts
    │       ├── llm_evaluators.py   # LLM Top 10 evaluator prompts
    │       ├── models.py           # LlmEvaluatorEntity, etc.
    │       └── evaluatorq_bridge.py # Static dataset loading + scoring
    ├── reports/             # Report generation
    │   ├── converters.py    # Result → report conversion
    │   ├── display.py       # Rich terminal display
    │   ├── executive_summary.py # Executive summary section builder
    │   ├── export_html.py   # HTML report export
    │   ├── export_md.py     # Markdown report export
    │   ├── guidance.py      # Remediation guidance text
    │   ├── recommendations.py # Recommendation generation
    │   └── sections.py      # Report section builders
    ├── runtime/             # Job execution
    │   └── jobs.py          # Async job runner
    └── ui/                  # Streamlit dashboard for red-team results
        └── dashboard.py
```

## Key Patterns

### Data Model

- **Vulnerability is the atomic primitive** — strategies, evaluators, and datapoints all bind to `Vulnerability` enum values
- Framework categories (ASI01, LLM01) are a derived mapping layer via `VulnerabilityDef.framework_mappings`
- `passed=True` means RESISTANT (attack failed), `passed=False` means VULNERABLE (attack succeeded)

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

### Dependencies

- Runtime (required): `pydantic`, `httpx`, `rich`, `loguru`, `typer`, `openai`
- Red team extra: `huggingface-hub`, `streamlit`, `plotly`, `watchdog`, `vl-convert-python` (install as `evaluatorq[redteam]`)
- Simulation extra: `orq-ai-sdk`, `streamlit`, `plotly`, `watchdog`, `vl-convert-python` (install as `evaluatorq[simulation]`)
- Dashboard extra: `python-fasthtml`, `uvicorn`, `vl-convert-python` (install as `evaluatorq[dashboard]`)
- Other extras: `orq` (`orq-ai-sdk`), `otel` (OpenTelemetry SDK/exporter), `langchain`, `langgraph`, `openai-agents`, `pydantic-ai`, `crewai`; `all` installs every extra
- Dev: `pytest`, `pytest-asyncio`, `basedpyright`, `ruff`
- Package manager: `uv` (not pip)
- Build system: `hatchling`

### Environment Variables

- `ORQ_API_KEY` — ORQ platform authentication
- `ORQ_BASE_URL` — ORQ API base URL (optional override; default `https://my.orq.ai`)
- `OPENAI_API_KEY` — for direct OpenAI backend or pipeline LLM calls
- `ORQ_WORKSPACE` (or `ORQ_WORKSPACE_SLUG`) — workspace slug for dashboard→Orq trace deep-links; buttons hidden when unset
- `ORQ_UI_BASE_URL` — optional Orq UI base for deep-links (defaults to `ORQ_BASE_URL` or `https://my.orq.ai`)
- Simulation recommendation limits (all optional, defaults preserve prior behaviour):
  - `EVALUATORQ_RECOMMENDATION_MAX_SUGGESTIONS`: max suggestions per result (default 3)
  - `EVALUATORQ_RECOMMENDATION_MAX_TRANSCRIPT_CHARS`: transcript char budget in the prompt (default 3000)
  - `EVALUATORQ_RECOMMENDATION_FACTUAL_ACCURACY_BELOW`: trigger threshold (default 0.5)
  - `EVALUATORQ_RECOMMENDATION_HALLUCINATION_RISK_ABOVE`: trigger threshold (default 0.5)
  - `EVALUATORQ_RECOMMENDATION_TONE_APPROPRIATENESS_BELOW`: trigger threshold (default 0.5)

### Code Style

- Python 3.10+ compatible (use `from __future__ import annotations` for newer typing syntax)
- `StrEnum` polyfill for Python 3.10 (native in 3.11+)
- Linting: ruff
- Type checking: basedpyright (lenient config — many rules disabled)
- Logging: `loguru` everywhere (core runtime dependency since 1.3)

### Releases

Releases are **tag-driven**. The package version comes from the latest git tag via
`hatch-vcs` (`[tool.hatch.version] source = "vcs"`) — there is **no `version` field
in `pyproject.toml`** and nothing is committed back to `main` on release. The
release workflow (`.github/workflows/release.yml`, on push to `main`) only pushes a
**tag**, which the `Protect-main` branch ruleset does not govern, so a plain
`GITHUB_TOKEN` is enough (no deploy key, no bypass). **You do not bump the version
or tag by hand on the normal path** — commit messages drive it.

- **Commits MUST follow [Conventional Commits](https://www.conventionalcommits.org).** python-semantic-release (used only as a *version calculator* — `semantic-release version --print`) maps the commit types since the last tag to the next version:
  - `feat:` → minor; `fix:`/`perf:` → patch; `feat!:`/`fix!:`/`BREAKING CHANGE:` → major; `docs:`/`chore:`/`ci:`/`test:`/`refactor:`/`style:`/`build:`/`revert:` → no release.
- On a release-worthy push the workflow: computes the next version, **pushes the tag `vX.Y.Z`**, builds the wheel/sdist (version derived from the tag), **publishes to PyPI via token auth** (the `PYPI_TOKEN` repo secret, passed to `pypa/gh-action-pypi-publish`), then creates a GitHub Release with PR-based auto-notes (`.github/release.yml` controls the categories — merged PRs + contributor attributions).
- **Never write a breaking commit without explicit approval.** Do not use `feat!:`/`fix!:` or a `BREAKING CHANGE:` footer unless the user has explicitly approved a major release for that change. A single one halts **all** releases (see next bullet) until a human forces a bump — in July 2026 three `!` commits froze PyPI on `v1.10.1` for 24 days and 317 commits, and the client bug that surfaced it was already fixed in an unreleased commit. If a change is genuinely breaking, ask first; otherwise land it as `feat:`/`fix:`/`refactor:` and describe the break in the PR body.
- **Accidental majors are refused.** A computed `major` bump is skipped unless you re-run via **workflow_dispatch** with `force_level=major`. Use `force_level=minor`/`patch` to override the computed level (e.g. to ship breaking changes as a minor deliberately). The refusal **fails the run** (`::error::` + `exit 1`) so a blocked release is visible; a genuine no-op (e.g. a `docs:`-only push, nothing to release) still exits 0 and stays green. The `!` commits stay in range until someone releases, so the block is permanent, not transient — a red Release run means act, not retry.
- There is no committed `CHANGELOG.md`; the human-readable changelog is the GitHub Release notes. Release notes are created last and are non-blocking — a notes failure never blocks the PyPI publish.
- PyPI publishing uses the **`PYPI_TOKEN`** repo secret (an API token). To switch to OIDC trusted publishing later, configure a **Trusted Publisher** on PyPI for this repo + `release.yml` (PyPI → project → Publishing) and delete the `with: password:` block in the publish step — `id-token: write` is already granted.

### Docs

**Docs ship in the same PR as the feature.** If a change touches public surface
(`__all__`, CLI flags or defaults, env vars, enum/registry members) or adds a
feature, update the docs before opening the PR:

1. `docs-drift` skill, scoped to the diff — finds claims the change made untrue.
2. `docs-coverage` skill — if the change opens a new usage path (a new mode,
   backend, surface, entry point), write the prose now rather than deferring it.

The interaction surface those skills reason about — the axes, the tier rules, and
the impossible-combination list — is
[`.claude/skills/docs-coverage/axes.md`](.claude/skills/docs-coverage/axes.md).
Read it when adding anything users choose between; **a new dimension means editing
that file in the same PR**, or coverage checking silently stops seeing it.

Rules live in the skills, not here. Both are under `.claude/skills/`.
