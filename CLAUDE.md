# CLAUDE.md — evaluatorq-py

This file provides guidance to Claude Code when working in `packages/evaluatorq-py`.

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

# Type check
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

## Package Structure

```
src/evaluatorq/
├── __init__.py              # Public API: evaluate(), DataPoint, EvaluationResult
├── cli.py                   # CLI entry point (evaluatorq / eq commands)
├── evaluatorq.py            # Core evaluation runner
├── evaluators.py            # Built-in evaluator definitions
├── types.py                 # Shared types (ScorerParameter, etc.)
├── deployment.py            # ORQ deployment integration
├── fetch_data.py            # Dataset fetching
├── integrations/            # Third-party integrations (LangChain, etc.)
├── tracing/                 # OpenTelemetry tracing
├── openresponses/           # OpenAI Responses API integration
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
│   ├── redteam_views.py     # HTMX fragment routes for 4 interactive redteam views
│   ├── redteam_charts.py    # Interactive breakdown chart + agent heatmap fragments
│   ├── redteam_transcripts.py # Conversation viewer + disagreement viewer fragments
│   ├── sim_views.py         # HTMX fragment routes: sim row list, transcript viewer, filter plumbing
│   ├── launch.py            # CLI launch helper (uvicorn entry point)
│   └── static/              # Vendored JS: htmx, vega trio, dashboard.js
└── redteam/                 # Red teaming subpackage
    ├── contracts.py         # All data models, enums, Pydantic schemas
    ├── vulnerability_registry.py  # Single source of truth for vulnerabilities
    ├── runner.py            # Unified red_team() entry point
    ├── cli.py               # Typer CLI for red teaming
    ├── hooks.py             # Pipeline lifecycle hooks (DefaultHooks, RichHooks)
    ├── tracing.py           # OTel span helpers
    ├── exceptions.py        # Custom exceptions
    ├── adaptive/            # Dynamic pipeline components
    │   ├── pipeline.py      # Datapoint generation pipeline
    │   ├── orchestrator.py  # Attack execution orchestrator
    │   ├── evaluator.py     # OWASPEvaluator wrapper
    │   ├── strategy_planner.py    # Strategy selection + LLM generation
    │   ├── strategy_registry.py   # Strategy lookup by vulnerability/category
    │   ├── attack_generator.py    # Adversarial prompt generation
    │   ├── objective_generator.py # Attack objective generation
    │   ├── capability_classifier.py # LLM-based agent capability classification
    │   └── agent_context.py # Agent context retrieval
    ├── backends/            # Target backends (ORQ agents, OpenAI models)
    │   ├── base.py          # AgentTarget protocol
    │   ├── orq.py           # ORQ agent backend
    │   ├── openai.py        # Direct OpenAI backend
    │   └── registry.py      # Backend/client factory
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
    │   └── display.py       # Rich terminal display
    └── runtime/             # Job execution
        ├── jobs.py          # Async job runner
        └── orq_agent_job.py # ORQ-specific job implementation
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

- Runtime: `pydantic`, `httpx`, `rich`, `loguru`
- Red team extra: `openai`, `typer`, `python-dotenv`, `huggingface-hub`
- Dashboard extra: `python-fasthtml`, `uvicorn` (install as `evaluatorq[dashboard]`)
- Dev: `pytest`, `pytest-asyncio`, `basedpyright`, `ruff`
- Package manager: `uv` (not pip)
- Build system: `hatchling`

### Environment Variables

- `ORQ_API_KEY` — ORQ platform authentication
- `ORQ_BASE_URL` — ORQ API base URL (optional override; default `https://my.orq.ai`)
- `OPENAI_API_KEY` — for direct OpenAI backend or pipeline LLM calls

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
- **Accidental majors are refused.** A computed `major` bump is skipped unless you re-run via **workflow_dispatch** with `force_level=major`. Use `force_level=minor`/`patch` to override the computed level (e.g. to ship breaking changes as a minor deliberately).
- There is no committed `CHANGELOG.md`; the human-readable changelog is the GitHub Release notes. Release notes are created last and are non-blocking — a notes failure never blocks the PyPI publish.
- **Release-note categories come from PR *labels*, not commit prefixes.** GitHub's auto-notes group merged PRs by label (`feature`/`fix`/`docs`/`chore`/`ci`/`refactor`), mapped to sections in `.github/release.yml`; an unlabeled PR falls into *Other Changes*. You don't label by hand: `.github/workflows/label-pr.yml` reads each PR's conventional-commit title prefix on open/edit and applies the matching label automatically (`feat→feature`, `fix`/`perf`/`revert`→`fix`, `docs→docs`, `chore`/`build`/`test`/`style`→`chore`, `ci→ci`, `refactor→refactor`). A non-conventional title gets no label and lands in *Other*.
- **When you (Claude) open a PR, label it at creation time.** Don't rely solely on the autolabeler — pass `--label` to `gh pr create` using the same mapping, derived from the PR's conventional-commit title prefix: `feat→feature`, `fix`/`perf`/`revert`→`fix`, `docs→docs`, `chore`/`build`/`test`/`style`→`chore`, `ci→ci`, `refactor→refactor`. E.g. a PR titled `fix: …` → `gh pr create --label fix …`. If the title isn't conventional, pick the closest label or leave it unlabeled (it lands in *Other*).
- PyPI publishing uses the **`PYPI_TOKEN`** repo secret (an API token). To switch to OIDC trusted publishing later, configure a **Trusted Publisher** on PyPI for this repo + `release.yml` (PyPI → project → Publishing) and delete the `with: password:` block in the publish step — `id-token: write` is already granted.
