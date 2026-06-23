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
├── dashboard/               # FastHTML web dashboard (eq ui / redteam ui / sim ui)
│   ├── app.py               # build_app(roots) — ASGI app factory + all routes
│   ├── shell.py             # page() — full HTML page shell with head assets
│   ├── view.py              # HTML fragment helpers (index, filter form, downloads)
│   ├── library.py           # File discovery, sniff_kind(), report_id(), scan()
│   ├── surfaces.py          # SurfaceAdapter registry (redteam + sim adapters)
│   ├── filters.py           # FilterDef registry (redteam 7-dim, sim 4-dim)
│   ├── redteam_views.py     # HTMX fragment routes for 4 interactive redteam views
│   ├── sim_views.py         # HTMX fragment route for sim transcript viewer
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
- `ORQ_API_URL` — ORQ API base URL (optional override)
- `EVALUATORQ_OWASP_DATASET_ID` — default dataset ID for static mode
- `OPENAI_API_KEY` — for direct OpenAI backend or pipeline LLM calls

### Code Style

- Python 3.10+ compatible (use `from __future__ import annotations` for newer typing syntax)
- `StrEnum` polyfill for Python 3.10 (native in 3.11+)
- Linting: ruff
- Type checking: basedpyright (lenient config — many rules disabled)
- Logging: `loguru` everywhere (core runtime dependency since 1.3)
