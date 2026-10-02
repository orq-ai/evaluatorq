# Contributing to evaluatorq

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Install all dependencies (dev group + all optional extras)
uv sync --all-extras --all-groups

# Verify the setup
uv run pytest -m 'not integration' --co  # list tests without running
uv run ty check                           # type check the whole repository
uv run ruff check src                     # lint
```

## Development Workflow

### Running Tests

```bash
# Quick local profile (no external services; excludes deliberately slow tests)
uv run pytest

# Complete non-integration profile (run before pushing; matches CI selection)
uv run pytest -m 'not integration'

# Specific test file
uv run pytest tests/redteam/test_vulnerability_first.py -v

# The default marker still applies; opt a targeted slow test back in explicitly
uv run pytest -m 'not integration' path/to/test.py::test_name

# With coverage
uv run pytest -m 'not integration' --cov=src/evaluatorq

# Integration tests (requires ORQ_API_KEY in .env)
uv run pytest -m integration
```

### Linting and Formatting

```bash
# Check for lint issues
uv run ruff check src

# Auto-fix lint issues
uv run ruff check src --fix

# Format code
uv run ruff format src

# Type check
uv run ty check
```

## Project Structure

Runtime modules live under `src/evaluatorq/`. See `CLAUDE.md` for the shared machinery map, and inspect the package directories for the current surface list.

## Code Conventions

Read [CODING_STANDARDS.md](CODING_STANDARDS.md) for the rules used to review code. The conventions below cover Python-specific details.

### Python Version

Target Python 3.10+. Use `from __future__ import annotations` at the top of files for modern type syntax. The codebase includes a `StrEnum` polyfill for Python 3.10 compatibility.

### Imports

- Use absolute imports (`from evaluatorq.redteam.contracts import ...`)
- Use `TYPE_CHECKING` blocks for imports only needed at type-check time
- Ruff handles import sorting

### Data Models

- Cross-package/cross-surface shared data models live in top-level `contracts.py` (Pydantic BaseModel); red-team-specific data models live in `redteam/contracts.py`
- Enums use `StrEnum` for JSON serialization compatibility
- Semantic convention: `passed=True` = RESISTANT, `passed=False` = VULNERABLE

### Error Handling

- Custom exceptions in `redteam/exceptions.py`
- Use `loguru.logger` for logging in the redteam subpackage
- Evaluator failures should return inconclusive results (`passed=None`), not raise

### Testing

- Unit tests go in `tests/unit/`, integration tests in `tests/integration/`, redteam tests in `tests/redteam/`
- Mark integration tests: `@pytest.mark.integration`
- Use `pytest-asyncio` for async test functions
- Default timeout: 120s per test

## Adding Features

### New Vulnerability / Evaluator / Framework

See `docs/custom-evaluators-and-frameworks.md` for a step-by-step guide.

### New Backend (Target)

Implement the `AgentTarget` abstract base class in `evaluatorq.contracts`: `async def respond(messages: list[Message]) -> AgentResponse` and `def new() -> AgentTarget`. `new()` returns an independent target for each concurrent datapoint.

For the full red-team target lifecycle, subclass `Backend` in `evaluatorq.redteam.backends.base`, implement `create_target()` and `cleanup_memory()`, and register its factory with `register_backend()` in `evaluatorq.redteam.backends.registry`.

### New Integration

Add integration modules under `src/evaluatorq/integrations/`. Add the dependency as an optional extra in `pyproject.toml`.

## Pull Requests

- Branch from `main`
- Run the complete `uv run pytest -m 'not integration'` profile and `uv run ty check` before pushing; the bare local pytest command also excludes deliberately slow tests, while CI runs the complete non-integration profile and runs the whole-repository ty check once on Ubuntu with Python 3.10
- Run `uv run python scripts/audit_ty_migration.py --refresh` when changing the exact ty pin so the pinned mapping and gap audit cannot drift
- Use conventional commit format for commit messages (e.g., `feat(redteam): ...`, `fix(evaluatorq): ...`)
- Keep PRs focused — one feature or fix per PR when possible
