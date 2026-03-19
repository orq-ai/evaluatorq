# Contributing to evaluatorq

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Access to the orqkit monorepo

### Setup

```bash
cd packages/evaluatorq-py

# Install all dependencies (dev group + all optional extras)
uv sync --all-extras --all-groups

# Verify the setup
uv run pytest -m 'not integration' --co  # list tests without running
uv run basedpyright                       # type check
uv run ruff check src                     # lint
```

## Development Workflow

### Running Tests

```bash
# Unit tests only (fast, no external services)
uv run pytest -m 'not integration'

# Specific test file
uv run pytest tests/redteam/test_vulnerability_first.py -v

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
uv run basedpyright
```

### Using Nx (from monorepo root)

```bash
bunx nx test @orq-ai/evaluatorq-py
bunx nx lint @orq-ai/evaluatorq-py
bunx nx typecheck @orq-ai/evaluatorq-py
bunx nx build @orq-ai/evaluatorq-py
```

## Project Structure

The package has two main areas:

1. **Core evaluation framework** (`src/evaluatorq/`) — the public `evaluate()` API, dataset fetching, scorers, and integrations
2. **Red teaming subpackage** (`src/evaluatorq/redteam/`) — adversarial testing pipeline with vulnerability-first data model

See `CLAUDE.md` for a detailed file tree.

## Code Conventions

### Python Version

Target Python 3.10+. Use `from __future__ import annotations` at the top of files for modern type syntax. The codebase includes a `StrEnum` polyfill for Python 3.10 compatibility.

### Imports

- Use absolute imports (`from evaluatorq.redteam.contracts import ...`)
- Use `TYPE_CHECKING` blocks for imports only needed at type-check time
- Ruff handles import sorting

### Data Models

- All shared data models live in `redteam/contracts.py` (Pydantic BaseModel)
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

Implement the `AgentTarget` protocol from `backends/base.py`:

```python
class AgentTarget(Protocol):
    async def send_prompt(self, prompt: str) -> str: ...
    def reset_conversation(self) -> None: ...
```

Optionally implement `SupportsClone`, `SupportsTokenUsage`, or `SupportsTargetMetadata` for advanced features. Register your backend by creating a `BackendBundle` in `backends/registry.py`.

### New Integration

Add integration modules under `src/evaluatorq/integrations/`. Add the dependency as an optional extra in `pyproject.toml`.

## Pull Requests

- Branch from `main`
- Run `uv run pytest -m 'not integration'` and `uv run basedpyright` before pushing
- Use conventional commit format for commit messages (e.g., `feat(redteam): ...`, `fix(evaluatorq): ...`)
- Keep PRs focused — one feature or fix per PR when possible
