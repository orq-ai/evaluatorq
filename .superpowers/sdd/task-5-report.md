# Task 5 report: DockerOptions and container argv helpers

## RED

Command: `uv run pytest tests/backends/test_container.py -q`

Result: collection failed with the expected `ImportError: cannot import name 'DockerOptions' from 'evaluatorq.backends'`. This confirmed the new API was not present before implementation.

## GREEN

Command: `uv run pytest tests/backends/test_container.py -q`

Result: `13 passed in 0.07s`.

Command: `uv run pytest tests/backends -q`

Result: `93 passed in 11.31s`.

Command: `uv run ruff check src/evaluatorq/backends/container.py src/evaluatorq/backends/coding_agent.py src/evaluatorq/backends/__init__.py`

Result: `All checks passed!`.

Command: `uv run ruff format --check src/evaluatorq/backends/container.py src/evaluatorq/backends/coding_agent.py src/evaluatorq/backends/__init__.py`

Result: `3 files already formatted`.

## Changed files

- `src/evaluatorq/backends/container.py`: frozen `DockerOptions`, image/version constants, CLI, run and exec argv builders, environment selection, watchdog command, isolation-flag detection, and unsafe mount detection.
- `src/evaluatorq/backends/coding_agent.py`: required `AgentSpec` container defaults, container option retention for `new()`, permission and extra-argument defaults, and construction-time warnings.
- `src/evaluatorq/backends/__init__.py`: public `DockerOptions` export.
- `tests/backends/test_container.py`: the specified tests for options, argv, environment, run-argument checks, and per-agent defaults.

## Self-review

Confirmed the test coverage checks default image tag shape, frozen options, absolute workdir validation, both container argv builders, optional privilege escalation, launcher-specific environment forwarding, reserved `HOME` and `PATH`, unsafe and safe mount sources, agent defaults, clone propagation, and construction-time warnings. Container lifecycle and process execution wiring were not added.

## Concerns

None within Task 5 scope. Container lifecycle and execution wiring remain for Tasks 6–7.
