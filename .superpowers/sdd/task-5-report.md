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

## Review fixes

Added focused regression tests before changing the implementation. The RED command `uv run pytest tests/backends/test_container.py -q` reported `4 failed, 12 passed in 0.16s`: Podman accepted a Docker context, `--mount` ignored a plain relative bind source, `--privileged=true` was not reported, and container defaults had overwritten original constructor values in `_kwargs`.

The GREEN command `uv run pytest tests/backends/test_container.py -q` reported `16 passed in 0.10s`.

The backend suite command `uv run pytest tests/backends -q` reported `96 passed in 12.23s`.

The source lint command `uv run ruff check src/evaluatorq/backends/container.py src/evaluatorq/backends/coding_agent.py src/evaluatorq/backends/__init__.py` reported `All checks passed!`.

The formatting command `uv run ruff format --check src/evaluatorq/backends/container.py src/evaluatorq/backends/coding_agent.py src/evaluatorq/backends/__init__.py` reported `3 files already formatted`.

The changes reject Podman `context` with a validation error directing callers to `CONTAINER_CONNECTION`, flag plain relative sources on `--mount` bind mounts while allowing named volumes, recognize `--privileged=true`, and retain caller inputs in `_kwargs` before calculating container defaults so `new()` resolves defaults afresh. The `docker run` test now checks contiguous argv token sequences instead of joining tokens into a string.

Review self-check: these changes stay in the Task 5 option, helper, target-initialization, test, and report paths; no container lifecycle wiring was added. No remaining concerns.
