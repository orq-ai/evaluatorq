# Task 8 report: coding-agent image and build command

Status: DONE.

## RED

Command: `uv run pytest tests/backends/test_coding_agent_cli.py -q`.

Output: test collection failed as expected with `ModuleNotFoundError: No module named 'evaluatorq.backends.coding_agent_cli'`.

## GREEN

Command: `uv run pytest tests/backends/test_coding_agent_cli.py -q`.

Output: `3 passed in 0.57s`.

Command: `uv run ruff check src/evaluatorq/backends/coding_agent_cli.py src/evaluatorq/cli.py`.

Output: `All checks passed!`.

Command: `uv run ruff format --check src/evaluatorq/backends/coding_agent_cli.py src/evaluatorq/cli.py`.

Output: `2 files already formatted`.

Command: `uv run basedpyright`.

Output: `0 errors, 0 warnings, 0 notes`.

Command: `uv run eq coding-agent build-image --help`.

Output includes `Usage: eq coding-agent build-image [OPTIONS]` and the `--tag`, `--binary`, `--context`, and repeatable `--build-arg` options.

## Package evidence

Checked current npm versions with `npm view <package> version`: `@anthropic-ai/claude-code` 2.1.282, `@openai/codex` 0.157.0, `opencode-ai` 1.18.32, and `@orq-ai/cli` 11.0.0. These exact versions are pinned by the Dockerfile defaults.

Command: `uv build`.

Output: wheel and source distribution built successfully as `dist/evaluatorq-1.41.1.dev13+g887b88743.d20260925-py3-none-any.whl` and `dist/evaluatorq-1.41.1.dev13+g887b88743.d20260925.tar.gz`.

Command: `unzip -l dist/*.whl | grep 'evaluatorq/backends/docker/'`.

Output listed `evaluatorq/backends/docker/Dockerfile` (695 bytes) and `evaluatorq/backends/docker/entrypoint.sh` (341 bytes).

## Files

- Added `src/evaluatorq/backends/docker/Dockerfile` and `entrypoint.sh`.
- Added `src/evaluatorq/backends/coding_agent_cli.py` with the packaged build directory, build arguments, context and binary support, streamed subprocess output, and propagated exit status.
- Registered the `coding-agent` Typer sub-app in `src/evaluatorq/cli.py`.
- Added Docker assets to the wheel include list in `pyproject.toml`.
- Added focused tests in `tests/backends/test_coding_agent_cli.py`.

## Self-review

The runtime entrypoint is not an image `ENTRYPOINT`; it repairs a missing passwd entry when needed and then uses `exec`. The image installs the four pinned CLIs, marks the script executable, and creates a non-root default user. The command uses `cli_prefix` and returns the container builder's exit code. The wheel contains both assets. `dist/` did not exist before this build and was left in place.

Concerns: none.
