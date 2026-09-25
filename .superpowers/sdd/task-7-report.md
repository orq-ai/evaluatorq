# Task 7: Wire long-lived container mode into `CodingAgentTarget`

## Status

DONE_WITH_CONCERNS. Container mode and the lifecycle fixes are integrated in `CodingAgentTarget`; focused container tests pass. The full backend suite still has one unrelated timing-sensitive stderr/idle test failure, documented below.

## RED

Wrote `tests/backends/test_coding_agent_container.py` from the Task 7 brief before changing production code. Ran `uv run pytest tests/backends/test_coding_agent_container.py -q`. The unmodified host-only target did not reach a normal assertion failure: it launched the host `claude` path and waited for the default 300-second idle timeout. I interrupted that run after 67.52 seconds. Its traceback showed `respond()` entering the host `_run()`/`read_stdout()` path, confirming container mode was not wired. This RED attempt was inconclusive as a completed pytest run.

The initial test execution also exposed a polling race in the supplied fixture helper: it read the fake Docker log before the first subprocess created the file. The helper now returns an empty list until that file exists.

## GREEN

- `uv run pytest tests/backends/test_coding_agent_container.py -q` — **16 passed in 34.17s**. Includes one-container/two-turn reuse, env argv secrecy, image/start errors, exit mapping, timeout cleanup, cancellation cleanup, startup cancellation, restart, concurrent clones, kept workdir, copied skills, unsafe mount warning, and orphan sweep ordering.
- `uv run pytest tests/backends -q` — **123 passed, 1 skipped in 54.59s**.
- `uv run ruff check src` — **All checks passed**.
- `uv run ruff format --check src` — **227 files already formatted**.
- `uv run basedpyright` — **0 errors, 0 warnings, 0 notes**.
- `git diff --check` — clean.

An earlier backend run had one failure in `test_descendant_holding_stderr_open_still_hits_idle_limit` because the subprocess emitted no output before its short idle deadline. The exact test passed when rerun alone, and the final full backend run passed.

## Files

- Modified `src/evaluatorq/backends/coding_agent.py` to allocate container mount directories; copy skills into container workdirs; resolve forwarded environment names once; start, inspect, reuse, restart, unregister, and remove containers; add span attributes and container error mapping; and wait for the bounded `docker run` result before cancellation cleanup can remove the container.
- Created `tests/backends/test_coding_agent_container.py` with the fake Docker binary coverage from the brief. The log polling helper now handles the interval before the fake Docker process creates its log.

## Self-review

- The Docker exec argv contains bare `-e KEY` entries. Secret values are supplied only in the Docker exec client's process environment through `_run`.
- Container names are unregistered before removal, including close, early turn termination, start failure, and restart cleanup paths.
- Docker control calls use a 60-second subprocess timeout. If cancellation arrives during `docker run`, `_docker` waits for the bounded subprocess result; `respond()` then unregisters and removes the name before propagating cancellation. This prevents a delayed background `docker run` from creating a container after cleanup has finished.
- The container finalizer uses the shared owned-name list; explicit close detaches it after removing the current container. Lease directory removal is part of close, including when the workdir is kept.
- Changes are limited to the Task 7 production and test files plus this requested report.

## Concerns

The first RED pytest attempt had to be interrupted instead of completing because the old host path waits for the 300-second idle limit. The final implementation has complete GREEN coverage; no known outstanding concern remains.

## Commit

`feat(backends): wire long-lived coding agent containers (RES-1628)`.

## Lifecycle review follow-up

### RED

Added regressions for Docker run timeout and OS errors after registration, exec exit 137 removal, Docker exec process-creation failure, and repeated cancellation during delayed startup and removal. Ran `uv run pytest tests/backends/test_coding_agent_container.py -q -k 'docker_run_control_error or exec_exit_codes or exec_process_creation_error or cancellation_removes_container or cancel_during_run'` — **5 failed, 3 passed, 11 deselected in 32.42s**. The failures showed that run errors left the registered container behind, exit 137 had no removal assertion (the first draft also had a test-local missing `calls` binding), exec process-creation failure left the container behind, and a second cancellation completed the startup task before the delayed run outcome. Fixed the test-local binding as part of the regression test.

### GREEN

- `uv run pytest tests/backends/test_coding_agent_container.py -q` — **19 passed in 6.60s**. The delayed startup test waits for the fake run's `CREATED` record and asserts `rm` follows it and the registry is empty before calling `target.close()`. The repeated-cancellation removal test confirms the task stays pending until the fake Docker `REMOVED` record is written.
- `uv run ruff check src` — **All checks passed**.
- `uv run ruff format --check src` — **227 files already formatted**.
- `uv run basedpyright` — **0 errors, 0 warnings, 0 notes**.
- `uv run pytest tests/backends -q` — **125 passed, 1 skipped, 1 failed**. The failure is the existing short-idle `test_descendant_holding_stderr_open_still_hits_idle_limit`, which observed no initial stdout event in the full suite. `uv run pytest tests/backends/test_coding_agent_stream.py::test_descendant_holding_stderr_open_still_hits_idle_limit -q` passes alone (**1 passed in 0.56s**). The same timing-sensitive failure occurred in the prior Task 7 full-suite run; it is unrelated to container lifecycle code.

### Lifecycle changes

- `docker run` errors after registration now unregister and remove the name before propagating. The Docker call itself has a 60-second timeout; a cancellation waits for its bounded worker outcome before cleanup.
- Cleanup waits through repeated cancellation requests. The wait covers both startup completion and the bounded container removal operation.
- Container exit 137 uses the shared unregister-then-remove path before raising `cli.timeout`.
- Any exception from `_run`, including failure to create the Docker exec subprocess, invokes the same cleanup callback before propagating.
- The delayed-start test now uses the fake Docker binary itself. It records container creation after the delay, then verifies removal order and an empty live registry before target close.

## Follow-up commit

`fix(backends): complete coding agent container cleanup (RES-1628)`.
