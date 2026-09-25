# Task 9: Live container tests

## Result

Status: DONE_WITH_CONCERNS. The image builds, all bundled CLIs start and print their pinned versions, and the runtime entrypoint resolves the host UID correctly. The live integration suite has two failures caused by provider access/runtime behavior and five skips; no containers were left behind.

## Image

Command: `uv run eq coding-agent build-image`

Result: PASS, exit 0. Image: `evaluatorq-coding-agent:1.41.1.dev13-g887b88743.d20260925`. Resolved CLI versions: Claude Code 2.1.282, Codex CLI 0.157.0, OpenCode 1.18.32, and Orq CLI 11.0.0. The base image is Node 22 slim on linux/arm64.

The first build failed because `node:22-slim` already uses UID 1000, so `useradd -u 1000 agent` returned `useradd: UID 1000 is not unique`. I changed the manual image user to UID 1001. A subsequent check found that running `chmod 0666 /etc/passwd` before `useradd` did not preserve the writable mode. The Dockerfile now creates `agent` first and applies the mode afterward.

CLI startup command: `docker run --rm evaluatorq-coding-agent:1.41.1.dev13-g887b88743.d20260925 sh -c 'claude --version; codex --version; opencode --version; orq --version'`

Output: Claude Code `2.1.282 (Claude Code)`; Codex `codex-cli 0.157.0`; OpenCode `1.18.32`; Orq `orq version 11.0.0` (API client build 4.14.20).

UID mapping command used the current host uid/gid and ran `evq-entrypoint node -e 'console.log(require("os").userInfo().uid)'` in the image. Output was `501`, matching the host UID. Without `evq-entrypoint`, Node's `os.userInfo()` fails because UID 501 has no passwd entry in the base image, as expected before runtime initialization.

## Live tests

Command: `uv run pytest tests/integration/test_coding_agent_container_live.py -m integration -v`

Result: 2 failed, 5 skipped in 143.46s. Exact case outcomes:

- `test_claude_turn[direct]`: SKIPPED, `ANTHROPIC_API_KEY not set`.
- `test_claude_turn[orq]`: FAILED after pytest's 120-second test timeout; isolated rerun with `-o timeout=300` also FAILED after 197.64s. The isolated failure was `CodingAgentError: cli.exit.1: docker exited 1:` with no stderr excerpt.
- `test_idle_limit_is_real[claude]`: SKIPPED, `ANTHROPIC_API_KEY not set`.
- `test_idle_limit_is_real[codex]`: FAILED. Codex returned HTTP 401 Unauthorized while connecting to `wss://api.openai.com/v1/responses`; the test expected an idle timeout. No credential value was printed.
- `test_idle_limit_is_real[opencode]`: SKIPPED, `ANTHROPIC_API_KEY not set`.
- `test_isolation`: SKIPPED, `ANTHROPIC_API_KEY not set`.
- `test_written_file_owned_by_host_uid`: SKIPPED on macOS because ownership is mapped by the local engine.

The suite's autouse teardown found no remaining containers after each test. A final `docker ps -aq --filter 'label=evaluatorq.coding-agent=1'` was empty.

## Changed files

- `tests/integration/test_coding_agent_container_live.py`: added live, marked integration cases from the Task 9 brief.
- `src/evaluatorq/backends/docker/Dockerfile`: changed the image's manual non-root `agent` UID to 1001 because the Node base image reserves 1000, and moved the `/etc/passwd` permission change after user creation so the runtime entrypoint can add a passwd record for the host UID.

## Concerns

The provider-backed cases are not all green in this environment: Orq-backed Claude exited 1 without stderr after a long wait, and the configured OpenAI key was rejected with 401. Direct Claude, OpenCode, and isolation coverage could not run because the Anthropic key is absent. These are reported as failures/skips rather than weakened expectations.
