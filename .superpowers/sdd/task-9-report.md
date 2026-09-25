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

## Follow-up: test quality and Orq diagnosis

The integration teardown now filters by both `evaluatorq.coding-agent=1` and `evaluatorq.host-pid=<pytest process pid>`, so another local worktree's containers cannot fail this process's teardown. The isolation test now checks completed tool calls for the outside path, the host-only environment variable, and an in-workdir readable control file. It requires the outside path command to report `No such file or directory`, the environment command to produce its `HOST_ENV_VALUE=` marker, the positive control to return `WORKDIR-READABLE-CONTROL`, and neither host secret to appear in returned text/results. The steady idle check requires six completed, separate shell calls each containing exactly one `sleep 10`, rejects loops and compound shell commands, and requires elapsed time to exceed the configured 20-second idle limit; it no longer skips when the model fails to produce the required calls.

Orq diagnosis: a bounded 30-second live run used `launcher='orq'`, forwarded environment names from the normal container path, and captured the agent's stdout with the API key redacted. The Orq CLI and `docker exec` started successfully. Claude Code emitted its init event, then repeated these exact API retry events before the 30-second hard cap:

```text
{"type":"system","subtype":"api_retry","attempt":1,"max_retries":10,"retry_delay_ms":599,"error_status":401,"error":"authentication_failed","session_id":"fafe1f36-0392-44a5-a1ca-b67b2ce7f698","uuid":"d09d15ba-c1e0-4313-a439-0b7f065006cf"}
{"type":"system","subtype":"api_retry","attempt":2,"max_retries":10,"retry_delay_ms":1071,"error_status":401,"error":"authentication_failed","session_id":"fafe1f36-0392-44a5-a1ca-b67b2ce7f698","uuid":"4b166cb1-2017-4c15-bdd8-8a74bded13cb"}
{"type":"system","subtype":"api_retry","attempt":3,"max_retries":10,"retry_delay_ms":2192,"error_status":401,"error":"authentication_failed","session_id":"fafe1f36-0392-44a5-a1ca-b67b2ce7f698","uuid":"ed1f6411-9365-4181-bc2f-fa8ecd98d8f6"}
{"type":"system","subtype":"api_retry","attempt":4,"max_retries":10,"retry_delay_ms":4907,"error_status":401,"error":"authentication_failed","session_id":"fafe1f36-0392-44a5-a1ca-b67b2ce7f698","uuid":"7762ee1e-b1e7-4ff0-bfaf-289e0b6bed74"}
{"type":"system","subtype":"api_retry","attempt":5,"max_retries":10,"retry_delay_ms":8835,"error_status":401,"error":"authentication_failed","session_id":"fafe1f36-0392-44a5-a1ca-b67b2ce7f698","uuid":"3e20a119-465c-4466-af8c-71b2ce6e1ce7"}
{"type":"system","subtype":"api_retry","attempt":6,"max_retries":10,"retry_delay_ms":19342,"error_status":401,"error":"authentication_failed","session_id":"fafe1f36-0392-44a5-a1ca-b67b2ce7f698","uuid":"3bf42bd8-6c66-4f1b-91f7-57d7f819ecb4"}
```

The final bounded-run error was `CodingAgentUnavailableError('cli.timeout: turn exceeded 30s (7 events)')`. Docker container logs were empty. The `docker exec` argv contains the bare `-e ORQ_API_KEY` name through `build_exec_argv`; the existing unit test `test_env_names_follow_launcher` confirms Orq forwards `ORQ_API_KEY` when present. These results point to an Orq API authentication/configuration rejection, not a container startup or environment-forwarding defect. The host key's value was not inspected or printed, so whether it is expired, invalid, or tied to a different workspace remains unproven.

Verbatim excerpts from the initial pytest failure output:

```text
FAILED tests/integration/test_coding_agent_container_live.py::test_claude_turn[orq]
E           Failed: Timeout (>120.0s) from pytest-timeout.
```

```text
FAILED tests/integration/test_coding_agent_container_live.py::test_idle_limit_is_real[codex]
E               evaluatorq.backends.coding_agent.CodingAgentError: cli.exit.1: docker exited 1: 2026-09-25T15:40:07.532874Z ERROR codex_api::endpoint::responses_websocket: failed to connect to websocket: HTTP error: 401 Unauthorized, url: wss://api.openai.com/v1/responses
E               2026-09-25T15:40:07.674240Z ERROR codex_api::endpoint::responses_websocket: failed to connect to websocket: HTTP error: 401 Unauthorized, url: wss://api.openai.com/v1/responses
E               2026-09-25T15:40:07.908305Z ERROR codex_api::endpoint::responses_websocket: failed to connect to websocket: HTTP error: 401 Unauthorized, url: wss://api.openai.com/v1/responses
E               2026-09-25T15:40:08.387691Z ERROR codex_api::endpoint::responses_websocket: failed to connect to websocket: HTTP error: 401 Unauthorized, url: wss://api.openai.com/v1/responses
E               2026-09-25T15:40:09.241081Z ERROR codex_api::endpoint::responses_websocket: failed to connect to websocket: HTTP error: 401 Unauthorized, url: wss://api.openai.com/v1/responses
E               2026-09-25T15:40:10.830775Z ERROR codex_api::endpoint::responses_websocket: failed to connect to websocket: HTTP error: 401 Unauthorized, url: wss://api.openai.com/v1/responses
E               2026-09-25T15:40:14.197011Z ERROR codex_api::endpoint::responses_websocket: failed to connect to websocket: HTTP error: 401 Unauthorized, url: wss://api.openai.com/v1/responses
```

Focused checks after the test changes: `uv run pytest tests/backends/test_coding_agent_container.py tests/backends/test_coding_agent_argv.py tests/backends/test_coding_agent_respond.py tests/backends/test_container.py -q` passed (80 passed, 1 skipped in 18.87s). `uv run pytest tests/integration/test_coding_agent_container_live.py -m integration --collect-only -q` collected all 7 live cases. The revised key-dependent isolation and six-call idle assertions could not be exercised live because `ANTHROPIC_API_KEY` is unset and the configured OpenAI key is rejected with HTTP 401.

Final follow-up validation after annotating parametrized arguments with `AgentName` and `Launcher`: `uv run basedpyright` passed with 0 errors, warnings, or notes; the focused unit suite passed (80 passed, 1 skipped in 20.55s); and collection found all 7 integration cases. The live Anthropic-dependent assertions remain unexercised in this environment because the key is unset. No production code changed in this follow-up.
