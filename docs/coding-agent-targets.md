# Coding agents as targets

`CodingAgentTarget` runs Claude Code, Codex CLI or OpenCode as the system under test, either on the host or inside a container. Each turn starts an agent process and replays the conversation; container mode reuses one container per target clone, so files and agent configuration persist between turns. `red_team()` and `simulate()` treat it like any other target.

Use this guide when you want evaluatorq to run a local coding-agent CLI as a target; use the [targets guide](guides/targets.md) to choose among target kinds or configure other backends.

Three agents are supported: `claude` (Claude Code), `codex` (Codex CLI) and `opencode` (OpenCode).

## Two ways to launch

| `launcher` | What runs | Use it when |
|---|---|---|
| `'direct'` (default) | The agent binary, with your environment and your own provider credentials | You are testing the agent as your users run it |
| `'orq'` | `orq launch <agent>`, so every model call routes through the Orq gateway with workspace skills and MCP server attached by default | You are testing the Orq launcher, skills or gateway themselves |

Under `launcher='orq'`, `model` becomes `orq launch --model provider/id` and `OrqLaunchOptions` tunes the remaining flags. Profile and workspace are not launch flags: set `ORQ_PROFILE` or `ORQ_API_KEY` through `env`.

`OrqLaunchOptions` has four fields: `mcp=False` renders `--no-mcp`, `skills=False` renders `--no-skills`, `base_url` renders `--base-url <url>`, and `fetch_models=False` renders `--no-fetch-models`.

## Running in a container

Container mode keeps the agent in a Docker-compatible container for the lifetime of one target clone. Use it when the agent has shell access and should not see the host filesystem or host environment; `launcher` still chooses direct provider access or `orq launch` inside that container.

Build the version-matched image once with `eq coding-agent build-image`. The default tag is `evaluatorq-coding-agent:<evaluatorq version>`, which is also the default `DockerOptions.image`. `CodingAgentTarget` does not pull or build automatically when the image is missing; run `eq coding-agent build-image` yourself. After upgrading evaluatorq, rebuild because the expected image tag changes. When developing from a source checkout, build with the same Dockerfile directory that the installed wheel packages:

```bash
image_tag=$(uv run python -c 'from evaluatorq.backends.container import DEFAULT_CODING_AGENT_IMAGE; print(DEFAULT_CODING_AGENT_IMAGE)')
docker build -t "$image_tag" src/evaluatorq/backends/docker
```

The CLI form uses the correct version tag automatically:

```bash
eq coding-agent build-image
```

Pass `container=DockerOptions()` to enable the default container setup. `DockerOptions` is frozen and accepts these fields:

| Field | Default | Purpose |
|---|---|---|
| `image` | `evaluatorq-coding-agent:<version>` | Local image to run; the runtime treats it as opaque and does not pull it. |
| `binary` | `docker` | Container CLI executable; use `podman` for Podman. |
| `context` | `None` | Docker context, such as `orbstack`; the CLI adds `--context <name>` to each Docker command. |
| `workdir` | `/work` | Absolute container path where the private host workdir is mounted. |
| `name_prefix` | `evq` | Prefix for unique container names; include a task identifier when you need to correlate containers. |
| `run_args` | `()` | Extra arguments appended verbatim to `docker run`, for resource and network controls. |
| `pass_env` | `None` | Explicit environment variable names to forward; with `None`, only launcher credentials are selected. |
| `allow_privilege_escalation` | `False` | When true, omit `--security-opt no-new-privileges`. |

The image needs the selected agent binary on `PATH`, plus `orq` for `launcher='orq'`, POSIX `sh`, `cat` and `sleep`, and `evq-entrypoint` on `PATH`. A Debian-based task image without Node needs Node installed before its agent package. This Dockerfile layer installs Node and an agent, and copies the runtime entrypoint from the default image built above:

```dockerfile
ARG EVQ_IMAGE=evaluatorq-coding-agent:replace-me
FROM ${EVQ_IMAGE} AS evaluatorq_runtime

FROM node:22-slim AS agent_layer
RUN npm install -g @anthropic-ai/claude-code

FROM your-task-image:latest
COPY --from=agent_layer /usr/local /usr/local
COPY --from=evaluatorq_runtime /usr/local/bin/evq-entrypoint /usr/local/bin/evq-entrypoint
ENV PATH="/usr/local/bin:${PATH}"
USER root
RUN chmod 0666 /etc/passwd
```

This layer provides Node 22, which current Claude Code packages require, along with the selected agent binary and the packaged entrypoint. Use a Debian-compatible task image so it can run the copied Node runtime. `USER root` and the writable `/etc/passwd` are required because `evq-entrypoint` adds an entry for the host uid when the task image does not already have one. Pass the matching default image tag as the `EVQ_IMAGE` build argument so Docker can use that image as the `evaluatorq_runtime` stage and copy its packaged entrypoint. If you saved the fragment as `Dockerfile` beside your task files and replaced `your-task-image:latest` with the task image, build it with:

```bash
image_tag=$(uv run python -c 'from evaluatorq.backends.container import DEFAULT_CODING_AGENT_IMAGE; print(DEFAULT_CODING_AGENT_IMAGE)')
docker build --build-arg "EVQ_IMAGE=$image_tag" -t my-coding-agent:latest .
```

Install `@openai/codex`, `opencode-ai` or `@orq-ai/cli` as needed for the selected agent and launcher. The entrypoint adds the runtime uid to `/etc/passwd` when needed, which prevents Node agents from failing when the host uid has no passwd entry in the image.

Container mode defaults each agent to its bypass mode: Claude Code uses `bypassPermissions`, Codex uses `danger-full-access`, and OpenCode uses `--auto`. A caller-provided `permission_mode` wins. Privilege escalation remains blocked by `no-new-privileges` unless `allow_privilege_escalation=True`; this controls Linux privilege escalation and is separate from the agent's own permission mode.

The coding-agent idle timeout is 300 seconds by default and resets whenever the process writes output. A 2 hour hard cap ends any turn that runs longer, even if it keeps writing. A single tool call does not stream its output while it runs, so set `timeout_ms` above the longest individual tool call you expect. For example, a 10 minute build needs a timeout greater than 600000 ms. The runner's `target_agent_timeout_ms` limit does not apply to `CodingAgentTarget` in either host or container mode.

The launcher forwards only its needed credentials by default: `ORQ_API_KEY` and `ORQ_BASE_URL` for `launcher='orq'`; for direct launch it forwards `ANTHROPIC_API_KEY` to Claude Code, `OPENAI_API_KEY` to Codex, and both to OpenCode. `pass_env=('NAME', ...)` replaces that selection. Values supplied through the target's `env` are also forwarded, with caller values winning; `PATH` and `HOME` are never forwarded from the host. `CONTAINER_CONNECTION` is inherited by the Podman client, so leave `context=None` with `binary='podman'` and select the connection using that variable.

One container belongs to each target clone. Its workdir and container home are host mounts, so files, agent configuration, trust state and caches survive a container recreation. Background processes do not survive; if a container disappears between turns evaluatorq recreates it, logs a warning and resumes with the files that remain. `close()`, cancellation and early turn termination remove the container. If the Python host process crashes, a heartbeat lease ends its container within five minutes. `keep_workdir=True` preserves the workdir and mounted home after close for an external scorer, while removing the container and lease file.

`run_args` is an escape hatch for Docker flags such as memory or CPU limits. Isolation-breaking options such as `--privileged`, host networking, `--volumes-from`, or mounts outside the private workdir trigger warnings and can defeat the boundary. Flags that replace the managed entrypoint or user, set the container name or reserved lifecycle labels, disable automatic removal, or mount over `/evq-home` or `/evq-lease` are rejected because they can prevent the five-minute crash lease from cleaning up safely. `--security-opt no-new-privileges=false` is rejected unless `allow_privilege_escalation=True`. Raw `-e`, `--env` and `--env-file` flags trigger a warning because they bypass `pass_env` filtering. Use `allow_privilege_escalation=True` only when the agent needs Linux capabilities that `no-new-privileges` blocks. Add packages to a custom image when they need to persist or be available before the agent starts; this is the preferred route. Alternatively, use an image with `sudo` and set `DockerOptions(allow_privilege_escalation=True)` so the agent can install packages at runtime. This removes `no-new-privileges`; because `/etc/passwd` is writable in the default setup, the agent can become root inside the container. Use this route only when that access is acceptable. Keep the existing host bind mount option for exposing specific host files when needed; mounts outside the private workdir can weaken isolation. Device flags expose host devices when the agent needs them.

On macOS, OrbStack works through Docker's context support. Select it with `DockerOptions(context='orbstack')` as described in the [OrbStack Docker documentation](https://docs.orbstack.dev/docker/); commercial use requires a paid OrbStack Pro licence ([pricing](https://orbstack.dev/pricing)). A remote Docker context can start the container, but host workdir bind mounts resolve on the remote machine and usually cannot see the local temporary directory. Embedded applications should call `await target.close()` when finished; SIGTERM cleanup is also registered, while SIGKILL cannot be handled and relies on the five-minute lease.

```python
from evaluatorq.backends import CodingAgentTarget, DockerOptions

target = CodingAgentTarget(
    'claude',
    launcher='orq',
    model='anthropic/claude-sonnet-5',
    container=DockerOptions(),
    timeout_ms=900_000,
)
```

## Red-teaming Claude Code with one injected skill

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluatorq.backends import CodingAgentTarget
from evaluatorq.redteam import red_team

async def main() -> None:
    with TemporaryDirectory(prefix='coding-agent-example-') as directory:
        root = Path(directory)
        workdir = root / 'sample-repo'
        workdir.mkdir()
        (workdir / 'README.md').write_text('A small sample repository.\n')
        skill = root / 'grill-me'
        skill.mkdir()
        (skill / 'SKILL.md').write_text(
            '---\nname: grill-me\ndescription: Ask for missing requirements before editing.\n---\n'
            'Before changing this repository, ask one concise question about any unclear requirement.\n'
        )
        target = CodingAgentTarget(
            'claude',
            model='claude-sonnet-5',
            permission_mode='acceptEdits',
            skills=[skill],
            workdir=workdir,
            system_prompt='You only work inside this repository.',
        )
        report = await red_team(target=target, max_concurrency=2)
        print(report.summary.resistance_rate)


asyncio.run(main())
```

The example creates a temporary repository and a one-file `grill-me` skill before the run, so it has no project-specific path prerequisites. Each parallel job gets its own repository copy with that skill added under `.claude/skills/`. The copies are deleted when the run ends; pass `keep_workdir=True` to keep them and log their paths, which is the fastest way to see what the agent actually changed.

## Through the Orq gateway

```python
from evaluatorq.backends import CodingAgentTarget, OrqLaunchOptions

target = CodingAgentTarget(
    'claude',
    launcher='orq',
    model='anthropic/claude-sonnet-5',
    orq=OrqLaunchOptions(mcp=True, skills=True),
    permission_mode='acceptEdits',
)
```

## Privilege

In host mode, each agent starts with its own default permission behaviour, and this target does not change it. In container mode, evaluatorq selects each agent's bypass mode unless you pass `permission_mode`; container Linux privilege escalation remains blocked separately by default. Two knobs use the agent's own vocabulary:

- `permission_mode`: claude `--permission-mode` (`default`, `acceptEdits`, `plan`, `bypassPermissions`, `dontAsk`); codex `--sandbox` (`read-only`, `workspace-write`, `danger-full-access`). OpenCode has no such flag, so a value raises `ValueError`.
- `extra_args`: appended to the agent's argv untouched. OpenCode blocks on its first permission prompt when run headless; pass `extra_args=['--auto']` to auto-approve, knowing that grants everything.

A tool call the harness refused still appears in the response as a tool call whose result is `[denied by claude]`, so the judge sees the attempt.

The private copy preserves symlinks from `workdir` as symlinks, so a link that pointed outside the original still points there and the agent can read or write through it. Point `workdir` at a tree whose links you are willing to expose.

Under `launcher='orq'`, orq launch adds provider and MCP configuration only; it passes no sandbox or permission flag of its own. In host mode, Codex therefore runs with `permission_mode` when given, otherwise with the sandbox configured in the user's own codex config. In container mode, its default is `danger-full-access` unless you set `permission_mode`.

## What the response carries

Tool calls in order, then the agent's final message as text. `response_id` is the agent's session id. Token usage comes from the agent's own usage events, summed across OpenCode's per-step reports with `calls` counting the steps, and is `None` with a warning when the agent reported none. Claude's reported cost lands on the target span as `evaluatorq.coding_agent.cost_usd`.

## Errors

Failures surface as `cli.*` error codes on the result, in this order of precedence:

| Code | Meaning | Retried by the runner |
|---|---|---|
| `cli.not_found` | The binary is not on `PATH` | No |
| `cli.timeout` | The 300 s default idle limit or 2 h hard cap is reached; the process group is killed and a container is removed. `timeout_ms` controls the idle limit, which must exceed the longest single tool call. The runner timeout does not apply. | No |
| `cli.image_missing` | The configured container image is not available locally; build it with `eq coding-agent build-image` or `docker build` | No |
| `cli.container_start` | The container CLI cannot start the container, for example because the daemon, context or `run_args` is invalid | No |
| `cli.agent_not_found` | The selected agent or `evq-entrypoint` is missing from the image `PATH` | No |
| `cli.prompt_too_long` | The OS refused the argv; under `launcher='orq'` codex and opencode take the rendered transcript as one argument, so a long conversation can exceed the limit | No |
| `cli.exit.<code>` | Non-zero exit, even if a result was printed | Yes |
| `cli.parse_error` | Stdout contained no JSON events | Yes |
| `cli.agent_error` | Exit 0 but the agent reported failure (`is_error`, `turn.failed`) | Yes |
| `cli.no_result` | Exit 0 with no final assistant message | Yes |

## Cost and concurrency

A tool-using turn is a full coding-agent session: minutes of wall clock and, on Claude Code, tens of cents. Run one agent per job and keep `max_concurrency` low. `simulate()` and `red_team()` each clone the target per conversation, so concurrency equals the number of live agent processes.

## Not in this version

Session resume between turns, a `cli:` string target for the `eq` CLI, and agents beyond the three named.
