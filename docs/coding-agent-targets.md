# Coding agents as targets

`CodingAgentTarget` runs a coding-agent CLI you already have installed as the system under test. Every turn spawns a fresh process in a private working directory, hands it the conversation so far, and reads its JSON output back as text, tool calls and token usage. `red_team()` and `simulate()` treat it like any other target.

Three agents are supported: `claude` (Claude Code), `codex` (Codex CLI) and `opencode` (OpenCode).

## Two ways to launch

| `launcher` | What runs | Use it when |
|---|---|---|
| `'direct'` (default) | The agent binary, with your environment and your own provider credentials | You are testing the agent as your users run it |
| `'orq'` | `orq launch <agent>`, so every model call routes through the Orq gateway with workspace skills and MCP server attached by default | You are testing the Orq launcher, skills or gateway themselves |

Under `launcher='orq'`, `model` becomes `orq launch --model provider/id` and `OrqLaunchOptions` tunes the remaining flags. Profile and workspace are not launch flags: set `ORQ_PROFILE` or `ORQ_API_KEY` through `env`.

`OrqLaunchOptions` has four fields: `mcp=False` renders `--no-mcp`, `skills=False` renders `--no-skills`, `base_url` renders `--base-url <url>`, and `fetch_models=False` renders `--no-fetch-models`.

## Red-teaming Claude Code with one injected skill

```python
from pathlib import Path

from evaluatorq.backends import CodingAgentTarget
from evaluatorq.redteam import red_team

target = CodingAgentTarget(
    'claude',
    model='claude-fable-5-1',
    permission_mode='acceptEdits',
    skills=[Path('skills/grill-me')],
    workdir=Path('fixtures/sample-repo'),
    system_prompt='You only work inside this repository.',
)

report = await red_team(target=target, max_concurrency=2)
```

Each parallel job gets its own copy of `fixtures/sample-repo` with `skills/grill-me` linked into `.claude/skills/`. The copies are deleted when the run ends; pass `keep_workdir=True` to keep them and log their paths, which is the fastest way to see what the agent actually changed.

## Through the Orq gateway

```python
from evaluatorq.backends import CodingAgentTarget, OrqLaunchOptions

target = CodingAgentTarget(
    'claude',
    launcher='orq',
    model='anthropic/claude-fable-5-1',
    orq=OrqLaunchOptions(mcp=True, skills=True),
    permission_mode='acceptEdits',
)
```

## Privilege

Each agent starts with its own default permission behaviour, and this target does not change it. Two knobs, both in the agent's own vocabulary:

- `permission_mode`: claude `--permission-mode` (`default`, `acceptEdits`, `plan`, `bypassPermissions`, `dontAsk`); codex `--sandbox` (`read-only`, `workspace-write`, `danger-full-access`). OpenCode has no such flag, so a value raises `ValueError`.
- `extra_args`: appended to the agent's argv untouched. OpenCode blocks on its first permission prompt when run headless; pass `extra_args=['--auto']` to auto-approve, knowing that grants everything.

A tool call the harness refused still appears in the response as a tool call whose result is `[denied by claude]`, so the judge sees the attempt.

The private copy preserves symlinks from `workdir` as symlinks, so a link that pointed outside the original still points there and the agent can read or write through it. Point `workdir` at a tree whose links you are willing to expose.

Under `launcher='orq'`, orq launch adds provider and MCP configuration only; it passes no sandbox or permission flag of its own. Codex therefore runs with `permission_mode` when given, otherwise with the sandbox configured in the user's own codex config. Requires an orq CLI that no longer injects `--full-auto` (orq 10.0.0-rc.1 still does, and codex-cli 0.153.4 rejects it with `cli.exit.2`).

## What the response carries

Tool calls in order, then the agent's final message as text. `response_id` is the agent's session id. Token usage comes from the agent's own usage events, summed across OpenCode's per-step reports with `calls` counting the steps, and is `None` with a warning when the agent reported none. Claude's reported cost lands on the target span as `evaluatorq.coding_agent.cost_usd`.

## Errors

Failures surface as `cli.*` error codes on the result, in this order of precedence:

| Code | Meaning | Retried by the runner |
|---|---|---|
| `cli.not_found` | The binary is not on `PATH` | No |
| `cli.timeout` | No result within `timeout_ms` (default 210 s, 30 s under the retry helper's 240 s so this ceiling fires first and is not retried); the process group is killed | No |
| `cli.prompt_too_long` | The OS refused the argv; under `launcher='orq'` codex and opencode take the rendered transcript as one argument, so a long conversation can exceed the limit | No |
| `cli.exit.<code>` | Non-zero exit, even if a result was printed | Yes |
| `cli.parse_error` | Stdout contained no JSON events | Yes |
| `cli.agent_error` | Exit 0 but the agent reported failure (`is_error`, `turn.failed`) | Yes |
| `cli.no_result` | Exit 0 with no final assistant message | Yes |

## Cost and concurrency

A tool-using turn is a full coding-agent session: minutes of wall clock and, on Claude Code, tens of cents. Run one agent per job and keep `max_concurrency` low. `simulate()` and `red_team()` each clone the target per conversation, so concurrency equals the number of live agent processes.

## Not in this version

Session resume between turns, a `cli:` string target for the `eq` CLI, sandboxed or remote execution, and agents beyond the three named.
