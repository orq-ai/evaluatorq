# CLI Reference

Both `evaluatorq` and `eq` are aliases for the same entry point:

```toml
# pyproject.toml [project.scripts]
evaluatorq = "evaluatorq.cli:main"
eq         = "evaluatorq.cli:main"
```

Subcommands are registered at startup. `eq redteam` requires the `redteam` extra; `eq sim` requires the `simulation` extra.

---

## Top-level options

`eq --version` prints the installed version (e.g. `evaluatorq 1.3.2`) and exits. Running `eq` with no arguments prints help and exits.

---

## `eq ui`

Launch the FastHTML dashboard showing all red team **and** simulation runs.

```bash
eq ui [PATH] [--host HOST] [--port PORT]
```

`PATH` is optional. Omit it to scan both default run stores (`.evaluatorq/runs/` and `.evaluatorq/sim-runs/`). Pass a directory to restrict the scan; pass a file to scope to that file's parent directory and print a direct report URL.

| Flag / Argument | Type / Default | Description |
|---|---|---|
| `PATH` | `Path \| None` / `None` | Optional path to scan. |
| `--host` | `str` / `127.0.0.1` | Host to bind the dashboard server to. |
| `--port` | `int` / `8080` | Port for the dashboard server. |

Requires `evaluatorq[dashboard]` (`python-fasthtml`, `uvicorn`).

---

## `eq redteam`

Red teaming subcommand group. Registered only when `evaluatorq[redteam]` is installed.

### `eq redteam run`

Run adversarial red teaming against one or more targets.

```bash
eq redteam run --target agent:<key> [OPTIONS]
```

| Flag | Type / Default | Description |
|---|---|---|
| `--target` / `-t` | `str` (repeatable) | Target identifier(s). Use `agent:<key>` for Orq agents or `deployment:<key>`. Repeatable. |
| `--name` / `-n` | `str \| None` / `None` | Experiment name (defaults to `red-team`). |
| `--mode` | `str` / `dynamic` | Execution mode: `dynamic`, `static`, or `hybrid`. |
| `--category` / `-c` | `str` (repeatable) | OWASP categories to test (e.g. `ASI01`). Repeatable and/or comma-separated. Defaults to all. |
| `--vulnerability` / `-V` | `str` (repeatable) | Vulnerability IDs to test (e.g. `goal_hijacking`). Repeatable and/or comma-separated. Also accepts OWASP codes. Takes precedence over `--category`. |
| `--strategy` / `-s` | `str` (repeatable) | Restrict to named attack strategies. Repeatable and/or comma-separated. Unknown registry names are rejected. |
| `--delivery-method` / `-d` | `str` (repeatable) | Restrict to one or more delivery methods. Repeatable and/or comma-separated. |
| `--max-turns` | `int` / `5` | Maximum conversation turns for multi-turn attacks. |
| `--max-per-category` | `int \| None` / `None` | Cap strategies per category. |
| `--attack-model` | `str` / `gpt-5-mini` | Model for adversarial prompt generation. |
| `--attacker-instructions` | `str \| None` / `None` | Domain-specific context to steer attack generation. |
| `--evaluator-model` | `str` / `gpt-5-mini` | Model for OWASP evaluation scoring. |
| `--parallelism` | `int` / `10` | Maximum concurrent jobs. |
| `--generated-strategy-count` | `int` / `2` | Number of LLM-generated strategies per category. |
| `--no-generate-strategies` | `bool` / `False` | Disable LLM-based strategy generation. |
| `--max-dynamic-datapoints` | `int \| None` / `None` | Cap dynamically generated datapoints. |
| `--max-static-datapoints` | `int \| None` / `None` | Cap static (dataset) datapoints. |
| `--no-cleanup-memory` | `bool` / `False` | Skip memory entity cleanup after dynamic runs. |
| `--dataset` | `str \| None` / `None` | Dataset source: local path, `hf:org/repo`, or `hf:org/repo/file.json`. |
| `--output-dir` | `Path \| None` / `None` | Directory for saved JSON files. Required when `--save detail`. |
| `--save` | `none \| final \| detail` / `final` | What to persist: `none` (no files), `final` (summary only), or `detail` (all stage artifacts). |
| `--save-report` | `Path \| None` / `None` | Path to write the report JSON. |
| `--export-md` | `Path \| None` / `None` | Directory for an auto-named Markdown report. |
| `--export-html` | `Path \| None` / `None` | Directory for an auto-named HTML report. |
| `--system-prompt` | `str \| None` / `None` | System prompt for the target model/agent. |
| `--yes` / `-y` | `bool` / `False` | Skip confirmation prompt. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. `-v` info logs; `-vv` debug logs. |
| `--quiet` / `-q` | `bool` / `False` | Suppress progress bars and non-error output. |

**Delivery methods** (`--delivery-method`): `DAN`, `role-play`, `skeleton-key`, `base64`, `leetspeak`, `multilingual`, `character-spacing`, `crescendo`, `many-shot`, `authority-impersonation`, `refusal-suppression`, `direct-request`, `code-elicitation`, `code-assistance`, `tool-response`, `word-substitution`.

**Saving results.** Persistence is controlled by two flags. `--save` accepts `none` (no files), `final` (summary JSON only), or `detail` (all per-stage artifacts). `--output-dir DIR` sets where JSON is written and is **required** when `--save detail`.

---

### `eq redteam ui`

Launch the FastHTML dashboard scoped to red team runs only.

```bash
eq redteam ui [--host HOST] [--port PORT]
```

| Flag | Type / Default | Description |
|---|---|---|
| `--host` | `str` / `127.0.0.1` | Host to bind the dashboard server to. |
| `--port` | `int` / `8080` | Port for the dashboard server. |

---

### `eq redteam validate-dataset`

Validate the shape of a red team dataset.

```bash
eq redteam validate-dataset [DATASET]
```

| Argument | Type / Default | Description |
|---|---|---|
| `DATASET` | `str \| None` / `None` | Local path, `hf:org/repo`, or `hf:org/repo/file.json`. Defaults to the official `orq/redteam-vulnerabilities` HuggingFace dataset. |

---

### `eq redteam runs`

List previously saved red team runs.

```bash
eq redteam runs [PATH] [--limit N]
```

| Flag / Argument | Type / Default | Description |
|---|---|---|
| `PATH` | `Path \| None` / `None` | Directory containing run reports. Defaults to `.evaluatorq/runs/`. |
| `--limit` / `-n` | `int` / `20` | Maximum number of runs to show. |

---

## `eq sim`

Agent simulation subcommand group. Registered only when `evaluatorq[simulation]` is installed.

Three main verbs: `generate` (datapoints only), `simulate` (run against pre-built datapoints), `run` (generate then simulate in one shot).

### `eq sim run`

Generate personas and scenarios, then run simulations.

```bash
eq sim run --agent-description "..." --openai-model gpt-4o-mini
eq sim run --target agent:<key>
```

Targets — provide **exactly one**:

| Flag | Description |
|---|---|
| `--target` | `agent:<key>` or `deployment:<key>`. Bare values default to `agent:<key>`. |
| `--agent-key` | Deprecated alias for Orq deployment key (requires `ORQ_API_KEY`). |
| `--vercel-url` | Vercel AI SDK HTTP endpoint URL. |
| `--openai-model` | OpenAI-compatible model name. Provider resolved from env: `ORQ_API_KEY` → Orq AI Router; `OPENAI_API_KEY` → OpenAI-compatible. |

| Flag | Type / Default | Description |
|---|---|---|
| `--agent-description` | `str \| None` / `None` | Free-text description of the agent. May be omitted when `--target` is an Orq agent (fetched automatically). |
| `--name` / `-n` | `str` / `sim` | Run name for the run-store entry. |
| `--sim-model` | `str` / `openai/gpt-5.4-mini` | Model for user-simulator, judge, and generation. |
| `--max-turns` | `int` / `10` | Maximum conversation turns. |
| `--parallelism` | `int` / `5` | Concurrent simulations. |
| `--num-personas` | `int` / `5` | Number of personas to generate. |
| `--num-scenarios` | `int` / `5` | Number of scenarios to generate. |
| `--evaluator` | `str` (repeatable) / API defaults | Evaluator name(s). Repeatable. |
| `--no-save` | `bool` / `False` | Skip writing to `.evaluatorq/sim-runs/`. |
| `--save-datapoints` | `Path \| None` / `None` | Write generated datapoints to JSONL for reproducible re-runs. |
| `--output` / `-o` | `Path \| None` / `None` | Path to write results JSONL. |
| `--report-output` | `Path \| None` / `None` | Path to write full SimulationRun report JSON. |
| `--export-md` | `Path \| None` / `None` | Directory for an auto-named Markdown report. |
| `--export-html` | `Path \| None` / `None` | Directory for an auto-named HTML report. |
| `--yes` / `-y` | `bool` / `False` | Skip interactive confirmation prompt. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. `-v` info; `-vv` debug. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |

---

### `eq sim simulate`

Run simulations from a pre-built datapoints JSONL file.

```bash
eq sim simulate --datapoints dp.jsonl --target agent:<key>
```

Targets — same four flags as `eq sim run`. All other flags match `eq sim run` except `--num-personas`, `--num-scenarios`, and `--save-datapoints` are absent (datapoints are already provided).

| Flag | Type / Default | Description |
|---|---|---|
| `--datapoints` / `-d` | `Path` (required) | Path to datapoints JSONL file. |
| `--name` / `-n` | `str` / `sim` | Run name for the run-store entry. |
| `--sim-model` | `str` / `openai/gpt-5.4-mini` | Model for user-simulator and judge. |
| `--max-turns` | `int` / `10` | Maximum conversation turns. |
| `--parallelism` | `int` / `5` | Concurrent simulations. |
| `--evaluator` | `str` (repeatable) / API defaults | Evaluator name(s). Repeatable. |
| `--no-save` | `bool` / `False` | Skip writing to `.evaluatorq/sim-runs/`. |
| `--output` / `-o` | `Path \| None` / `None` | Path to write results JSONL. |
| `--report-output` | `Path \| None` / `None` | Path to write full SimulationRun report JSON. |
| `--export-md` | `Path \| None` / `None` | Directory for an auto-named Markdown report. |
| `--export-html` | `Path \| None` / `None` | Directory for an auto-named HTML report. |
| `--yes` / `-y` | `bool` / `False` | Skip interactive confirmation prompt. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |

---

### `eq sim generate`

Generate simulation datapoints only — no simulation is run.

```bash
eq sim generate --output dp.jsonl --agent-description "..."
```

| Flag | Type / Default | Description |
|---|---|---|
| `--output` / `-o` | `Path` (required) | Path to write generated datapoints JSONL. |
| `--agent-description` | `str \| None` / `None` | Free-text description of the agent. |
| `--target` | `str \| None` / `None` | Agent target used to fetch the description when `--agent-description` is omitted. Accepts `agent:<key>`. |
| `--sim-model` | `str` / `openai/gpt-5.4-mini` | Model for persona/scenario/first-message generation. |
| `--num-personas` | `int` / `5` | Number of personas to generate. |
| `--num-scenarios` | `int` / `5` | Number of scenarios to generate. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |

---

### `eq sim export`

Convert simulation results JSONL to OpenResponses payload JSON.

```bash
eq sim export --input results.jsonl --output payload.json
```

| Flag | Type / Default | Description |
|---|---|---|
| `--input` / `-i` | `Path` (required) | Path to results JSONL file. |
| `--output` / `-o` | `Path` (required) | Path to write OpenResponses payload JSON. |

---

### `eq sim validate-dataset`

Validate a simulation datapoints JSONL file.

```bash
eq sim validate-dataset dp.jsonl
```

| Argument | Type / Default | Description |
|---|---|---|
| `PATH` | `Path` (required) | Path to datapoints JSONL file to validate. |

---

### `eq sim runs`

List recent simulation runs.

```bash
eq sim runs [DIRECTORY] [--limit N]
```

| Flag / Argument | Type / Default | Description |
|---|---|---|
| `DIRECTORY` | `Path \| None` / `None` | Directory to scan. Defaults to `.evaluatorq/sim-runs/`. |
| `--limit` / `-n` | `int` / `20` | Maximum number of runs to show. |

---

### `eq sim ui`

Launch the FastHTML dashboard scoped to simulation runs only.

```bash
eq sim ui [--host HOST] [--port PORT]
```

| Flag | Type / Default | Description |
|---|---|---|
| `--host` | `str` / `127.0.0.1` | Host to bind the dashboard server to. |
| `--port` | `int` / `8080` | Port for the dashboard server. |

---

## Recipes

```bash
# CI smoke run — one strategy per category, no LLM-generated strategies, quiet
eq redteam run -t agent:my-agent --max-per-category 1 --no-generate-strategies -q

# Save full per-stage artifacts to a directory
eq redteam run -t agent:my-agent --save detail --output-dir ./runs

# Quick simulation — two personas, two scenarios
eq sim run -t agent:my-agent --num-personas 2 --num-scenarios 2
```
