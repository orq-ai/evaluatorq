# Red Teaming (`eq redteam`)

Red teaming subcommand group. Registered only when `evaluatorq[redteam]` is installed.

## `eq redteam run`

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
| `--verbose` / `-v` | count / `0` | Increase verbosity. `-v` per-attack progress + info logs; `-vv` debug logs. |
| `--quiet` / `-q` | `bool` / `False` | Suppress progress bars and non-error output. |

**Delivery methods** (`--delivery-method`): `DAN`, `role-play`, `skeleton-key`, `base64`, `leetspeak`, `multilingual`, `character-spacing`, `crescendo`, `many-shot`, `authority-impersonation`, `refusal-suppression`, `direct-request`, `code-elicitation`, `code-assistance`, `tool-response`, `word-substitution`.

**Saving results.** Persistence is controlled by two flags. `--save` accepts `none` (no files), `final` (summary JSON only), or `detail` (all per-stage artifacts). `--output-dir DIR` sets where JSON is written and is **required** when `--save detail`.

---

## `eq redteam ui`

Launch the Streamlit dashboard for a saved red-team run.

```bash
eq redteam ui [REPORT_PATH] [--latest] [--host HOST] [--port PORT]
```

| Flag / Argument | Type / Default | Description |
|---|---|---|
| `REPORT_PATH` | `Path \| None` / `None` | Saved run to open. Omit to use the latest auto-saved run. |
| `--latest` / `-l` | `bool` / `False` | Open the most recent run without passing a path. |
| `--host` | `str` / `localhost` | Host to bind the Streamlit server to. |
| `--port` | `int` / `8501` | Port for the Streamlit server. |

Requires `evaluatorq[redteam]`.

---

## `eq redteam validate-dataset`

Validate the shape of a red team dataset.

```bash
eq redteam validate-dataset [DATASET]
```

| Argument | Type / Default | Description |
|---|---|---|
| `DATASET` | `str \| None` / `None` | Local path, `hf:org/repo`, or `hf:org/repo/file.json`. Defaults to the official `orq/redteam-vulnerabilities` HuggingFace dataset. |

---

## `eq redteam runs`

List previously saved red team runs.

```bash
eq redteam runs [PATH] [--limit N]
```

| Flag / Argument | Type / Default | Description |
|---|---|---|
| `PATH` | `Path \| None` / `None` | Directory containing run reports. Defaults to `.evaluatorq/runs/`. |
| `--limit` / `-n` | `int` / `20` | Maximum number of runs to show. |
