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
| `--from-run` | `str \| None` / `None` | Replay a previous run instead of generating data: pass its file name, run id, path, or `latest`. Re-runs the exact same attacks, so only the target and models may differ. Cannot be combined with `--mode`, `--dataset`, `--category`, `--vulnerability`, `--strategy`, `--delivery-method`, or the `--max-*-datapoints` caps. |
| `--artifacts-dir` | `Path \| None` / `None` | Directory for saved JSON files. Required when `--save detail`. (`--output-dir` was removed; use `--artifacts-dir`.) |
| `--save` | `none \| final \| detail` / `final` | What to persist: `none` (no files), `final` (summary only), or `detail` (all stage artifacts). |
| `--report` | `Path \| None` / `None` | Path to write the report JSON. |
| `--report-md` | `Path \| None` / `None` | Directory for an auto-named Markdown report. |
| `--report-html` | `Path \| None` / `None` | Directory for an auto-named HTML report. |
| `--executive-summary` / `--no-executive-summary` | `bool` / `--executive-summary` | Generate an LLM narrative executive summary at the top of the report (needs LLM credentials). Pass `--no-executive-summary` to skip the extra LLM call. |
| `--recommendations` / `--no-recommendations` | `bool` / `--recommendations` | Generate LLM remediation recommendations for the top focus areas (needs LLM credentials). Pass `--no-recommendations` to skip the extra LLM call. |
| `--system-prompt` | `str \| None` / `None` | System prompt for the target model/agent. |
| `--yes` / `-y` | `bool` / `False` | Skip confirmation prompt. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. `-v` per-attack progress + info logs; `-vv` debug logs. |
| `--quiet` / `-q` | `bool` / `False` | Suppress progress bars and non-error output. |

**Delivery methods** (`--delivery-method`): `DAN`, `role-play`, `skeleton-key`, `base64`, `leetspeak`, `multilingual`, `character-spacing`, `crescendo`, `many-shot`, `authority-impersonation`, `refusal-suppression`, `direct-request`, `code-elicitation`, `code-assistance`, `tool-response`, `word-substitution`.

**Saving results.** Persistence is controlled by two flags. `--save` accepts `none` (no files), `final` (summary JSON only), or `detail` (all per-stage artifacts). `--artifacts-dir DIR` sets where JSON is written and is **required** when `--save detail` (`--output-dir` was removed; use `--artifacts-dir`).

**Exit codes.** `eq redteam run` exits `1` — after writing any requested report
artifacts — in two cases, both read off `report.summary`:

- **Zero verdicts** (`summary.no_verdict`): attacks ran but not one could be
  evaluated. Always fails; there is no setting that disables this.
- **Coverage below the floor** (`summary.coverage_below_minimum`): fewer than
  `--min-evaluation-coverage` (default **`0.8`**) of attacks got a verdict.
  **A run that finishes at 79% coverage now exits `1`**, not `0` with a warning —
  the same run used to pass. Pass `--min-evaluation-coverage 0` to warn instead
  of failing, or a higher value to be stricter. The Python equivalent is
  `EvaluatorConfig.min_evaluation_coverage` (`None` there also means warn-only)
  — see [Red Teaming › In CI](../guides/red-teaming.md#in-ci).

Both cases print the dominant failure cause with a sample message before
exiting: an `evaluation/<code>` (timeout / parse / api_connection / api_status /
scorer_exception) when the judge failed, or an `execution/<code>` when the
target failed and there was nothing to judge. Either way a systematically
blocked run is diagnosable from the CLI output alone.

---

## `eq redteam ui` (deprecated)

!!! warning "Deprecated — use `eq dashboard`"
    `eq redteam ui` is a deprecated legacy Streamlit command. The primary UI for
    browsing red-team runs is the multi-run FastHTML dashboard: `eq dashboard`
    (both stores) or `eq dashboard .evaluatorq/runs` (red team only). Passing a
    single JSON report file to `eq dashboard` is an optional direct deep-link.

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
eq redteam runs [PATH] [--limit N] [--json]
```

| Flag / Argument | Type / Default | Description |
|---|---|---|
| `PATH` | `Path \| None` / `None` | Directory containing run reports. Defaults to `.evaluatorq/runs/`. |
| `--limit` / `-n` | `int` / `20` | Maximum number of runs to show. |
| `--json` | `bool` / `False` | Emit runs as a JSON array on stdout (machine-readable). |
