# evaluatorq CLI clig.dev Alignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the `evaluatorq`/`eq` CLI to ~9/10 clig.dev compliance via ten additive, non-breaking fixes.

**Architecture:** Small, focused edits to the three CLI modules (`cli.py`, `redteam/cli.py`, `simulation/cli.py`) plus a handful of tiny shared helpers under `common/` so redteam and sim stop diverging. No behavior removed; one behavior flips (non-TTY redteam confirm now auto-proceeds, matching sim).

**Tech Stack:** Python 3.10+, Typer + Click, Rich, pytest, `typer.testing.CliRunner`, `uv`.

## Global Constraints

- Python 3.10+ compatible; use `from __future__ import annotations`.
- Package manager is `uv` (never pip for repo tasks). Run tests with `uv run pytest`.
- Lint: `uv run ruff check src`. Types: `uv run basedpyright`. Both must stay green.
- All changes additive / non-breaking. No major version bump. No new runtime dependency.
- `passed=True` = RESISTANT, `passed=False` = VULNERABLE (unchanged; not touched here).
- Only introduce a `common/` helper where ≥2 call sites use it.
- Design spec: `docs/superpowers/specs/2026-07-17-evaluatorq-cli-cligdev-design.md`.

---

## File Structure

- Create `src/evaluatorq/common/cli_help.py` — shared `CONTEXT_SETTINGS` (item 1).
- Create `src/evaluatorq/common/cli_tty.py` — `should_skip_confirm()` (item 2).
- Create `src/evaluatorq/common/cli_epilog.py` — `examples()` epilog builder, moved from sim (item 8).
- Create `src/evaluatorq/common/cli_json.py` — `echo_json()` (item 5).
- Create `src/evaluatorq/common/cli_errors.py` — `emit_error()` + `run_guarded()` (item 6).
- Modify `src/evaluatorq/cli.py` — context settings, root epilog, global error guard, ImportError stub.
- Modify `src/evaluatorq/redteam/cli.py` — context settings, `--mode` enum, TTY guard, `run` epilog, `runs --json`.
- Modify `src/evaluatorq/simulation/cli.py` — context settings, Sphinx strip, use `examples()` from common, `runs --json`, use `should_skip_confirm`.
- Create/extend tests under `tests/redteam/` and `tests/unit/`.

---

## Task 1: Trivial doc fixes (items 4 + D)

**Files:**
- Modify: `src/evaluatorq/simulation/cli.py:1330`
- Modify: `docs/orq-deployment.md:25`

No test — pure string/docstring edits.

- [ ] **Step 1: Strip the leaked Sphinx role.**

In `simulation/cli.py`, the `validate-dataset` command docstring (line ~1330) reads:

```python
    """Deprecated alias for :command:`validate`."""
```

Change to:

```python
    """Deprecated alias for `validate`."""
```

- [ ] **Step 2: Fix the `.env` doc contradiction.**

`docs/orq-deployment.md:25` currently:

```markdown
| `ORQ_API_KEY` | Required. Set in environment or `.env` file. |
```

Change to (matches `docs/configuration.md`, which correctly states the library never calls `load_dotenv()`):

```markdown
| `ORQ_API_KEY` | Required. Set in the environment (the library does not auto-load `.env`; call `load_dotenv()` yourself first — see configuration.md). |
```

- [ ] **Step 3: Verify nothing else leaks the role.**

Run: `grep -rn ':command:' src/ docs/`
Expected: no matches (the one occurrence is now gone).

- [ ] **Step 4: Commit.**

```bash
git add src/evaluatorq/simulation/cli.py docs/orq-deployment.md
git commit -m "fix(cli): strip Sphinx role from help text; correct .env doc"
```

---

## Task 2: `-h` short help alias (item 1)

**Files:**
- Create: `src/evaluatorq/common/cli_help.py`
- Modify: `src/evaluatorq/cli.py:26`, `src/evaluatorq/redteam/cli.py:20`, `src/evaluatorq/simulation/cli.py:49`
- Test: `tests/unit/test_cli_help_option.py`

**Interfaces:**
- Produces: `evaluatorq.common.cli_help.CONTEXT_SETTINGS: dict[str, list[str]]`

- [ ] **Step 1: Write the failing test.**

```python
# tests/unit/test_cli_help_option.py
from typer.testing import CliRunner

from evaluatorq.redteam.cli import app as redteam_app


def test_dash_h_shows_help_on_subcommand():
    result = CliRunner().invoke(redteam_app, ['run', '-h'])
    assert result.exit_code == 0
    assert 'Usage' in result.output
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run pytest tests/unit/test_cli_help_option.py -v`
Expected: FAIL — `-h` is currently an unknown option, exit code 2.

- [ ] **Step 3: Create the shared constant.**

```python
# src/evaluatorq/common/cli_help.py
"""Shared Typer context settings for the evaluatorq CLIs."""

from __future__ import annotations

# clig.dev: both -h and --help must show help. Typer only wires --help by
# default, so add -h explicitly. Defined once to keep the three Typer apps in sync.
CONTEXT_SETTINGS: dict[str, list[str]] = {'help_option_names': ['-h', '--help']}
```

- [ ] **Step 4: Apply to the root app.**

In `cli.py`, add the import near the other imports and pass `context_settings`:

```python
from evaluatorq.common.cli_help import CONTEXT_SETTINGS

app = typer.Typer(
    name='evaluatorq',
    help='Evaluation framework for AI systems.',
    rich_markup_mode='rich',
    context_settings=CONTEXT_SETTINGS,
)
```

- [ ] **Step 5: Apply to the redteam sub-app.**

In `redteam/cli.py`:

```python
from evaluatorq.common.cli_help import CONTEXT_SETTINGS

app = typer.Typer(
    name='redteam',
    help='Red teaming CLI for evaluatorq.',
    no_args_is_help=True,
    rich_markup_mode='rich',
    context_settings=CONTEXT_SETTINGS,
)
```

- [ ] **Step 6: Apply to the sim sub-app.**

In `simulation/cli.py`, add the import and `context_settings=CONTEXT_SETTINGS` to the `app = typer.Typer(...)` at line ~49 (same edit shape as Step 5).

- [ ] **Step 7: Run the test + a manual check.**

Run: `uv run pytest tests/unit/test_cli_help_option.py -v`
Expected: PASS
Run: `uv run evaluatorq -h` and `uv run evaluatorq redteam run -h`
Expected: both print help, exit 0.

- [ ] **Step 8: Commit.**

```bash
git add src/evaluatorq/common/cli_help.py src/evaluatorq/cli.py src/evaluatorq/redteam/cli.py src/evaluatorq/simulation/cli.py tests/unit/test_cli_help_option.py
git commit -m "feat(cli): add -h short help alias across all apps"
```

---

## Task 3: `--mode` enum on `redteam run` (item 7)

**Files:**
- Modify: `src/evaluatorq/redteam/cli.py:18,195-198`
- Test: `tests/redteam/test_cli_mode_enum.py`

**Interfaces:**
- Consumes: `evaluatorq.redteam.contracts.Pipeline` (existing `StrEnum`: `static`/`dynamic`/`hybrid`).

- [ ] **Step 1: Write the failing test.**

```python
# tests/redteam/test_cli_mode_enum.py
from typer.testing import CliRunner

from evaluatorq.redteam.cli import app as redteam_app


def test_bogus_mode_rejected_early():
    result = CliRunner().invoke(redteam_app, ['run', '-t', 'agent:x', '--mode', 'bogus'])
    assert result.exit_code == 2  # usage error, not a late runtime failure
    assert 'bogus' in result.output or 'dynamic' in result.output
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run pytest tests/redteam/test_cli_mode_enum.py -v`
Expected: FAIL — `mode` is currently a free `str`, so `bogus` is accepted at parse time and fails later (not exit 2 at parse).

- [ ] **Step 3: Add `Pipeline` to the imports.**

In `redteam/cli.py:18`, extend the contracts import:

```python
from evaluatorq.redteam.contracts import DEFAULT_PIPELINE_MODEL, DeliveryMethod, Pipeline, SaveMode, Vulnerability
```

- [ ] **Step 4: Change the `mode` parameter type.**

Replace the current param (lines ~195-198):

```python
    mode: Annotated[
        Pipeline,
        typer.Option(help='Execution mode.'),
    ] = Pipeline.DYNAMIC,
```

The `red_team(mode=mode)` call at line ~429 needs no change — `Pipeline` is a `StrEnum`, so it already coerces where a string was passed.

- [ ] **Step 5: Run the test to verify it passes.**

Run: `uv run pytest tests/redteam/test_cli_mode_enum.py -v`
Expected: PASS

- [ ] **Step 6: Commit.**

```bash
git add src/evaluatorq/redteam/cli.py tests/redteam/test_cli_mode_enum.py
git commit -m "feat(redteam): make --mode a Pipeline enum for early validation"
```

---

## Task 4: TTY guard on redteam confirm (item 2)

**Files:**
- Create: `src/evaluatorq/common/cli_tty.py`
- Modify: `src/evaluatorq/redteam/cli.py:443`, `src/evaluatorq/simulation/cli.py:~612`
- Test: `tests/unit/test_cli_tty.py`

**Interfaces:**
- Produces: `evaluatorq.common.cli_tty.should_skip_confirm(yes: bool) -> bool`

- [ ] **Step 1: Write the failing test.**

```python
# tests/unit/test_cli_tty.py
import sys

from evaluatorq.common.cli_tty import should_skip_confirm


def test_skip_when_non_tty(monkeypatch):
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)
    assert should_skip_confirm(False) is True


def test_no_skip_when_tty_and_no_yes(monkeypatch):
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    assert should_skip_confirm(False) is False


def test_yes_always_skips(monkeypatch):
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    assert should_skip_confirm(True) is True
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run pytest tests/unit/test_cli_tty.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the helper.**

```python
# src/evaluatorq/common/cli_tty.py
"""Shared TTY helpers for the evaluatorq CLIs."""

from __future__ import annotations

import sys


def should_skip_confirm(yes: bool) -> bool:
    """Return True when the confirmation prompt must be skipped.

    Skip when the user passed --yes, or when stdin is not a TTY (CI, pipes) —
    otherwise ``typer.confirm`` blocks forever waiting on input nobody will send.
    """
    return yes or not sys.stdin.isatty()
```

- [ ] **Step 4: Use it at the redteam call site.**

In `redteam/cli.py`, add the import and change line ~443:

```python
from evaluatorq.common.cli_tty import should_skip_confirm
```

```python
                hooks=RichHooks(skip_confirm=should_skip_confirm(yes)),
```

- [ ] **Step 5: Use it at BOTH sim call sites.**

`simulation/cli.py` has **two** identical inline occurrences (verified): line ~612
(inside `simulate`) and line ~900 (inside `run`):

```python
            skip_confirm=yes or not sys.stdin.isatty(),
```

Replace **both** with `skip_confirm=should_skip_confirm(yes)` and add the import.
(Migrating only one would leave the duplication this helper exists to remove.)

- [ ] **Step 6: Run tests to verify they pass.**

Run: `uv run pytest tests/unit/test_cli_tty.py -v`
Expected: PASS
Run: `echo | uv run evaluatorq redteam run -t agent:x --save none`
Expected: does NOT hang on the confirm prompt (fails/exits for other reasons, e.g. missing creds, but no blocking read).

- [ ] **Step 7: Commit.**

```bash
git add src/evaluatorq/common/cli_tty.py src/evaluatorq/redteam/cli.py src/evaluatorq/simulation/cli.py tests/unit/test_cli_tty.py
git commit -m "fix(redteam): skip confirm on non-TTY to avoid CI hangs"
```

---

## Task 5: Examples epilogs (items 8 + 9)

**Files:**
- Create: `src/evaluatorq/common/cli_epilog.py`
- Modify: `src/evaluatorq/simulation/cli.py:393-406` (remove local `_examples`, import from common)
- Modify: `src/evaluatorq/redteam/cli.py` (`run` epilog), `src/evaluatorq/cli.py` (root epilog + `dashboard` epilog)
- Test: `tests/unit/test_cli_epilog.py`

**Interfaces:**
- Produces: `evaluatorq.common.cli_epilog.examples(*lines: str) -> str`

- [ ] **Step 1: Write the failing test.**

```python
# tests/unit/test_cli_epilog.py
from evaluatorq.common.cli_epilog import examples


def test_examples_dims_comment_lines():
    out = examples('# a comment', 'eq redteam run -t agent:x')
    assert 'Examples' in out
    assert '[dim]' in out  # comment line is dimmed
    assert 'eq redteam run -t agent:x' in out
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run pytest tests/unit/test_cli_epilog.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the shared builder (moved verbatim from sim).**

```python
# src/evaluatorq/common/cli_epilog.py
"""Shared ``--help`` epilog builder for the evaluatorq CLIs."""

from __future__ import annotations


def examples(*lines: str) -> str:
    """Build a command ``--help`` epilog from example lines.

    Under ``rich_markup_mode='rich'`` the epilog is flowed like HTML — single
    newlines collapse to spaces — so each visual line must be its own paragraph
    (blank line between) to render one-per-row. Lines starting with ``#`` are
    dimmed as comments; command lines render verbatim.
    """
    from rich.markup import escape

    def render(line: str) -> str:
        return f'[dim]{escape(line)}[/]' if line.lstrip().startswith('#') else escape(line)

    return '\n\n'.join(['[bold]Examples[/]', *(render(line) for line in lines)])
```

- [ ] **Step 4: Point sim at the shared builder.**

In `simulation/cli.py`, delete the local `def _examples(...)` (lines ~393-406) and add:

```python
from evaluatorq.common.cli_epilog import examples as _examples
```

(Alias keeps the existing `_SIMULATE_EPILOG = _examples(...)` call sites unchanged.)

- [ ] **Step 5: Add the `redteam run` epilog.**

In `redteam/cli.py`, add near the top:

```python
from evaluatorq.common.cli_epilog import examples

_RUN_EPILOG = examples(
    '# dynamic run against an orq agent',
    'eq redteam run -t agent:my-agent',
    '# static + generated attacks from an OWASP dataset',
    'eq redteam run -t agent:my-agent --mode hybrid',
    '# scope to one OWASP category, machine-readable later via `eq redteam runs --json`',
    'eq redteam run -t agent:my-agent -c ASI01',
)
```

Then add `epilog=_RUN_EPILOG` to the `run` command decorator:

```python
@app.command(no_args_is_help=True, epilog=_RUN_EPILOG)
def run(
```

- [ ] **Step 6: Add root + dashboard epilog and docs link (item 9).**

In `cli.py`, add:

```python
from evaluatorq.common.cli_epilog import examples

_ROOT_EPILOG = examples(
    '# red team an agent',
    'eq redteam run -t agent:my-agent',
    '# explore saved runs in the dashboard',
    'eq dashboard',
    '# docs: https://github.com/orq-ai/evaluatorq',
    '# report issues: https://github.com/orq-ai/evaluatorq/issues',
)
```

Add `epilog=_ROOT_EPILOG` to the root `app = typer.Typer(...)` and to the `@app.command()` decorator on `dashboard`.

- [ ] **Step 7: Run tests + manual check.**

Run: `uv run pytest tests/unit/test_cli_epilog.py -v`
Expected: PASS
Run: `uv run evaluatorq -h` and `uv run evaluatorq redteam run -h`
Expected: both show an "Examples" section; root help shows the docs/issues links.

- [ ] **Step 8: Commit.**

```bash
git add src/evaluatorq/common/cli_epilog.py src/evaluatorq/cli.py src/evaluatorq/redteam/cli.py src/evaluatorq/simulation/cli.py tests/unit/test_cli_epilog.py
git commit -m "feat(cli): add example epilogs and docs link; share epilog builder"
```

---

## Task 6: `--json` on the `runs` listing commands (item 5)

**Files:**
- Create: `src/evaluatorq/common/cli_json.py`
- Modify: `src/evaluatorq/redteam/cli.py:656-758` (`runs`)
- Modify: `src/evaluatorq/simulation/cli.py:1410-...` (`runs`)
- Test: `tests/redteam/test_cli_runs_json.py`

**Interfaces:**
- Produces: `evaluatorq.common.cli_json.echo_json(obj: object) -> None`

- [ ] **Step 1: Write the failing test.**

```python
# tests/redteam/test_cli_runs_json.py
import json

from typer.testing import CliRunner

from evaluatorq.redteam.cli import app as redteam_app


def test_runs_json_empty_dir_emits_empty_array(tmp_path):
    result = CliRunner().invoke(redteam_app, ['runs', str(tmp_path), '--json'])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == []


def test_runs_json_serialises_raw_fields(tmp_path):
    (tmp_path / 'r.json').write_text(json.dumps({
        'run_name': 'demo',
        'created_at': '2026-07-17T12:00:00Z',
        'pipeline': 'dynamic',
        'tested_agents': ['agent:x'],
        'summary': {'total_attacks': 3, 'vulnerability_rate': 0.42},
    }))
    result = CliRunner().invoke(redteam_app, ['runs', str(tmp_path), '--json'])
    assert result.exit_code == 0
    records = json.loads(result.stdout)
    assert records[0]['run_name'] == 'demo'
    assert records[0]['created_at'] == '2026-07-17T12:00:00Z'  # NOT truncated
    assert records[0]['vulnerability_rate'] == 0.42  # a number, not "42%"
    assert records[0]['total_attacks'] == 3
    assert records[0]['report_id']  # stable id correlating to dashboard /r/<id>
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run pytest tests/redteam/test_cli_runs_json.py -v`
Expected: FAIL — `--json` is an unknown option (exit 2).

- [ ] **Step 3: Create the shared JSON helper.**

```python
# src/evaluatorq/common/cli_json.py
"""Canonical machine-readable JSON output for the evaluatorq CLIs."""

from __future__ import annotations

import json
from typing import Any

import typer


def echo_json(obj: Any) -> None:
    """Print ``obj`` as indented JSON to stdout (nothing else on stdout)."""
    typer.echo(json.dumps(obj, indent=2, default=str))
```

- [ ] **Step 4: Add `--json` to redteam `runs`.**

Add the option to the signature:

```python
    json_output: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--json', help='Emit runs as a JSON array on stdout (machine-readable).'),
    ] = False,
```

Add the imports and thread `json_output` through the early-return branches and the main body. `report_id` correlates each row with the dashboard `/r/<id>` deep-link (reuses `dashboard/library.py`, which is stdlib-only at import — safe without the dashboard extra). Replace the two "no runs" branches so they honor `--json`:

```python
    from evaluatorq.common.cli_json import echo_json
    from evaluatorq.dashboard.library import report_id

    runs_dir = Path(path) if path is not None else get_runs_dir()
    if not runs_dir.exists():
        if json_output:
            echo_json([])
            raise typer.Exit(code=0)
        typer.echo(f'No runs found (directory {runs_dir} does not exist).')
        raise typer.Exit(code=0)

    run_files = sorted(runs_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_files:
        if json_output:
            echo_json([])
            raise typer.Exit(code=0)
        typer.echo(f'No runs found in {runs_dir}.')
        raise typer.Exit(code=0)

    run_files = run_files[:limit]

    if json_output:
        records: list[dict[str, Any]] = []
        skipped = 0
        for f in run_files:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                skipped += 1
                continue
            summary = data.get('summary', {})
            records.append({
                'report_id': report_id(f),
                'run_name': data.get('run_name', f.stem),
                'created_at': data.get('created_at'),
                'pipeline': data.get('pipeline'),
                'tested_agents': data.get('tested_agents', []),
                'total_attacks': summary.get('total_attacks', data.get('total_results')),
                'vulnerability_rate': summary.get('vulnerability_rate'),
                'file': f.name,
            })
        echo_json(records)
        if skipped:
            typer.echo(f'Warning: {skipped} file(s) could not be parsed and were skipped.', err=True)
        raise typer.Exit(code=0)

    # ...existing Rich-table rendering unchanged below...
```

Leave the existing Rich-table code (lines ~681-758) as-is; the `--json` branch returns before it.

- [ ] **Step 5: Add `--json` to sim `runs`.**

Add the same `json_output` option and honor it in the two no-runs branches
(`echo_json([])`, `raise typer.Exit(0)`). Insert the JSON branch as an **early
return immediately after the `if not run_files:` guard (~line 1440), BEFORE the
existing `rows = []` display loop** — this avoids re-parsing every file twice and
avoids shadowing the display loop's own `malformed` counter (use `malformed_json`).
Match redteam's exact exception tuple `(json.JSONDecodeError, OSError)` — do not
use a bare `except Exception`:

```python
    if json_output:
        from evaluatorq.common.cli_json import echo_json
        from evaluatorq.dashboard.library import report_id

        records: list[dict[str, Any]] = []
        malformed_json = 0
        for run_file in run_files:
            try:
                data = json.loads(run_file.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                malformed_json += 1
                continue
            records.append({
                'report_id': report_id(run_file),
                'run_name': data.get('run_name'),
                'created_at': data.get('created_at'),
                'mode': data.get('mode'),
                'target_kind': data.get('target_kind'),
                'total_results': data.get('total_results'),
                'scorer_averages': data.get('scorer_averages', {}),
                'file': run_file.name,
            })
        echo_json(records)
        if malformed_json:
            typer.echo(f'Warning: {malformed_json} malformed file(s) skipped.', err=True)
        raise typer.Exit(0)
```

- [ ] **Step 6: Run tests + pipe check.**

Run: `uv run pytest tests/redteam/test_cli_runs_json.py -v`
Expected: PASS
Run: `uv run evaluatorq redteam runs --json | python -m json.tool`
Expected: parses (empty array if no runs).

- [ ] **Step 7: Commit.**

```bash
git add src/evaluatorq/common/cli_json.py src/evaluatorq/redteam/cli.py src/evaluatorq/simulation/cli.py tests/redteam/test_cli_runs_json.py
git commit -m "feat(cli): add --json to redteam/sim runs with raw serializable fields"
```

---

## Task 7: Global unexpected-error handler (item 6)

**Files:**
- Create: `src/evaluatorq/common/cli_errors.py`
- Modify: `src/evaluatorq/cli.py:130-146` (`main`)
- Modify: `src/evaluatorq/simulation/cli.py:202-205` (`_handle_cli_error` → use shared `emit_error`)
- Test: `tests/unit/test_cli_errors.py`

**Interfaces:**
- Produces: `evaluatorq.common.cli_errors.emit_error(exc: object) -> None`
- Produces: `evaluatorq.common.cli_errors.run_guarded(app_callable: Callable[[], object]) -> None`

- [ ] **Step 1: Write the failing test.**

```python
# tests/unit/test_cli_errors.py
import pytest

from evaluatorq.common.cli_errors import run_guarded


def test_unexpected_exception_becomes_exit_1(capsys):
    def boom():
        raise RuntimeError('kaboom')

    with pytest.raises(SystemExit) as exc_info:
        run_guarded(boom)
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert 'kaboom' in err
    assert 'EQ_DEBUG=1' in err


def test_systemexit_code_preserved():
    def usage_error():
        raise SystemExit(2)

    with pytest.raises(SystemExit) as exc_info:
        run_guarded(usage_error)
    assert exc_info.value.code == 2


def test_keyboardinterrupt_propagates():
    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_guarded(interrupted)


def test_debug_env_reraises(monkeypatch):
    monkeypatch.setenv('EQ_DEBUG', '1')

    def boom():
        raise RuntimeError('kaboom')

    with pytest.raises(RuntimeError):
        run_guarded(boom)
```

Also add an end-to-end test that exercises the real entrypoint (`main()` wires
`run_guarded(app)`), so the feature has regression coverage beyond the isolated
function. Register a throwaway command that raises, then drive it through `main()`
with patched argv:

```python
def test_main_converts_unexpected_error(monkeypatch, capsys):
    import typer

    from evaluatorq import cli as cli_module

    @cli_module.app.command('boom-test', hidden=True)
    def _boom() -> None:  # pragma: no cover - registered for the test only
        raise RuntimeError('e2e kaboom')

    monkeypatch.setattr('sys.argv', ['evaluatorq', 'boom-test'])
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 1
    assert 'e2e kaboom' in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run pytest tests/unit/test_cli_errors.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the shared error module.**

```python
# src/evaluatorq/common/cli_errors.py
"""Shared CLI error helpers for evaluatorq."""

from __future__ import annotations

import os
from collections.abc import Callable  # NOT typing.Callable — repo enforces UP035

import typer

_ISSUES_URL = 'https://github.com/orq-ai/evaluatorq/issues'


def emit_error(exc: object) -> None:
    """Print a one-line ``Error: ...`` to stderr (shared CLI error format).

    Uses ``typer.echo(err=True)`` to match the stderr convention every other
    line in these CLIs already uses (Click stream handling, color-stripping).
    """
    typer.echo(f'Error: {exc}', err=True)


def run_guarded(app_callable: Callable[[], object]) -> None:
    """Run a Typer app, turning unexpected exceptions into a clean one-liner.

    Click's ``main(standalone_mode=True)`` already converts usage/click errors
    (and Ctrl-C, which it turns into exit 1) to ``SystemExit`` inside the app
    call, so anything reaching here is either control flow to re-raise
    (``SystemExit``/``KeyboardInterrupt``) or a genuinely unexpected exception.
    Unexpected exceptions print a one-line message plus a debug/report pointer
    and exit 1; set ``EQ_DEBUG=1`` to see the full traceback instead.

    Note: the ``redteam run`` command keeps its own ``KeyboardInterrupt`` →
    exit 130 handler; this wrapper does not override it.
    """
    try:
        app_callable()
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:  # noqa: BLE001 — this is the top-level backstop
        if os.environ.get('EQ_DEBUG'):
            raise
        emit_error(exc)
        typer.echo(f'Set EQ_DEBUG=1 to see the full traceback; report at {_ISSUES_URL}', err=True)
        raise SystemExit(1) from exc
```

- [ ] **Step 4: Wrap `app()` in `main()`.**

In `cli.py`, change the `app()` call at the end of `main()`:

```python
    from evaluatorq.common.cli_errors import run_guarded

    run_guarded(app)
```

- [ ] **Step 5: Fold sim's known-error handler onto `emit_error`.**

In `simulation/cli.py`, change `_handle_cli_error` to reuse the shared emitter:

```python
from evaluatorq.common.cli_errors import emit_error


def _handle_cli_error(exc: Exception) -> NoReturn:
    emit_error(exc)
    raise typer.Exit(1) from None
```

- [ ] **Step 6: Run tests + full suite.**

Run: `uv run pytest tests/unit/test_cli_errors.py -v`
Expected: PASS
Run: `uv run pytest -m 'not integration' -q`
Expected: green (confirms exit-code behavior of existing CLI tests is unchanged).

- [ ] **Step 7: Commit.**

```bash
git add src/evaluatorq/common/cli_errors.py src/evaluatorq/cli.py src/evaluatorq/simulation/cli.py tests/unit/test_cli_errors.py
git commit -m "feat(cli): add global unexpected-error handler with EQ_DEBUG traceback"
```

---

## Task 8: Stop silently hiding sub-commands (item 3)

**Files:**
- Modify: `src/evaluatorq/cli.py:130-146` (`main`)
- Modify: `src/evaluatorq/redteam/__init__.py:17-31` (remove dead dep check)
- Test: `tests/unit/test_cli_subapps.py`

**Interfaces:**
- Consumes: `run_guarded` (Task 7).
- Produces: `evaluatorq.cli._register_subapps(app: typer.Typer) -> None`

**Design note (decision B):** `openai`/`typer` are now **core** deps, so
`import evaluatorq.redteam.cli` / `...simulation.cli` always succeeds on a normal
install — the "missing extra" `ImportError` path the ticket imagined is
unreachable. The only thing the old `except ImportError: pass` can catch today is a
**genuinely broken install**, which it silently swallows (the subcommand vanishes
with no hint). Per decision B we **drop the swallow entirely and import directly**,
so a broken install surfaces (routed through `run_guarded` for a clean one-liner +
`EQ_DEBUG` traceback) instead of hiding. This satisfies item 3 ("stop silently
hiding subcommands") without a stub whose install hint would be misleading (it
would fire only on broken installs, never on a real missing extra). The dead
`_check_redteam_deps()` (it checks core deps) is removed as the spec directed.

- [ ] **Step 1: Confirm imports succeed on a normal install.**

Run: `uv run python -c "import evaluatorq.redteam.cli, evaluatorq.simulation.cli; print('ok')"`
Expected: `ok` — proves the ImportError path is not exercised on a good install, so swallowing it only ever hides breakage.

- [ ] **Step 2: Write the failing test.**

```python
# tests/unit/test_cli_subapps.py
import typer
from typer.testing import CliRunner

from evaluatorq import cli as cli_module


def test_subapps_are_registered():
    app = typer.Typer()
    cli_module._register_subapps(app)
    result = CliRunner().invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'redteam' in result.output
    assert 'sim' in result.output
```

- [ ] **Step 3: Run test to verify it fails.**

Run: `uv run pytest tests/unit/test_cli_subapps.py -v`
Expected: FAIL — `_register_subapps` does not exist yet.

- [ ] **Step 4: Replace the swallowing registration with a direct helper.**

In `cli.py`, replace the two `try/except ImportError: pass` blocks in `main()`:

```python
def _register_subapps(app: typer.Typer) -> None:
    """Register the redteam and sim sub-apps.

    Their deps are core, so a failing import here means a broken install — let it
    surface (via run_guarded) rather than silently dropping the subcommand
    (clig.dev: no silent failure).
    """
    from evaluatorq.redteam.cli import app as redteam_app
    from evaluatorq.simulation.cli import app as sim_app

    app.add_typer(redteam_app, name='redteam', help='Red teaming commands.')
    app.add_typer(sim_app, name='sim', help='Agent simulation pipeline.')


def main() -> None:
    """Entry point that assembles sub-commands and runs the CLI under a guard."""
    from evaluatorq.common.cli_errors import run_guarded

    def _run() -> None:
        _register_subapps(app)
        app()

    run_guarded(_run)
```

- [ ] **Step 5: Remove the dead dep check.**

In `redteam/__init__.py`, delete the `_check_redteam_deps()` definition (lines
~17-29) **and** its module-level call (`_check_redteam_deps()`, ~line 31). It only
verifies `openai`/`typer`, which are core deps and always importable — so it can
never fire, and removing it keeps `import evaluatorq.redteam` free of dead guards.

- [ ] **Step 6: Run the test + happy-path check.**

Run: `uv run pytest tests/unit/test_cli_subapps.py -v`
Expected: PASS
Run: `uv run evaluatorq --help`
Expected: `redteam` and `sim` listed as normal commands.
Run: `uv run pytest -m 'not integration' -q`
Expected: green.

- [ ] **Step 7: Commit.**

```bash
git add src/evaluatorq/cli.py src/evaluatorq/redteam/__init__.py tests/unit/test_cli_subapps.py
git commit -m "fix(cli): surface broken sub-app imports instead of hiding them; drop dead dep check"
```

---

## Task 9: Final gate

**Files:** none (verification only).

- [ ] **Step 1: Lint.**

Run: `uv run ruff check src`
Expected: no errors. Fix any (e.g. import ordering) and amend the relevant commit.

- [ ] **Step 2: Type check.**

Run: `uv run basedpyright`
Expected: no new errors introduced by these changes.

- [ ] **Step 3: Full unit suite.**

Run: `uv run pytest -m 'not integration'`
Expected: all green.

- [ ] **Step 4: Ticket acceptance smoke checks.**

```bash
uv run evaluatorq -h
uv run evaluatorq redteam run -h
uv run evaluatorq redteam runs --json | python -m json.tool
echo | uv run evaluatorq redteam run -t agent:x --save none   # must not hang
uv run evaluatorq redteam run --mode bogus                     # exit 2, lists choices
```

Expected: all behave per the ticket's verification list.

- [ ] **Step 5: Confirm the "without extras" criterion is not applicable.**

The ticket lists "without extras installed, `evaluatorq --help` still explains how
to enable redteam/sim." With `openai`/`typer` now **core** deps, the sub-app
imports always succeed, so `redteam`/`sim` are always listed and there is no
"extras missing" state to explain. Record this in the PR/ticket as *resolved-moot
by the core-deps change*, not silently dropped. (Item 3's real fix is: broken
installs now surface instead of hiding — Task 8.)

---

## Self-Review notes

- **Spec coverage:** items 1 (Task 2), 2 (Task 4), 3 (Task 8 — decision B: surface broken imports; missing-extra is unreachable with core deps, so no stub), 4 (Task 1), 5 (Task 6, with `report_id` for dashboard deep-link correlation), 6 (Task 7), 7 (Task 3), 8+9 (Task 5), dropped 11 (n/a), doc-fix D (Task 1). All ten in-scope items mapped.
- **Type consistency:** helper names used across tasks — `CONTEXT_SETTINGS`, `should_skip_confirm`, `examples`, `echo_json`, `emit_error`/`run_guarded`, `_register_subapps` — are defined before first use. `Callable` is imported from `collections.abc` (repo enforces UP035).
- **Ordering:** Task 8 depends on Task 7 (`run_guarded`); Task 5 defines `examples` used only within its own task; do Task 7 before Task 8.
- **Reuse:** `report_id` (Task 6) reuses `dashboard/library.py` (stdlib-only at import); `emit_error` uses `typer.echo(err=True)` matching repo convention; no bare `print` to stderr.
