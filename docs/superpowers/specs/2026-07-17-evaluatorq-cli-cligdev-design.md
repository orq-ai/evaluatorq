# evaluatorq CLI — clig.dev alignment (RES-1117)

**Status:** design approved, ready for implementation plan **Ticket:** [RES-1117](https://linear.app/orqai/issue/RES-1117) **Branch:** `bauke/res-1117-evaluatorq-cli-align-with-cligdev-command-line-interface`

## Goal

Raise `evaluatorq` / `eq` CLI clig.dev compliance from ~7.5/10 to ~9/10. All changes additive and non-breaking. No major version bump. CLI surface only.

## Scope

**In:** `src/evaluatorq/cli.py`, `src/evaluatorq/redteam/cli.py`, `src/evaluatorq/redteam/hooks.py`, `src/evaluatorq/simulation/cli.py`, `src/evaluatorq/common/` (new shared helpers), one docs line, unit tests.

**Ten items** = ticket P1+P2+P3 **minus** P3.10 (`.env` autoload — rejected as an unexpected input surface; the doc-contradiction half is fixed instead, see item D).

**Out (deferred to a separate ticket, P4):** `--dry-run`; removal of deprecated `ui` / `validate-dataset` aliases (breaking — needs a major).

This design was hardened by a four-critic adversarial review (`/hate`). Several ticket items rested on assumptions the code disproves; the corrected versions are below. Where the fix belongs in `common/` rather than duplicated per-CLI, that is called out — sim and redteam already diverge and this pass consolidates rather than widens the gap.

---

## Items

### P1

**1. `-h` short help alias.** Set `context_settings={'help_option_names': ['-h', '--help']}`. Define the dict **once** as a shared constant (e.g. `common/cli_help.py::HELP_OPTION_NAMES` or an inline shared dict) rather than three literal copies.
- First verify empirically whether a root-only setting inherits to the `redteam` / `sim` sub-apps via Click's parent-context fallback (`uv run eq redteam -h`, `uv run eq sim -h`). If root-only suffices, set it once on the root `app` (`cli.py:26`) and drop the sub-app edits. If not, apply the shared constant to all three `typer.Typer()` sites.
- `dashboard` is a command on the root app, so it inherits the root setting; no separate edit.
- No existing `-h` collision anywhere (verified).

**2. TTY guard on the redteam confirm.** `redteam/cli.py:443` passes `RichHooks(skip_confirm=yes)` with no TTY check; a non-TTY invocation (CI, pipe) that forgets `--yes` blocks on `typer.confirm` (`hooks.py:430`). Mirror sim (`simulation/cli.py:612`) verbatim: `skip_confirm=yes or not sys.stdin.isatty()`. `sys` already imported.
- **Behavior change (accepted):** non-TTY now auto-proceeds — fail-open — instead of hanging. This matches sim and the ticket's explicit call. Noted that redteam attacks live targets, so auto-proceed carries more weight than in sim; decision stands (mirror sim) per ticket.

**3. Stop silently dropping sub-commands on ImportError — corrected.** `cli.py:136,143` wrap sub-app registration in `except ImportError: pass`. The ticket's "distinguish missing-extra by module name" mechanism is **infeasible as written and no longer needed:** `openai` and `typer` are now **core** dependencies (not extras), and `redteam/__init__.py::_check_redteam_deps()` raises a hand-built `ImportError` whose `.name` is `None` — name-based discrimination cannot work. With the deps core, a bare `ImportError` escaping the sub-app import now almost always means a **genuinely broken install**, not a missing extra.
- **Implementation (investigate-first, then typed-catch):**
  1. Reproduce: determine what, if anything, actually raises `ImportError` from
     `import evaluatorq.redteam.cli` / `...simulation.cli` today given the deps are
     core (e.g. uninstall a real optional dep like `huggingface-hub` and observe).
  2. If a legitimate "extra not installed" path survives, have the dep-check raise
     a dedicated subclass (e.g. `RedteamExtraMissing(ImportError)`) and **catch it
     by class** in `cli.py` — register a stub command that prints the install hint
     and exits 1.
  3. For any **other** `ImportError` (broken install), **re-raise** — never mask it
     behind an install hint. This is the actual bug being fixed: today's bare
     `except ImportError: pass` hides real breakage.
- `_check_redteam_deps()` checking core deps is now effectively dead; clean it up or repoint it at the real optional deps as part of this item.

**4. Strip Sphinx `:command:` markup leak.** `simulation/cli.py:1330` docstring `Deprecated alias for :command:\`validate\`` renders literally. Remove the role. One-line, isolated (only occurrence).

### P2

**5. `--json` on `redteam runs` + `sim runs` (listing commands) — corrected.** clig: machine-readable output where it doesn't hurt usability. **Serialize the raw underlying fields, not the display row dicts.** The existing row dicts hold human-formatted, lossy values (`asr` as `"42%"`, timestamps truncated/tz-stripped, missing fields as the literal string `'—'`, sim's pre-formatted `scores`). Dumping those is a table with `.json()` slapped on — hostile to `jq`. Emit the pre-format `data.get(...)` values (numbers as numbers, ISO-8601 timestamps, `null` for missing).
- Route all non-JSON chatter to stderr (or suppress) under `--json` so stdout is pure JSON: redteam no-runs `typer.echo` (`cli.py:671,676`), skipped-file warning `console.print` (`cli.py:728`); sim equivalents (`cli.py:1429,1439`).
- Empty runs dir under `--json` → emit `[]` on stdout, exit 0 (not a bare text message). Skipped/malformed files → count to stderr, keep JSON parseable.
- Rich table remains the default (no flag).
- Factor the serialization through one shared `common/` helper (e.g. `common/cli_json.py::echo_json(obj)` → `json.dumps(..., default=str)` to stdout) so redteam and sim don't each grow their own copy.
- Verify: `uv run eq redteam runs --json | python -m json.tool` parses.

**6. Global unexpected-error handler — wrap `app()` once.** Wrap the single `app()` call in `main()` (`cli.py:146`), **not** per-command. Click's `main(standalone_mode=True)` already converts `ClickException` / `Abort` / usage errors to `sys.exit()` inside `app()`, so anything reaching the outer wrapper is either a genuinely-unexpected exception (catch it) or `SystemExit` / `KeyboardInterrupt` (re-raise / let propagate — preserves exit codes 0/1/2/130). The ticket's `typer.Exit` re-raise list is unnecessary at this layer (Click already handled it).
- On an unexpected exception: print one line to stderr — `Error: <msg>` + `Set EQ_DEBUG=1 to see the full traceback; report at https://github.com/orq-ai/evaluatorq/issues` — and exit 1.
- Full traceback shown when `EQ_DEBUG=1` (env var, **command-agnostic**; `-v` was rejected because it exists only on `redteam run` and three sim commands, so "re-run with -v" would be false advice on `dashboard`, `runs`, etc.).
- Consolidate with sim's existing `_handle_cli_error` / `_clean_cli_error_types` (`simulation/cli.py:202`): move the shared shape into `common/` and have both CLIs use it, rather than adding a third bespoke error path.

**7. `--mode` → enum on `redteam run` — reuse existing `Pipeline`.** Do **not** define a new enum. `redteam/contracts.py:266` already defines `Pipeline(StrEnum)` = `static` / `dynamic` / `hybrid`, and `runner.py` already coerces via `Pipeline(mode)` (raising `ValueError` on bad input, already caught). Change the param at `redteam/cli.py:195` from `mode: Annotated[str, ...] = 'dynamic'` to `mode: Annotated[Pipeline, ...] = Pipeline.DYNAMIC` — mirroring the `SaveMode` pattern already in this file (`cli.py:304`). This is a **UX** fix (Typer renders choices in `--help` and rejects typos early with a list); the underlying validation already existed.
- Verify: `uv run eq redteam run --mode bogus` fails fast listing valid choices.

### P3

**8. Examples epilog on `redteam run` + `dashboard` — reuse, don't copy.** Extract sim's `_examples()` builder (`simulation/cli.py:393-406`, dependency-free rich-markup formatter) into `common/` (e.g. `common/cli_epilog.py`) and import from sim, redteam, and root. Do **not** copy-paste it into `redteam/cli.py`. Add 1–2 concise example invocations for `redteam run` and `dashboard`.

**9. Docs/support link in root epilog.** Add an epilog to the root `app`: docs URL + `https://github.com/orq-ai/evaluatorq/issues` (the repo already exposes a `Documentation` URL in `pyproject.toml`).

**(dropped) 11. Announce auto-save path.** **Dropped — already exists.** `redteam/runner.py::_auto_save_run` persists to `.evaluatorq/runs/{name}_{timestamp}.json` and `RichHooks.on_complete` (`hooks.py:760`) already prints a path-referencing tip. The ticket's `<id>` filename is also wrong. Adding a second "Saved run to …" line would duplicate existing output. No action.

### Docs (cheap half of dropped P3.10)

**D. Fix the live doc contradiction.** `docs/orq-deployment.md:25` states `ORQ_API_KEY … Set in environment or .env file.` while `docs/configuration.md:28` correctly says the library never calls `load_dotenv()`. Since we are **not** adding autoload, correct the deployment doc line to match `configuration.md` (point users at calling `load_dotenv()` themselves). One line; closes the exact gap that justified P3.10.

---

## Shared `common/` helpers introduced

Consolidation is a first-class goal of this pass (sim and redteam already drift):

| Helper | Purpose | Consumers |
| --- | --- | --- |
| `HELP_OPTION_NAMES` (item 1) | `['-h','--help']` context-settings dict | root + (maybe) sub-apps |
| `echo_json(obj)` (item 5) | canonical JSON → stdout | redteam `runs`, sim `runs` |
| shared error handler (item 6) | one-line error + `EQ_DEBUG` traceback | root wrapper; folds sim's `_handle_cli_error` |
| `_examples()` moved to common (item 8) | epilog builder | root, redteam, sim |

Only introduce a helper where ≥2 call sites use it; a single-use extraction is not worth the indirection.

## Testing

Unit tests (`tests/redteam/`, `tests/unit/` as appropriate):
- `-h` shows help on root and `redteam run` (item 1).
- `redteam runs --json` and `sim runs --json` emit valid JSON parseable by `json.loads`; raw numeric/timestamp fields, `null` for missing; empty dir → `[]` on stdout, exit 0; no Rich table on stdout (item 5).
- Non-TTY confirm path: `sys.stdin.isatty()` mocked `False`, `yes=False` → redteam run proceeds without blocking (item 2); mirror sim's existing test.
- Missing-extra stub message vs broken-import re-raise (item 3) — assert the two branches are distinguishable (subclass-caught vs re-raised).
- `--mode bogus` exits with usage error listing valid choices (item 7).
- Global handler: unexpected `ValueError` in a command → exit 1 + one-line message (no traceback); `EQ_DEBUG=1` → traceback shown; `KeyboardInterrupt` → 130; `typer.Exit(2)` preserved (item 6).

Gates (must be green): `uv run pytest -m 'not integration'`, `uv run ruff check src`, `uv run basedpyright`.

## Verification (ticket acceptance)

- `uv run evaluatorq -h` and `uv run evaluatorq redteam run -h` show help.
- `uv run evaluatorq redteam runs --json | python -m json.tool` parses.
- `echo | uv run evaluatorq redteam run -t agent:x --save none` does not hang.
- `uv run evaluatorq redteam run --mode bogus` fails fast listing valid choices.
- Broken sub-app import surfaces the real error (not an install hint); a genuine missing extra, if reachable, shows the install hint.

## Sequencing (smallest-diff first)

1. Item 4 (docstring) + item D (doc line) — trivial, isolated.
2. Item 1 (`-h`) — verify inheritance, then shared constant.
3. Item 7 (`Pipeline` enum) — one-line type change.
4. Item 2 (TTY guard) — one-line, mirrors sim.
5. Item 8 (`_examples` → common) + item 9 (root epilog).
6. Item 5 (`--json` + `echo_json` helper).
7. Item 6 (error handler + fold sim's `_handle_cli_error`).
8. Item 3 (ImportError — investigate, then typed-catch) — last, needs reproduction.
