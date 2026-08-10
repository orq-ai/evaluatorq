---
name: docs-drift
description: Use when checking whether evaluatorq docs still match the code — verifying CLI flag tables, code examples, imported symbols, defaults, enum/registry members and env vars against current source. Triggers on "check the docs", "are the docs still right", "docs drift", "docs out of date", and before opening a PR that changes public surface (`__all__`, CLI flags, env vars, registry members).
---

# docs-drift

Docs assert facts. This skill verifies each fact against current source and reports
what no longer holds.

**Truth-first, not diff-first.** A wrong flag default from six months ago is still
wrong, and no git range will surface it. Verify everything; use the diff only to
order the work.

Companion skill: `docs-coverage` answers the opposite question — what the code does
that no page documents.

## Scope

**In:** `docs/**/*.md` (hand-written), `README.md`, `CLAUDE.md`.

**Out, and why:**

| Excluded | Owned by |
|---|---|
| `docs/reference/**` (API pages) | generated from `__all__` by `docs/gen_pages.py` — cannot drift |
| `examples/` | `scripts/check_examples.py` in CI — executable code is checked by running it, not by reading it |
| nav / links / llms.txt parity | `mkdocs build --strict` + `docs/hooks.py` |
| mermaid label defects | `scripts/validate_mermaid.py` |
| `docs/superpowers/` | historical plans and specs; snapshots of intent, not claims about today |

## Step 0 — env guard (do not skip)

```bash
uv sync --all-extras --all-groups
uv run eq --help >/dev/null && uv run eq redteam --help >/dev/null && echo "ENV OK"
```

`eq redteam` registers only when the `redteam` extra is installed. Without it every
red-team flag looks deleted and this skill will confidently "fix" correct docs into
wrong ones. **If `ENV OK` does not print, stop and report the env problem.** Never
report findings from a partial install.

## Step 1 — collect claims

Read the in-scope files and extract every falsifiable assertion. Five classes, in
descending order of yield:

1. **Symbols** — every `from evaluatorq... import X`, `eq.foo(...)`, `evaluatorq.Bar`
   in prose or fenced blocks.
2. **CLI flags** — every row of every flag table in `docs/cli-reference/*.md`:
   flag name, short alias, type, default.
3. **Signatures** — kwargs passed in doc examples; required params that examples omit.
4. **Membership** — vulnerability IDs, strategy names, delivery methods, backend
   keys, env var names mentioned anywhere in prose.
5. **Behavioral prose** — "defaults to X", "runs concurrently", "only when
   `evaluatorq[redteam]` is installed", "takes precedence over".

## Step 2 — establish ground truth

Hybrid. Execute where a command exists; read source where it doesn't.

**CLI flags — execute.** `--help` is authoritative and prints defaults; extracting
Typer defaults statically is guesswork.

```bash
uv run eq redteam run --help
uv run eq simulate --help
uv run eq --help          # enumerate command groups first
```

**Symbols and signatures — execute.**

```bash
uv run python -c "
import inspect, evaluatorq
print(inspect.signature(evaluatorq.red_team))
"
```

**Membership and prose — read source.** Registries under
`src/evaluatorq/redteam/*_registry.py`, enums in `contracts.py`, env vars via
`grep -rn 'environ\|getenv' src`.

## Step 3 — classify each finding

| Confidence | Means | Action |
|---|---|---|
| **mechanical** | one correct value, verified by execution — wrong default, renamed flag, dead import path | **auto-apply** |
| **judgement** | needs a decision about what to *say* — behavioral prose, restructured explanation | report, ask, then apply |
| **removal** | a documented thing appears not to exist | **propose only, never apply** |

Removal is never automatic. An absent symbol is more often a bad check — a lazy
import, an optional extra, a re-export — than a real deletion. Deleting correct
docs is the one failure mode of this skill that is expensive to undo.

Class-5 behavioral findings are inherently noisy: an LLM reading code to judge
whether "defaults to X" still holds is the weakest check here. Always mark them
low-confidence. Never auto-apply them.

## Step 4 — report

Write `.context/docs-drift-report.md`:

```markdown
# docs-drift — <date>

Env: OK · Scope: full sweep | diff <range>

## Auto-applied (N)
- `docs/cli-reference/redteam.md:24` — `--parallelism` default `5` → `10`
  (verified: `eq redteam run --help`)

## Needs a decision (N)
- `docs/guides/red-teaming.md:88` — "runs sequentially"; `runner.py:142` fans out
  under `--parallelism`. Proposed rewrite: ...

## Proposed removals — NOT applied (N)
- `docs/configuration.md:12` — `EVALUATORQ_CACHE_DIR` not found in source.
  Confirm it is gone before deleting.
```

Then summarise to the user and ask before touching anything in the last two sections.

`.context/` is gitignored — reports never reach the published site.

## Diff-scoped mode

Default is the full sweep. For PR use, take a git range and verify only claims
touching symbols, flags, env vars or registry members that the diff changes:

```bash
git diff --name-only origin/main... -- src
git diff origin/main... -- src | grep -E '^[+-].*(__all__|@app\.|Option\(|environ)'
```

Same steps, smaller claim set. Step 0 still applies.
