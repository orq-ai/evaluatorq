---
name: docs-autofill
description: Weekly unattended run that finds the top docs coverage gap, writes the page, runs every code block in it, has it reviewed by adversarial critics and roleplaying users, and opens a PR. Triggers on "docs autofill", "fill a docs gap", and from the docs-autofill cloud routine.
---

# docs-autofill

One gap per run, or no PR at all. A week with nothing worth writing is the
expected outcome, not a failure.

Runs unattended in a cloud session. Every decision that would normally be a
question to the user has to be a rule here instead.

## Step 0 — env guard

```bash
uv sync --all-extras --all-groups
uv run eq --help >/dev/null && uv run eq redteam --help >/dev/null && echo "ENV OK"
```

No `ENV OK` → stop, Slack the failure (step 9), open nothing. Without the
`redteam` extra whole axes vanish and the matrix reports phantom gaps, so a
run on a broken env produces a confidently wrong PR.

Also confirm keys are present:

```bash
[ -n "$ORQ_API_KEY" ] && [ -n "$OPENAI_API_KEY" ] && echo "KEYS OK"
```

No `KEYS OK` → keep going, but step 5 drops to the no-key tier and the PR body
says so.

## Step 1 — don't stack on yourself

```bash
gh pr list --repo orq-ai/evaluatorq --label docs-autofill --state open --json number,title,headRefName
```

- ≥2 open → stop. Slack one line saying the queue is full and name the PRs.
  Unreviewed docs PRs pile up faster than anyone reads them.
- 1 open → note the gap it fills; that gap is off the table this run.

Create the label once if it does not exist:
`gh label create docs-autofill --color 0e8a16 --description "Docs gap filled by the weekly autofill routine" 2>/dev/null || true`

Also `git pull` so the matrix is computed against current `main`.

## Step 2 — find the gap

Read `.claude/skills/docs-coverage/SKILL.md` and run it. It writes
`.context/docs-coverage-matrix.md`.

Pick **one** gap:

1. Tier 1 only. Tier 2 gaps do not justify an unattended PR.
2. Skip any gap already covered by an open `docs-autofill` PR (step 1).
3. Among the rest, prefer the gap whose usage path a user is most likely to hit
   first — an entry point or CLI command over a niche axis crossing.

**No qualifying Tier 1 gap → write no PR.** Slack the no-op line (step 9) with
what you checked, and end the run. Do not downgrade to a Tier 2 gap to have
something to show.

The docs-coverage skill says to ask the user before writing. That instruction
is superseded here — the gap-selection rules above are the standing answer, and
the PR is the place the decision gets reviewed.

## Step 3 — write it

Branch: `docs/autofill-<short-gap-slug>`.

Scope: one page, or one section on an existing page. Not two. If the gap is
genuinely too large for one page, write the page that covers the most common
half and say in the PR body what you deliberately left out.

Rules that are not negotiable, all from `CLAUDE.md` and the docs skills:

- **A new page goes in `mkdocs.yml` twice** — under `nav:` and under
  `plugins.llmstxt.sections`. The build fails on the mismatch, but fix it at
  write time rather than discovering it in step 6.
- **Every code sample is fenced with a language.** No indented blocks, no bare
  `Example::`.
- **Every code sample runs.** If you cannot make it run, it does not go in the
  page. A sample that needs a key it does not have is not "runnable" — cut it or
  reduce it to the part that does run.
- If a cell resolved `N/A` for a reason not yet listed, add it to
  `.claude/skills/docs-coverage/axes.md` in the same branch. That is how the
  false-gap list stays useful.
- Do not touch `src/` beyond fixing a docstring fence that the build rejects.
- No drive-by edits to unrelated pages.

## Step 4 — the receipt

Extract every fenced block from the new/changed docs into `.context/` and run
each one. Keep a receipt table — command, exit code, first ~5 lines of output —
you will paste it into the PR body verbatim. A docs PR whose examples were never
executed is the exact thing this routine exists to stop shipping.

Bash blocks run as-is. Python blocks run with `uv run python`. A block that is
deliberately illustrative (a directory tree, a config excerpt, a truncated
response) is exempt — mark it `-` in the receipt with a one-word reason, do not
silently omit the row.

If `KEYS OK` did not print in step 0: run the blocks that need no network, and
mark the rest `SKIPPED (no key)` in the receipt. Say it in the PR body too.

## Step 5 — validate

```bash
uv run --group docs mkdocs build --strict
uv run python scripts/validate_mermaid.py
```

If anything under `src/` changed, also run CI's four, verbatim:

```bash
uv run ruff check src
uv run ruff format --check src
uv run basedpyright
uv run pytest -m 'not integration'
```

`basedpyright` is never scoped to a path. Everything must be green before review.

## Step 6 — review, in parallel

Two review tracks, dispatched together, sonnet for all of them. Wait for both
before revising.

### Track A — adversarial critique

Read the hate skill and follow it, in `--apply --sonnet` mode, targeting the
branch diff:

The skill lives in the `orq-ai/research` checkout, not this one, so locate it
rather than assuming a path:

```bash
find .. "$HOME/.claude/skills" -path '*skills/hate/SKILL.md' -print -quit 2>/dev/null
```

Not found → skip track A, say so in the PR body in one line, and run track B
alone. A missing critic is not a reason to skip the personas.

Two overrides for unattended use:

- Its "ask, then STOP" branch does not apply — `--apply` mode is in force, so
  Recommendations get applied and Decisions stay open.
- Its Linear-comment step is replaced by the PR body. There is no ticket.

### Track B — roleplaying users

Four agents. Each gets a task **derived from the gap being filled**, not a
generic one — the point is to test whether the new page makes *its own* usage
path achievable. Write the four tasks yourself from the gap, one per persona,
phrased as something a person wants to accomplish, never as "review this page".

| Persona | Angle the task should take |
|---|---|
| First-timer | Never used evaluatorq. Task starts from zero, arrives at the docs root. |
| Platform user | Works through Orq — a `deployment:` target, no local model code. |
| Practitioner | Already uses the surface the gap belongs to; wants this specific path against their own target. |
| CI engineer | Needs it non-interactive in a GitHub Action — exit codes, env vars, no prompts. |

Rules given to every persona agent:

- You are a user, not a reviewer. You have the docs site and a terminal. You have
  **not** seen the diff and must not read it.
- Start from `docs/index.md` and navigate. If you cannot find the page, that is
  the finding — say so and stop.
- **Run the commands. Do not read them and assume.** Paste what actually happened.
- Report: where you got stuck, what you had to guess, what sent you to the source,
  and what you expected the page to say that it did not.
- End with one line: COMPLETED / COMPLETED WITH GUESSWORK / BLOCKED.

Their finding class is the one the critics structurally cannot produce: *the page
is correct and I still could not do it.*

## Step 7 — revise, once

Apply the hate Recommendations and every persona finding that is a real
navigability or completeness defect. One round only — docs quality does not
improve on round three, cost does.

Re-run steps 4 and 5 after revising. The receipt in the PR is the post-revision
one.

Anything you did not apply goes in the PR body under **Not addressed**, with the
reason. A persona that ended BLOCKED and was not unblocked must be called out in
the PR title — that is a PR the human should read first.

## Step 8 — PR

Conventional commit, `docs:` type. Never `feat!`/`fix!`.

```bash
gh pr create --assignee @me --label docs-autofill --base main
```

CODEOWNERS requests reviewers; do not pass `--reviewer`. Never merge.

PR body, in this order:

1. **The gap** — the matrix cell, quoted, and why it ranked first.
2. **What the page covers**, and what it deliberately does not.
3. **Receipt** — the step 4 table.
4. **Persona verdicts** — one line each, persona → task → verdict.
5. **Open decisions** — the hate Decisions, unresolved, each with its Lean.
6. **Not addressed** — findings skipped, with reasons.
7. One line noting this was opened by the weekly docs-autofill routine.

## Step 9 — Slack

Channel `research-assistant` (`C0BNKRH9FGR`) via `slack_send_message`.
**Standard Markdown, not Slack mrkdwn** — the tool converts it itself. Links as
`[#123](url)`, bold as `**bold**`, bullets as `• `.

Post exactly one message per run, in every case:

- **PR opened:** the gap in one clause, the PR link, persona verdict counts, and
  the number of open decisions.
- **No-op:** one line — what was checked, why no gap cleared the bar.
- **Aborted:** env guard failed, or the PR queue was full. Name the cause.

A cloud run that dies silently is worse than one that fails loudly. If the
channel post fails with `not_in_channel` or `channel_not_found`, DM
`U09BR0B0Q7P` and say the channel post failed.

## Caps

One gap. One revision round. Nine agents (4 critics + 4 personas + 1 author).
Never merge. Never touch `src/` beyond a docstring fence.
