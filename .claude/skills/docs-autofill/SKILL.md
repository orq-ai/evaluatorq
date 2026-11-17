---
name: docs-autofill
description: Weekly unattended run that finds the top docs coverage gap, writes the page, runs every code block in it, has it reviewed by adversarial critics and roleplaying users until they can complete the task, and opens a PR. Triggers on "docs autofill", "fill a docs gap", and from the docs-autofill cloud routine.
---

# docs-autofill

One gap per run, or no PR at all. A week with nothing worth writing is the expected outcome, not a failure.

Runs unattended in a cloud session. Every decision that would normally be a question to the user has to be a rule here instead.

The routine's own config — cron, model, environment, sources, tools — is committed next to this file as `routine.json`. It is applied by hand, so a change there is not live until someone asks Claude to push it via the `RemoteTrigger` API. Say which you changed in the PR body.

## Step 0 — where am I, and what keys do we have

The session starts in `/home/user`, not in a checkout. `cd` into the evaluatorq one before anything else; every relative path in this file is relative to it.

```bash
cd /home/user/evaluatorq 2>/dev/null || cd "$(git rev-parse --show-toplevel)"
```

```bash
for k in ORQ_API_KEY OPENAI_API_KEY; do
  [ -n "${!k}" ] && echo "HAVE $k" || echo "MISSING $k"
done
```

This is an inventory, not a gate. It feeds step 2: a gap whose examples cannot run without a key we do not have is not eligible this week. There is no "documented but unverified" tier — a page whose central example never executed is the exact artifact this routine exists to stop shipping.

The environment guard proper lives in `docs-coverage` step 0 and runs at step 2. Do not repeat it here.

**Say what this run will spend and touch, before it spends it.** Docs examples here are live: a red-team page's blocks make real API calls and upload experiment records to the Orq workspace, and steps 4/6/7 multiply that by rounds and personas. State the estimate — roughly how many live runs, against which workspace — in `.context/` at step 0 and in the PR body. Outward side effects a reader discovers for themselves were undisclosed, not disclosed.

Note what `ORQ_WORKSPACE` does and does not do: it is a **display** setting, read only by `dashboard/` to build trace deep-links. It does not route anything. Which workspace a run's records land in is decided by `ORQ_API_KEY` alone, so the only way to isolate this routine's output is to give `routine.json` a key for a workspace kept for it. That is worth doing, and it is **not a gate** — a run against the normal key is allowed, it just has to say so in the disclosure above.

## Step 1 — the ledger

`ledger.md`, next to this file, is the only memory this routine has. One row per attempt: date, matrix cell, branch, outcome.

Read it from `main` **and** from every unmerged autofill branch — an attempt that was rejected never merged its own row, and that is exactly the attempt you most need to remember:

```bash
git fetch --prune origin
git pull --ff-only
git show origin/main:.claude/skills/docs-autofill/ledger.md
for b in $(git branch -r --list 'origin/docs/autofill-*' --format='%(refname:short)'); do
  git show "$b:.claude/skills/docs-autofill/ledger.md" 2>/dev/null
done
```

Every cell that appears in any of those is off the table, **except a `[remainder]` cell that only has `follow-up` rows** (step 3): that is a gap an earlier run deliberately left for later, so it is eligible. Eligibility is a set difference with no ordering: a cell string whose every row is `follow-up` is eligible; the run that picks it writes an ordinary row for the same string, and from then on the string is off the table like any other. A gap this routine has already tried is not a gap it gets to try again — if the attempt was wrong, a human closed it, and repeating it weekly is the worst failure mode available.

This routine never inspects pull requests. It creates one. Branch names and the ledger are its whole view of history.

## Step 2 — find the gap

Read `.claude/skills/docs-coverage/SKILL.md` and run it, including its step 0 environment guard. Its abort is this routine's abort: no `ENV OK`, stop and Slack the failure (step 9). Without the `redteam` extra whole axes vanish and the matrix reports phantom gaps, so a run on a broken env produces a confidently wrong PR.

It writes `.context/docs-coverage-matrix.md`. Pick **one** gap:

1. Tier 1 only. Tier 2 gaps do not justify an unattended PR.
2. Not in the ledger (step 1), except an eligible `[remainder]` cell.
3. Every example it needs must be runnable with the keys from step 0. A gap that needs a provider we have no key for is skipped — record it in the ledger with outcome `skipped-no-key` so next week does not re-derive the same dead end.
4. Among the rest, prefer the gap whose usage path a user is most likely to hit first — an entry point or CLI command over a niche axis crossing. A `[remainder]` cell competes on the same terms, and must still be Tier 1; it gets no priority for being old.

**No qualifying gap → write no PR.** Slack the no-op line (step 9) with what you checked, and end the run. Do not downgrade to a Tier 2 gap to have something to show.

The docs-coverage skill says to ask the user before writing. That instruction is superseded here — the rules above are the standing answer, and the PR is where the decision gets reviewed.

## Step 3 — write it

Branch: `docs/autofill-<short-gap-slug>`.

Scope: **one page, or one section on an existing page, plus a bounded set of adjacent edits.** The page is the unit that gets run, receipted and reviewed; adjacent edits are the small, obviously-related things a reader of that page would otherwise trip over. If the gap is genuinely too large for one page, write the page that covers the most common half and record the other half as a follow-up (below).

An **adjacent edit** is allowed when all of these hold:

- It is prose only, with no code block added or changed. That keeps the receipt and the reader personas unchanged: nothing new to run, nothing new to test.
- It is directly tied to the new page: a link from a page that already covers the neighbouring topic, a `see also` row, a nav or `llms.txt` entry, or a factual slip you noticed on a page the new one links to (a wrong flag name, a stale default).
- It fits a rough budget, counted apart from the page: **about 3 files besides the new page and `mkdocs.yml`, and about 60 added lines across them.** `scope_check.sh` (step 5) says how far over or under you are; it advises and never blocks, so you decide, and what you drop goes to a follow-up.
- A factual-slip fix names what it was checked against, a `src/` line or `--help` output, in its **Beyond the page** line. Nothing else executes or reviews that claim.

Not adjacent, whatever the size: anything in `src/`, `.claude/skills/docs-coverage/axes.md` (that one keeps its `[axes]` tag), a rewrite of another page's structure, and anything that needs a human decision (that is an Open decision).

Every adjacent edit is listed in the PR body (step 8), one line each, so a reviewer can reject one without losing the page. Everything else you wanted to do and did not is a **follow-up**: add a ledger row with outcome `follow-up`, whose matrix cell is the cell it would fill followed by ` [remainder]` (for example `entry point (red_team()) × data source (replay) [remainder]`). The distinct string is what keeps it from tying with the `opened` row for the cell this run is filling. Only Tier 1 gaps get a row, in the same commit; step 8 files a Linear ticket for it before the commit and writes the ticket id into that row. A "left out" sentence in the PR that no later run can find is how useful work gets lost.

**Invoke the `docs-writing` skill and follow it — that skill does the writing, this step only scopes it.** Call it with the `Skill` tool (`skill: docs-writing`) before you draft a word; it holds the reader, the three page genres and their shapes, the voice, the four things every page carries, and the reviewer loop that gates a draft. Its instructions win over any writing habit you would otherwise fall back on. This routine writes unattended, so a page drafted without it is a page nobody read for voice before it shipped.

Rules that are not negotiable:

- **A new page goes in `mkdocs.yml` twice** — under `nav:` and under `plugins.llmstxt.sections`. The build fails on the mismatch, but fix it at write time rather than discovering it in step 5.
- **Every code sample is fenced with a language.** No indented blocks, no bare `Example::`.
- **Every code sample runs.** If you cannot make it run, it does not go in the page.
- If a cell resolved `N/A` for a reason not yet listed, add it to `.claude/skills/docs-coverage/axes.md` in the same branch — and mark it in the **PR title** with the bare tag `[axes]`, nothing more; the detail goes in the body. That file decides what counts as a gap at all; an agent quietly narrowing its own future workload is the one edit here a human must see.
- **Never cut a sentence off, and do not hard-wrap the prose.** Both rules live in `CLAUDE.md` and bind every author, not just this routine — and they cover the PR title and body, the Slack message and any issue or review comment this routine writes, not only the `.md` pages. Code, and the comments and docstrings inside it, is the one place that still wraps. The one thing that is specific here: the receipt's captured output *is* the allowed truncation — the first 40 lines of a command's output (`RECEIPT_OUT_LINES`), marked when cut, never a sentence you wrote.
- **Never edit `src/`. No exception, not even a one-character docstring fence.** A routine that can reach the code it documents will eventually document its way into changing behaviour, and the reviewer of a docs PR is not reading it for that. This is the hard boundary of the whole routine: it writes prose about code, and it never writes code.
- **A code defect you find gets reported, not fixed, and never papered over.** When the code and the docs disagree — a flag the parser accepts but the help text denies, an argument silently swallowed, a docstring fence that breaks the build — you have three moves, in this order: say it in the PR body as an open Decision with its file and line; say it in the **step 9 Slack message** as a named inconsistency, because a human who never opens the PR still needs to know; and only then decide whether the page can honestly be written around it. If it cannot, drop the gap and record `skipped-code-defect` in the ledger. Writing a `!!! note` that explains an inconsistency as though it were a design is the failure mode here — it launders a bug into documented behaviour, and once documented it is much harder to fix.
- No drive-by edits to pages the new one has no link to. Adjacent edits (above) are the only exception.

The rest of the rules that bind a docs page in this repo are in `CLAUDE.md` under **Docs**. Read it; do not expect this file to restate it.

## Step 4 — the receipt

**Write the runner before you run anything.** `.claude/skills/docs-autofill/run_receipt.sh` — a tracked path, committed with the branch — extracts the blocks, runs them, and writes `.context/receipt.txt`. The runner is tracked and the receipt is not, on purpose: `.gitignore` ignores `.context/`, so a runner written there cannot be committed and a reviewer cannot see what actually ran. Ad-hoc shell functions are how a receipt and reality drift apart: an out-of-band re-run that fixes a block leaves the file on disk still saying it failed, and every downstream reader — including four critics — reviews the stale copy.

**Never hand-write or hand-patch receipt state.** The receipt is a generated artifact, not a summary you author. Changed a block? Re-run the whole runner. If what you are about to say about the receipt is not in the file the runner just wrote, it is not true yet.

How the runner runs blocks:

- **One persistent shell per page, blocks in document order** — a page where block 2 uses an export from block 1 is normal prose, and a per-block runner would force the author to write worse pages to satisfy the runner.
- Python blocks under `uv run python`, in that session's working directory.
- **Bash blocks get the project venv on `PATH`** (`source .venv/bin/activate`, or prefix with `uv run`). A bash block calling `eq`/`evaluatorq` in a bare shell exits `127`, and `127` looks like a broken example rather than a broken runner.
- **Against a fresh run store**: `export EVALUATORQ_DIR=$(mktemp -d)` before the first block. Blocks that run against a store your own probes already populated will pass while hiding a state dependency — a `previous_run="latest"` that only resolves because you happened to leave a run lying around.

The receipt opens with a `RESULT:` line (block counts, ran vs illustrative, pass or fail), then one row per block: its exit code, or `— illustrative (<why>)`, with the first 40 lines of output (`RECEIPT_OUT_LINES`) indented under it, and an `[output truncated at N lines]` marker when it is cut. Paste it into the PR body verbatim, folded (step 8). A docs PR whose examples were never executed is the thing this routine exists to stop shipping.

**Nonzero exit is a hard gate.** Any block that exits nonzero and is not exempt stops the run here: fix the page, regenerate the receipt, and only then go to step 5. Never carry a red row into review — a table nobody is required to act on is not a check.

A block that cannot be executed *by nature* — a directory tree, a config excerpt, a truncated API response — is exempt. Mark it `— illustrative (<what it is>)` in the receipt; never drop the row. The exemption is a claim the critics in step 6 are told to check, so do not use it to retire a block that merely failed.

There is a second exempt class, and it is narrow on purpose: **an environment-setup block, whose commands configure the reader's own project rather than assert anything about this repo.** `uv add`, `pip install`, and a placeholder `export SOME_KEY=...` qualify. These are exempt because executing them is *wrong*, not because it is hard: `uv add` mutates `pyproject.toml` (and cannot resolve at all inside this package's own checkout, where the requirement is the project itself), and a placeholder `export ORQ_API_KEY=...` would overwrite the real key with the literal `...` for every later block in the shared shell. The runner decides this by pattern, never by an author's say-so, and the receipt row names the class. The rule against retiring a block that merely failed binds here exactly as above — a block that does real work and happens to fail is not setup.

## Step 5 — validate

```bash
uv run --group docs mkdocs build --strict
uv run python scripts/validate_mermaid.py
```

Then the advisory scope check:

```bash
.claude/skills/docs-autofill/scope_check.sh docs/guides/<page>.md
```

It reports adjacent-edit files and lines against the budget in step 3 and flags code-fence changes outside the page. It always exits 0: read it, trim or defer what it says is over, and leave a short justification in **Beyond the page** if you keep an overage.

If anything under `src/` changed, **the run is already broken** — step 3 forbids it, so a dirty `src/` means an edit slipped through. Revert that file, report it as an inconsistency (step 9), and re-run the build. The CI-four below exist only for that recovery path; on a correct run they never execute:

```bash
uv run ruff check src
uv run ruff format --check src
uv run ty check
uv run pytest -m 'not integration'
```

`ty check` is never scoped to a path. Everything green before review.

## Step 6 — review, in parallel

The reviewer loop is `docs-writing`'s, not this routine's. Run it as written there — the four lenses, the reader lens widened to all four personas from its **Reader personas** table, the artifacts-not-conclusions rule, validate-before-fixing, the blocking/non-blocking split. This step adds one track on top of it and one extra instruction to the lenses; everything it does not restate is governed there.

Derive each persona's task from the gap being filled, one per persona — the point is to test whether the new page makes *its own* usage path achievable.

Tell the lenses to check the receipt's `illustrative` exemptions against the page. That claim is the only part of step 4 nothing else verifies.

Dispatch the track below together with those lenses; the same no-reviewer-holds-the-barrier rule covers all of them, and a missing one is named in the PR body **and** in the Slack line.

### Track A — adversarial critique

The `hate` skill lives in the `orq-ai/research` checkout, which `routine.json` mounts as a second source. In the cloud sandbox the two checkouts are siblings under `/home/user`, so it is at `/home/user/research/.claude/skills/hate/SKILL.md` — verified from a real run, not assumed. Fall back to a search only if that path is wrong:

```bash
find .. "$HOME/.claude/skills" -path '*skills/hate/SKILL.md' -print -quit 2>/dev/null
```

Not found → run the `docs-writing` lenses alone, say so in the PR body, and **say so in the Slack message** — half the review pipeline vanishing must not be visible only to someone who opens the PR.

Run it in `--apply --sonnet` mode. Three overrides for unattended use:

- Its "ask, then STOP" branch does not apply; `--apply` is in force, so Recommendations get applied and Decisions stay open for the PR body.
- Its Linear-comment step is replaced by the PR body. There is no ticket for the run itself; the only tickets this routine files are the follow-ups in step 8.
- Its Context Resolution table has no pattern for "diff against main". Give it the diff explicitly: commit step 3's work as a single commit before dispatch, so its `git diff HEAD~1` fallback resolves to the whole page.

If one of your own checks reports that *everything* is broken — every anchor missing, every link dead — suspect the check first. Confirm against the built artifact before you state it anywhere.

## Step 7 — iterate

`docs-writing`'s iteration rules apply unchanged: apply the blockers, leave the non-blocking findings reported and unfixed, re-dispatch only the personas that did not end COMPLETED, and stop at three rounds. What this routine adds to a round: apply the hate Recommendations too, and re-run steps 4 and 5 before re-dispatching. The receipt in the PR is the one from the final round.

**Still blocked after round 3:** open the PR anyway and surface it in the four places `docs-writing` names — the bare `[BLOCKED]` title tag, the `## Blocking, unresolved` body section with the session URL, and the DM to Bauke on Slack (`U09BR0B0Q7P`) rather than only the channel post. The persona's name and its verdict go in the body under persona verdicts, never in the title. The work is worth keeping; the judgement is not the routine's to make.

Anything not applied is a `Left out` row in the PR's review table (step 8), with the reason — never a finding that quietly disappears.

## Step 8 — PR

**Follow-up tickets first.** Before the ledger row and the commit, file one Linear ticket per follow-up (step 3), with the Linear connector in `routine.json`: team `BACK`, project `EvaluatOrq`, title `docs: <what the follow-up would cover>`. Filing first means the ticket id is in the committed ledger row and the PR body from the start, so the only edit after `gh pr create` is `prepared` → `opened`. The cost is an orphan ticket when the run later aborts; the search below absorbs that on the retry. First search that project for an open issue about the same matrix cell and skip the filing if one exists, so a gap does not collect a ticket per week. The description is two headers and short prose, no links to the PR's discussion:

```markdown
## Situation
<what the page does not cover yet and who hits it; the `gap: <axis> × <axis>` line>

## Desired state
<what the missing page or section should let a reader do>
```

The id goes in the ledger row (`follow-up BACK-1234`), the PR body's **Beyond the page** line and the step 9 Slack message. **If the Linear connector is missing or the call fails, do not stall the run:** write the row as plain `follow-up` and say once, in the Slack message only, that no ticket was filed and why. Until `routine.json` is applied to the live routine this is the normal path, so it must stay quiet. When a later run picks up a `[remainder]` cell that has a ticket id, its PR title takes the id in trailing parentheses (`docs: <what the page covers> (BACK-1234)`) as `CLAUDE.md` requires, and the PR body says the ticket is to be closed by hand.

Then append the ledger row, in the same commit as the docs, with outcome `prepared`:

```markdown
| 2026-08-24 | `entry point × hybrid` | docs/autofill-hybrid-mode | prepared |
```

Amend it to `opened` only after `gh pr create` returns a URL. Writing `opened` up front means a failed PR creation still tells next week the gap was handled, and the gap is then skipped forever.

Then, body to a file — `gh pr create` prompts for a title and body it is not given, and a headless session has no terminal to prompt at:

**Title: short.** `docs: <what the page covers>` in **at most ~60 characters** — one clause, no parentheses, no em-dash aside, no flag names, no list of secondary edits. `docs: document red-team replay` is the whole title; the `previous_run` / `--from-run` detail and the axes.md edit belong in the body. Only `[BLOCKED]` (step 7) and `[axes]` (step 3) may be appended, as bare tags.

```bash
gh pr create --title "docs: <what the page covers>" \
             --body-file .context/pr-body.md \
             --assignee @me --label docs-autofill --base main
```

Conventional commit, `docs:` type. Never `feat!`/`fix!`. CODEOWNERS requests reviewers; do not pass `--reviewer`. Never merge.

**A title tag goes in the commit subject too.** `.github/workflows/pr-title.yml` runs `validateSingleCommitMatchesPrTitle`, and this routine always opens a one-commit PR, so a `[axes]` or `[BLOCKED]` tag appended to the title alone fails the **Conventional title** check with `The pull request title doesn't match the commit`. Amend the commit subject to the exact title before pushing. Found the hard way on #200.

**PR body: short.** A reviewer should see what changed and what needs them on one screen. Prose sections are two or three sentences each, not paragraphs; detail a reviewer only needs to check something goes in a `<details>` block. In this order:

0. **Status** — one line, before any heading, that says whether the PR is waiting on a human: `Adds docs/guides/<page>.md. <line 2 of .context/receipt-summary.txt>. 2 open decisions.` or `... 0 open decisions, nothing needs you but a read.` The receipt part is pasted from that file, never retyped; the decision count comes from the review.
1. **The gap** — the literal `gap: <axis> × <axis>` line, then one sentence on why it ranked first. Put `.context/docs-coverage-matrix.md` in a folded `<details>` block under it, verbatim: `.context/` is gitignored, so a reviewer cannot open the file, and the matrix is the evidence for the ranking.
2. **What the page covers** — one sentence, plus one on what it deliberately leaves out.
3. **Beyond the page** — one line per adjacent edit (file, what changed, why it belongs here, and for a factual fix what it was checked against), then one line per follow-up (the gap and its ticket id). Omit the heading when there are neither.
4. **Run** — three lines at most: the step 0 spend (live calls, which key/workspace, records written or not), the persona verdicts as `persona → verdict` with rounds taken, and any missing reviewer or whether `routine.json` changed.
5. **Open decisions** — the hate Decisions and anything the review table marked `Open`, each with its Lean, in the plain-words form `docs-writing` requires. Omit the heading when there are none.
6. **Review** — the table exactly as `docs-writing`'s **Reporting the review in the PR** defines it: `Impact` column, `Left out` and `Open` rows visible, `Fixed` rows folded.
7. **Receipt** — always folded. The summary is line 1 of `.context/receipt-summary.txt`, the block carries the step 4 file verbatim:

   ````markdown
   <details>
   <summary>Receipt — <line 1 of .context/receipt-summary.txt></summary>

   ```text
   <.context/receipt.txt, verbatim>
   ```

   </details>
   ````

8. One line saying the weekly docs-autofill routine opened this.

Every sentence in the body is a whole sentence — the no-truncation rule from step 3 applies here too. Short means fewer sentences, not clipped ones.

## Step 9 — Slack

Channel `research-assistant` (`C0BNKRH9FGR`) via `slack_send_message`. **Standard Markdown, not Slack mrkdwn** — the tool converts it itself. Links as `[#123](url)`, bold as `**bold**`, bullets as `• `.

Exactly one message per run, in every case:

- **PR opened** — the PR's own gap line, the PR link, the persona verdicts line and rounds taken, open decision count. Reuse the PR body's wording rather than re-summarising it. List each follow-up ticket id, or say none was filed and why. Name any reviewer that did not run.
- **PR blocked after 3 rounds** — as above, plus a DM to `U09BR0B0Q7P`.
- **No-op** — what was checked, why no gap cleared the bar.
- **Aborted** — env guard failed, or `gh pr create` failed after the work was done. Name the cause; a failure at the last step is the most expensive one.

Whatever the outcome, if the run found a **code/docs inconsistency** (step 3), name it in that same message with its file and line — one line each, and say the routine did not fix it. This is the only channel by which a defect the routine is forbidden to touch reaches someone who can touch it. A no-op run that found one is not a quiet week.

A cloud run that dies silently is worse than one that fails loudly. If the channel post fails with `not_in_channel` or `channel_not_found`, DM `U09BR0B0Q7P` and say the channel post failed.

## Caps

One gap. Three revision rounds. Never merge. Never touch `src/`, full stop.
