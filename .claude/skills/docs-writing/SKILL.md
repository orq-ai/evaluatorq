---
name: docs-writing
description: How to write a page on the evaluatorq docs site — reader, genre, voice, the four things every page carries, and the reviewer loop that gates it. Use when drafting a new docs page or rewriting an existing one, when a feature PR needs its docs, and when asking how a page should be structured. Invoked deterministically by docs-autofill before it drafts.
---

# docs-writing

Governs pages under `docs/`. It does not govern PR bodies, commit messages, tickets or code comments — those follow `CLAUDE.md` and `ticket-writing`.

Two rules bind every prose surface, this one included, and live in `CLAUDE.md` because they bind outside it too: **never cut a sentence off**, and **do not hard-wrap prose**. Read them there.

## The reader

An ML or AI engineer already shipping LLM applications. They know Python, async, providers, tokens, and what a prompt is. They do not know evaluatorq.

That sets the jargon budget: a term from the *evaluation* domain — judge, jury, datapoint, vulnerability, target, attack strategy — gets defined on first use on each page, because a reader lands on one page from a search result, not on page one of a book. General Python and LLM terms are free.

Two pages carry a second reader: `docs/index.md` and `docs/guides/getting-started.md` are also read by a backend engineer new to evaluation. On those two, define the eval-domain terms in a sentence rather than a clause, and let the pace slow down.

## Three genres

Every page is one of three. Pick before drafting — the genre decides the opening, the shape, and the length.

| Genre | Answers | Opening | Length | Examples |
|---|---|---|---|---|
| **Tutorial** | "get me a working thing" | A promise with a bound: what you will have, how long it takes, what you do not need | ≤ 800 words | `guides/getting-started.md` |
| **Guide** | "how do I do X" | One line naming X, then the scope line | ≤ 2500 words | `guides/red-teaming.md`, `guides/targets.md` |
| **Reference** | "what are the knobs" | One line naming the surface, then a table that partitions it | Uncapped — bounded by the surface, not by prose | `tuning.md`, `configuration.md` |

The caps are guidance, not gates. `dashboard.md` runs long because the dashboard is large. A guide running long because it is padded is a different thing, and the cutter reviewer will say so.

**Tutorial pacing.** One path, no choices. Every branch you offer is a place the reader stops to decide, and a tutorial's job is to reach a working result before they have to. Mention the alternatives in one line at the end, or not at all.

**Reference shape.** A reference page's real work is partitioning: the reader has a symptom and needs to know which group of knobs owns it. Lead with the table that splits the surface into groups, then take each group in order. `tuning.md` is the model — three groups, and a note that confusing them is the usual cause of a setting that does nothing.

## Four non-negotiables

Every page, all three genres, no exceptions.

1. **An opening one-line definition.** The first sentence says what the thing is, in one line, without a preamble. `A **target** is the thing evaluatorq attacks or converses with.`
2. **A scope line.** When to reach for this, and when not. `Every knob on this page has a working default. Reach for it when a run is slow, flaky, or spending more than you want — not before.` A page without one gets read by people it cannot help.
3. **At least one runnable example.** Runnable means you ran it. Not a fragment, not pseudocode, not a snippet that needs a variable defined three pages away.
4. **A named failure mode.** What goes wrong, and — where it applies — that it goes wrong *silently*. `Setting the wrong one is silent — the call simply runs at the default.` The reader's next hour is decided by this sentence more than by anything else on the page.

## Voice

House voice is already on the page in `guides/targets.md`, `tuning.md`, `index.md` and `guides/getting-started.md`. Read one before drafting. What follows is what those pages do, stated as rules.

**Define, then qualify.** State the thing flatly, then add the caveat as its own sentence. Not one sentence carrying both.

> ❌ It's important to note that targets, which can be specified either as strings or as objects depending on your use case, are what red teaming and simulation both operate against.
> ✅ A **target** is the thing evaluatorq attacks or converses with. `red_team()` and `simulate()` both take one.

**Tables partition choices; prose explains one thing.** When the reader is choosing between kinds, modes or backends, a table with a `Use it when` column beats three paragraphs. When there is one thing, prose beats a one-row table.

> ✅ `| Kind | Use it when | Context discovery |`

**Put the gotcha where it bites, not in a section at the bottom.** The install caveat belongs under the install command.

> ✅ ```uv add evaluatorq``` followed immediately by: `uv add` installs into the current project — run `uv init` first if you don't have one.

**Second person for actions, third for facts.** "You pass a target" when the reader does it. "evaluatorq resolves the identifier" when the system does. Mixing these mid-paragraph is the fastest way to lose which side of the API a sentence describes.

**No throat-clearing.** Cut the opening move that says the page is about to begin.

> ❌ In this guide, we will explore the various ways in which you can configure timeouts.
> ✅ Every knob on this page has a working default.

**Cut the reassurance.** "Simply", "just", "easy", "powerful", "seamlessly", "it's worth noting that", "in today's landscape". They add no information and quietly tell a stuck reader the problem is them.

**Concrete numbers and names over hedges.** "Three groups" not "several groups". "`gpt-5.6-luna`" not "a small model". "Fails after 120 seconds" not "may time out".

**One idea per paragraph, and the idea is in the first sentence.** The reader skims first sentences. A paragraph whose point arrives in the fourth sentence is a paragraph they skip.

## Mechanics

- **Fence every code sample on a page, with a language.** An indented block reaches Pygments with no lexer and renders as grey text. `docs/hooks.py` fails `--strict` on any unhighlighted block, on every page; the two exemptions there are prose diagrams, not an escape hatch. The docstring version of this rule — griffe section indentation, and the backtick width in `examples/*.py` — lives in `CLAUDE.md`, because it binds when you are editing Python and this skill is not loaded.
- **A new page goes in `mkdocs.yml` twice** — once under `nav:`, once under `plugins.llmstxt.sections`. The llmstxt plugin silently drops any nav page it does not list from `llms.txt` and `llms-full.txt` without failing the build, so `docs/hooks.py` fails `--strict` on the mismatch instead. The check runs one way (nav → sections) and accepts globs, which is why `reference/evaluatorq/*.md` covers every generated API page with one entry.
- **Name a current model in every example.** `gpt-4o-mini` dates the page the moment a reader sees it, and the reader copies it. The catalog is the source of truth: `orq models list --json` returns a `created` timestamp per entry, so `model_developer == 'openai' and model_type == 'chat'` sorted by `created` descending gives the latest OpenAI family in one command. Prefer the cheap member of that family — it is also the value of `DEFAULT_PIPELINE_MODEL` — unless the sample is specifically about a bigger model. Run the query rather than copying a model name out of a file that does not know today's date.

## Process

1. **Pick the genre and name the reader's task in one sentence.** If you cannot, the page has no scope line yet and drafting it will not find one.
2. **Draft**, following the genre template and the voice rules.
3. **Run every code block as written**, in a scratch file. Not "it looks right" — run it. Fix the page, not the snippet's environment.
4. **Self-check the four non-negotiables.** Missing one is a defect, not a nit.
5. **Reviewer loop** — below.
6. **Hand off**: run the `docs-drift` skill scoped to the page (are its claims still true of the code?), confirm both `mkdocs.yml` entries, then `uv run --group docs mkdocs build --strict`.

### Reviewer loop

Four reviewers in parallel on every draft — three narrow sonnet lenses and one broad opus editor:

- **reader** (sonnet) — roleplay the assumed reader. Attempt the task using only this page. Report the first place you got stuck and what you needed that was not there. Where more than one kind of reader lands on the page, run one agent per persona from **Reader personas** below instead of a single generic reader; the other three lenses stay one agent each.
- **conformance** (sonnet) — the four non-negotiables, the genre template, the voice rules. Quote the offending line.
- **cutter** (sonnet) — filler, reassurance, throat-clearing, paragraphs that delete without loss. Quote what to cut.
- **`content-evaluator` agent** (opus) — the standing editor agent, unmodified. It reads for argument quality, audience fit and structural problems the three narrow lenses are not looking for. Take its Critical findings as candidate blockers and its rating as a signal, not a gate.

Dispatch all four together. **No reviewer holds the barrier forever**: once the others are in, wait a little longer, then proceed without the missing one and name which lens was skipped wherever you report the result. A review someone knows is partial beats a loop that never ends.

#### Reader personas

The reader lens is the only one that can produce *the page is correct and I still could not do it*. Widen it when the page serves more than one kind of reader — one agent per row, each given a task **derived from what the page is for**, phrased as something a person wants to accomplish, never as "review this page".

| Persona | Angle the task takes |
|---|---|
| First-timer | Never used evaluatorq. Task starts from zero, arriving at the docs root. |
| Platform user | Works through Orq — a `deployment:` target, no local model code. |
| Practitioner | Already uses the surface this page belongs to; wants this specific path against their own target. |
| CI engineer | Needs it non-interactive in a GitHub Action — exit codes, env vars, no prompts. |

Rules every persona agent is given:

- You are a user, not a reviewer. You have the docs site and a terminal.
- **Do not read the diff, the git log, or the raw markdown source.** Read the built site. This is on your honour — nothing enforces it, and a persona that peeks produces a report that looks fine and is worthless.
- Start from `docs/index.md` and navigate. If you cannot find the page, that is the finding — say so and stop.
- **Run the commands. Do not read them and assume.** Paste what actually happened.
- Report: where you got stuck, what you had to guess, what sent you to the source, and what you expected the page to say that it did not.
- End with one line: COMPLETED / COMPLETED WITH GUESSWORK / BLOCKED.

Re-dispatch only the personas that did not end COMPLETED, with the same task.

**Hand every reviewer artifacts, never your conclusions.** Give paths — the page, the receipt of what ran, the diff for the lenses allowed to see it — and let them read. Writing "all blocks pass" or "verified correct" into a reviewer prompt contaminates the one thing a reviewer is for. The same holds in reverse: a reviewer's claim you did not check is a claim, not a finding.

Then: **validate each finding yourself before fixing it.** A reviewer that misread the page produces a finding that makes the page worse. Check the claim against the draft; drop the ones that do not hold.

**Blocking** — fix these, then re-run the loop:

- a reader lens ended anything other than COMPLETED. `COMPLETED WITH GUESSWORK` is a blocker: the guesswork *is* the gap, and a persona that guessed right this round is a reader who guesses wrong next round
- a non-negotiable is missing
- a code block does not run
- a claim is false

Everything else — voice nits, a clunky sentence, a suggested extra example — is **non-blocking**: report it, do not fix it, do not re-run for it. Without that line the loop never ends, because there is always one more sentence to improve.

**Maximum three iterations.** A blocker surviving three rounds means the page has a design problem underneath it — a missing prerequisite, an undocumented feature it stands on — and a fourth round will not find it. Stop; the page is still worth keeping, and the judgement about it is not yours.

**Ship it and make the blockers impossible to miss.** Open the PR, and surface what is unresolved in three places, because each one reaches a reader the others do not:

1. **The PR title** carries the bare tag `[BLOCKED]`. Nothing else — details go in the body. A reviewer scanning the list sees it without opening anything.
2. **The PR body** opens with a `## Blocking, unresolved` section: each blocker, which lens raised it, what you tried across the three rounds, and what you think it needs. This is what the reviewer reads.
3. **The session that produced it** gets linked from that section. A blocker's whole context — the drafts, the findings you dropped as invalid, the three rounds of reasoning — lives in the session and nowhere else, and a reviewer who has to reconstruct it will instead just merge or close. In a cloud or scheduled run, paste the session URL. Where there is no shareable URL, say so in that line and name what the reviewer would have to re-derive, so the gap is visible rather than assumed away.

Running unattended (the `docs-autofill` routine), add its fourth channel: DM Bauke on Slack rather than only posting to the channel. A `[BLOCKED]` PR nobody opens is a blocker nobody read.

The aim above all of this is that someone unfamiliar can complete the task from the page alone. That is the goal, not the gate: no agent can certify it, which is why the reader lens exists.

---

*Keeping this file useful: it stays one file while it is under ~250 lines. When the examples push past that, split them into `patterns.md` and point at it from here — a drafting agent needs them, a reviewing one does not.*
