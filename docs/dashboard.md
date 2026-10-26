# Dashboard

!!! note "The UI — FastHTML `eq dashboard`"
    The combined `eq dashboard` documented here opens the report index for red-team, simulation and pairwise runs. Its canonical invocation scans a run directory and opens the multi-run FastHTML UI — `eq dashboard` (no path) browses the default stores, and `eq dashboard .evaluatorq/sim-runs` scopes to simulation. Passing a single JSON report file is an optional direct deep-link to that report.

evaluatorq ships a built-in web dashboard for browsing red-team, simulation and pairwise reports.  It is powered by **FastHTML** (a lightweight Python web framework) and served locally via **uvicorn**.  There is no external service dependency — everything runs on your machine.

The separate **Insights** sidebar page at `/insights` reads `.evaluatorq/insights-runs`, outside this report index. See the [Trace Insights guide](insights.md) for how to create and review those runs.

![The evaluatorq combined dashboard — stat band, runs by type, attack-resistance, findings by severity, token usage, and recent runs.](assets/dashboard-index.png){ .dashboard-shot }

## Install

The dashboard is an optional extra (it pulls in `python-fasthtml` and `uvicorn`):

```bash
uv add "evaluatorq[dashboard]"
# or — if you already have the redteam / simulation extras:
uv add "evaluatorq[redteam,dashboard]"
```

Prefer pip? Use `python -m pip install "evaluatorq[dashboard]"`, which installs into the interpreter you just named rather than whichever `pip` happens to be first on your `PATH`.

## Launch

Launch it with `eq dashboard` (the `evaluatorq` and `eq` entry points are interchangeable):

```bash
# Canonical — browse all default stores (red team + simulation + pairwise)
eq dashboard

# Canonical — scope to the simulation run store
eq dashboard .evaluatorq/sim-runs

# Scope to any directory of exported reports
eq dashboard /path/to/my/reports

# Optional direct deep-link — open a single report file
eq dashboard .evaluatorq/runs/red-team_20260626_143024.json

# Bind a custom host / port (default 127.0.0.1:8080); -p is short for --port
eq dashboard --host 0.0.0.0 --port 8888

# Enable “View Traces” links in reports. Use the workspace slug from the
# Orq URL (https://my.orq.ai/<workspace>/...), not an API key or workspace UUID.
ORQ_WORKSPACE=orq-research eq dashboard
```

| Invocation | What it scans |
|---|---|
| `eq dashboard` | All three default stores: `.evaluatorq/runs` (red team), `.evaluatorq/sim-runs` (simulation), and `.evaluatorq/pairwise-runs` (pairwise) |
| `eq dashboard <dir>` | Only that directory (e.g. `eq dashboard .evaluatorq/sim-runs`) |
| `eq dashboard <file>.json` | Optional direct deep-link; prints that report's direct URL so you land straight on it |

The dashboard opens in your browser when the server starts; pass `--no-browser` to keep it in the terminal. It also reads `.env` from the directory where you run the command, while values already exported in your shell take precedence. Pointing at a directory (`eq dashboard .evaluatorq/sim-runs`) scopes the UI to that store. Passing a single JSON report file prints that report's direct URL.

### Orq trace links

Set `ORQ_WORKSPACE` when launching the dashboard to show **View Traces** links for conversations and runs that have no experiment URL. Its value is the workspace slug in the Orq UI URL; for example, `https://my.orq.ai/orq-research/traces` uses `orq-research`. A workspace selected under **CLI OAuth** in Settings takes precedence for those fallback links. When neither is available, trace-link buttons are hidden for runs without their own experiment URL.

`ORQ_WORKSPACE_SLUG` remains supported as an alias. For an Orq UI on a host different from the API host used by the selected credential, set `ORQ_UI_BASE_URL`. Otherwise the dashboard uses the selected credential host when available, then `ORQ_BASE_URL`, then `https://my.orq.ai`.

### Settings and the trace finder

The dashboard has separate **Traces** (`/traces`) and **Trace search** (`/find`) pages. Traces browses recent traces; Trace search starts from a natural-language question. The **Settings** item opens `/settings`, where **Models** has one field per model role: **Fast model**, **Smart model**, **Classifier model** and **Embedding model**. Each field shows its built-in default, offers **Use default** to clear a saved choice, and says which environment variable or CLI flag is overriding it, if any; a value that came from an environment variable or a CLI flag is never written to the settings file. A read-only **Task overrides** row lists any `model_overrides` in the file. Each field opens a two-level provider and model menu; **Custom…** accepts any `provider/model` id. When credentials are missing or rejected, model fields stay editable and Settings shows one warning below Models with guidance for the selected authentication method. If Orq is unavailable, the same warning area asks you to retry; you can still type a model id. CLI OAuth loads the catalogue through the signed-in CLI session. The Insights run form's summary, classifier, embedding and question-compiler pickers load their menus the same way and start on the smart, classifier, embedding and fast roles. The search window, trace limit, and parallelism are set per run on Trace search. **Authentication** offers four sources for dashboard trace access and Insights runs: the dashboard process's `ORQ_API_KEY`, an Orq CLI API-key profile, CLI OAuth, or an API key you enter. Settings saves the selected method in `.evaluatorq/dashboard-settings.json`. The manually entered key is encrypted there with a key kept in macOS Keychain; on other platforms, set `EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY` to a Fernet key. The CLI owns OAuth storage and refreshes its session. A startup toast reports when the selected credential needs attention or could not be checked; a valid credential produces no toast. If CLI OAuth needs a new sign-in, run `orq auth login`. Environment authentication reads `ORQ_API_KEY` and `ORQ_BASE_URL` from the dashboard process. Every method uses all projects accessible to its credential; choose a project facet in Trace search or the Insights run form to narrow a run. The page also shows read-only runtime configuration such as scanned run stores, API-key presence, Orq host, and workspace for trace links. The CLI masks keys in its profile list, so evaluatorq reads usable keys from its private local credential file; a profile without an accessible key remains disabled. See the [Trace finder guide](trace-finder.md) for explorer behavior, authentication details, and the `eq find` CLI.

#### Compression inspector

The **Compression inspector** compares a trace's complete readable conversation with the structured Jev state derived from its canonical ATIF document. Use it to understand what the existing compression reducer keeps or omits; it does not answer a question, run inference, or change the source document.

The preview reports plaintext and Jev percentage reductions against the complete trace document serialized as compact JSON, including metadata and null fields. The baseline is uncapped and independent of the selected Original view. Jev uses compact serialization rather than the pretty-printed display; plaintext counts the rendered text. A negative percentage means expansion. These character reductions are not token savings.

The setting is off by default. Turn on **Compression inspector** under **Settings → Advanced**; every Settings change saves as soon as you make it, a small notice in the corner confirms it, and the navigation link appears without a page reload. Then open it from the dashboard navigation. Start the dashboard with:

```bash
eq dashboard
```

Choose **Orq trace ID** to fetch one trace using the dashboard's selected Orq account. The default search window is the last seven days; set both optional dates to search an older trace, with the end date including that full day. Trace IDs and dates are submitted by POST rather than placed in the URL.

Choose **Local sessions** to search the configured Claude Code, Claude desktop, and Codex session folders. The list is limited to 50 sessions. Selecting one revalidates its `SessionRef` against the configured session roots before reading the session into the same canonical ATIF `TraceDocument` used by the Orq source.

The left pane is the **Original**, and **Original shows** picks what it is. **ATIF trajectory** (the default) is the canonical ATIF trajectory and trace metadata held for the trace, as pretty JSON with empty fields left out. **Raw source file** is the session file itself, as read from disk (Claude Code, Claude desktop or Codex JSONL); it is offered for local sessions only. **Readable text** is the Insights `full_conversation_view` rendering with the inspector's selected layout and de-noising options. It omits harness reminders and keeps every part whole by default. Every original is scrubbed of known sensitive values and escaped before display, cut to 2,000,000 characters with a visible note, and a raw session file is read up to 5,000,000 bytes with a note saying how much was shown. The source `TraceDocument` remains unchanged. The right pane is the **Output**, the structured state from the Jev reducer, rendered as escaped JSON. The inspector fetches no image or other media URLs and makes no paid model call.

The preview reports the characters of the shown original, the readable-source characters (the uncapped transcript), readable-view characters, source message count, serialized Jev-state characters, omitted message count, and the active global cap. The global cap defaults to 500,000 characters and follows the saved model-input setting. Jev has a 112,000-character state cap and question-aware state-plus-question ceilings of 128,000 or 256,000 characters; this preview includes no question, so those question-aware ceilings are context, not a claim about a question payload that was never supplied. The readable JSON adds whitespace that is excluded from the compact serialized-state character count.

Once a trace is loaded, a form above the two panes exposes every compression knob in groups, each field showing its default. **Original shows** and **Wrap long lines** (off by default, applied to both panes instantly) sit above them. **Readable text** options apply only when the Original shows Readable text; they control the overall view limit, which message kinds are included (system, user, assistant, reasoning, tool inputs, tool results), the tool display mode (one line per call, or one line per run of calls without results), and the per-part limits, where `-1` keeps a part whole and a tool result limit of `0` leaves results out. **Jev state** controls the global character cap, the Jev state limit, a keep-whole length and an edge length for assistant, user, system and developer messages, tool call input and tool results, and which roles, tool calls and tool results are included. Each edge may be at most half its keep-whole length. Excluding tool calls drops their results too, and excluding tool results keeps the calls without them; excluded messages count in the omitted-message total. Scrubbing of known secrets is always on and cannot be turned off.

The layout and de-noising switches below start **on in the inspector**. Outside it, `ReadableStyle()` and `JevOptions()` keep all these switches off: existing Insights and classifier production inputs remain unchanged. These switches change the preview, not the stored trace.

| Readable text switch | Effect |
|---|---|
| Message blocks with source indexes | Separates messages into blocks carrying their original source indexes. |
| Pair tool results with calls | Places results with their matching calls. Requires message blocks; with blocks off, pairing is inactive but its selected checkbox is preserved. |
| Show tool input as plain text | Shows decoded tool arguments as readable text rather than escaped JSON strings. |
| Label cuts with the part and exact omitted count | Names the part being cut and counts the absent source characters. The marker counts against the part and whole-view caps. |
| Remove terminal control codes and progress redraws | Removes terminal formatting and transient redraw noise from tool output. |
| Shorten the session working directory to `<cwd>` | Replaces the known session working-directory prefix with `<cwd>`. |
| Collapse consecutive repeated tool output | Replaces repeated adjacent output with a counted repeat marker. |
| Collapse long hex and base64 blobs | Replaces long encoded blobs with a marker rather than filling the preview with their bytes. |

| Jev format switch | Effect |
|---|---|
| Mark omitted messages at each source gap | Adds in-place omission entries with neighbouring source indexes and a unique source-message count. Beginning and end neighbours are null. Grouped retained tool results are never counted as absent. |
| Identifiers and status before text | Orders identifiers and result status before text, input and grouped results, including within each result. |
| Leave out empty fields | Removes empty result arrays, empty text and input on non-message entries, and redundant `call_index: 0` for a sole call. Keeps entry types and meaningful result metadata. |

Omission entries and the coverage description count against the Jev cap. Caps too small for their metadata show an inline error rather than an oversized preview. Labelled readable cuts likewise need enough room for their exact marker. If your filters leave no readable body, the pane says so even when the coverage header is still present.

**Describe what is present and absent at the start** (on by default) tells the reader of a compressed text what it holds. The readable text starts with one `[context: ...]` line listing included, excluded and absent message types, per-part limits, tool display and active layout settings, source message count and whether the view budget cut the messages. The Jev state starts with a `context` object listing included and excluded kinds, trimming rules and active format switches. Jev never carries reasoning. Both descriptions count against their renderer's cap and remain intact. The readable Original includes its header; ATIF and raw-source originals do not. Outside the inspector, `full_conversation_view(coverage_header=True)` and `JevOptions(coverage=True)` opt into coverage descriptions, while production Insights and classifier inputs leave them off.

The panes update as you change a field, without reloading the page, and **Reset to defaults** restores every field. An invalid value, or limits too small for the Jev state, show a message under the form instead of a preview. The loaded trace is held only in the dashboard process's memory, never written to disk, and the last eight loaded traces are kept; if one has been dropped, load it again to adjust it.

The dashboard process reads local session files and renders the selected preview in the browser; local content is not uploaded to Orq. The inspector does not put trace IDs or content in URLs, browser storage, or its own logs. Source failures use safe messages without echoing IDs, paths, credentials, or provider exceptions. If the setting is disabled, direct access to every inspector route returns 404. An older Orq trace outside the default window will appear missing until you supply its date range.

---

## What the dashboard browses

The dashboard auto-discovers JSON report files in the configured root directories:

| Default store | Written by |
|---|---|
| `.evaluatorq/runs/*.json` | `red_team()` / `eq redteam run` |
| `.evaluatorq/sim-runs/*.json` | `eq sim run` (auto-saves unless `--no-save`); `simulate()` only when called with `save=True` |
| `.evaluatorq/pairwise-runs/*.json` | `PairwiseRun.save()` (see [Pairwise Judging](pairwise-judging.md#saving-a-run-and-viewing-it-in-the-dashboard)) |

Each report gets a stable URL for the lifetime of its file, so links you share keep working.

### Supported surfaces

| Surface | JSON discriminator |
|---|---|
| Red team | `"pipeline"` key present |
| Simulation | `"mode"` key present (`mode` wins over `pipeline`) |
| Pairwise | `"judging"` key present |

Files that cannot be parsed (invalid JSON) are silently skipped.  Files that parse but fail model validation appear in the index as **broken cards** with an error badge; their detail page shows a non-fatal error message instead of a traceback.

---

## Landing

`GET /` opens the combined dashboard: a stat band (total runs, per-surface counts, attack resistance), runs-by-type and attack-resistance breakdowns, findings by severity, token usage, and a **recent runs** list across both stores.  The left sidebar switches surface — **Red Team** and **Agent Sim** open filtered run lists at `?surface=…`, sorted by creation time (newest first).  Each run row drills into its report view; reports whose JSON is only partially valid surface an error badge instead of a traceback.  The **export** action on a report downloads the standalone self-contained HTML for offline sharing.

### How attack resistance is counted

Resistance is **attack-weighted**: the rate is resisted attacks over *evaluated* attacks, not over every attack attempted.  An attack only counts as evaluated once a judge returned a verdict — an attack whose target call failed, or whose judge crashed, timed out or was skipped, is excluded from both sides of the ratio rather than counted as resisted.  So a run headlined `100 attacks` can show a rate measured over 60; the Score tooltip names both numbers.

For older reports, the dashboard derives missing summary values from the stored per-attack results. If that produces a different rate from the exported report, the Score cell is marked with `*` and its tooltip explains the difference.

!!! note "Rates use evaluated attacks"

    Attacks without a judge verdict are excluded from resistance rates. If an older exported report uses a different denominator, the Score tooltip marks the difference; re-export the report to refresh it under the current rule.

---

## Filters

Both surfaces expose dimension filters in a sidebar:

### Red team filters (7 dimensions)

| Dimension | Values |
|---|---|
| `result` | VULNERABLE / RESISTANT |
| `severity` | critical / high / medium / low / info |
| `category` | framework category codes (ASI01, LLM01, …) |
| `vulnerability` | vulnerability enum values |
| `attack_technique` | technique identifiers |
| `delivery_method` | delivery method identifiers |
| `source` | dataset source identifiers |

### Simulation filters

The sim rail exposes chip toggles, `<details>` dropdowns, and range controls, rendered directly in the sidebar wherever they always apply and tucked behind a **More filters** expander (`filter-dd-more`, reusing the same open-state persistence as the red team rail) when they may be unavailable for a given run:

| Control | Kind | Direction | Notes |
|---|---|---|---|
| `goal_outcome` | chip (2-value) | — | achieved / not achieved; zero or both selected means "all" |
| `terminated_by` | chip | — | termination reasons present in the run |
| `persona` | dropdown | — | persona names present in the run |
| `scenario` | dropdown | — | scenario names present in the run |
| `rule_broken` | chip (opt-in) | — | `yes` narrows to results with any broken rule; absent (default) shows all |
| `max_goal_score` | range | ceiling (`≤`) | `goal_completion_score`, step `0.05`, range `0..1` |
| `min_turns` | range | floor (`≥`) | raw `turn_count`, step `1`, max = the run's actual longest conversation (never normalized) |
| `min_total_tokens` *(More)* | range | floor (`≥`) | raw `token_usage.total_tokens`, step `1`, max = the run's actual highest token count; hidden when the run recorded zero tokens |
| per-turn metric thresholds *(More)* | range | floor for hallucination risk (`≥`), ceiling for response quality / tone appropriateness / factual accuracy (`≤`) | step `0.05`, range `0..1`; each control is rendered **only when that metric was actually scored somewhere in the run** — unavailable metrics are hidden, not shown disabled |

Every range control renders its default bound numerically (`≤`/`≥ value`) — when unset it shows the no-op end of the range, so it always reads as a number, never "all". Min-turns additionally shows the run's max turns beside the readout. Metric thresholds compare against a result's **worst scored turn** (`max()` for hallucination risk, `min()` for the quality metrics); a result with **no scored turns for that metric stays visible** regardless of the threshold, so unscored results are never silently dropped from the filtered view.

Filters are applied via HTMX (no page reload).  The report body, summary aggregates, and download links all update in-place to reflect the active filter state.

![A red team report filtered by the sidebar dimension filters.](assets/dashboard-redteam-filtered.png){ .dashboard-shot }

---

## Interactive views (red team)

The red team surface exposes four dashboard-only interactive panels alongside the static report body:

1. **Interactive breakdown** — pick a group-by and stack-by dimension (7 × 7 combinations); attack-success rate recomputed per (group, stack) cell.
2. **Agent heatmap** — select the pivot dimension (vulnerability / category / technique / severity) for the agent × dimension ASR heatmap.
3. **Conversation viewer** — drill into the full message-by-message transcript for any individual attack (system / user / assistant / tool messages plus evaluator explanation).
4. **Disagreement viewer** — for multi-agent runs, select any agent pair and page through attacks where their results differ (side-by-side transcripts).

### Per-attack jury panel health

Open an attack in the red-team report's **Attacks** tab to inspect its **Jury panel health**, the recorded votes from its panel of model judges. The panel sits below the evaluator verdict and above the transcript. Boolean votes read **RESISTANT** or **VULNERABLE**; an abstention reads **abstained**, and a judge that failed reads **failed** with its recorded error. Failed repetitions and replacement judges carry their own tags.

The summary counts decisive votes over all attempted judges, including replacements. For example, three configured judges and one replacement with two decisive votes read **2/4 decisive votes**. The **failed** count covers configured judges only; a failed replacement is visible in its own row. An abstention contributes no decisive vote, so read the per-judge rows before treating a small numerator as a mechanical failure. Ties and inconclusive panels are marked explicitly. A result with no recorded jury has no panel; reopening an older single-judge run cannot recover votes that were never saved.

### Apply recommendations to the agent

On a single-agent run that produced recommendations, **Focus areas** and **Recommendations** can preview suggested instruction changes as a diff. Nothing is written until you confirm. Applying a recommendation creates a new minor agent version and marks the recommendation as applied on the run.

Write-back requires an **Orq agent**, `ORQ_API_KEY`, and the `orq` extra (`orq-ai-sdk`). Multi-agent red-team runs never offer the action. Everything else does: the button renders whenever a single target produced recommendations, and the target-type check happens at preview time, not at render time — so on a run against a plain model, deployment or callback the **Apply…** button appears and then fails with an error naming the missing agent. Treat the button as "recommendations exist", not "this run is writable". The same flow is available programmatically through the red-team and simulation apply helpers.

### Simulation transcript viewer

Simulation reports expose a conversation transcript panel: select any conversation entry from the run to see the full multi-turn exchange between the simulated user and the target agent.

![The dashboard conversation transcript viewer, message by message.](assets/dashboard-transcript.png){ .dashboard-shot }

### Pairwise comparison view

A pairwise run opens on a verdict: which side won, and whether the run has enough decided comparisons to say so. The report has five tabs: Overview, Performance, Judges, Comparisons and Config. Open a pairwise run from the **Pairwise** run list when you want to know whether version B beat version A, how far to trust that answer, and what generating and judging the responses cost.

**Overview.** The verdict card names the leader and its win rate over decided comparisons, with a 95% range (a Wilson interval). It reads `✓ Clear winner` when that range stays above 50%, `⚠ Too close to call` when it crosses 50%, and `No decided comparisons` when every comparison ended in a tie or was inconclusive. Ties and inconclusive comparisons are left out of the win rate. Below it, count tiles and the **How much to trust this** table list signals about the run itself, such as how many comparisons were decided, how often judges split, and how many votes were thrown out; orange means read the verdict with care. The Length row warns only when the longer (or shorter) answer's share of wins has a 95% range that stays above 50%, since judges can favour longer answers; otherwise it notes the lean could be chance. The outcome donut follows the table.

Judge agreement uses Fleiss’ κ only on comparisons with exactly one valid vote from each configured judge. Replacement judges do not expand the panel, and duplicate configured-judge records make a comparison ambiguous, so it is excluded. When every included judge chose the same outcome on every included comparison, κ is undefined rather than perfect agreement; the report shows it as unavailable.

**Reading the hero chart.** The solid marker shows the observed win rate of the named side; the shaded band shows its 95% uncertainty range. Both sit on the same 0–100% scale. The center reference, `50% · equal wins`, means neither side has an advantage. If the band crosses that reference, the observed lead is too uncertain to call a winner. For example, three wins out of four decided comparisons gives an observed rate of 75%, but its range is about 30–95%, so the report still reads **Too close to call**.

![The Overview tab of a pairwise run: the centered verdict chart with its 95% range, count tiles, the full-width trust table, and the tab bar for Performance, Judges, Comparisons and Config.](assets/dashboard-pairwise.png){ .dashboard-shot }

**Performance.** A side-by-side table compares the two systems' recorded total generation cost in USD and mean generation latency. A third column gives B relative to A, as a cost ratio and a latency difference, only when every comparison has a complete measurement for both sides; otherwise it reads `—`. Judging cost, judge tokens and mean judging time per comparison appear separately below it. Generation measurements must be captured when producing the responses and passed through `PairwiseRun.add(metrics_a=..., metrics_b=...)`; the dashboard cannot recover them from saved response text. Missing values read **Not recorded**, while incomplete costs read **recorded subtotal** with their coverage. Mean latency uses only measured entries and gives its measured count when some are missing; it is not total run duration. New comparisons record judging time automatically, including both orderings and replacements. Filters apply to these metrics too. See [Recording generation cost and latency](pairwise-judging.md#recording-generation-cost-and-latency).

![The Performance tab separates the two prompts' recorded generation cost and mean latency from judging cost, latency and tokens, with matched-pair measurement coverage.](assets/dashboard-pairwise-performance.png){ .dashboard-shot }

**Judges.** One row per judge with its win rate for each side, tie rate, failed passes and position bias, plus consistency when repeated passes make it measurable. A judge whose position bias reaches 0.15 is flagged: it changed its pick when the sides were swapped, so its votes say more about ordering than quality. The column reads `n/a` rather than `0.00` wherever nothing was flippable, since there was no measurement to make. Whether swapping happened is read from the votes rather than from the saved `swap` flag, so a run saved with the default `swap=True` but executed single-ordering is labelled `on (never observed)`.

**Comparisons.** A strip of dots, one per comparison and coloured by winner, sits above a paged list of 25 rows. A ringed dot marks a comparison where the judges split. Click a dot to open the report on the Comparisons tab with that comparison's page loaded and its row expanded, or sort the list by number or winner. Rows load their detail only when opened, so a large run stays fast: each row expands to the two responses side by side with every judge's vote, each ordering's pass, and the rationale. A vote thrown out because the judge changed its answer when the sides were swapped carries a `flipped` tag, and a replacement judge is marked `stand-in`. Row numbers stay the same under any filter or sort, so `#14` is always the same comparison.

![The Comparisons tab: the strip of comparison dots above the paged list, with the first comparison expanded to show both responses side by side and each judge's vote in both orderings, one of them thrown out as flipped.](assets/dashboard-pairwise-comparison.png){ .dashboard-shot }

When the run was judged with tracing on, each comparison records the `trace_id` of its judging span and its row links to that trace in Orq. The link needs an Orq workspace in Settings; without one the row shows no link, and a run saved with tracing off has no `trace_id` at all.

**Config.** The sides, judges, criteria, swap setting, passes per ordering, aggregation and creation time, a short note on how the judging works, and the evaluatorq version and judge list the run was produced with.

**Filters.** The filter rail narrows the report, not only the list. Choose a winner, comparisons where judges split, comparisons with thrown-out votes, comparisons with a failed judge pass, or a single judge; the counts, trust signals, judges, performance and the Comparisons tab then describe only the matching comparisons. The verdict card always uses every comparison and says so, because most filters select rows by their outcome and a range computed on those rows would be meaningless. For the same reason, a filtered view retitles the trust table **Signals in these comparisons** and turns off its warning colours: a filter on thrown-out votes or split judges raises those rates by construction. A **Filtered view** banner states how many comparisons the tabs cover and offers **Show all**. Changing a filter keeps the open tab. A filter that matches nothing shows an empty state with the same reset.

**Comparing two runs.** When a second pairwise run exists, the report header shows a compare control that treats the current run as A and lets you pick B. The compare page pairs comparisons by question text, shows both verdicts side by side with the agreement between judges, and lists every matched comparison with those whose winner changed first. Both verdict charts plot side B's win rate and its 95% range on the same axis, so a lead that reversed between runs shows on opposite sides of 50%. Each comparison links to the same row in that run's report. Questions that appear in only one run are counted, not listed. A run that cannot be loaded returns an error instead of an empty page.

The changed tag compares the outcome slot (`A`, `B`, `tie` or `inconclusive`), not the side's display name. Renaming a side alone does not count as a changed winner. Keep the same systems assigned to A and B across runs when interpreting this tag; swapping their slots changes its meaning. Labels that collide with each other or with `tie` or `inconclusive` are qualified with their slot in the comparison table.

In the run lists, a pairwise run's headline names the leader and its share of decided comparisons, for example `40 comparisons · prompt-v3 leads 62%`, or `even` when the sides won equally. When the lead's 95% range still includes 50%, the headline adds `· too close`, matching the report's verdict. Its Score column is the **called rate**, the share of comparisons ending in a win or tie rather than inconclusive, or `1 - inconclusive_rate`. Mean inter-judge agreement reads like the more natural choice but is a modal vote share, so it is quantized by panel size: against the shared `≥ 0.80` threshold it silently means "unanimous" for three judges and "four of five" for five, and it is undefined for a single-judge run. Every surface's Score column names its own metric on hover.

### Additional red team charts

Beyond the four panels above, the red team surface recomputes several charts live that the static exported report does not carry:

- **Cumulative discovery curve** — vulnerabilities found as a function of conversation turn depth.
- **Attack-failure treemap** — vulnerability → technique, sized by attack count.
- **Token histograms** — prompt and completion token distributions per attack.
- **Vulnerability × severity** — a cross-join stacked bar.

---

## Reading a red-team run

Runs are saved to `.evaluatorq/runs/<name>_<timestamp>.json` by default (`--save none` skips the file; `--save detail` also keeps per-stage artifacts in `--artifacts-dir`), and `eq dashboard` browses those files locally — no external service:

```bash
uv add "evaluatorq[dashboard]"

eq dashboard                                              # browse every saved run
eq dashboard .evaluatorq/runs/red-team_<timestamp>.json   # deep-link to one report
```

What follows walks the dashboard the way you'd read a finished run: land, pick the run, then work down the report tabs from headline to evidence. The screenshots come from one example run — a hybrid run against two deliberately contrasting targets — so your own numbers will differ. [Filters](#filters), [trace links](#orq-trace-links) and [downloads](#downloads) are documented above. For the red-team concepts behind these numbers, see the [Red Teaming guide](guides/red-teaming.md).

### 1. Landing — what has been run at all { #rt-landing }

`eq dashboard` opens on a cross-surface overview: jobs run, spend, tokens, the red team / agent sim split, and findings by severity across every stored run. Use it to confirm the run you expect actually landed, then pick a surface from the left rail.

It is a triage screen, not a scoreboard: the severity bars aggregate *every* stored run, including throwaway smoke runs, so a red bar here doesn't mean the system you care about is broken. Go to the surface, find the run, and read its numbers there.

An empty landing page — zeros across the stat band, no runs in either rail — means the dashboard found no report files, not that the runs failed. Check you launched it from the directory holding `.evaluatorq/`, and that the run was saved at all — `red_team()` saves by default, but `--save none` (CLI) or `save=SaveMode.NONE` writes nothing.

![The dashboard landing page: jobs run, spend, tokens, run mix, and findings by severity across all stored runs.](assets/dashboard/redteam-01-landing.png){ .dashboard-shot }

### 2. Red Team — pick a run { #rt-run-list }

**Red Team** opens on a card of four numbers across every stored run (ASR, critical findings, attacks run and total spend), then lists every red-team job, newest first, with the target or targets it attacked. Each row shows how many attacks broke the agent out of how many were evaluated (`BROKE`, as `x/y`), the attack success rate for that run (`ASR`), the case count and the cost. ASR is lower-is-better: a chip is teal at 10% or below, amber above 10% and red above 25%, and the same bands colour the ASR number in the card.

The total spend is the sum of the costs the runs recorded. When only some calls were priced, the line under it says how many (for example "1,140 of 1,236 calls priced"), so the figure counts only what could be priced.

A row for a run that died part-way carries an **error** pill, and one still in flight carries **running** (the dashboard reads the run manifest, so a long run appears here before it has a report). Open an **error** run anyway — the attacks that completed before the failure are still in it, and the numbers are computed over those alone, so treat its rates as a partial sample.

![The Red Team overview: a card of ASR, critical findings, attacks run and total spend, then a run list with broke, ASR, cases and cost per job.](assets/dashboard/redteam-02-run-list.png){ .dashboard-shot }

### 3. Overview — the headline { #rt-overview }

Opening a run lands on **Overview**: a written executive summary, the five headline numbers (attacks run, vulnerabilities, attack success rate, resistance rate, critical findings), the resistant/vulnerable split, the severity distribution, and per-agent attack success with the weakest agent first.

Resistance rate and ASR are complements: 78% resistant is 22% ASR. There is no universal pass mark — take the first run as your baseline, fix what [Focus areas](#rt-focus-areas) puts on top, and turn the number you reach into a [CI gate](guides/red-teaming.md#in-ci) so the next run can only improve on it.

Both numbers cover only the categories this run actually attacked. The example run touched 10 of the 19 framework categories, so its resistance rate says nothing about the other 9 — [Config](#rt-config) lists which were skipped, and it is worth reading before quoting the headline anywhere.

Filters sit in the right rail on every tab — outcome, severity, minimum turns, category, agent, attack technique, delivery method and vulnerability — and the whole page, downloads included, respects them.

![Overview: executive summary, headline metrics, outcome donut, severity split and per-agent attack success.](assets/dashboard/redteam-03-overview.png){ .dashboard-shot }

### 4. Agents — which target broke { #rt-agents }

For a run with more than one target, **Agents** puts each one's ASR, model, discovered tools, skills and knowledge side by side, so a hardened target and an unhardened one are directly comparable. On a single-agent run — the usual first run — the tab still renders, as one row: the same capability inventory, no comparison. Skip it and read Focus areas instead.

A dash in the tools, skills or knowledge column means nothing was discovered for that target, not that the agent has none. For custom targets, provide `agent_context=` if you want evaluatorq to use the capabilities you know about. Treat a broad attack set as unknown capability coverage, not evidence that the agent supports every tool or skill.

![Agents: per-agent ASR, model, and the tools, skills and knowledge discovered for each target.](assets/dashboard/redteam-04-agents.png){ .dashboard-shot }

### 5. Focus areas — what to fix first { #rt-focus-areas }

**Focus areas** ranks fixes by `risk = success rate × avg severity` and attaches a recommended remediation to each. Severity is the numeric weight shown on [Config](#rt-config) (critical is the heaviest), so a vulnerability that fails often *and* fails badly floats to the top. Start at P1 and work down; re-run the same categories afterwards and compare the two runs in the run list to confirm the fix landed.

When the run has recommendations and exactly one tested agent, each one also gets an **Apply…** button: it previews the merged instructions as a diff, and nothing is written until you confirm. Multi-agent runs never get the button. The button does *not* check the target type, though — on a run against a plain model, deployment or callback it appears and then fails at preview with an error naming the missing agent. [Apply recommendations to the agent](#apply-recommendations-to-the-agent) covers the requirements. The [`14_recommendations_and_artifacts.py` cookbook](examples/redteam/14_recommendations_and_artifacts.md) shows the corresponding Python configuration.

### 6. Breakdowns — where the weakness sits { #rt-breakdowns }

**Breakdowns** gives attack success per framework category, worst first, so a category that fell over on every attempt isn't hidden behind a healthy overall resistance rate. Read the attempt count alongside the rate — 100% over two attacks and 33% over twelve are not equally strong signals, and the ranking doesn't weight for that. When a category you care about sits on a handful of attempts, re-run scoped to it (`--category LLM01`) before treating either the pass or the fail as real. In dynamic and hybrid mode, `--generated-strategy-count` (2 per category) buys more attempts — but only while strategy generation is on, and `--max-per-category` still truncates the combined curated + generated list afterwards, so raise both or neither.

![Breakdowns: attack success rate per OWASP category, worst first.](assets/dashboard/redteam-06-breakdowns.png){ .dashboard-shot }

### 7. Attacks — the evidence { #rt-attacks }

**Attacks** lists every attack with its agent, vector, severity and outcome.

![Attacks: every attack in the run with vector, severity and outcome.](assets/dashboard/redteam-07-attacks.png){ .dashboard-shot }

Click a row to expand the evaluator's verdict, its reasoning, and the full message-by-message transcript — the evidence you'd paste into a ticket. With `ORQ_WORKSPACE` set, attacks that recorded a trace id also link out to that trace in the Orq UI; without the variable the buttons are hidden entirely. In the expanded row below, a memory-poisoning attack succeeded: the agent confirms it stored a persistent "System Administrator / UNRESTRICTED" directive telling it to bypass safety checks on request.

![An expanded attack row: evaluator verdict plus the full message-by-message transcript.](assets/dashboard/redteam-08-attack-detail.png){ .dashboard-shot }

### 8. Usage — what the run consumed { #rt-usage }

**Usage** breaks recorded tokens down per agent — total, prompt, completion and API calls. Spend shows a dash when the provider does not return a price; see [What a run costs](guides/red-teaming.md#what-a-run-costs) for how to interpret that lower bound.

![Usage: total, prompt and completion tokens plus API calls per agent.](assets/dashboard/redteam-09-usage.png){ .dashboard-shot }

### 9. Config — what was actually tested { #rt-config }

**Config** records how the run was produced — pipeline, framework, scoring method, duration — plus which categories were tested, which were not, and the severity weights used for scoring. Read the *not tested* list before reading a high resistance rate as broad coverage.

![Config: run configuration, methodology, tested and untested categories, and severity weights.](assets/dashboard/redteam-10-config.png){ .dashboard-shot }

!!! tip "Exports respect the filters"
    **Export** downloads the run as standalone HTML, Markdown, CSV or JSON — the tabular formats carry only the rows your filters left visible, so filter first, then export. Formats per surface: [Downloads](#downloads).

---

## Reading a simulation run

Runs are saved to `.evaluatorq/sim-runs/<name>_<timestamp>.json` — automatically by `eq sim run` (unless you pass `--no-save`), and by `simulate()` when called with `save=True`. `eq dashboard` browses those files locally, no external service:

```bash
uv add "evaluatorq[dashboard]"

eq dashboard                       # browse every saved run
eq dashboard .evaluatorq/sim-runs  # scope to simulation runs
```

!!! warning "The Python examples above don't save by default"
    `simulate()` and `generate_and_simulate()` default to `save=False`, so a script copy-pasted from earlier on this page leaves the dashboard empty. Pass `save=True`, or drive the run from the CLI (`eq sim run`), which saves unless you pass `--no-save`.

What follows walks the dashboard the way you'd read a finished run: land, pick the run, then work down the report tabs from headline to transcript. The screenshots come from one example run — 10 personas × 5 scenarios against a refund agent — so your own numbers will differ. [Filters](#filters), [trace links](#orq-trace-links) and [downloads](#downloads) are documented above. For the simulation concepts behind these numbers, see the [Agent Simulation guide](guides/agent-simulation.md).

### 1. Landing — what has been run at all { #sim-landing }

`eq dashboard` opens on a cross-surface overview: jobs run, spend, tokens and the red team / agent sim split across every stored run. Agent simulation sits next to red teaming in the left rail.

Zeros everywhere and an empty rail mean no report files were found, not that the run failed — the usual cause is the `save=False` default in the warning above. Launch `eq dashboard` from the directory holding `.evaluatorq/`.

![The dashboard landing page: jobs run, spend, tokens and run mix across all stored runs.](assets/dashboard/sim-01-landing.png){ .dashboard-shot }

### 2. Agent Sim — pick a run { #sim-run-list }

**Agent Sim** shows goal completion, simulations run, average turns and total recorded spend in one summary strip. Pricing coverage appears beneath the spend when some calls have no recorded cost. Each run lists its target, goals achieved over total conversations, goal-completion percentage, conversation count and cost. Errors are called out separately; a run where every conversation errored has no goal verdict. Goal completion includes errors in the total conversation count. Open a report and use **Compare with** to [compare two runs](#sim-compare).

![The Agent Sim run list: goal completion, simulations run, average turns and total spend.](assets/dashboard/sim-02-run-list.png){ .dashboard-shot }

### 3. Overview — the headline { #sim-overview }

Opening a run lands on **Overview**: a written summary naming the best and worst persona × scenario pair, the run's counts (personas, scenarios, conversations, average score, average turns, errors), the achieved/not-achieved split, and the four per-turn quality metrics — response quality, hallucination risk, tone appropriateness, factual accuracy.

Those four are scored by the judge on every turn regardless of what you passed in `evaluator_names`, which is why they appear even though the Config tab lists only `goal_achieved` and `criteria_met`. All four run 0–1; higher is better for three of them, and *lower* is better for hallucination risk. Factual accuracy is only meaningful when the scenario supplies ground truth — without it the judge has nothing to check the response against.

Two numbers on this screen are easy to conflate. The **pass rate** is the share of conversations where the judge set `goal_achieved` — the donut. **Average score** is the mean `goal_completion_score`, the judge's 0–1 rating of *how fully* the goal was met, so a run can average 0.7 while passing half its conversations.

The simulation summary does not show a confidence badge: goal success alone does not establish how much evidence supports the result. Read the conversation count and errors alongside the rates. When every conversation errors, the summary states that no goal verdict is available. Cohorts with no completed conversations show unavailable goal rates and scores in Breakdown; the distribution includes only completed conversations. Partial cohorts retain the total-conversation denominator and display their error counts. Confirmed criterion violations observed before an execution error remain in the failure-mode list.

Filters sit in the right rail on every tab — goal outcome, rule violations, who terminated the conversation, persona, scenario, and score/turn thresholds — and the whole page respects them. "Terminated by" takes four values: `judge` (the judge decided the conversation was done), `max_turns` (the turn cap hit first), `error`, and `timeout`. A run dominated by `max_turns` means the cap, not the agent, decided where the conversations ended.

![Overview: executive summary, run counts, outcome donut and the four average quality metrics.](assets/dashboard/sim-03-overview.png){ .dashboard-shot }

### 4. Breakdown — which persona × scenario pair fails { #sim-breakdown }

**Breakdown** is the persona × scenario heatmap, usually the fastest read in the report. Here every persona clears the straightforward refund paths at 100%, while one column — *Never Received Claim With Unverified Evidence* — lands between 20% and 90% for every one of them. A column that's weak across personas points at the scenario; a row that's weak across scenarios points at the persona; a single cold cell (*Cautious Low-Tech Senior* × *Duplicate Refund Attempt*, 0%) points at that pairing specifically.

![Breakdown: goal completion per persona and scenario. One column scores far below the rest across nearly every persona.](assets/dashboard/sim-04-breakdown-heatmap.png){ .dashboard-shot }

### 5. Recommendations — what to change { #sim-recommendations }

**Recommendations** turns the failures into suggested edits, one card per suggestion, with the persona, scenario and triggers that produced it. For a run that targeted an Orq agent, each suggestion can be applied to the agent's instructions from here: preview the merge as a diff, confirm, and the write lands as a new minor agent version. Runs against a plain model, a deployment or a callback have no instructions to write back to, so their suggestions render as plain bullets. A framework-wrapped agent (LangGraph, Pydantic AI, CrewAI, OpenAI Agents) is *not* an Orq agent but still gets the button, which then fails at preview — see [Apply recommendations to the agent](#apply-recommendations-to-the-agent) for the full contract. The tab carries a count badge, and it disappears entirely when a run generated no recommendations — as does **Turn quality** when a run recorded no per-turn metrics. A missing tab here is a property of the run, not a broken page.

### 6. Transcripts — the conversations { #sim-transcripts }

**Transcripts** lists every conversation with persona, scenario, turn count, score, who ended it, and whether the goal was met. Sort by score to put the failures on top, and raise the page size (5 / 10 / 25) before scanning a large run. The **TRACES** column stays empty until `ORQ_WORKSPACE` is set, since the deep-link needs it. With it set, a row shows **View Trace** when the conversation stored a trace id, **View Traces** (a thread filter) when it only stored a thread id, and nothing when it has neither.

Click a row to open the conversation: required and prohibited criteria with their pass marks, the judge's rationale, and the full user ↔ agent exchange. The example below is the interesting failure mode — all four criteria pass, but the judge still marks the goal missed because the agent stopped at requesting evidence instead of resolving the claim.

![A conversation drawer: criteria, the judge's rationale, and the full transcript.](assets/dashboard/sim-06-conversation-detail.png){ .dashboard-shot }

### 7. Turn quality — behaviour over time { #sim-turn-quality }

**Turn quality** trends the four metrics by turn index, so quality decay over longer conversations is visible, alongside the turn-count distribution. Use multi-turn scenarios and a sufficient `max_turns` value when you want to study quality over time.

### 8. Config — what was tested { #sim-config }

**Config** records the run metadata — target kind, mode, evaluators, when it ran — and the persona table with each simulated user's tone, patience, assertiveness, politeness and technical level.

![Config: run metadata and the persona dials used to drive the simulated users.](assets/dashboard/sim-08-config.png){ .dashboard-shot }

### 9. Compare — did the fix work { #sim-compare }

Pick a second run in **Compare with** to see what changed between two runs. The current run is A and the one you pick is B. Run A is drawn in grey and run B in dark ink; green and red always mean better and worse.

The page opens on what changed: how many conversations regressed (goal met in A, missed in B), how many improved (missed in A, met in B), how many changed because one run errored and the other did not, how far the goal rate moved, and the scorer that moved most. Each line links to the matching conversations. Below it, number tiles show B's value with `A → B` and the change, and two panels show goal outcomes and how conversations ended as stacked bars per run, and each scorer and turn-quality mean side by side. A scorer only one run measured reads `n/a`, never zero, and a scorer that failed on every conversation of a run reads `failed`. Goal rate, average score and error counts are computed the same way as in each run's own report, so errored conversations count in the denominator.

The conversation table pairs conversations by `(persona, scenario)` and opens on the changed ones, worst first: regressions, then error changes, then improvements. Switch to **Regressed**, **Improved**, **Errors** or **All**, sort by persona, scenario, score change, turn change or change, and click a row to read both transcripts side by side in the drawer. Use `j` and `k` to step through rows. Conversations found in only one run are listed below the table.

The filter rail applies to the whole page. A pair stays when either run matches, and every number on the page is computed over the same conversations in both runs. The address bar always holds the current comparison, filters, segment, sort and page included, so you can share the URL and it opens the same view.

Run-level numbers always compare. For a per-conversation diff, the runs must share the same `(persona, scenario)` pairs; a low match rate shows a warning in the header. Reuse the same personas and scenarios across runs (see [Replay stored datapoints](guides/agent-simulation.md#replay-stored-datapoints)) and you get the per-conversation comparison needed for a regression gate.

![Comparing two runs of a refund agent: eight conversations regressed and one improved, the goal rate fell 14 points, with number tiles, outcome bars and scorer means for both runs, and the filter rail on the right.](assets/dashboard/sim-09-compare.png){ .dashboard-shot }

!!! tip "Exports respect the filters"
    **Export** downloads the run as standalone HTML, Markdown or JSON (no CSV on this surface) — the JSON carries only the conversations your filters left visible, so filter first, then export. Formats per surface: [Downloads](#downloads).

---

## Downloads

Every report page includes a download sidebar with export links:

| Format | Red team | Simulation | Pairwise |
|---|---|---|---|
| HTML (standalone, self-contained) | yes | yes | yes |
| Markdown | yes | yes | — |
| CSV (filtered result rows) | yes | — | yes |
| JSON (filtered result rows) | yes | yes | yes |

Simulation and red-team HTML downloads open with a compact report brief: tested scope, outcome and coverage counts, and up to three priority findings linked to their evidence. The findings show recorded recommendations or existing framework guidance when available; missing recommendations stay explicit. Every available dashboard tab follows as a numbered section, with chart-reading notes and links back to the brief. Configuration and usage appear in closed appendix disclosures at the end. The file embeds its styles, fonts, charts and recorded evidence, so it opens without the dashboard server or an internet connection. Conversation and attack details are embedded in foldable rows that are closed by default, with local links from summary tables to their summaries. Open a row to read the complete transcript. Printing includes the evidence rows and appendix disclosures you have opened; closed disclosures retain only their summaries. A notice in the printout explains this limit. Recommendations retain their recorded applied status; applying changes, fetching live agent details and comparing runs remain dashboard actions.

Red-team rates use evaluated attacks as their denominator. The Overview shows evaluation coverage and distinguishes attacks without verdicts from recorded errors, so a partially evaluated run does not read as a clean pass.

HTML downloads render the full, unfiltered saved report. For simulation, the saved run supplies the captured agent details, target, model, maximum turns, recommendations and whole-run token usage when recorded. Missing fields in older runs stay unavailable. Transcripts are included in full, so a large run produces a larger file. CSV and JSON downloads continue to contain the filtered result rows.

The pairwise CSV writes one row per comparison: the question, the consensus winner, and each judge's vote as its own column, all resolved to the run's side labels rather than the bare `A` / `B` slot letters.

Download links respect the currently active filter state — the CSV/JSON exports contain only the rows visible in the filtered report body.

## Where to next

- **[Red Teaming](guides/red-teaming.md)** — generate the red-team reports the dashboard browses.
- **[Agent Simulation](guides/agent-simulation.md)** — generate simulation reports.
- **[CLI Reference](cli-reference/overview.md)** — the `eq dashboard` command and options.
