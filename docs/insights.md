# Trace Insights

**Trace Insights** selects a population of Orq traces or local trace snapshots, answers fixed label questions, and discovers groups in the traces by clustering summary text. Use it when you want to review patterns in trace traffic; use the trace finder alone when you only need to locate individual Orq matches.

## What a run produces

An Insights run keeps three different things separate. A **classifier** is the model that answers fixed-form questions about each trace and, when you provide a semantic query, decides whether the trace matches it.

| Result | How it is chosen | What it contains |
|---|---|---|
| Population | Query and filters, Finder export, or local trace snapshot | The traces included in the run. Labels never change population membership. |
| Labels | Fixed classifier questions | One answer per trace for each yes/no (`noul`), choice, or score question, with confidence and probabilities. |
| Discovered dimensions | Text fields from a per-trace summary | Two-level clusters and saved 3D UMAP coordinates for traces in groups large enough to project. |

The built-in discovered dimensions are `intent` (summary field `request`), `failure` (`assistant_errors`), and `sentiment` (`sentiment_explanation`). The summary model writes these text fields from the compact conversation view described below, including assistant tool calls but excluding tool-result bodies. Tool result bodies are omitted from saved projections because they may contain credentials the projector cannot recognize. The classifier answers labels such as sentiment, customer satisfaction, or your own questions. When you request the discovered `sentiment` dimension without a sentiment label, Insights adds that label so it can group the explanations by sentiment.

The run writes its JSON result to `.evaluatorq/insights-runs/` and its manifest under `.evaluatorq/insights-runs/.manifests/`; `EVALUATORQ_DIR` changes the base directory. It does not store message bodies or embedding vectors in the run JSON.

## Run from Python

Call `insights()` with an `InsightsPopulation` and optional labels and discovered dimensions. This example restricts the population to one agent and adds a custom label alongside two presets. It reads live traces and makes model requests through Orq, so set `ORQ_API_KEY` before running it.

Install the `insights` extra before importing this package path with `uv add "evaluatorq[insights]"`.

```python
import asyncio

from evaluatorq.insights import InsightsPopulation, LabelSpec, insights, presets
from evaluatorq.trace_finder.models import FacetSelection, NumericFilters


async def main() -> None:
    run = await insights(
        population=InsightsPopulation(
            query="customers asking about refunds",
            facets=FacetSelection(agent_name=frozenset({"support-bot"})),
            numeric=NumericFilters(),
            window_days=7,
            limit=500,
        ),
        labels=[
            presets.SENTIMENT,
            presets.CUSTOMER_SATISFACTION,
            LabelSpec(
                name="needs_human_follow_up",
                kind="noul",
                instructions="Decide whether a human should follow up on this conversation.",
                criteria={
                    "true": "The customer needs a human to resolve the request.",
                    "false": "The assistant resolved the request without human help.",
                },
            ),
        ],
        dimensions=["intent", "failure", "sentiment"],
        summary_model="openai/gpt-6-sol",
        classifier_model="typesafe/jev-latest",
        embedding_model="openai/text-embedding-3-small",
    )
    print(run.status, run.counts)


asyncio.run(main())
```

`InsightsPopulation` also accepts just facets and numeric bounds, `InsightsPopulation.from_finder_export(path)` for the matches from an `eq find --json` export, or `InsightsPopulation.from_snapshot(path)` for a local JSON file validated as `trace_finder.models.Snapshot`. Explicit `start` and `end` bounds must include a timezone offset, and `start` cannot be later than `end`; invalid bounds fail before traces are loaded. File sources already fix the population, so do not combine them with a query, facets, numeric bounds, time window, or limit. A Finder export contains trace IDs, classification metadata, and its original query window; Insights fetches the matched messages from Orq using that window and the exact trace IDs. Older exports without a recorded window cannot be reloaded safely and must be exported again from Trace Finder. A local snapshot contains the messages and needs no Orq trace fetch. A full Insights run needs an Orq credential for model requests.

## Run from the CLI

The `eq insights` command accepts trace population filters, label presets or JSON files, and discovered dimensions. Install `evaluatorq[insights]` and use `ORQ_API_KEY`, the profile selected in dashboard Settings, or an explicit `--profile NAME`. An explicit profile wins over Settings, and Settings wins over the environment. The selected profile supplies its API key and host for both traces and model calls; if it is unavailable, the command exits with an error instead of using another key. With `--query`, Insights asks the classifier to match traces and removes non-matches. Requested labels are sent with the same per-trace request. A trace whose match judgment fails is excluded from summaries and dimensions; the run counts the failed judgment and explains it in a warning. Filter-only runs and `--from-finder` runs do not ask a new match question.

```bash
eq insights --query "customers asking about refunds" --agent support-bot --label sentiment --label customer_satisfaction --dimension intent --dimension failure --limit 500 --json refunds-insights.json
```

The command prints cluster and label summaries, failed-trace counts, projection coverage, and the stored run path. The priority matrix uses `intent` when selected, or your first `--dimension` otherwise; `--priority-dimension` must name one of the selected dimensions. Use `--from-finder PATH` to use matched trace IDs from a finder export instead of `--query`, or `--from-snapshot PATH` to use local traces with embedded messages. Both file sources reject other population filters, `--window-days`, and `--limit`. Run `eq insights --from-snapshot PATH --preview-input` to measure truncation without model requests or an Orq credential. A full Insights run still needs an Orq credential for model requests. `--no-cache` disables the local summary and embedding cache. Each model comes from a role when you leave its option out: `--summary-model` from the `smart` role (`openai/gpt-6-sol`), `--classifier-model` from the `classifier` role (`typesafe/jev-latest`), and `--embedding-model` from the `embedding` role (`openai/text-embedding-3-small`). Set a role with `EVALUATORQ_SMART_MODEL`, `EVALUATORQ_CLASSIFIER_MODEL`, `EVALUATORQ_EMBEDDING_MODEL`, the dashboard Settings page, or the global `--smart-model`, `--classifier-model` and `--embedding-model` flags, as [Configuration › Models](configuration.md#models) describes. An explicit `--summary-model`, `--classifier-model` or `--embedding-model` overrides all of them.

After a finder run completes, its **Analyze matches** section lets you download the finder export. Pass the downloaded filename to `--from-finder`; Insights selects only the matched trace IDs from that export. For a file named `trace-finder-42.json`, this command reviews those matches:

```bash
eq insights --from-finder trace-finder-42.json --label sentiment --dimension intent --json insights.json
```

## Add a fixed label

Save one label object or a list of label objects as JSON, then pass its path with repeatable `--label`. A choice label must provide its answer names and descriptions in `criteria`.

```json
{
  "name": "resolution_status",
  "kind": "choice",
  "instructions": "Classify how the assistant handled the customer's request.",
  "criteria": {
    "resolved": "The assistant completed the requested task.",
    "partially_resolved": "The assistant made progress but left part of the request open.",
    "unresolved": "The assistant did not resolve the request."
  }
}
```

```bash
eq insights --query "customers asking about refunds" --label ./resolution-label.json --dimension intent
```

## Analyze coding agents

Pass `--coding` (or `coding_analysis=True` in Python, or tick individual **Coding agent questions** in the dashboard run form) to add coding-agent labels to a run. Insights first asks the classifier whether each trace comes from a coding agent, using only a count of the tools, shell programs, and skills the trace called. Traces it answers yes for get two more sets of questions; other traces get only the labels you selected.

| Label | Kind | Read from | Answers |
|---|---|---|---|
| `coding_agent` | yes/no | Tool counts | Whether the trace is a coding agent's session. |
| `task_type` | choice | Conversation | `bugfix`, `feature`, `refactor`, `docs`, `maintenance`, `investigation`, `review_followup`, `infra_ops`, `planning`, or `question`. |
| `outcome` | choice | Conversation | `done`, `partial`, `not_done`, `cut_off`, or `inconclusive`. |
| `verified` | choice | Conversation | `verified`, `claimed_without_check`, `unverified`, or `not_applicable`. |
| `scope_creep` | yes/no | Conversation | Whether the agent changed things the user did not ask for. |
| `user_corrections` | score | Conversation | How often the user had to correct the agent, from none to repeatedly. |
| `unfixed_error` | yes/no | Tool calls | Whether a tool error was left unresolved at the end. |
| `risky_action` | yes/no | Tool calls | Whether the agent did something destructive or hard to undo without being asked, such as a force push, `rm`, or `gh pr merge`. A plain push is routine. |

The conversation labels are asked in the same classifier call as your selected labels. The tool-call labels get their own call, which reads each tool call's input, status, and the start and end of its output. If a trace's tool calls are too long for one call, Insights cuts the middle only when that loses at most 30% of the text; otherwise it splits the calls into consecutive parts, asks each part separately with the opening user messages repeated, and merges the answers. `risky_action` is yes when any part says yes, and `unfixed_error` takes the answer from the last part, because an error only counts as unfixed if it is still broken at the end. If a part fails and no other part answered yes, the label fails rather than reading as no. You cannot name your own label after one of these. A coding label a trace was not asked counts as not asked, not as failed.

```bash
eq insights --from-snapshot claude-traces.json --label sentiment --label user_frustration --coding
```

## What the classifier reads

The classifier does not read the summary. It reads a compact view of the conversation: every user message with injected `<system-reminder>` blocks removed, the start and end of each assistant message, and one line per group of tool calls. Tool outputs are left out. A tool call shows its file, path, or URL; a shell call shows its program, such as `pytest` or `git status`, and a risky shell command appears in full. When a view is longer than 75,000 characters (about 25,000 tokens), Insights keeps the opening user messages and the end of the trace and cuts the middle.

Every trace also records how often it called each tool, shell program, and skill, counted from its messages. The trace detail page lists them under **Tool use**.

## Trace signals

A **trace signal** is a deterministic measurement of recorded structure, tool use, or autonomy, with evidence and checks for missing data. Insights computes the complete signal report for every selected trace before labeling and summarizing it. A failed labeling or summary request does not discard that report. Signals are not added to the classifier or summary prompts.

Inside the pipeline, a `TraceDocument` holds `metadata: TraceMetadata` and `trajectory: AtifTrajectory`. The conversation lives in the ATIF trajectory; the complete typed signal report lives in `metadata.signals`. Existing message snapshots are converted at the source boundary. Conversion cannot recover information that the source never recorded. For benchmark datasets, the same wrapper carries typed dataset provenance and ground truth outside the trajectory; see [Wrap a dataset trajectory](formats.md#wrap-a-dataset-trajectory) for a runnable example and parser checks.

Message-only snapshots support message and tool-call counts, argument repetition, and result sizes. The measurements below need additional recorded fields; absent fields produce **No basis** under the default signal settings.

| Measurements | Missing input in a message-only snapshot |
|---|---|
| `total_input_tokens`, `total_output_tokens`, `total_tokens`, `cache_read_token_share`, `peak_context_tokens` | Per-model-call token usage, including cached tokens where required. |
| `finish_reason_length_count` | Recorded model finish reasons. |
| `tool_error_count`, `tool_error_rate` | Explicit error or success status on tool results. Content-based error detection can provide an estimate when enabled in `SignalsConfig`. |
| `tool_retry_count`, `tool_succeeded_after_retry_count` | Explicit error or success status on tool results when a matching prior call exists. Content-based detection is optional when results are recorded. |
| `invalid_schema_tool_call_count` | Tool definitions active when each call was made. |
| `wall_time_ms`, `active_time_ms`, `llm_time_ms`, `tool_time_ms` | Recorded model and tool timestamps. Some measurements can be approximate when only model-step timestamps exist. |
| `max_autonomous_duration_ms` | A timestamp on the root user step that starts each segment, plus timestamps on subsequent root agent activity. |

Tags report **No basis** when missing measurements leave the rule's result undetermined. A known false mandatory condition makes a tag **Clear**, even when other measurements are missing; a tag is also Clear when too few conditions could still fire to satisfy its rule. With complete tool-call coverage, retry counts are zero when no recorded call can match a prior call under the configured retry definition, without needing error status. Otherwise missing error status remains unknown under the default settings. Known shell, web, skill, and subagent tools have built-in roles in `SignalsConfig.tool_roles`; add mappings for custom names to count them in those roles. Delegation measurements describe only subagent structure actually captured in ATIF. See the [signal reference](signals.md#signal-reference) for each measurement's inputs. A snapshot that carries richer capture metadata may support more measurements; its saved preconditions show the actual coverage.

In the dashboard, open **Insights**, choose a saved run, and select a row in its **Traces** list. This opens the trace sidebar. Choose **Open full trace**, the primary action, for the trace's dedicated page; use the ↗ action beside it to open the trace in Orq when a live Orq trace link is available. The full trace page opens on the **Trace** tab when the dashboard can read the run's conversations again, and on the **Analysis** tab otherwise, for example when the snapshot file the run read is gone. When the conversation cannot be read again, the **Trace** tab is greyed out with the reason beside it.

The trace page has two tabs. **Trace** shows the conversation and the span tree as the Traces explorer shows them, under its own **Conversation** and **Spans** tabs. **Analysis** shows what the run saved for the trace: its summary, labels, dimensions, tool use, and signals.

The **Trace** tab reads the conversation again from the source the run analysed each time it opens. For a run that read Orq through a query, filters, or a Finder export, it runs the trace query again for the exact trace ID, with the run's filters and recorded time window. It displays the conversation only when Orq returns the same span ID recorded by the run; if Orq now selects a different span, the tab reports that the analysed conversation is unavailable instead of showing a different transcript. This can happen when spans change after the run. Older runs without a recorded time window cannot be reloaded safely. The **Spans** tab loads the current trace span tree independently of the conversation, so it remains available when the conversation cannot be read, including a missing timestamp, source query failure, or span mismatch. A trace without a recorded time cannot be read from Orq again. A run started from the dashboard remembers the Orq account and workspace it read. When the account selected in Settings is that account, the tab reads with the run's workspace and project, even if Settings now selects another workspace. When another account is selected, the tab makes no Orq request and names the account mismatch instead.

New dashboard runs record a versioned identity verified from Orq: API keys use the single authenticated workspace ID, or the single visible project ID when Orq rejects workspace listing with a project-key 403; OAuth uses the signed-in user ID. Truncated or ambiguous workspace/project listings cannot verify an identity, so launch logs a warning and keeps the strict legacy credential fingerprint instead. A rotated key continues to read the same provider principal without treating profile names or key bytes as identity. Older saved scopes contain only the original credential fingerprint and remain strict; the dashboard does not infer identity from profile labels. Runs without a recorded scope use the current Settings scope.

A run started from the CLI or the Python API, or before the dashboard recorded this, reads with the account and workspace currently selected in Settings, so a trace from another workspace shows that Orq did not return it until you switch back. For a run read from a snapshot file, it reads that file and has no **Spans** tab, because span data only exists in Orq. The conversation is unavailable once that file is moved or changed. A run started from a dashboard upload loses its conversation when the dashboard deletes the uploaded file after the run, and the tab says so. Any other run, such as a dataset run, cannot read its trace source again, so its **Trace** tab is greyed out. When the conversation cannot load, including when the run or trace no longer exists or the dashboard cannot be reached, the tab shows the reason instead.

For a trace with a saved report, both the sidebar and the full trace page's **Analysis** tab show L4 tags that fired as prominent chips that open the matching signal; when no tags fired, they say so explicitly. The smaller **All signals** foldout starts collapsed below the chips. Expand **All signals** to reveal four independently expandable groups: **L4 Tags**, **L3 Autonomy**, **L2 Tools**, and **L1 Structure**, in that order. Each group starts collapsed and expands independently. Level headers have distinct backgrounds, signal rows are indented beneath them, and preconditions and evidence are inset within each signal. Open signal rows stay highlighted, and disclosure chevrons are aligned on the left. Each signal row expands to show its primary value and outcome, with precondition checks visible inside the opened row. Inside an expanded signal row, open **Evidence** for step or call references; this foldout starts collapsed. L4 rows show the tag outcome and supporting evidence without a technical attribute-name block. Flagged L4 rows keep their outcome in the row header and open directly to the preconditions, without a duplicate result card. **Flagged** means a tag fired, **Clear** means it did not fire, **Approximate** marks an estimate, and **No basis** means required data is missing. Older runs without a saved report show that signals were not measured. A measured zero is a real value. Unsupported custom or MCP tool activity makes affected tool measurements unavailable instead of displaying an incomplete count as zero. Inspecting the original conversation still requires its source.

New runs use storage schema version 2 and save each trace's full `signals` report and `source_coverage` summary, plus the effective `config.signals` settings and resolved tag thresholds. The coverage summary retains known provenance, span counts, missing call IDs, and enrichment failures; arbitrary source payload sidecars are excluded. Version 1 runs remain readable and show their signals as unmeasured. Insights does not save a separate local ATIF document or conversation body, and opening a saved signal report makes no source or model requests. The trace page's **Trace** tab re-reads the conversation from the run's source on request instead.

Read the saved reports from Python. This prints one line per saved trace; if there are no saved runs, it prints nothing.

```python
from evaluatorq.insights.models import InsightsRun
from evaluatorq.insights.store import list_runs

for path, run in list_runs():
    if not isinstance(run, InsightsRun):
        print(path.name, "Unreadable run:", run)
        continue
    for trace in run.traces:
        if trace.signals is None:
            print(trace.trace_id, trace.span_id, "Unmeasured")
        else:
            print(trace.trace_id, trace.span_id, trace.signals.values())
```

Pass `signals_config=SignalsConfig(...)` to `insights()` to set tool roles, retry windows, or tag thresholds. See [Signals](signals.md) for the available measurements and configuration. Insights does not classify tool roles automatically. Provide known roles through `SignalsConfig.tool_roles`, or prepare a config with `classify_tool_roles()` as described in Signals and pass that config as `signals_config`. Computing signals itself makes no model calls.

## Review a run

The Insights review dashboard turns a saved run into four linked views: Themes, Activity, Map, and Compare. Use it to move from a pattern in the run to the traces that make up that pattern; use the **Trace** tab of the trace detail page or the Orq link to inspect the original conversation.

Open **Insights** in the dashboard navigation and select a saved run. The new review page shows its run status, cost coverage, warnings, highlights, and matching trace list. Choose **All Insights runs** to return to the overview. The page fetches saved run data from a same-origin JSON route; if that request fails, it shows an error and a retry action. Snapshot runs have no Orq trace link because their traces are local.

The four views use one shared set of filters. Selecting a theme, label value, map region, or Compare cell narrows the same trace list and updates counts and highlights. The address bar stores the view, dimension, filters, and selected detail; copy the URL to share the view, and use Back or Forward to restore earlier selections. The page validates restored values against the saved run and ignores selections the run does not contain. Search narrows the trace list without changing the shared population filters.

**Themes** groups traces by the selected discovered dimension and shows label distributions for the visible cohort. The Frustration score uses the saved 1–5 `user_frustration` answers: theme averages show one decimal place, trace answers show whole numbers, and `—` means there is no measured answer. Its tooltip gives the number of measured traces. A missing answer is not a score of zero.

**Activity** summarizes saved tool, shell-command, and skill counts for the current filtered cohort. A trace is eligible for an Activity count only when it has recorded `tool_stats`; a missing record is unknown, while a recorded empty set means no use was recorded. Each category reports distinct items and traces with recorded use. Tools and Shell commands also show call totals; Skills show trace coverage only because a skill load is a recorded call, not proof that the skill succeeded. Coverage percentages use eligible traces as their denominator, and the weekly chart uses UTC weeks. Selecting an Activity item narrows the trace list to traces that used it; rankings and pairings still describe the whole filtered cohort. Pairings show items recorded on the same traces, with overlap counts and percentages among traces using the selected item. Co-occurrence describes observed overlap and does not establish that one item caused another. A `Potentially state-changing` badge is a heuristic based on a shell-command name, not a finding that a trace changed state.

**Map** draws a two-dimensional display projection from the run's saved three-dimensional UMAP coordinates. It does not rerun embeddings or change the saved run. Five valid points are required; invalid coordinates are skipped, and unavailable projections show why the map is empty. Filtered points remain visible in a muted style so you can compare them with the selected cohort. Select a point or brush an area to filter the shared trace list.

The dashboard uses the source selected under **Settings → Authentication** for facet values and Insights runs. You can use `ORQ_API_KEY` from the dashboard process, a local Orq CLI API-key profile, the CLI's OAuth session, or an API key entered in Settings. The selected method persists in the dashboard settings file. A manually entered key is encrypted in that file with its encryption key stored in macOS Keychain; on other platforms, configure `EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY`. CLI OAuth keeps token storage and refresh inside the `orq` CLI. If the selected credential needs attention, a startup toast links to Settings; if CLI OAuth is rejected, sign in again with `orq auth login`. The `eq insights` command uses API-key profiles or environment credentials, and accepts `--profile` to choose a profile explicitly. Without that flag, it uses the saved profile only when **CLI API-key profile** is selected in dashboard Settings, and otherwise uses `ORQ_API_KEY` and `ORQ_BASE_URL`. The dashboard's OAuth and manually entered API-key methods do not change credentials used by `eq insights`. The Python API uses environment credentials unless you pass clients explicitly.

**Compare** shows how the selected dimension's themes break down by assistant errors, a label, or another discovered dimension. Cells show their counts and denominators; selecting a cell applies both values as filters. Categories that do not fit are grouped as `Other`, which cannot identify an individual category to filter.

Select a trace to open its saved summary, label answers, dimension assignments, activity counts, signal report, and errors. Select a cluster to inspect its members and examples. **View matching traces** selects that cluster's saved members in the review list. **Open in Orq** is available only when the run has a live Orq trace link. **Export** downloads the saved JSON; **Re-run** opens the launch sheet with available choices prefilled.

### Start a run in the dashboard

Open **Insights → New run**, or click **+ New run** or **Re-run** on a run page. The new-run page and the run-page dialog render the same three-step form on the server, so both offer the same choices. **Re-run** prefills the form from the saved run. A rejected submission comes back with everything you entered and the error, on the page and in the dialog.

The step bar shows `1 · Traces`, `2 · Analysis` and `3 · Review`. **Continue** validates the current step, and **Start run** appears on the last. The same bar carries a one-line estimate of traces, cost and time that updates as you change the form.

**1 · Traces.** Choose a source:

| Source | Population | Window, limit and filters |
|---|---|---|
| Recent | Recent traces from Orq | Apply |
| Question | Traces from the window that the classifier says match your question | Apply |
| Finder export | The traces in a Trace Finder JSON export | Do not apply; the file defines the population |
| Local file | A trace snapshot with embedded messages | Do not apply; the file defines the population |

For the two file sources, click **Browse…** and choose a file. There is no path field: the dashboard stores the upload under a random name and keeps the original file name for the run. The dashboard accepts Finder exports up to 10 MiB and snapshots up to 100 MiB. A larger file does not upload; run it from the terminal with `eq insights --from-finder PATH` or `eq insights --from-snapshot PATH`. For a snapshot, the form reports the measured truncation before you start.

For Recent and Question, **+ Filter** opens the same filter menu as Traces and Trace search, and each chosen value appears as a removable chip. Every value shows how many traces in the selected window carry it, and values are ordered from most to least frequent. Multiple values within one facet match any of them; different facets must all match, and the filters apply before the trace limit. The menu loads from Orq for the selected window. If Orq rejects the credential or cannot be reached, the form says so and keeps your chosen filters; fix the credential in **Settings → Authentication** and click **Retry**.

**2 · Analysis.** A **Preset** ticks a starting selection that you can then change. The presets are **Find failures** (group by failure and intent; ask about assistant mistakes and frustration), **Understand intents** (group by intent), and **Coding agent** (group by intent and failure; ask about frustration; ask the task type, outcome, unfixed error and risky action coding questions). The rest of the step is:

- **Group traces by** offers the discovered dimensions `intent`, `failure` and `sentiment`.
- **Ask about every trace** offers the labels `sentiment`, `customer_satisfaction`, `made_errors` and `user_frustration`. **+ Write your own question** adds a custom yes/no, choice, or 1–5 score question. A choice question needs answer names and descriptions, and a score question needs five criteria. Custom names cannot reuse a preset or coding-label name, and the form validates them before launch.
- **Coding agent questions** are individual toggles. Insights still uses its `coding_agent` classifier gate and asks the selected coding questions only for traces it identifies as coding-agent sessions. A coding answer that was not asked is not a failed answer.

Without **Assistant mistakes**, the run estimates the error share from the summaries instead of asking it directly. Insights adds the `sentiment` label itself when you group by sentiment and have not selected it.

**3 · Review.** Name the run (optional), choose models, set the number of parallel requests, and read the plan and the estimate. **Models** are four grouped Orq pickers, the same menus as the model fields in Settings:

| Picker | Used for | Default |
|---|---|---|
| Summary model | The per-trace summary | The `summary_model` default of `InsightsConfig` |
| Classifier model | Label questions; must serve `/classify` | The classifier model from Settings |
| Embedding model | Embeddings of the summaries | The `embedding_model` default of `InsightsConfig` |
| Question compiler | Compiling a Question source into a population filter; shown only for that source | The compiler model from Settings |

The run uses the models you pick and records them in its config, so **Re-run** prefills them. Starting a run is rejected when the Orq catalogue says the classifier cannot serve `/classify` or the embedding model is not an embedding model. When the catalogue is unavailable, the pickers become text boxes, a typed id is accepted, and a warning is logged. A blank field falls back to the default.

**Expected stages** lists the stages the run will report, produced by the same plan the run uses, so the progress list on the run page matches it. The estimate is described below.

### Read the estimate

The estimate is a ceiling computed before any request is made. It states a basis beside every number, and it shows `unknown` with the reason instead of a number it cannot ground.

- **Traces.** A Finder export or snapshot gives an exact count, read from the file. Recent and Question give `up to N`: the smallest of the trace limit and the Orq count for your filters. With no filters it uses the workspace's status counts. With several facets it takes the smallest of the per-facet sums, because a trace must match them all. If Orq truncated a facet's values, or you filter by a project id Orq cannot count, the bound falls back to the trace limit and says why. If the file's count cannot be read, the count is `unknown`.
- **Cost.** Per stage, traces multiplied by tokens per trace multiplied by the stage model's price in the Orq catalogue. The token figures are caps, not measurements: the classifier and summary read at most the conversation view cap of 75,000 characters at three characters per token, a classify answer is four tokens, the summary writes up to its `max_tokens`, and an embedding is as long as the summary cap. A stage whose model has no price shows `unknown` and is left out of the total, which then reads `(priced stages only)`.
- **Time.** Median per-trace seconds for each stage from your earlier Insights runs, scaled to the parallelism you chose when those runs recorded theirs. It reads `estimated after your first run` until a completed run exists, and `(timed stages only)` when some stages have no earlier timing. The figure is rough.

A Question source labels every trace in range, but only the matching traces reach the summary and embedding stages. Those stages therefore show a range from zero to every trace, and the total shows `$low to $high`. The review step also lists the stages with their traces, cost, time and basis, names what is unknown, and states what the estimate leaves out: trace selection (question compiling and filter choice), and cluster naming and merging.

Choosing coding questions adds one more classify call per trace, shown as its own row. It is an upper bound, because the call runs only for traces the gate identifies as coding agents, and it is timed at the whole label stage.

The exact cost and progress become visible as the run proceeds. The dashboard continues the run in the background if you restart the dashboard, and saved stage status remains on the run page.

A raw array of session objects is not a snapshot; convert it to `Snapshot` JSON before uploading it.

### Local trace file format

A local trace file contains a `traces` array. Each trace needs an ID, a span ID, a timestamp with a timezone, a non-empty `messages` array, and the metadata fields shown below. Empty strings mean the source did not provide a value; do not fill unknown model or provider names by guessing. Save this example as `traces.json`, start `eq dashboard`, choose **Local file** in the new-run form, click **Browse…** and choose the file. The dashboard validates the file before starting the run.

```json
{
  "traces": [
    {
      "schema_version": 1,
      "trace_id": "session-1",
      "span_id": "session-1-root",
      "timestamp": "2026-09-26T07:28:08Z",
      "messages": [
        {"role": "user", "content": "Summarize the release notes."},
        {"role": "assistant", "content": "I will read the release notes."}
      ],
      "project": "",
      "model": "",
      "provider": "",
      "status": "unknown",
      "product": "local",
      "trace_type": "session"
    }
  ]
}
```

When converting a local session export, keep each session's message order and include assistant `tool_calls` and tool messages in `messages`. The dashboard and CLI reject an empty snapshot. A Finder export cannot replace this file: it does not contain message content and its trace IDs must already exist in Orq.

The summary model receives the compact conversation view, capped at 75,000 characters, plus the analysis prompt. The run form preview and saved population coverage describe a separate Finder projection capped at 50,000 UTF-8 bytes; those counts do not measure the summary prompt. For a local trace file, the run form reports projection truncation before you start the run. For live traces or a Finder export, the completed run reports it after loading. These counts show whole-message omissions and source and projected byte totals. Split long sessions into shorter traces when more of the conversation needs to influence the analysis.

Each run has a state file under `.evaluatorq/insights-runs/.manifests/<run-id>.json` and a report in `.evaluatorq/insights-runs/`. The state file records the stage plan, current stage, outcomes, errors, and report path. The report's `evaluatorq_version` field records the evaluatorq version that produced it; reports saved before this field existed have `null`. The report's `population` block records the source: the filters and time window for a live selection, or the file path and a SHA-256 of its contents (`snapshot_sha256` or `finder_export_sha256`) for a file. A file uploaded in the dashboard is stored under a random name, so its original file name is kept in `source_name`. If a run fails before writing its report, it still appears in the dashboard with the failed stage and error. The worker log is under `.evaluatorq/insights-runs/.logs/<run-id>.log`.

Known provider cost appears in the run header. The JSON report records usage and cost for tracked label, summary, describe, merge, and embed calls in `cost_by_stage`. Semantic-query population compiler and filter-selection calls are excluded.

A missing stage key means no usage entry was recorded; it can reflect a cache hit, no call, or an embedding request that failed before usage was available. A recorded call with no readable usage counts as unpriced. Token totals include only calls with reported usage, so zero tokens can mean that usage was unavailable. Older reports may contain a `null` stage value; its call count is unknown and is omitted from the header's price-coverage count.

“Cost unknown” means no recorded call has a known price. If only some recorded calls have a known price, the run header says “priced for N of M calls” and the displayed amount is a lower bound. Failed calls may have unknown billing, and excluded calls and unpriced or unreported usage mean displayed dollars can be below whole-run spend.

On the run overview, stage dots show each stage's status and a count shows how many are complete. The run page shows the current stage and completed stages out of the plan. Label and Summary stages also show completed traces out of the total, such as `running 27/100`; those counts update while the run is active and remain in the saved manifest.

For the finder’s **Analyze matches** handoff, see [Trace Finder](trace-finder.md).

## Failures and cache

A per-trace label, summary, or dimension failure is recorded on that trace and counted as failed; it does not discard the rest of the run. When a dimension has at least five signal traces overall, a sentiment group with one trace is marked unclassified for that dimension. Sentiment groups with two to four traces form one cluster, and Insights records a warning because UMAP coordinates are unavailable below five traces. A dimension with fewer than five signal traces is skipped with a warning. An embedding batch failure marks that discovered-dimension stage as failed. Any whole-stage failure marks the run `error` and preserves partial results. The CLI exits with status 1 for an `error` run and status 0 for a completed run, including one with failed traces. Check the failed count and stage message before treating a result as complete. An empty population completes with a warning and empty results.

If every summary request fails, Insights marks the summary stage failed and sets the run status to `error`.

Placeholder answers such as “no errors” mean there is no failure text to cluster, so the trace has no signal for the failure dimension.

By default, summaries and embeddings are cached per item in `.evaluatorq/cache/insights.sqlite`. Summary cache keys include the trace, span, summary model, and prompt hash; embedding keys include the model and text hash. This cache is local to the working directory. Pass `cache=False` to Python `insights()` or `--no-cache` to the CLI to bypass it.

The `insights` extra installs numpy, scipy, and umap-learn for vector processing, clustering, and the 3D map; the embedding requests go through the Orq router. Add the `dashboard` extra to the Insights extra to review saved runs locally with `eq dashboard`.
