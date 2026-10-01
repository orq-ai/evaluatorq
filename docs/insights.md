# Trace Insights

**Trace Insights** selects a population of Orq traces or local trace snapshots, answers fixed label questions, and discovers groups in the traces by clustering summary text. Use it when you want to review patterns in trace traffic; use the trace finder alone when you only need to locate individual Orq matches.

## What a run produces

An Insights run keeps three different things separate. A **classifier** is the model that answers fixed-form questions about each trace and, when you provide a semantic query, decides whether the trace matches it.

| Result | How it is chosen | What it contains |
|---|---|---|
| Population | Query and filters, Finder export, or local trace snapshot | The traces included in the run. Labels never change population membership. |
| Labels | Fixed classifier questions | One answer per trace for each yes/no (`noul`), choice, or score question, with confidence and probabilities. |
| Discovered dimensions | Text fields from a per-trace summary | Two-level clusters and saved 3D UMAP coordinates for traces in groups large enough to project. |

The built-in discovered dimensions are `intent` (summary field `request`), `failure` (`assistant_errors`), and `sentiment` (`sentiment_explanation`). The summary model writes these text fields from the trace's conversation, including assistant tool calls and fixed diagnostic categories for failed tool results. Tool result bodies are omitted from saved projections because they may contain credentials the projector cannot recognize. The classifier answers labels such as sentiment, customer satisfaction, or your own questions. When you request the discovered `sentiment` dimension without a sentiment label, Insights adds that label so it can group the explanations by sentiment.

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
        summary_model="openai/gpt-6-luna",
        classifier_model="typesafe/jev-latest",
        embedding_model="openai/text-embedding-3-small",
    )
    print(run.status, run.counts)


asyncio.run(main())
```

`InsightsPopulation` also accepts just facets and numeric bounds, `InsightsPopulation.from_finder_export(path)` for the matches from an `eq find --json` export, or `InsightsPopulation.from_snapshot(path)` for a local JSON file validated as `trace_finder.models.Snapshot`. Explicit `start` and `end` bounds must include a timezone offset, and `start` cannot be later than `end`; invalid bounds fail before traces are loaded. File sources already fix the population, so do not combine them with a query, facets, numeric bounds, time window, or limit. A Finder export contains trace IDs and classification metadata; Insights fetches its messages from Orq. A local snapshot contains the messages and needs no Orq trace fetch. A full Insights run needs an Orq credential for model requests.

## Run from the CLI

The `eq insights` command accepts trace population filters, label presets or JSON files, and discovered dimensions. Install `evaluatorq[insights]` and use `ORQ_API_KEY`, the profile selected in dashboard Settings, or an explicit `--profile NAME`. An explicit profile wins over Settings, and Settings wins over the environment. The selected profile supplies its API key and host for both traces and model calls; if it is unavailable, the command exits with an error instead of using another key. With `--query`, Insights asks the classifier to match traces and removes non-matches. Requested labels are sent with the same per-trace request. A trace whose match judgment fails is excluded from summaries and dimensions; the run counts the failed judgment and explains it in a warning. Filter-only runs and `--from-finder` runs do not ask a new match question.

```bash
eq insights --query "customers asking about refunds" --agent support-bot --label sentiment --label customer_satisfaction --dimension intent --dimension failure --limit 500 --json refunds-insights.json
```

The command prints cluster and label summaries, failed-trace counts, projection coverage, and the stored run path. The priority matrix uses `intent` when selected, or your first `--dimension` otherwise; `--priority-dimension` must name one of the selected dimensions. Use `--from-finder PATH` to use matched trace IDs from a finder export instead of `--query`, or `--from-snapshot PATH` to use local traces with embedded messages. Both file sources reject other population filters, `--window-days`, and `--limit`. Run `eq insights --from-snapshot PATH --preview-input` to measure truncation without model requests or an Orq credential. A full Insights run still needs an Orq credential for model requests. `--no-cache` disables the local summary and embedding cache. The summary model defaults to `openai/gpt-6-luna`. If you omit `--classifier-model`, Insights uses `EVALUATORQ_CLASSIFIER_MODEL`, then the classifier model saved in dashboard Settings, then `typesafe/jev-latest`; an explicit option overrides those defaults.

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

Pass `--coding` (or `coding_analysis=True` in Python, or tick **Coding agents** in the dashboard wizard) to add coding-agent labels to a run. Insights first asks the classifier whether each trace comes from a coding agent, using only a count of the tools, shell programs, and skills the trace called. Traces it answers yes for get two more sets of questions; other traces get only the labels you selected.

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

## Review a run

The Insights review dashboard turns a saved run into four linked views: Themes, Activity, Map, and Compare. Use it to move from a pattern in the run to the traces that make up that pattern; use the trace detail page or Orq link to inspect the original conversation.

Open **Insights** in the dashboard navigation and select a saved run. The new review page shows its run status, cost coverage, warnings, highlights, and matching trace list. Choose **All Insights runs** to return to the overview. The page fetches saved run data from a same-origin JSON route; if that request fails, it shows an error and a retry action. Snapshot runs have no Orq trace link because their traces are local.

The four views use one shared set of filters. Selecting a theme, label value, map region, or Compare cell narrows the same trace list and updates counts and highlights. The address bar stores the view, dimension, filters, and selected detail; copy the URL to share the view, and use Back or Forward to restore earlier selections. The page validates restored values against the saved run and ignores selections the run does not contain. Search narrows the trace list without changing the shared population filters.

**Themes** groups traces by the selected discovered dimension and shows label distributions for the visible cohort. The Frustration score uses the saved 1–5 `user_frustration` answers: theme averages show one decimal place, trace answers show whole numbers, and `—` means there is no measured answer. Its tooltip gives the number of measured traces. A missing answer is not a score of zero.

**Activity** summarizes saved tool, shell-command, and skill counts for the current filtered cohort. A trace is eligible for an Activity count only when it has recorded `tool_stats`; a missing record is unknown, while a recorded empty set means no use was recorded. Each category reports distinct items and traces with recorded use. Tools and Shell commands also show call totals; Skills show trace coverage only because a skill load is a recorded call, not proof that the skill succeeded. Coverage percentages use eligible traces as their denominator, and the weekly chart uses UTC weeks. Selecting an Activity item narrows the trace list to traces that used it; rankings and pairings still describe the whole filtered cohort. Pairings show items recorded on the same traces, with overlap counts and percentages among traces using the selected item. Co-occurrence describes observed overlap and does not establish that one item caused another. A `Potentially state-changing` badge is a heuristic based on a shell-command name, not a finding that a trace changed state.

**Map** draws a two-dimensional display projection from the run's saved three-dimensional UMAP coordinates. It does not rerun embeddings or change the saved run. Five valid points are required; invalid coordinates are skipped, and unavailable projections show why the map is empty. Filtered points remain visible in a muted style so you can compare them with the selected cohort. Select a point or brush an area to filter the shared trace list.

The dashboard uses the source selected under **Settings → Authentication** for facet values and Insights runs. You can use `ORQ_API_KEY` from the dashboard process, a local Orq CLI API-key profile, the CLI's OAuth session, or an API key entered in Settings. The selected method persists in the dashboard settings file. A manually entered key is encrypted in that file with its encryption key stored in macOS Keychain; on other platforms, configure `EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY`. CLI OAuth keeps token storage and refresh inside the `orq` CLI. If the selected credential needs attention, a startup toast links to Settings; if CLI OAuth is rejected, sign in again with `orq auth login`. The `eq insights` command uses API-key profiles or environment credentials, and accepts `--profile` to choose a profile explicitly. Without that flag, it uses the saved profile only when **CLI API-key profile** is selected in dashboard Settings, and otherwise uses `ORQ_API_KEY` and `ORQ_BASE_URL`. The dashboard's OAuth and manually entered API-key methods do not change credentials used by `eq insights`. The Python API uses environment credentials unless you pass clients explicitly.

**Compare** shows how the selected dimension's themes break down by assistant errors, a label, or another discovered dimension. Cells show their counts and denominators; selecting a cell applies both values as filters. Categories that do not fit are grouped as `Other`, which cannot identify an individual category to filter.

Select a trace to open its saved summary, label answers, dimension assignments, activity counts, and errors. Select a cluster to inspect its members and examples. **View matching traces** selects that cluster's saved members in the review list. **Open in Orq** is available only when the run has a live Orq trace link. **Export** downloads the saved JSON; **Re-run** opens the launch sheet with available choices prefilled.

### Start a run in the dashboard

Click **+ New Run** to open the launch sheet. Choose one of the starting templates, then select recent traces, traces matching a question, a Finder export, or a local trace snapshot. A question asks the classifier to keep only traces that match, from the selected time window. Recent traces and question matches can use the window, trace limit, and available facet filters. Finder exports and snapshots already define their population, so they cannot be combined with those filters.

Choose discovered dimensions and fixed questions in the sheet. The templates set an initial selection that you can adjust. You can choose individual coding-agent labels; Insights still uses its `coding_agent` classifier gate and asks the selected coding questions only for traces it identifies as coding-agent sessions. Alternatively, keep the full coding bundle enabled. Coding-label answers that were not asked are not failed answers.

You can add a custom yes/no, choice, or 1–5 score question. A choice question needs answer names and descriptions; a score question needs five criteria. The sheet validates question names and criteria before launch, and custom questions cannot reuse preset or coding-label names. The source upload accepts validated Finder exports up to 10 MiB and trace snapshots up to 100 MiB. Use the JSON export from Trace Finder for a Finder source; a snapshot must contain a non-empty `traces` array in the format below.

The sheet estimates trace count, cost, and duration from a priced prior run when it has a usable basis. If the prior run has partial price coverage, the sheet says actual cost may be higher. Without a reliable prior cost, it shows that the estimate is unavailable. The exact cost and progress become visible as the run proceeds. The dashboard continues the run in the background if you restart the dashboard, and saved stage status remains on the run page.

For recent traces or a semantic query, choose any project, agent, model, provider, status, product, trace type, or tool values shown under **Filter by facets**. The values come from Orq for the selected window and reload when you change it. Multiple values within one facet include any matching value; different facets must all match. The choices are applied before the trace limit. If Orq cannot provide facet values, the sheet explains that filters are unavailable. A raw array of session objects is not a snapshot; convert it to `Snapshot` JSON before uploading it.

### Local trace file format

A local trace file contains a `traces` array. Each trace needs an ID, a span ID, a timestamp with a timezone, a non-empty `messages` array, and the metadata fields shown below. Empty strings mean the source did not provide a value; do not fill unknown model or provider names by guessing. Save this example as `traces.json`, start `eq dashboard`, choose **Local trace file** in the new-run wizard, and enter its absolute path. The dashboard validates the file before starting the run.

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

Insights sends at most 50,000 projected UTF-8 bytes per trace to its summary model, plus the analysis prompt. The byte count conservatively bounds tokenizer tokens but does not measure them with the selected model's tokenizer. For a local trace file, the wizard measures truncation before you start the run. For recent traces, a question, or a Finder export, the completed run page reports it after loading the traces from Orq. The report shows how many traces hit the budget, how many whole messages were omitted, and the source and projected byte totals. Tool-result excerpts are shortened separately, so the whole-message count does not describe every byte removed. Split sessions into meaningful shorter traces if the full conversation needs to influence the analysis.

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
