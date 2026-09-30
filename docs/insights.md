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

Start the dashboard with the `dashboard` and `insights` extras, then open **Insights** in the dashboard navigation. Its overview lists saved runs, their status, trace counts, and dimensions. The Stages column beside each run shows its saved progress; the current stage appears beneath the name of an active run. Hover over a stage dot to read its label and status. Select a run to open a dedicated results page without a second run list; **All Insights runs** returns to the overview. In a run, inspect discovered dimensions as a cluster tree or 3D map, review label distributions, compare dimensions or labels in a crosstab, or filter the trace table. Click a trace ID in the table, a cluster example, or a selected map point to see its saved summary, labels, dimension assignments, and errors. The detail page contains the saved analysis rather than the original conversation; use the Orq link for a full live trace or the source JSON for a local snapshot. The priority matrix is available when the run includes the customer-satisfaction label; otherwise the page explains why it is empty.

The priority matrix uses error share from summaries when you did not request the `made_errors` label. It divides traces whose `assistant_errors` summary field contains a non-placeholder entry by traces in that cluster that have a summary; traces without a summary are excluded, and a cluster without summaries is skipped.

When you request `made_errors`, its successful answers determine error share instead. The share divides true answers by answers that did not fail; missing or failed answers are excluded, and a cluster without any successful answers is skipped.

The **3D Map** tab shows each discovered dimension in its own UMAP projection. Choose **Projection** to change the coordinates, then **Colour by** to compare that shape with another cluster dimension, a fixed label, agent, or project. The colour selection does not move points. Click **Full screen** to explore the map across the window, and select a point to inspect its trace summary and cluster. UMAP needs at least five traces in a clustering group; smaller groups have no coordinates, and a dimension with no coordinates shows an empty state with the reason.

To start a run from the dashboard, click **+ New Run**. Choose recent traces, a semantic query, a Finder JSON export, or a local trace snapshot stored on the dashboard machine. Then select built-in labels and discovered dimensions, review the expected stages, and click **Start run**. The dashboard opens the new run immediately. The overview shows an active run's current stage; its page shows every stage as pending, running, completed, error, or skipped and refreshes while the run is active. The stages reflect your source, labels, and dimensions; each dimension stage embeds, clusters, and maps its traces. The worker continues if the dashboard is restarted. For more filters and custom labels on live or Finder sources, use `eq insights` or the Python API. For custom labels on a local snapshot, use the Python API with `InsightsPopulation.from_snapshot(path)`.

To use a Finder export in the dashboard, complete a Finder run and click **Download and save**. The dashboard saves a copy under `.evaluatorq/finder-exports/`; enter its filename or full path in the Insights wizard. If `EVALUATORQ_DIR` is set, use its `finder-exports/` subdirectory instead. To use another valid Finder export, place it in that directory first. The wizard rejects paths outside this directory and files larger than 10 MiB. If the server cannot save the copy, the download shows an error so you can fix its storage directory before starting Insights.

The dashboard uses the Orq profile selected in **Settings** for both facet values and the run worker, including the profile's API key and host. With **Environment** selected, it uses `ORQ_API_KEY` and `ORQ_BASE_URL`. If a saved profile is unavailable, the wizard cannot start a run until you select another profile or Environment. `eq insights` also reads the saved profile and accepts `--profile` to override it. The Python API uses environment credentials unless you pass clients explicitly.

For recent traces or a semantic query, set the window and choose any project, agent, model, provider, status, product, trace type, or tool values shown under **Filter by facets**. The values come from Orq for that window and reload when you change it. Multiple values within one facet include any matching value; different facets must all match. Your choices are shown on the review step and applied before the trace limit. Finder exports and local snapshots already fix their populations, so the wizard does not add facet filters to those sources. If Orq cannot provide the facet list, the wizard says so; check the connection before starting a run that needs filters. A raw array of session objects is not a snapshot; convert it to `Snapshot` JSON before entering its path.

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

Each run has a state file under `.evaluatorq/insights-runs/.manifests/<run-id>.json` and a report in `.evaluatorq/insights-runs/`. The state file records the stage plan, current stage, outcomes, errors, and report path. If a run fails before writing its report, it still appears in the dashboard with the failed stage and error. The worker log is under `.evaluatorq/insights-runs/.logs/<run-id>.log`.

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
