# Trace signals

A **signal** is a deterministic measurement or tag computed from an agent trajectory, with evidence showing which steps or tool calls contributed to it.

Use signals when you need repeatable measurements of an agent run's structure, tool use, or autonomy. They do not judge whether an answer is correct or helpful; use an evaluator or judge for answer quality.

## Compute signals from a trace

`compute_signals()` takes an `AtifTrajectory` and returns a `SignalReport`. Convert a trace's raw Orq span records to ATIF first with `OtelTrace.from_orq(...).to_atif()`. The raw span records are the `v3spans` list from Orq's trace endpoint; `evaluatorq.formats` converts records but does not fetch them.

Responses conversion preserves each response's tool definitions on its originating ATIF step. Schema checks use those definitions for that step, including an explicitly empty list, so a later response's schema does not change how an earlier call is checked. Unsupported custom, MCP, or builtin tool activity remains recorded as source data; signals that depend on complete tool activity return `no_basis`, and independent signals remain available. [Trace Insights](insights.md#trace-signals) computes these reports automatically and saves them with its trace results.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluatorq.formats import OtelTrace
from evaluatorq.signals import SignalsConfig, compute_signals

raw_spans = [
    {
        'span_id': 'chat-1',
        'parent_span_id': None,
        'name': 'chat-completion',
        'started_at': '2026-09-29T10:00:00Z',
        'ended_at': '2026-09-29T10:00:02Z',
        'attributes': {
            'gen_ai.operation.name': 'chat-completion',
            'gen_ai.response.model': 'eu.gpt-6.1-sol',
            'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'Hi'}]}],
            'gen_ai.output.messages': [{'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello.'}]}],
        },
    }
]

trajectory = OtelTrace.from_orq(raw_spans).to_atif(agent_name='support-agent')

with TemporaryDirectory() as directory:
    config_path = Path(directory) / 'signals.json'
    config_path.write_text('{"tool_roles": {"FetchPage": "webview"}}', encoding='utf-8')
    config = SignalsConfig.from_file(config_path)
    report = compute_signals(trajectory, config=config)

print(report.values())
print(report.results['tool_error_count'].preconditions)
print(report.results['assistant_message_count'].evidence)
```

The example uses a local span fixture with the same shape as an Orq export, so it runs without credentials. To measure a recorded run, get its trace ID from the Orq trace URL, set `ORQ_API_KEY`, and fetch the span list:

```bash
TRACE_ID=trace_123
curl -fsS "${ORQ_BASE_URL:-https://my.orq.ai}/v2/traces/${TRACE_ID}/v3spans" -H "Authorization: Bearer ${ORQ_API_KEY}" > spans.json
```

Replace the fixture's `raw_spans` list with `json.loads(Path('spans.json').read_text(encoding='utf-8'))` after importing `json`. `SignalsConfig.from_file()` accepts a JSON object containing any subset of config fields; omitted fields keep their defaults, and unknown fields raise a validation error.

You can run one signal by passing `only=['tool_error_count']` to `compute_signals()`. Group D tags load their metric dependencies automatically. `report.values()` returns only signals with a basis; inspect `report.results[name]` for evidence, preconditions, the approximation flag, or a `no_basis` explanation.

To use a signal as an evaluatorq scorer, create one with `signal_evaluator('tool_error_count', threshold=2)` or create all tag scorers with `signal_evaluators()`, then pass the result in `evaluators=[...]` to `evaluatorq()`. Group D tags fail when they fire; numeric measurements need a threshold to produce a pass or fail. `signal_evaluators()` also accepts `names='all'`, one group with `names='A'` through `'D'`, or an explicit list of names.

```python
from evaluatorq.signals import signal_evaluators

evaluators = signal_evaluators()  # Every trajectory tag, using the bundled thresholds.
print([evaluator['name'] for evaluator in evaluators])
```

## Results and evidence

Each `SignalResult` carries a value, step and call evidence, preconditions, and an `approximate` flag. Evidence identifies the ATIF step and, for tool-call measurements, the call id; `agent_path` identifies the root or subagent trajectory. A failed precondition is recorded explicitly. If the signal cannot be computed, `no_basis` explains why and its value is `None`.

Some signals can be approximated when a trace lacks precise timestamps. For example, consecutive model-call step timestamps can bound an approximate model interval when OTel spans lack start and end pairs. The result keeps that value marked `approximate=True`. Compaction and copied-context steps are excluded from step and call measurements; embedded subagent trajectories still contribute to depth and invocation counts.

## Inputs and coverage

Signals use information represented by the trajectory. Tool timing and tool definitions are preserved by the OTel-to-ATIF conversion when present in the OTel spans. Other ATIF sources can still support signals from their own fields, but they do not gain timing or schemas they never recorded. Missing tool result status, finish reasons, timestamps, or tool definitions can make a signal's precondition fail or qualify its value as approximate.

Jev classification is an optional preparation step for tools whose roles are not in `SignalsConfig.tool_roles`. `classify_tool_roles(trajectories, config)` classifies each unknown tool name once and returns a config with the roles merged in; it sends tool names to Jev without sending tool arguments. It uses `SignalsConfig.classifier.model`, which defaults to `typesafe/jev-latest` and can be set through `EVALUATORQ_CLASSIFIER_MODEL`, and resolves a client from your configured credentials unless you pass `client=`; signal computation itself makes no model calls. Existing role entries take precedence, and unsuccessful classifications are recorded as `other` with a warning. You can instead provide role names directly in the JSON config.

`SignalsConfig` also controls error detection, argument canonicalisation, retry definitions, empty results, and tag percentiles. For example, the same JSON file can set `{"error_detection": "status_and_content", "retry_definition": "same_tool_args_within_n", "retry_window": 3, "tag_percentiles": {"error_heavy.tool_error_rate": 90}}` to detect errors in result text, look back three calls for retries, and use the cohort's 90th percentile for that tag clause. Its `classifier.enabled` field is descriptive configuration; call `classify_tool_roles()` explicitly when you want model-assisted classification.

## Signal reference

The groups separate measurements by the kind of trajectory behavior they describe. Groups A–C return counts, ratios, durations, booleans, or per-tool breakdowns. Group D returns named tags whose fired rule clauses are recorded in `reason` and whose supporting steps and calls appear in `evidence`.

The Insights dashboard displays these groups as **L1 Structure**, **L2 Tools**, **L3 Autonomy**, and **L4 Tags**, respectively, and shows L4 by default. Select another level or **All** to inspect the other measurements. Saved reports and Python group arguments use `A`, `B`, `C`, and `D`.

### A — Structure

| Signal | Measures | Additional input needed |
|---|---|---|
| `max_depth` | Deepest subagent nesting level, with the root at depth zero. | Embedded subagent trajectories |
| `llm_call_count` | Explicit model-call counts, or one call for each agent step when no count is recorded. | Agent steps; explicit call counts when available |
| `user_message_count` | Number of user steps. | None |
| `assistant_message_count` | Number of agent steps. | None |
| `turn_count` | Number of user-to-agent turns. | None |
| `total_input_tokens` | Input tokens after subtracting cached tokens from prompt tokens. | Token metrics |
| `total_output_tokens` | Completion tokens across model calls. | Token metrics |
| `total_tokens` | Input plus output tokens. | Token metrics |
| `cache_read_token_share` | Cached tokens divided by prompt tokens. | Cached and prompt token metrics |
| `peak_context_tokens` | Largest prompt token count on one model call. | Prompt token metrics |
| `tool_call_count` | Number of tool calls. | Tool calls |
| `unique_tools_used` | Number of distinct tool names called. | Tool calls |
| `tool_call_value_count` | Number of calls for each tool name. | Tool calls |
| `bash_command_value_count` | Shell command calls by command family. | Tool calls classified as bash |
| `webview_value_count` | Web browsing and fetch calls. | Tool calls classified as webview |
| `loaded_skill_value_count` | Calls that load a named skill. | Tool calls classified as skill |
| `subagent_invocation_count` | Embedded subagent trajectories, including empty or unlinked children. | Embedded subagent trajectories |
| `total_subagent_messages` | Steps across embedded subagent trajectories. | Embedded subagent trajectories |
| `avg_messages_per_subagent_invocation` | Average subagent steps per invocation. | Embedded subagent trajectories |
| `model_count` | Number of distinct models used. | Model name on the step or trajectory |
| `provider_count` | Number of distinct providers inferred from model names. | Model name on the step or trajectory |
| `finish_reason_length_count` | Model calls that ended because of a length limit. | Invocation finish reasons |

### B — Tools

| Signal | Measures | Additional input needed |
|---|---|---|
| `tool_error_count` | Tool results marked as errors. | Tool results with status or error metadata |
| `tool_error_rate` | Error results divided by tool calls. | Tool results with status or error metadata |
| `duplicate_tool_call_count` | Repeated calls with the same tool and canonical arguments. | Tool calls |
| `tool_retry_count` | Calls that repeat a prior failed call under the configured retry definition. | Tool calls; error status when a matching prior call exists |
| `tool_succeeded_after_retry_count` | Retries that follow a failed call and then succeed. | Tool calls; error status when a matching prior call exists |
| `invalid_schema_tool_call_count` | Calls that do not conform to a recorded tool definition. | Tool calls and tool definitions |
| `consecutive_same_tool_max` | Longest run of calls to one tool. | Tool calls |
| `consecutive_command_family_max` | Longest run of shell commands in one command family. | Tool calls classified as bash |
| `identical_tool_call_run_count` | Runs of identical tool calls. | Tool calls |
| `tool_oscillation_count` | Alternating tool-and-argument patterns, including two alternating argument sets on one tool. | Tool calls and canonical arguments |
| `tool_loop_count` | Detected repeated or oscillating tool-call loops. | Tool calls |
| `distinct_tool_arg_ratio` | Distinct canonical tool arguments divided by calls. | Tool calls |
| `empty_tool_result_count` | Results matching configured empty values or literals. | Tool results |
| `max_tool_result_bytes` | Largest tool result size in bytes. | Tool results |
| `total_tool_result_bytes` | Total tool result size in bytes. | Tool results |

### C — Autonomy

Autonomous step counts include delegated subagent work between root user messages. `max_autonomous_duration_ms` uses root-agent timestamps, so a child step's later timestamp does not extend the root segment.

| Signal | Measures | Additional input needed |
|---|---|---|
| `autonomous_segment_count` | Runs of agent work between user messages. | User and agent steps |
| `max_autonomous_steps` | Most model responses plus tool calls in one autonomous segment. | User and agent steps, tool calls |
| `avg_autonomous_steps` | Average model responses plus tool calls per autonomous segment. | User and agent steps, tool calls |
| `max_llm_tool_cycles` | Longest consecutive run of tool-calling agent responses, measured separately along each agent path across the trace. | Agent steps grouped by path, tool calls |
| `terminal_answer_present` | Whether the final root agent step contains a terminal answer. | Root agent steps |
| `human_interruption_count` | User messages arriving before the root agent has given a terminal answer. | User and agent steps |
| `subagent_step_share` | Share of model responses plus tool calls occurring in subagent trajectories. | Embedded subagent trajectories |
| `subagent_message_share` | Share of messages occurring in subagent trajectories. | Embedded subagent trajectories |
| `parallel_tool_batch_count` | Agent steps containing multiple parallel tool calls. | Tool calls grouped by step |
| `max_parallel_tool_calls` | Largest parallel tool-call batch. | Tool calls grouped by step |
| `wall_time_ms` | Elapsed time across the recorded run. | Start and end timestamps |
| `active_time_ms` | Time spent in recorded model and tool operations. | Model and tool timing |
| `llm_time_ms` | Time spent in model calls. | Model call bounds or consecutive model-call step timestamps |
| `tool_time_ms` | Tool time that does not overlap model-call time. | Tool result start and end timestamps |
| `max_autonomous_duration_ms` | Longest root-agent segment duration between user messages. | Root user and agent steps with timing |

### D — Tags

A tag is false when a mandatory condition is known to be false, or when the remaining unknown conditions cannot supply enough matches for its rule. It is true when its mandatory conditions and enough alternatives are known to match. Missing measurements produce **No basis** only when the result could still be either true or false.

| Tag | Meaning |
|---|---|
| `long_autonomous_run` | Extended execution without user input. |
| `delegation_heavy` | Subagent use is high relative to the calibration cohort. |
| `error_heavy` | Tool errors dominate execution. |
| `tool_churn` | High tool activity coincides with low argument diversity or frequent retries. |
| `tool_loop` | Repeated calls, oscillation, parameter drift, or a retry storm. |
| `stalled` | The run ends without an answer after looping or repeated failure. |
| `output_heavy` | Tool results are unusually large. |
| `inefficient_execution` | At least two structural inefficiency indicators fire together. |

The bundled thresholds are calibrated on local Claude Code sessions. The `cohort` field in `tag_thresholds.json` records the source, session counts, calibration date, and holdout count. These thresholds describe that cohort; treat the tags as cohort-relative indicators and recalibrate or override thresholds before applying them as universal limits to a different agent or workload.
