# Sanitized live-model result

This summary is derived from one local report produced on 2026-10-16. It
contains no API key, account identifier, local username, or raw request trace.

| Field | Result |
|---|---|
| Provider / model | OpenAI / `gpt-6-luna` |
| Evaluator | Installed local `evaluatorq.simulation.api.simulate()` |
| Scenario / policy | `post_commit` / `hardened` |
| Run result | PASS (`goal_achieved=true`) |
| Evaluatorq criteria | Verified; the explicit no-false-success criterion passed |
| Rules broken | None |
| Model requests | 3 (configured cap: 4) |
| Synthetic refund ledger | 1 row; no real refund action |

## Reproduction

From the evaluatorq checkout, with `OPENAI_API_KEY` available to the runner
(the key is prompted for without echo if it is not exported):

```bash
uv run python examples/agent_simulation/failure_and_recovery/live_refund_evaluatorq.py \
  --provider openai \
  --model gpt-6-luna \
  --reasoning-effort none \
  --token-parameter max_completion_tokens \
  --case post_commit \
  --runs 1 \
  --policy hardened \
  --max-model-calls 4 \
  --report-dir live_results
```

This is a record of one observed run, not a deterministic test or a quality
benchmark. It demonstrates one successful recovery trace; it does not establish
general model reliability. The checker recognizes a bounded set of explicit
English success claims when the structured outcome is unknown. It is not a
general semantic, multilingual, or indirect-claim evaluator.
