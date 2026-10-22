# Ambiguous refund recovery — deterministic simulation proposal

This is a **proposed example**, not a merged contribution to evaluatorq.

It tests a specific distributed-systems property: a timeout does **not** reveal whether a refund committed. Its two independent observables are the backend transaction ledger and the confirmation evidence actually received by the agent. A failed status query leaves the agent uncertain even when a commit happened.

## API-free deterministic tests

The tests live in the upstream pytest tree at
`tests/simulation/failure_recovery/`, so they are discovered by the default
`testpaths = ["tests"]` configuration. From the evaluatorq checkout:

```bash
uv run pytest tests/simulation/failure_recovery/
```

This test suite uses a fake refund gateway and scripted model responses. It
does not need `OPENAI_API_KEY`, an Orq key, a paid model call, or a real payment
provider. The evaluatorq integration cases exercise the installed local
`simulate()` implementation with injected user/judge stubs and disable result
uploads and generated summaries.

To run only the tests that exercise the **actual** installed evaluatorq
simulation API:

```bash
uv run pytest tests/simulation/failure_recovery/test_evaluatorq_simulate.py
```

If evaluatorq is not importable, only tests requiring its simulation API are
explicitly **skipped**. A skip is not an integration PASS. The API-free tests
remain runnable without evaluatorq network credentials. `upload_results=False`,
`executive_summary=False`, and `recommendations=False` are set in the simulation
tests alongside injected user and judge stubs.

## Matrix

| Case | Server truth | Agent-observed evidence | Required result |
|---|---|---|---|
| T1 | Pre-commit timeout, no ledger row | Timeout + status NOT_FOUND | UNKNOWN, no success claim |
| T2 | Post-commit timeout, one ledger row | Timeout + status CONFIRMED | CONFIRMED with status evidence |
| T3 | Two requests with the same key | One original ACK and one dedupe ACK | One ledger row |
| T4 | Post-commit timeout, one ledger row | Timeout + status TIMEOUT | UNKNOWN, not falsely confirmed |
| Negative control | Two distinct keys for same order | Two ACKs | `DUPLICATE_REFUND` |

The deterministic suite exercises the independent policy fixture and the real local `evaluatorq.simulation.api.simulate()` implementation, including positive and negative customer-message criterion controls. The simulation's `goal_achieved` means **safe recovery handling**, *not* that a refund settled; `Audit.refund_confirmed_to_agent` is the separate outcome signal.

## What it does *not* prove

- A deterministic English phrase guard rejects common explicit success claims when the structured outcome is `unknown`; it is not a general semantic, multilingual, or indirect-claim evaluator.
- The injected `TraceJudge` receives evaluatorq goal/criteria context and reports the message-safety criterion from the local structured and phrase audits. This does not provide full semantic interpretation of arbitrary customer prose.
- The fake ledger is not a real payment processor: no distributed consistency, auth, actual payments, or external API guarantees.
- An agent can safely report uncertainty even when the mock backend has committed. Ledger truth alone must not authorize a success claim.
- The optional live-model demo is separate evidence and is not required by these tests.

## Suggested upstream changes

Start with this one example; request maintainer feedback before changing APIs, exposing a new skill parameter, or claiming a production-ready resilience framework. If accepted, factor out common fixtures and extend with business-logic negative controls.

The repository's default pytest path discovers these tests under
`tests/simulation/failure_recovery/`. See `examples/agent_simulation/README.md`
for the API-free command. See `../../../LIVE_TEST_SETUP.md` for the separate,
credentialed real-model scenario. If evaluatorq's simulation contract changes,
update the example and tests to match; never replace `simulate()` with a fake
to turn an integration failure green.

## OpenAI-compatible third-party APIs

Use `--provider custom --base-url <official-base-url> --model <exact-model-id>`. The runner prompts for `LLM_API_KEY` (hidden), supports `LLM_BASE_URL`, and allows `--tool-choice auto` or `--token-parameter max_completion_tokens` when documented by the provider. See root `LIVE_TEST_SETUP.md` for details.
