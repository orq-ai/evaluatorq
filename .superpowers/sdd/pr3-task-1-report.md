# PR3 task 1 report (RES-1875)

Status: DONE_WITH_CONCERNS. Commit: see `git log -1` on branch anthropic-env-resolution-5a9a158f.

## Changes
- `common/llm_client.py`: third resolution step from `ANTHROPIC_API_KEY` (`AsyncAnthropic(max_retries=retries)` wrapped by the adapter, `owned=True`, `routes_through_orq=False`); ImportError naming `evaluatorq[anthropic]` when the extra is missing; both `MissingLLMCredentialsError` messages name the keys; docstrings updated.
- `common/env_config.py`: `ANTHROPIC_API_KEY` row (secret). `docs/configuration.md` row and "Go Anthropic-direct" section; `docs/anthropic.md` "Without injecting a client"; CHANGELOG; drift fixes in README, faq, tuning, guides/targets, guides/agent-simulation, cli-reference/simulation, cli_help.
- `redteam/runner.py` pre-flight, `simulation/cli.py` (`--openai-model` check, provider label), `dashboard/app.py` config panel accept the key.
- Tests: `tests/common/test_llm_client.py` (six new), `tests/anthropic/test_env_cli.py`.

## Verification
- `uv run ruff check src`: All checks passed. `ruff format --check src`: 368 already formatted. `uv run ty check`: All checks passed. `mkdocs build --strict`: built.
- `uv run pytest tests/common tests/anthropic tests/test_reuse_guardrails.py`: 1143 passed before the CLI test fix; rerun with redteam, simulation and dashboard: 6568 passed, 24 failed.
- The 24 failures are all in `tests/dashboard/test_insights_*`; the same files fail 19 on the untouched baseline checkout at 12db921b, so they are not from this change.
- The full `uv run pytest -m 'not integration'` aborts at collection with PermissionError on /data/.env in the main dir (tests/unit/test_crewai_target.py), so I ran the directories above instead.

## CLI snippets
`/tmp/claude-1004/-data-superset-repos-worktrees-evaluatorq-create-ticket-plan-support-5a9a158f/7f9e8736-108c-4cd5-bf25-0621668f7e18/scratchpad/pr3-cli-snippets.md`

## Concerns
- The red-team CLI cannot target a plain Claude model (only `agent:`/`deployment:`), so its end-to-end 401 proof is only on `eq sim run --openai-model`.
- Missing extra raises `ImportError`, not `CredentialError`, so the red-team CLI shows it as a raw error.
- `model_catalogue.py:451` message left alone: it is about the Orq catalogue, where Anthropic does not work.

## Review-fix round

1. Added `MissingExtraError(ImportError)` in `common/llm_client.py`, raised for the missing anthropic extra; `redteam/backends/registry.py` re-raises it as `BackendError(str(exc))` before the generic openai hint. Test `test_a_missing_anthropic_extra_names_the_extra_not_openai` (sys.modules['anthropic']=None via `create_async_llm_client`). CLI-level variant not added.
2. Three-key precedence in docstrings: registry.py, openresponses/client.py (steps 4 and 5 disabled by `require_orq`), simulation/agents/base.py, simulation/api.py (4 places).
3. `tests/anthropic/test_env_cli.py`: sim CLI test intercepts `anthropic.AsyncAnthropic` with an httpx.MockTransport and asserts `/v1/messages` with `x-api-key: sk-ant-test`; red-team test patches `red_team` and asserts the runner's attacker client is an `AnthropicLLMClient`.
4. `tests/common/test_model_catalogue.py`: no-ORQ-key lookup returns `{}`/None/unchanged usage.
5. `docs/anthropic.md`: sim CLI example, Orq-only red-team CLI targets, Python-API line.

Checks: ruff check/format clean, bare `ty check` clean, mkdocs strict built. pytest suite: 4857 passed, 5 failed, 2 skipped; the 5 failures are PermissionError on /data/.env (test_examples_smoke x2, test_orq_agent_tailscale_script x3), environmental and unrelated.
