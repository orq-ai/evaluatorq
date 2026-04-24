"""Centralized configuration with LLMConfig.

LLMConfig is a single object that controls model selection, LLM call
tuning, and extra kwargs — replacing the need to pass many individual
parameters. Backend routing is now inferred from the target type:
string targets like ``"agent:<key>"`` route through the ORQ platform,
while :class:`OpenAIModelTarget` instances call OpenAI directly.

Key features:
    Model prefixing — When routing through the ORQ router, models are
                       auto-prefixed (e.g. "gpt-5-mini" → "openai/gpt-5-mini").
    llm_kwargs      — Extra kwargs merged into every chat.completions.create()
                       call, useful for fixing API compatibility issues.
    Flat structure  — Fine-tune temperatures, token limits, timeouts, and
                       retry settings directly on LLMConfig.

Prerequisites:
    - ORQ_API_KEY or OPENAI_API_KEY set in environment

Usage:
    ORQ_API_KEY=orq-... python 11_redteam_config.py
    # or
    OPENAI_API_KEY=sk-... python 11_redteam_config.py
"""

import asyncio

from evaluatorq.redteam import LLMConfig, OpenAIModelTarget, red_team


async def main() -> None:
    # --- Example 1: Basic config -------------------------------------------
    # Backend is inferred from the target: "agent:" strings go through the
    # ORQ platform, OpenAIModelTarget objects call OpenAI directly.
    config = LLMConfig(
        attack_model="gpt-5-mini",
        evaluator_model="gpt-5-mini",
    )

    # --- Example 2: Fix reasoning model compatibility ----------------------
    # OpenAI reasoning models (gpt-5-mini, o1, o3) require
    # max_completion_tokens instead of max_tokens when called directly.
    # The pipeline already uses max_completion_tokens, but if you need to
    # pass additional model-specific parameters:
    config_with_kwargs = LLMConfig(
        attack_model="gpt-5-mini",
        llm_kwargs={
            # Any extra kwargs are merged into every LLM call
            "reasoning_effort": "medium",
        },
    )

    # --- Example 3: Tune pipeline LLM settings (flat structure) -----------
    # LLMConfig now has a flat structure with all settings at the top level.
    config_tuned = LLMConfig(
        attack_model="gpt-4.1-mini",
        evaluator_model="gpt-4.1-mini",
        # Lower temperature for more deterministic attacks
        adversarial_temperature=0.7,
        # Increase timeout for slow models
        llm_call_timeout_ms=90_000,
        target_agent_timeout_ms=180_000,
        # Retry configuration for the ORQ router
        retry_count=5,
        retry_on_codes=[429, 500, 502, 503, 504],
    )

    # --- Run with the config -----------------------------------------------
    # Individual params (categories, max_turns, etc.) are still passed
    # directly to red_team(). Config handles the model/LLM layer.
    report = await red_team(
        "agent:your-agent-key",
        config=config,
        mode="dynamic",
        categories=["LLM07"],
        max_dynamic_datapoints=3,
        max_turns=2,
        generate_strategies=False,
    )

    print(f"Resistance rate: {report.summary.resistance_rate:.0%}")

    # --- Example: Testing an OpenAI model directly -------------------------
    # Use OpenAIModelTarget instead of the removed "llm:<model>" string prefix.
    # This allows you to test models directly without the ORQ router.
    report2 = await red_team(
        OpenAIModelTarget("gpt-4o", system_prompt="You are a helpful assistant."),
        config=LLMConfig(attack_model="gpt-4.1-mini"),
        mode="dynamic",
        categories=["LLM01"],
        max_dynamic_datapoints=3,
    )

    print(f"Direct model resistance rate: {report2.summary.resistance_rate:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
