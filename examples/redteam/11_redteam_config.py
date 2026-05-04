"""Centralized configuration with LLMConfig and LLMCallConfig.

LLMConfig controls model selection and LLM call tuning for each pipeline role
(attacker and evaluator). Backend routing is inferred from the target type:
string targets like ``"agent:<key>"`` route through the ORQ platform, while
:class:`OpenAIModelTarget` instances call OpenAI directly.

Key features:
    Role-based config — Use ``attacker`` and ``evaluator`` fields to configure
                         each pipeline role independently.
    LLMCallConfig      — Per-role settings: model, temperature, max_tokens,
                         timeout_ms, extra_kwargs, and an optional pre-built client.
    Retry config       — ``retry_count`` and ``retry_on_codes`` for ORQ router retries.

Prerequisites:
    - ORQ_API_KEY or OPENAI_API_KEY set in environment

Usage:
    ORQ_API_KEY=orq-... python 11_redteam_config.py
    # or
    OPENAI_API_KEY=sk-... python 11_redteam_config.py
"""

import asyncio

from evaluatorq.redteam import LLMCallConfig, LLMConfig, OpenAIModelTarget, red_team


async def main() -> None:
    # --- Example 1: Role-based config with custom models -------------------
    config = LLMConfig(
        attacker=LLMCallConfig(model="openai/gpt-4o", temperature=0.9),
        evaluator=LLMCallConfig(model="openai/gpt-4o-mini", temperature=0.0),
    )

    # --- Example 2: Tune per-role settings ---------------------------------
    # LLMCallConfig supports model, temperature, max_tokens, timeout_ms,
    # extra_kwargs (merged into every LLM call), and an optional pre-built client.
    config_tuned = LLMConfig(
        attacker=LLMCallConfig(
            model="openai/gpt-4o",
            temperature=0.7,
            max_tokens=4096,
            timeout_ms=90_000,
            extra_kwargs={"reasoning_effort": "medium"},
        ),
        evaluator=LLMCallConfig(
            model="openai/gpt-4o-mini",
            temperature=0.0,
        ),
        # Retry configuration for the ORQ router
        retry_count=5,
        retry_on_codes=[429, 500, 502, 503, 504],
    )

    # --- Example 3: Use defaults for both roles ----------------------------
    # LLMConfig() with no arguments uses the default model for both roles.
    config_defaults = LLMConfig()

    # --- Run with the config -----------------------------------------------
    # Individual params (categories, max_turns, etc.) are still passed
    # directly to red_team(). Config handles the model/LLM layer.
    report = await red_team(
        "agent:your-agent-key",
        llm_config=config,
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
        llm_config=LLMConfig(attacker=LLMCallConfig(model="openai/gpt-4o-mini")),
        mode="dynamic",
        categories=["LLM01"],
        max_dynamic_datapoints=3,
    )

    print(f"Direct model resistance rate: {report2.summary.resistance_rate:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
