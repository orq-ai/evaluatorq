"""Prompt building utilities for personas, scenarios, and datapoints."""

from __future__ import annotations

import uuid

from evaluatorq.simulation.types import (
    CULTURAL_CONTEXT_INSTRUCTIONS,
    EMOTIONAL_ARC_INSTRUCTIONS,
    INPUT_FORMAT_INSTRUCTIONS,
    STRATEGY_INSTRUCTIONS,
    CulturalContext,
    Datapoint,
    EmotionalArc,
    InputFormat,
    Persona,
    Scenario,
    ConversationStrategy,
)
from evaluatorq.common.sanitize import delimit

# ---------------------------------------------------------------------------
# Emotion instructions (local to prompt builders)
# ---------------------------------------------------------------------------

_EMOTION_INSTRUCTIONS: dict[str, str] = {
    "neutral": "You approach this conversation calmly.",
    "frustrated": "You are frustrated and may express irritation.",
    "confused": "You are confused and may ask for clarification.",
    "happy": "You are in a good mood and friendly.",
    "urgent": "This is urgent and you need a quick resolution.",
}


# ---------------------------------------------------------------------------
# Persona → system prompt
# ---------------------------------------------------------------------------


def build_persona_system_prompt(persona: Persona) -> str:
    """Convert a persona to a system prompt for the user simulator."""
    if persona.patience < 0.3:
        patience_desc = "very impatient"
    elif persona.patience > 0.7:
        patience_desc = "patient"
    else:
        patience_desc = "moderately patient"

    if persona.assertiveness > 0.7:
        assertive_desc = "very assertive and direct"
    elif persona.assertiveness < 0.3:
        assertive_desc = "passive"
    else:
        assertive_desc = "balanced"

    if persona.politeness < 0.3:
        polite_desc = "rude and curt"
    elif persona.politeness > 0.7:
        polite_desc = "very polite"
    else:
        polite_desc = "neutral in tone"

    if persona.technical_level < 0.3:
        tech_desc = "a complete novice"
    elif persona.technical_level > 0.7:
        tech_desc = "a technical expert"
    else:
        tech_desc = "somewhat technical"

    arc_text = ""
    arc_key = persona.emotional_arc or EmotionalArc.stable
    arc_instruction = EMOTIONAL_ARC_INSTRUCTIONS.get(arc_key, "")
    if arc_instruction:
        arc_text = f"\n\nEmotional Arc: {arc_instruction}"

    cultural_text = ""
    cultural_key = persona.cultural_context or CulturalContext.neutral
    cultural_instruction = CULTURAL_CONTEXT_INSTRUCTIONS.get(cultural_key, "")
    if cultural_instruction:
        cultural_text = f"\n\nCultural Communication Style: {cultural_instruction}"

    return (
        f"You are simulating a user with the following characteristics:\n\n"
        f"Name: {delimit(persona.name)}\n"
        f"Patience: You are {patience_desc}\n"
        f"Assertiveness: You are {assertive_desc}\n"
        f"Politeness: You are {polite_desc}\n"
        f"Technical Level: You are {tech_desc}\n"
        f"Communication Style: {delimit(persona.communication_style.value)}\n\n"
        f"Background: {delimit(persona.background)}\n"
        f"{arc_text}{cultural_text}\n"
        f"Stay in character throughout the conversation. Your responses should reflect these traits consistently.\n"
        f"Do not break character or acknowledge that you are a simulation."
    )


# ---------------------------------------------------------------------------
# Scenario → user context
# ---------------------------------------------------------------------------


def build_scenario_user_context(scenario: Scenario) -> str:
    """Convert a scenario to context for the user simulator."""
    criteria_text = ""
    if scenario.criteria:
        must_happen = [
            delimit(c.description) for c in scenario.criteria if c.type == "must_happen"
        ]
        must_not = [
            delimit(c.description)
            for c in scenario.criteria
            if c.type == "must_not_happen"
        ]

        if must_happen:
            criteria_text += f"\n\nYou expect the agent to: {', '.join(must_happen)}"
        if must_not:
            criteria_text += f"\n\nYou would be dissatisfied if: {', '.join(must_not)}"

    strategy_text = ""
    strategy_key = scenario.conversation_strategy or ConversationStrategy.cooperative
    strategy_instruction = STRATEGY_INSTRUCTIONS.get(strategy_key, "")
    if strategy_instruction:
        strategy_text = f"\n\nConversation Strategy: {strategy_instruction}"

    format_text = ""
    format_key = scenario.input_format or InputFormat.plain_text
    format_instruction = INPUT_FORMAT_INSTRUCTIONS.get(format_key, "")
    if format_instruction:
        format_text = f"\n\nMessage Format: {format_instruction}"

    emotion_text = _EMOTION_INSTRUCTIONS.get(
        (scenario.starting_emotion or "neutral")
        if isinstance(scenario.starting_emotion, str)
        else (
            scenario.starting_emotion.value if scenario.starting_emotion else "neutral"
        ),
        "",
    )

    return (
        f"Scenario: {delimit(scenario.name)}\n\n"
        f"Your Goal: {delimit(scenario.goal)}\n\n"
        f"Context: {delimit(scenario.context or '')}\n\n"
        f"Emotional State: {emotion_text}\n"
        f"{criteria_text}{strategy_text}{format_text}\n\n"
        f"Work towards your goal naturally. React authentically based on how the agent responds."
    )


# ---------------------------------------------------------------------------
# Datapoint helpers
# ---------------------------------------------------------------------------


def build_datapoint_system_prompt(persona: Persona, scenario: Scenario) -> str:
    """Build the combined system prompt from persona and scenario."""
    persona_prompt = build_persona_system_prompt(persona)
    scenario_context = build_scenario_user_context(scenario)
    return f"{persona_prompt}\n\n---\n\n{scenario_context}"


def generate_datapoint(
    persona: Persona, scenario: Scenario, first_message: str = ""
) -> Datapoint:
    """Generate a datapoint from persona and scenario."""
    dp_id = f"dp_{uuid.uuid4().hex[:12]}"
    return Datapoint(
        id=dp_id,
        persona=persona,
        scenario=scenario,
        user_system_prompt=build_datapoint_system_prompt(persona, scenario),
        first_message=first_message,
    )
