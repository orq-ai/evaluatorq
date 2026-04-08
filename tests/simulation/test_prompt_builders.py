"""Tests for prompt builders."""

from typing import Any

from evaluatorq.simulation.types import (
    CommunicationStyle,
    ConversationStrategy,
    Criterion,
    EmotionalArc,
    Persona,
    Scenario,
)
from evaluatorq.simulation.utils.prompt_builders import (
    build_datapoint_system_prompt,
    build_persona_system_prompt,
    build_scenario_user_context,
    generate_datapoint,
)


def _make_persona(**overrides: Any) -> Persona:
    defaults: dict[str, Any] = dict(
        name="Test User",
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background="A test background",
    )
    defaults.update(overrides)
    return Persona(**defaults)


def _make_scenario(**overrides: Any) -> Scenario:
    defaults: dict[str, Any] = dict(
        name="Test Scenario", goal="Get help", context="Need assistance"
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def test_build_persona_system_prompt_basic():
    p = _make_persona()
    prompt = build_persona_system_prompt(p)
    assert "Test User" in prompt
    assert "moderately patient" in prompt
    assert "balanced" in prompt
    assert "neutral in tone" in prompt
    assert "somewhat technical" in prompt
    assert "casual" in prompt
    assert "Stay in character" in prompt


def test_build_persona_system_prompt_extreme_traits():
    p = _make_persona(
        patience=0.1, assertiveness=0.9, politeness=0.1, technical_level=0.9
    )
    prompt = build_persona_system_prompt(p)
    assert "very impatient" in prompt
    assert "very assertive" in prompt
    assert "rude and curt" in prompt
    assert "technical expert" in prompt


def test_build_persona_system_prompt_emotional_arc():
    p = _make_persona(emotional_arc=EmotionalArc.escalating)
    prompt = build_persona_system_prompt(p)
    assert "escalates" in prompt


def test_build_scenario_user_context_basic():
    s = _make_scenario()
    ctx = build_scenario_user_context(s)
    assert "Test Scenario" in ctx
    assert "Get help" in ctx
    assert "Need assistance" in ctx
    assert "Work towards your goal" in ctx


def test_build_scenario_user_context_with_criteria():
    s = _make_scenario(
        criteria=[
            Criterion(description="Agent answers", type="must_happen"),
            Criterion(description="Agent is rude", type="must_not_happen"),
        ]
    )
    ctx = build_scenario_user_context(s)
    assert "Agent answers" in ctx
    assert "Agent is rude" in ctx


def test_build_scenario_user_context_with_strategy():
    s = _make_scenario(conversation_strategy=ConversationStrategy.topic_switching)
    ctx = build_scenario_user_context(s)
    assert "switch topics" in ctx


def test_build_datapoint_system_prompt():
    p = _make_persona()
    s = _make_scenario()
    prompt = build_datapoint_system_prompt(p, s)
    assert "Test User" in prompt
    assert "Test Scenario" in prompt
    assert "---" in prompt


def test_generate_datapoint():
    p = _make_persona()
    s = _make_scenario()
    dp = generate_datapoint(p, s, "Hello!")
    assert dp.id.startswith("dp_")
    assert dp.persona.name == "Test User"
    assert dp.scenario.name == "Test Scenario"
    assert dp.first_message == "Hello!"
    assert dp.user_system_prompt != ""


# --- Trait mapping tests ---


def test_low_patience_maps_to_impatient():
    p = _make_persona(patience=0.1)
    prompt = build_persona_system_prompt(p)
    assert "very impatient" in prompt


def test_high_assertiveness():
    p = _make_persona(assertiveness=0.9)
    prompt = build_persona_system_prompt(p)
    assert "very assertive" in prompt


def test_low_politeness():
    p = _make_persona(politeness=0.1)
    prompt = build_persona_system_prompt(p)
    assert "rude" in prompt


def test_high_technical_level():
    p = _make_persona(technical_level=0.9)
    prompt = build_persona_system_prompt(p)
    assert "technical expert" in prompt


# --- Edge case tests ---


def test_stable_arc_omits_emotional_section():
    p = _make_persona(emotional_arc=EmotionalArc.stable)
    prompt = build_persona_system_prompt(p)
    assert "Emotional Arc" not in prompt


def test_neutral_culture_omits_cultural_section():
    from evaluatorq.simulation.types import CulturalContext

    p = _make_persona(cultural_context=CulturalContext.neutral)
    prompt = build_persona_system_prompt(p)
    assert "Cultural Communication Style" not in prompt


def test_scenario_without_criteria():
    s = _make_scenario(criteria=None)
    ctx = build_scenario_user_context(s)
    assert "You expect" not in ctx
    assert "dissatisfied" not in ctx


def test_generate_datapoint_default_first_message():
    p = _make_persona()
    s = _make_scenario()
    dp = generate_datapoint(p, s)
    assert dp.first_message == ""


def test_generate_datapoint_unique_ids():
    p = _make_persona()
    s = _make_scenario()
    dp1 = generate_datapoint(p, s)
    dp2 = generate_datapoint(p, s)
    assert dp1.id != dp2.id
