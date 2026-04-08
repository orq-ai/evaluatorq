"""Tests for simulation types."""

from evaluatorq.simulation.types import (
    CommunicationStyle,
    ConversationStrategy,
    Criterion,
    CulturalContext,
    Datapoint,
    EmotionalArc,
    InputFormat,
    Message,
    Persona,
    Scenario,
    SimulationResult,
    StartingEmotion,
    TerminatedBy,
    TokenUsage,
    EMOTIONAL_ARC_INSTRUCTIONS,
    CULTURAL_CONTEXT_INSTRUCTIONS,
    STRATEGY_INSTRUCTIONS,
    INPUT_FORMAT_INSTRUCTIONS,
)


def test_enums():
    assert CommunicationStyle.formal.value == "formal"
    assert StartingEmotion.frustrated.value == "frustrated"
    assert EmotionalArc.escalating.value == "escalating"
    assert CulturalContext.indirect.value == "indirect"
    assert ConversationStrategy.evasive.value == "evasive"
    assert InputFormat.code_block.value == "code_block"
    assert TerminatedBy.judge.value == "judge"


def test_instruction_maps():
    assert EMOTIONAL_ARC_INSTRUCTIONS[EmotionalArc.stable] == ""
    assert "escalates" in EMOTIONAL_ARC_INSTRUCTIONS[EmotionalArc.escalating]
    assert CULTURAL_CONTEXT_INSTRUCTIONS[CulturalContext.neutral] == ""
    assert "direct" in CULTURAL_CONTEXT_INSTRUCTIONS[CulturalContext.direct]
    assert STRATEGY_INSTRUCTIONS[ConversationStrategy.cooperative] == ""
    assert (
        "switch topics" in STRATEGY_INSTRUCTIONS[ConversationStrategy.topic_switching]
    )
    assert INPUT_FORMAT_INSTRUCTIONS[InputFormat.plain_text] == ""
    assert "URLs" in INPUT_FORMAT_INSTRUCTIONS[InputFormat.with_url]


def test_persona_model():
    p = Persona(
        name="Test User",
        patience=0.5,
        assertiveness=0.7,
        politeness=0.3,
        technical_level=0.8,
        communication_style=CommunicationStyle.formal,
        background="A test user",
        emotional_arc=EmotionalArc.volatile,
    )
    assert p.name == "Test User"
    assert p.patience == 0.5
    assert p.communication_style == CommunicationStyle.formal
    assert p.emotional_arc == EmotionalArc.volatile
    assert p.cultural_context is None


def test_scenario_model():
    s = Scenario(
        name="Test Scenario",
        goal="Get help",
        context="Need assistance",
        starting_emotion=StartingEmotion.frustrated,
        criteria=[
            Criterion(description="Agent provides answer", type="must_happen"),
            Criterion(description="Agent is rude", type="must_not_happen"),
        ],
    )
    assert s.name == "Test Scenario"
    assert s.criteria is not None
    assert len(s.criteria) == 2
    assert s.criteria[0].type == "must_happen"


def test_simulation_result_model():
    r = SimulationResult(
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ],
        terminated_by=TerminatedBy.judge,
        reason="Goal achieved",
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        turn_metrics=[],
    )
    assert r.goal_achieved is True
    assert r.turn_count == 1
    assert r.token_usage.total_tokens == 15


def test_datapoint_model():
    p = Persona(
        name="User",
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background="bg",
    )
    s = Scenario(name="Scenario", goal="Goal")
    dp = Datapoint(
        id="dp_abc123",
        persona=p,
        scenario=s,
        user_system_prompt="prompt",
        first_message="Hello",
    )
    assert dp.id == "dp_abc123"
    assert dp.persona.name == "User"
    assert dp.first_message == "Hello"
