"""Core type definitions for the agent simulation framework.

Uses Pydantic models for maximum compatibility with generators/runner/agents.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_MODEL = "azure/gpt-4o-mini"


# ---------------------------------------------------------------------------
# Literal union types (StrEnum for Python 3.10 compat)
# ---------------------------------------------------------------------------


class CommunicationStyle(str, Enum):
    formal = "formal"
    casual = "casual"
    terse = "terse"
    verbose = "verbose"


class StartingEmotion(str, Enum):
    neutral = "neutral"
    frustrated = "frustrated"
    confused = "confused"
    happy = "happy"
    urgent = "urgent"


class EmotionalArc(str, Enum):
    stable = "stable"
    escalating = "escalating"
    de_escalating = "de_escalating"
    volatile = "volatile"
    manipulative = "manipulative"
    hostile = "hostile"


class CulturalContext(str, Enum):
    neutral = "neutral"
    direct = "direct"
    indirect = "indirect"
    high_context = "high_context"
    low_context = "low_context"
    hierarchical = "hierarchical"


class ConversationStrategy(str, Enum):
    cooperative = "cooperative"
    topic_switching = "topic_switching"
    contradictory = "contradictory"
    multi_intent = "multi_intent"
    evasive = "evasive"
    repetitive = "repetitive"
    ambiguous = "ambiguous"


class InputFormat(str, Enum):
    plain_text = "plain_text"
    with_url = "with_url"
    with_attachment = "with_attachment"
    form_data = "form_data"
    code_block = "code_block"
    mixed_media = "mixed_media"


class TerminatedBy(str, Enum):
    judge = "judge"
    max_turns = "max_turns"
    error = "error"
    timeout = "timeout"


# ---------------------------------------------------------------------------
# Constant instruction maps
# ---------------------------------------------------------------------------

EMOTIONAL_ARC_INSTRUCTIONS: dict[EmotionalArc, str] = {
    EmotionalArc.stable: "",
    EmotionalArc.escalating: (
        "Your emotional intensity escalates over the course of the conversation. "
        "Start relatively calm but become progressively more frustrated, impatient, or upset "
        "with each turn, especially if the agent is slow to resolve your issue. "
        "By the end, you should be noticeably more emotional than at the start."
    ),
    EmotionalArc.de_escalating: (
        "You start the conversation very upset, frustrated, or angry. As the agent responds "
        "and shows willingness to help, you gradually calm down. Your tone softens over time, "
        "and by the end you may even express gratitude. However, any misstep by the agent "
        "can temporarily reverse this trend."
    ),
    EmotionalArc.volatile: (
        "Your emotional state is volatile and unpredictable. Swing between being calm, "
        "frustrated, grateful, and confused within the same conversation. A small positive "
        "response from the agent might suddenly improve your mood, while a minor setback "
        "could trigger strong frustration. Your reactions should feel disproportionate."
    ),
    EmotionalArc.manipulative: (
        "You use emotional manipulation tactics to get what you want. Employ guilt-tripping "
        "('I've been a loyal customer for years and this is how you treat me?'), flattery "
        "('You seem so much more competent than the last agent'), threats to escalate "
        "('I'll leave a bad review'), or appeals to sympathy ('This is ruining my day'). "
        "Switch between these tactics based on what seems to be working."
    ),
    EmotionalArc.hostile: (
        "You are genuinely hostile and confrontational. Use aggressive language, make "
        "unreasonable demands, and challenge the agent's competence. Express contempt and "
        "refuse to cooperate with standard procedures. This tests the agent's ability to "
        "maintain professionalism and de-escalate under pressure."
    ),
}

CULTURAL_CONTEXT_INSTRUCTIONS: dict[CulturalContext, str] = {
    CulturalContext.neutral: "",
    CulturalContext.direct: (
        "You communicate in a very direct, low-context style typical of Northern European "
        "or North American cultures. Say exactly what you mean without hints or implication. "
        "Get straight to the point. 'No' means 'no' — you don't soften refusals."
    ),
    CulturalContext.indirect: (
        "You communicate indirectly, typical of many East Asian or Southeast Asian cultures. "
        "Avoid saying 'no' directly — instead use phrases like 'that might be difficult' or "
        "'I'll think about it.' Hint at problems rather than stating them outright. Use hedging "
        "language ('perhaps', 'maybe', 'it seems'). Preserving harmony is important."
    ),
    CulturalContext.high_context: (
        "You rely heavily on context and implication rather than explicit statements. "
        "You expect the agent to read between the lines and understand unstated needs. "
        "You may reference shared knowledge without explaining it. Use fewer words but "
        "expect more understanding. Silence can be meaningful."
    ),
    CulturalContext.low_context: (
        "You spell everything out explicitly and leave nothing to interpretation. "
        "Provide full context with every message. Repeat important details. "
        "Don't assume the agent remembers previous context. "
        "Be thorough and detailed in every message."
    ),
    CulturalContext.hierarchical: (
        "You approach the interaction with a strong sense of hierarchy, typical of many "
        "Middle Eastern, East Asian, or South Asian cultures. You may expect formal address, "
        "defer to authority ('can I speak to a manager?'), and be uncomfortable challenging "
        "the agent's statements directly. Status and titles matter to you."
    ),
}

STRATEGY_INSTRUCTIONS: dict[ConversationStrategy, str] = {
    ConversationStrategy.cooperative: "",
    ConversationStrategy.topic_switching: (
        "You frequently switch topics mid-conversation. After a few exchanges about your main goal, "
        "bring up unrelated questions or concerns before returning to your original topic. "
        "This tests the agent's ability to handle context switching."
    ),
    ConversationStrategy.contradictory: (
        "You contradict yourself during the conversation. Say one thing, then later say the opposite "
        "or change your requirements. For example, first ask for a refund, then say you actually want "
        "a replacement. This tests the agent's ability to handle inconsistent user input."
    ),
    ConversationStrategy.multi_intent: (
        "You have multiple goals packed into each message. Combine questions, requests, and complaints "
        "in single messages. For example, ask about your order status while also requesting a password "
        "reset and complaining about a previous experience."
    ),
    ConversationStrategy.evasive: (
        "You are evasive and avoid directly answering the agent's questions. Give vague or incomplete "
        "responses when asked for details. The agent needs to work harder to extract the information "
        "it needs to help you."
    ),
    ConversationStrategy.repetitive: (
        "You repeat your requests and questions even after the agent has addressed them. Ask the same "
        "thing in slightly different ways, as if you didn't understand or weren't satisfied with "
        "the response. This tests the agent's patience and ability to rephrase explanations."
    ),
    ConversationStrategy.ambiguous: (
        "You are deliberately vague and unclear in your requests. Use imprecise language, "
        "avoid giving specific details, and make the agent work to understand what you actually need. "
        "When the agent asks clarifying questions, give partial or still-ambiguous answers. "
        "For example, say 'the thing isn't working' instead of specifying which product or error."
    ),
}

INPUT_FORMAT_INSTRUCTIONS: dict[InputFormat, str] = {
    InputFormat.plain_text: "",
    InputFormat.with_url: (
        "Include relevant URLs in your messages. Reference links to products, order pages, "
        "screenshots, or documentation."
    ),
    InputFormat.with_attachment: (
        "Reference file attachments in your messages as if you're uploading them. "
        "Mention screenshots, receipts, photos, or documents."
    ),
    InputFormat.form_data: (
        "Structure your messages like filled-out forms or structured data. Include labeled fields, "
        "order details in a structured format, or table-like information."
    ),
    InputFormat.code_block: (
        "Include code snippets, error logs, stack traces, or technical output in your messages. "
        "Wrap technical content in code blocks."
    ),
    InputFormat.mixed_media: (
        "Mix different input types in your messages. Combine plain text with URLs, "
        "attachment references, structured data, or code blocks."
    ),
}


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------


class Persona(BaseModel):
    name: str
    patience: float
    assertiveness: float
    politeness: float
    technical_level: float
    communication_style: CommunicationStyle
    background: str
    emotional_arc: EmotionalArc | None = None
    cultural_context: CulturalContext | None = None


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------


class Criterion(BaseModel):
    description: str
    type: Literal["must_happen", "must_not_happen"]
    evaluator: str | None = None


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


class Scenario(BaseModel):
    name: str
    goal: str
    context: str | None = None
    starting_emotion: StartingEmotion | None = None
    criteria: list[Criterion] | None = None
    is_edge_case: bool | None = None
    conversation_strategy: ConversationStrategy | None = None
    ground_truth: str | None = None
    input_format: InputFormat | None = None


# ---------------------------------------------------------------------------
# Judgment
# ---------------------------------------------------------------------------


class Judgment(BaseModel):
    should_terminate: bool
    reason: str
    goal_achieved: bool
    rules_broken: list[str]
    goal_completion_score: float
    response_quality: float | None = None
    hallucination_risk: float | None = None
    tone_appropriateness: float | None = None
    factual_accuracy: float | None = None


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


# ---------------------------------------------------------------------------
# TurnMetrics
# ---------------------------------------------------------------------------


class TurnMetrics(BaseModel):
    turn_number: int
    token_usage: TokenUsage
    response_quality: float | None = None
    hallucination_risk: float | None = None
    tone_appropriateness: float | None = None
    factual_accuracy: float | None = None
    judge_reason: str


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------


class SimulationResult(BaseModel):
    messages: list[Message]
    terminated_by: TerminatedBy
    reason: str
    goal_achieved: bool
    goal_completion_score: float
    rules_broken: list[str]
    turn_count: int
    token_usage: TokenUsage
    turn_metrics: list[TurnMetrics]
    metadata: dict[str, Any] = Field(default_factory=dict)
    criteria_results: dict[str, bool] | None = None
    total_turns: int | None = None


# ---------------------------------------------------------------------------
# Datapoint
# ---------------------------------------------------------------------------


class Datapoint(BaseModel):
    id: str
    persona: Persona
    scenario: Scenario
    user_system_prompt: str
    first_message: str
