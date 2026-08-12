"""Core type definitions for the agent simulation framework.

Uses Pydantic models for maximum compatibility with generators/runner/agents.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from evaluatorq.contracts import Message, ResponseTrace, RunSummary, StrEnum, TokenUsage

DEFAULT_MODEL = 'openai/gpt-5.4-mini'

DEFAULT_MAX_TURNS = 10
"""Turn cap when the caller names none. The public ``max_turns`` defaults to
``None`` rather than to this value so a replay can tell "unset" from
"explicitly 10" and restore the replayed run's cap only in the former case."""


class AgentInfoSnapshot(TypedDict, total=False):
    """Best-effort snapshot of an ORQ agent's configuration, as fetched by
    ``fetch_agent_info`` and stored on ``SimulationRun.agent_info``.

    Deliberately excludes the agent's system prompt / instructions.
    """

    key: str
    id: str | None
    role: str | None
    description: str | None
    model: str | None
    tools: list[str]
    skills: list[str]
    knowledge_bases: list[str]
    memory_stores: list[str]
    sub_agents: list[str]
    version: str | None
    agent_type: str | None
    engine: str | None
    workspace_id: str | None
    workspace_key: str | None
    base_url: str
    url: str | None


# Default evaluators applied when none are explicitly requested. Single source
# of truth shared by ``api.simulate`` and the CLI run-store record.
DEFAULT_EVALUATOR_NAMES = ['goal_achieved', 'criteria_met']


# ---------------------------------------------------------------------------
# Literal union types (StrEnum for Python 3.10 compat)
# ---------------------------------------------------------------------------


class CommunicationStyle(StrEnum):
    formal = 'formal'
    casual = 'casual'
    terse = 'terse'
    verbose = 'verbose'


class StartingEmotion(StrEnum):
    neutral = 'neutral'
    frustrated = 'frustrated'
    confused = 'confused'
    happy = 'happy'
    urgent = 'urgent'


class EmotionalArc(StrEnum):
    stable = 'stable'
    escalating = 'escalating'
    de_escalating = 'de_escalating'
    volatile = 'volatile'
    manipulative = 'manipulative'
    hostile = 'hostile'


class CulturalContext(StrEnum):
    neutral = 'neutral'
    direct = 'direct'
    indirect = 'indirect'
    high_context = 'high_context'
    low_context = 'low_context'
    hierarchical = 'hierarchical'


class ConversationStrategy(StrEnum):
    cooperative = 'cooperative'
    topic_switching = 'topic_switching'
    contradictory = 'contradictory'
    multi_intent = 'multi_intent'
    evasive = 'evasive'
    repetitive = 'repetitive'
    ambiguous = 'ambiguous'


class InputFormat(StrEnum):
    plain_text = 'plain_text'
    with_url = 'with_url'
    with_attachment = 'with_attachment'
    form_data = 'form_data'
    code_block = 'code_block'
    mixed_media = 'mixed_media'


class TerminatedBy(StrEnum):
    judge = 'judge'
    max_turns = 'max_turns'
    error = 'error'
    timeout = 'timeout'


# ---------------------------------------------------------------------------
# Constant instruction maps
# ---------------------------------------------------------------------------

EMOTIONAL_ARC_INSTRUCTIONS: dict[EmotionalArc, str] = {
    EmotionalArc.stable: '',
    EmotionalArc.escalating: (
        'Your emotional intensity escalates over the course of the conversation. '
        'Start relatively calm but become progressively more frustrated, impatient, or upset '
        'with each turn, especially if the agent is slow to resolve your issue. '
        'By the end, you should be noticeably more emotional than at the start.'
    ),
    EmotionalArc.de_escalating: (
        'You start the conversation very upset, frustrated, or angry. As the agent responds '
        'and shows willingness to help, you gradually calm down. Your tone softens over time, '
        'and by the end you may even express gratitude. However, any misstep by the agent '
        'can temporarily reverse this trend.'
    ),
    EmotionalArc.volatile: (
        'Your emotional state is volatile and unpredictable. Swing between being calm, '
        'frustrated, grateful, and confused within the same conversation. A small positive '
        'response from the agent might suddenly improve your mood, while a minor setback '
        'could trigger strong frustration. Your reactions should feel disproportionate.'
    ),
    EmotionalArc.manipulative: (
        'You use emotional manipulation tactics to get what you want. Employ guilt-tripping '
        "('I've been a loyal customer for years and this is how you treat me?'), flattery "
        "('You seem so much more competent than the last agent'), threats to escalate "
        "('I'll leave a bad review'), or appeals to sympathy ('This is ruining my day'). "
        'Switch between these tactics based on what seems to be working.'
    ),
    EmotionalArc.hostile: (
        'You are genuinely hostile and confrontational. Use aggressive language, make '
        "unreasonable demands, and challenge the agent's competence. Express contempt and "
        "refuse to cooperate with standard procedures. This tests the agent's ability to "
        'maintain professionalism and de-escalate under pressure.'
    ),
}

CULTURAL_CONTEXT_INSTRUCTIONS: dict[CulturalContext, str] = {
    CulturalContext.neutral: '',
    CulturalContext.direct: (
        'You communicate in a very direct, low-context style typical of Northern European '
        'or North American cultures. Say exactly what you mean without hints or implication. '
        "Get straight to the point. 'No' means 'no' — you don't soften refusals."
    ),
    CulturalContext.indirect: (
        'You communicate indirectly, typical of many East Asian or Southeast Asian cultures. '
        "Avoid saying 'no' directly — instead use phrases like 'that might be difficult' or "
        "'I'll think about it.' Hint at problems rather than stating them outright. Use hedging "
        "language ('perhaps', 'maybe', 'it seems'). Preserving harmony is important."
    ),
    CulturalContext.high_context: (
        'You rely heavily on context and implication rather than explicit statements. '
        'You expect the agent to read between the lines and understand unstated needs. '
        'You may reference shared knowledge without explaining it. Use fewer words but '
        'expect more understanding. Silence can be meaningful.'
    ),
    CulturalContext.low_context: (
        'You spell everything out explicitly and leave nothing to interpretation. '
        'Provide full context with every message. Repeat important details. '
        "Don't assume the agent remembers previous context. "
        'Be thorough and detailed in every message.'
    ),
    CulturalContext.hierarchical: (
        'You approach the interaction with a strong sense of hierarchy, typical of many '
        'Middle Eastern, East Asian, or South Asian cultures. You may expect formal address, '
        "defer to authority ('can I speak to a manager?'), and be uncomfortable challenging "
        "the agent's statements directly. Status and titles matter to you."
    ),
}

STRATEGY_INSTRUCTIONS: dict[ConversationStrategy, str] = {
    ConversationStrategy.cooperative: '',
    ConversationStrategy.topic_switching: (
        'You frequently switch topics mid-conversation. After a few exchanges about your main goal, '
        'bring up unrelated questions or concerns before returning to your original topic. '
        "This tests the agent's ability to handle context switching."
    ),
    ConversationStrategy.contradictory: (
        'You contradict yourself during the conversation. Say one thing, then later say the opposite '
        'or change your requirements. For example, first ask for a refund, then say you actually want '
        "a replacement. This tests the agent's ability to handle inconsistent user input."
    ),
    ConversationStrategy.multi_intent: (
        'You have multiple goals packed into each message. Combine questions, requests, and complaints '
        'in single messages. For example, ask about your order status while also requesting a password '
        'reset and complaining about a previous experience.'
    ),
    ConversationStrategy.evasive: (
        "You are evasive and avoid directly answering the agent's questions. Give vague or incomplete "
        'responses when asked for details. The agent needs to work harder to extract the information '
        'it needs to help you.'
    ),
    ConversationStrategy.repetitive: (
        'You repeat your requests and questions even after the agent has addressed them. Ask the same '
        "thing in slightly different ways, as if you didn't understand or weren't satisfied with "
        "the response. This tests the agent's patience and ability to rephrase explanations."
    ),
    ConversationStrategy.ambiguous: (
        'You are deliberately vague and unclear in your requests. Use imprecise language, '
        'avoid giving specific details, and make the agent work to understand what you actually need. '
        'When the agent asks clarifying questions, give partial or still-ambiguous answers. '
        "For example, say 'the thing isn't working' instead of specifying which product or error."
    ),
}

INPUT_FORMAT_INSTRUCTIONS: dict[InputFormat, str] = {
    InputFormat.plain_text: '',
    InputFormat.with_url: (
        'Include relevant URLs in your messages. Reference links to products, order pages, '
        'screenshots, or documentation.'
    ),
    InputFormat.with_attachment: (
        "Reference file attachments in your messages as if you're uploading them. "
        'Mention screenshots, receipts, photos, or documents.'
    ),
    InputFormat.form_data: (
        'Structure your messages like filled-out forms or structured data. Include labeled fields, '
        'order details in a structured format, or table-like information.'
    ),
    InputFormat.code_block: (
        'Include code snippets, error logs, stack traces, or technical output in your messages. '
        'Wrap technical content in code blocks.'
    ),
    InputFormat.mixed_media: (
        'Mix different input types in your messages. Combine plain text with URLs, '
        'attachment references, structured data, or code blocks.'
    ),
}


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
    type: Literal['must_happen', 'must_not_happen']
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
    thread_id: str | None = Field(
        default=None,
        description='Orq observability thread id for this conversation (deterministic: f"{run_id}:{index}"). '
        'None for runs saved before thread grouping existed.',
    )
    response_traces: list[ResponseTrace] = Field(
        default_factory=list,
        description='Per-turn Orq trace/span handles for each successful target-agent response '
        '(excludes user-simulator and judge calls), in turn order. Empty for non-Orq targets and for '
        'runs saved before this field existed.',
    )

    @property
    def last_trace_id(self) -> str | None:
        """Trace id of the last successful target response, for the table deep-link."""
        for rt in reversed(self.response_traces):
            if rt.trace_id:
                return rt.trace_id
        return None


# ---------------------------------------------------------------------------
# SimulationDatapoint
# ---------------------------------------------------------------------------


class SimulationDatapoint(BaseModel):
    id: str
    persona: Persona
    scenario: Scenario
    user_system_prompt: str
    """Cached/serialized system prompt. Used for export only — the runner
    always rebuilds from persona + scenario via ``build_datapoint_system_prompt``."""
    first_message: str


# ---------------------------------------------------------------------------
# SimulationRecommendation
# ---------------------------------------------------------------------------


class SimulationRecommendation(BaseModel):
    """LLM-generated remediation suggestion for one failed simulation result.

    ``result_index`` is the position in ``SimulationRun.results`` /
    the exporter's results list; ``datapoint_id`` is carried from result
    metadata when available.
    """

    result_index: int
    datapoint_id: str | None = None
    persona: str
    scenario: str
    triggers: list[str]
    """What flagged this result, one ``<kind>: <evidence>`` entry each — e.g.
    ``rule_broken: quoted internal ticket ID`` or ``low_factual_accuracy:
    factual_accuracy averaged 0.30 across 4 turns``."""
    suggestions: list[str]


# ---------------------------------------------------------------------------
# SimulationRun  (run-store record)
# ---------------------------------------------------------------------------


class SimulationRun(BaseModel):
    run_name: str
    created_at: datetime
    mode: Literal['run', 'simulate', 'generate']
    target_kind: Literal['orq_agent', 'orq_deployment', 'vercel', 'openai_model', 'callback']
    # The concrete target under test: agent key / deployment key / model id /
    # 'callback'. ``target_model`` is the model that target used *when the client
    # knows it* (OpenAI-model targets); for orq agents/deployments the model is
    # server-side config, so it stays None. Optional for back-compat with runs
    # saved before these fields existed.
    target: str | None = None
    target_model: str | None = None
    # Configured turn cap for the run. Optional for back-compat with runs saved
    # before this field existed (older reports simply omit the Config row).
    max_turns: int | None = None
    # Snapshot of the ORQ agent's configuration at run time (orq_agent targets
    # only; None for other target kinds, fetch failures, and pre-existing runs).
    # Deliberately excludes the system prompt / instructions.
    agent_info: AgentInfoSnapshot | None = None
    orq_base_url: str | None = None
    """The Orq host that served this run (``ORQ_BASE_URL`` or the prod default),
    recorded so a saved run remembers which deployment — prod / staging / on-prem
    — it ran against. None when no Orq agent/deployment was used (plain callable /
    OpenAI-model targets) and for runs saved before this field existed."""
    evaluator_names: list[str]
    total_results: int
    scorer_averages: dict[str, float]
    results: list[SimulationResult]
    datapoints: list[SimulationDatapoint] | None = None
    """The exact cases this run simulated, stored so the run can be replayed
    verbatim (``previous_run=`` / ``--from-run``). Results alone don't carry
    enough — they keep persona/scenario *names*, not the objects. None for runs
    saved before this field existed, which therefore cannot be replayed."""
    replay_version: int | None = None
    """Format version of the replay payload above, stamped when ``datapoints`` is
    written so a future format change reports itself instead of failing
    structurally. None for runs saved before versioning, which read as v1."""
    executive_summary: str | None = None
    run_id: str | None = None
    """Client-minted run-grouping id (uuid hex, not an Orq-side run id) shared by every
    conversation's ``thread_id`` (``{run_id}:{index}``). Powers the dashboard's
    'View all run traces' deep link. None for older runs."""
    experiment_url: str | None = None
    """Absolute URL of the Orq experiment this run was uploaded to, captured from the
    results upload. Powers the terminal 'View on Orq' line and the dashboard's
    'Open experiment' button. None when upload was skipped/failed or for older runs."""
    recommendations: list[SimulationRecommendation] | None = None
    """LLM-generated remediation suggestions for remediable failures
    (see ``reports.recommendations``). None when never generated."""

    def manifest_summary(self) -> RunSummary:
        """Compact run-list summary stored on this run's ``RunManifest``.

        Single source of truth for the shape: the runner writes it on completion
        and the dashboard's backfill writes it for legacy runs. `eq sim runs`
        reads every field here, so a second hand-rolled shape silently blanks
        columns.
        """
        return {
            'mode': self.mode,
            'target_kind': self.target_kind,
            'total_results': self.total_results,
            'scorer_averages': dict(self.scorer_averages),
        }


# ---------------------------------------------------------------------------
# View models for individual_results section entries
# ---------------------------------------------------------------------------


class TranscriptMessage(BaseModel):
    """A single message in a conversation transcript (role + content)."""

    role: str
    content: str


class CriteriaRow(BaseModel):
    """One criterion row as emitted by _criteria_rows().

    Field order MUST match the key order of the dicts returned by
    ``_criteria_rows()`` so that ``model_dump(mode='json')`` is byte-identical
    to the hand-built dict.
    """

    id: str
    description: str
    type: Literal['must_happen', 'must_not_happen'] | None
    passed: bool
    safety: bool


class SimulationEntry(BaseModel):
    """View model for one entry in the individual_results section.

    Field declaration order matches the dict built in
    ``_build_individual_results_section`` exactly so that
    ``model_dump(mode='json')`` produces byte-identical output.
    """

    index: int
    persona: str
    scenario: str
    model: str
    target_model: str | None
    terminated_by: str  # stored as .value — plain str, never an enum repr
    goal_achieved: bool
    goal_completion_score: float
    rules_broken: list[str]
    criteria: list[CriteriaRow]
    turn_count: int
    total_tokens: int
    judge_reason: str
    error: str | None
    evaluator_scores: dict[str, float]
    transcript: list[TranscriptMessage]
    thread_id: str | None = None
    last_trace_id: str | None = None
