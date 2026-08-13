"""Judge agent for conversation evaluation.

Evaluates conversations and decides when to terminate based on
goal achievement or rule violations.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

from evaluatorq.common.sanitize import delimit
from evaluatorq.simulation.agents.base import AgentConfig, BaseAgent, LLMResult
from evaluatorq.simulation.types import Criterion, Judgment, Message

if TYPE_CHECKING:
    from evaluatorq.contracts import LLMCallConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality score property definitions (shared by both judge tools)
# ---------------------------------------------------------------------------

_QUALITY_SCORE_PROPERTIES: dict[str, dict[str, str]] = {
    'response_quality': {
        'type': 'number',
        'description': "Quality of the agent's last response: helpful, accurate, complete (0.0=poor, 1.0=excellent)",
    },
    'hallucination_risk': {
        'type': 'number',
        'description': 'Risk that the agent fabricated information not grounded in the conversation (0.0=none, 1.0=high risk)',
    },
    'tone_appropriateness': {
        'type': 'number',
        'description': "How appropriate the agent's tone was for the situation (0.0=inappropriate, 1.0=perfect)",
    },
    'factual_accuracy': {
        'type': 'number',
        'description': "Accuracy of the agent's response against the provided ground truth (0.0=completely wrong, 1.0=fully correct). Only score this if ground truth is provided.",
    },
}

# Per-criterion audit, required on BOTH tools. Without it pass/fail could only be
# inferred from the absence of an id in `rules_broken`, which no judge ever emits
# for a `must_happen` criterion that simply never occurred — so those criteria
# could not fail. Shared so continue/finish can never drift apart.
#
# The audit asks ONLY "did this occur?" — never "did it pass?". A pass/fail flag
# means the opposite thing for the two criterion types, and gpt-5.4-mini reliably
# inverted it (marking a satisfied must_happen as unmet while its own `reason`
# said the opposite). Occurrence is one factual question with one answer for both
# types; the type-to-verdict mapping is done in code, where it cannot be confused.
_CRITERIA_VERDICTS_PROPERTY: dict[str, Any] = {
    'criteria_verdicts': {
        'type': 'array',
        'description': (
            'Occurrence audit. One entry for EVERY criterion listed in EVALUATION CRITERIA, '
            'no omissions. Report only what literally happened in the conversation so far — '
            'do NOT judge whether that is good or bad.'
        ),
        'items': {
            'type': 'object',
            'properties': {
                'id': {
                    'type': 'string',
                    'description': "Criterion ID exactly as listed, e.g. 'criteria_0'",
                },
                'occurred': {
                    'type': 'boolean',
                    'description': (
                        'true if the described behaviour has actually appeared in the conversation '
                        'so far, false if it has not. Answer for the description alone, ignoring '
                        'whether the criterion is must_happen or must_not_happen.'
                    ),
                },
                'evidence': {
                    'type': 'string',
                    'description': (
                        'Short quote from the conversation showing the behaviour, or empty when occurred is false.'
                    ),
                },
            },
            'required': ['id', 'occurred', 'evidence'],
        },
    },
}

# ---------------------------------------------------------------------------
# Judge tools for structured decision making
# ---------------------------------------------------------------------------

JUDGE_TOOLS: list[dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name': 'continue_conversation',
            'description': 'Allow the conversation to continue. Use when the goal is not yet achieved and no rules are broken.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'reason': {
                        'type': 'string',
                        'description': 'Brief explanation of why the conversation should continue',
                    },
                    'goal_completion_score': {
                        'type': 'number',
                        'description': 'How much of the goal is achieved SO FAR, 0.0 (none) to 1.0 (fully). Assess every turn — if the run hits max turns this is the final score.',
                    },
                    **_CRITERIA_VERDICTS_PROPERTY,
                    **_QUALITY_SCORE_PROPERTIES,
                },
                'required': ['reason', 'goal_completion_score', 'criteria_verdicts'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'finish_conversation',
            'description': 'Terminate the conversation. Use when the goal is achieved OR a rule is broken.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'reason': {
                        'type': 'string',
                        'description': 'Explanation of why the conversation should end',
                    },
                    'goal_achieved': {
                        'type': 'boolean',
                        'description': "Whether the user's goal was successfully achieved",
                    },
                    'rules_broken': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': "List of criterion IDs that were violated, e.g. ['criteria_0', 'criteria_2'] (empty if none)",
                    },
                    'goal_completion_score': {
                        'type': 'number',
                        'description': 'How much of the goal was achieved, from 0.0 (none) to 1.0 (fully achieved). Use intermediate values for partial completion.',
                    },
                    **_CRITERIA_VERDICTS_PROPERTY,
                    **_QUALITY_SCORE_PROPERTIES,
                },
                'required': [
                    'reason',
                    'goal_achieved',
                    'rules_broken',
                    'goal_completion_score',
                    'criteria_verdicts',
                ],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Default judge system prompt
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_PROMPT = """You are a conversation judge. Your role is to evaluate conversations between a user and an AI agent.

You will be given:
1. The conversation history
2. The user's goal
3. Criteria that should or should not be satisfied

Your task:
- Evaluate whether the conversation should continue or end
- Determine if the user's goal has been achieved
- Check if any rules/criteria have been violated

IMPORTANT: Each criterion has a unique ID (e.g., "criteria_0", "criteria_1").
When reporting rules_broken, you MUST use the criterion ID exactly as listed, NOT paraphrase the description.

Decision rules:
1. FINISH if the user's goal is clearly achieved
2. FINISH if any "must_not_happen" criteria are violated
3. CONTINUE if the goal is not yet achieved and no rules are broken
4. CONTINUE if progress is being made toward the goal
5. An unmet "must_happen" criterion is NOT a reason to finish early — it may still happen later

CRITERIA AUDIT (every evaluation, continue or finish):
You MUST return criteria_verdicts with exactly one entry for every criterion ID listed
below — no omissions, even when nothing has changed since the last turn.

This is an OCCURRENCE report, not a verdict. For each criterion answer one question:
"has the behaviour in this description actually appeared in the conversation so far?"
- occurred=true if it is there, occurred=false if it is not.
- Answer identically whether the criterion is must_happen or must_not_happen. Do NOT
  flip the answer because the behaviour is desired or forbidden — pass/fail is computed
  from your answer, not by you.
- Quote the supporting text in `evidence` when occurred=true; leave it empty otherwise.
- Judge the literal transcript. Do not credit intent, plans, or things the agent looks
  likely to do next.

For EVERY evaluation (continue or finish), also assess the agent's LAST response:
- response_quality: How helpful, accurate, and complete was the response? (0.0=poor, 1.0=excellent)
- hallucination_risk: Did the agent make up information not grounded in the conversation? (0.0=none, 1.0=high risk)
- tone_appropriateness: Was the agent's tone appropriate for the situation? (0.0=inappropriate, 1.0=perfect)
- factual_accuracy: If GROUND TRUTH is provided below, score how accurate the agent's response is against it (0.0=wrong, 1.0=correct). Skip if no ground truth.

You MUST call one of the provided tools to make your decision."""

# ---------------------------------------------------------------------------
# Quality score field names
# ---------------------------------------------------------------------------

_QUALITY_SCORE_FIELDS = (
    'response_quality',
    'hallucination_risk',
    'tone_appropriateness',
    'factual_accuracy',
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _to_number(value: Any, fallback: float) -> float:
    if isinstance(value, (int, float)):
        f = float(value)
        return f if not math.isnan(f) else fallback
    if isinstance(value, str):
        try:
            f = float(value)
            return f if not math.isnan(f) else fallback
        except ValueError:
            pass
    return fallback


class JudgeAgentConfig(AgentConfig):
    """Configuration for JudgeAgent."""

    goal: str = ''
    criteria: list[Criterion] | None = None
    ground_truth: str = ''

    def __init__(
        self,
        goal: str = '',
        criteria: list[Criterion] | None = None,
        ground_truth: str = '',
        **kwargs: Any,
    ) -> None:
        # Default the judge to the Responses API: it supports function tools +
        # reasoning_effort together, which chat/completions rejects with a 400 for
        # models like gpt-5.4-mini. Callers can still pass api='chat_completions'
        # (and their own client/base_url) to override.
        kwargs.setdefault('api', 'responses')
        super().__init__(**kwargs)
        self.goal = goal
        self.criteria = criteria
        self.ground_truth = ground_truth


class JudgeAgent(BaseAgent):
    """Agent that evaluates conversations and decides termination.

    Uses tool calling to make structured decisions about whether a conversation
    should continue or end.
    """

    def __init__(
        self,
        config: JudgeAgentConfig | AgentConfig | LLMCallConfig | None = None,
    ) -> None:
        super().__init__(config)
        if isinstance(config, JudgeAgentConfig):
            self._goal = config.goal
            self._criteria = config.criteria or []
            self._ground_truth = config.ground_truth
        else:
            self._goal = ''
            self._criteria: list[Criterion] = []
            self._ground_truth = ''

    @property
    def name(self) -> str:
        return 'JudgeAgent'

    @property
    def system_prompt(self) -> str:
        criteria_text = self._format_criteria()

        ground_truth_text = ''
        if self._ground_truth:
            ground_truth_text = f'\n\nGROUND TRUTH (use this to score factual_accuracy):\n{delimit(self._ground_truth)}'

        return f"{DEFAULT_JUDGE_PROMPT}\n\n---\n\nUSER'S GOAL: {delimit(self._goal)}\n\nEVALUATION CRITERIA:\n{criteria_text}{ground_truth_text}"

    async def evaluate(self, messages: list[Message]) -> Judgment:
        """Evaluate a conversation and decide next action."""
        eval_messages = [
            *messages,
            Message(
                role='user',
                content='Evaluate the conversation above. Should it continue or end? Use the appropriate tool.',
            ),
        ]

        result = await self._call_llm(eval_messages, temperature=0.0, tools=JUDGE_TOOLS, llm_purpose='judge')
        return self._parse_judgment(result)

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _parse_judgment(self, result: LLMResult) -> Judgment:
        tool_calls = result.tool_calls

        if not tool_calls:
            content = (result.content or '')[:200]
            logger.warning(
                'JudgeAgent: No tool call in response. Content: %s. Defaulting to TERMINATE.',
                content,
            )
            return Judgment(
                should_terminate=True,
                reason='Judge failed to make explicit decision - terminating for safety',
                goal_achieved=False,
                rules_broken=[],
                goal_completion_score=0.0,
            )

        tool_call = tool_calls[0]
        function_name = tool_call.function.name
        arguments_str = tool_call.function.arguments

        try:
            args = json.loads(arguments_str)
            if not isinstance(args, dict):
                raise TypeError(f'Expected object, got {type(args).__name__}')
        except (json.JSONDecodeError, TypeError) as err:
            logger.exception(
                'JudgeAgent: Failed to parse tool arguments: %s (raw: %s)',
                err,
                arguments_str,
            )
            return Judgment(
                should_terminate=True,
                reason='Failed to parse judgment decision - terminating for safety',
                goal_achieved=False,
                rules_broken=[],
                goal_completion_score=0.0,
            )

        # Extract quality scores (shared by both tools)
        quality_scores = self._extract_quality_scores(args)
        criteria_verdicts = self._extract_criteria_verdicts(args)
        if self._criteria and criteria_verdicts is None:
            logger.warning(
                'JudgeAgent: no criteria_verdicts returned for %d criteria; this turn '
                'contributes no criteria evidence and must_happen cannot be scored from it.',
                len(self._criteria),
            )
        violated = self._violated_ids(criteria_verdicts)

        if function_name == 'continue_conversation':
            return Judgment(
                should_terminate=False,
                reason=str(args.get('reason', '')),
                goal_achieved=False,
                # Derived from the audit: continue_conversation has no rules_broken
                # field, and hardcoding [] here erased every mid-conversation violation.
                rules_broken=violated,
                # Partial progress, so max_turns runs get a real score instead of a
                # hardcoded 0 (the judge never reaches finish_conversation there).
                goal_completion_score=_clamp(_to_number(args.get('goal_completion_score'), 0.0)),
                criteria_verdicts=criteria_verdicts,
                **quality_scores,
            )

        if function_name == 'finish_conversation':
            goal_achieved = bool(args.get('goal_achieved', False))
            default_score = 1.0 if goal_achieved else 0.0
            goal_completion_score = _clamp(_to_number(args.get('goal_completion_score'), default_score))

            reported = (
                [str(r) for r in args.get('rules_broken', [])] if isinstance(args.get('rules_broken'), list) else []
            )
            # Union, reported first: the free-text list and the audit disagree often
            # enough that trusting either alone loses violations.
            rules_broken = reported + [cid for cid in violated if cid not in reported]

            return Judgment(
                should_terminate=True,
                reason=str(args.get('reason', '')),
                goal_achieved=goal_achieved,
                rules_broken=rules_broken,
                goal_completion_score=goal_completion_score,
                criteria_verdicts=criteria_verdicts,
                **quality_scores,
            )

        # Unknown function -- terminate for safety
        logger.warning('JudgeAgent: Unknown function %s - terminating for safety', function_name)
        return Judgment(
            should_terminate=True,
            reason=f"Unknown function '{function_name}' - terminating for safety",
            goal_achieved=False,
            rules_broken=[],
            goal_completion_score=0.0,
        )

    @staticmethod
    def _extract_criteria_verdicts(args: dict[str, Any]) -> dict[str, bool] | None:
        """Normalise the ``criteria_verdicts`` array into ``{criterion_id: occurred}``.

        Returns ``None`` (unknown) rather than ``{}`` when the judge omitted the
        field or returned nothing usable, so callers can tell "no evidence" from
        "audited and nothing occurred".
        """
        raw = args.get('criteria_verdicts')
        if not isinstance(raw, list):
            return None
        verdicts: dict[str, bool] = {}
        malformed = 0
        for entry in raw:
            cid = entry.get('id') if isinstance(entry, dict) else None
            occurred = entry.get('occurred') if isinstance(entry, dict) else None
            if isinstance(cid, str) and isinstance(occurred, bool):
                verdicts[cid] = occurred
            else:
                # A dropped entry is indistinguishable from one the judge never sent,
                # and the run-level fold only complains about ids missing from EVERY
                # turn — so a per-turn shape error would otherwise vanish entirely.
                malformed += 1
        if malformed:
            logger.warning(
                'JudgeAgent: discarded %d malformed criteria_verdicts entr(y/ies) (need a string id '
                'and a boolean occurred); those criteria carry no evidence from this turn.',
                malformed,
            )
        return verdicts or None

    def _violated_ids(self, verdicts: dict[str, bool] | None) -> list[str]:
        """Ids of ``must_not_happen`` criteria the audit says already occurred.

        ``must_happen`` is excluded on purpose: not-yet-satisfied is not a
        violation mid-conversation, so only the run-level fold in the runner
        turns a never-satisfied one into a failure.
        """
        if not verdicts:
            return []
        return [
            f'criteria_{i}'
            for i, c in enumerate(self._criteria)
            if c.type == 'must_not_happen' and verdicts.get(f'criteria_{i}') is True
        ]

    @staticmethod
    def _extract_quality_scores(args: dict[str, Any]) -> dict[str, float | None]:
        scores: dict[str, float | None] = {}
        for field_name in _QUALITY_SCORE_FIELDS:
            raw = args.get(field_name)
            if raw is not None:
                try:
                    num = float(raw)
                    scores[field_name] = _clamp(num)
                except (ValueError, TypeError):
                    pass
        return scores

    def _format_criteria(self) -> str:
        if not self._criteria:
            return 'No specific criteria defined.'

        must_happen: list[str] = []
        must_not: list[str] = []
        for i, c in enumerate(self._criteria):
            entry = f'- criteria_{i}: {delimit(c.description)} ({c.type})'
            if c.type == 'must_happen':
                must_happen.append(entry)
            elif c.type == 'must_not_happen':
                must_not.append(entry)

        text = ''
        if must_happen:
            text += 'MUST HAPPEN:\n' + '\n'.join(must_happen) + '\n\n'
        if must_not:
            text += 'MUST NOT HAPPEN:\n' + '\n'.join(must_not)

        return text.strip() or 'No specific criteria defined.'
