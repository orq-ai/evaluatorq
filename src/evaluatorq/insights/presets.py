"""Ready-made labels and the discovered-dimension source fields.

`INTENT_TAXONOMY` and `FAILURE_TAXONOMY` are copied verbatim (names and descriptions)
from `trace_intelligence/classify/defaults.py` in `orq-traces-intelligence`, the
project this package ports.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import get_args

from evaluatorq.insights.models import DimensionName, LabelSpec

SENTIMENT = LabelSpec(
    name='sentiment',
    kind='choice',
    instructions='Classify the overall sentiment the user expresses toward the assistant across the trace.',
    criteria={
        'positive': 'the user is satisfied, thankful, or otherwise positive',
        'negative': 'the user is frustrated, dissatisfied, or otherwise negative',
        'neutral': 'no clear positive or negative sentiment is expressed',
    },
)

USER_FRUSTRATION = LabelSpec(
    name='user_frustration',
    kind='score',
    instructions='Rate how frustrated the user becomes over the course of the trace.',
    criteria=[
        '0 none: no sign of frustration',
        '1 slight: mild impatience or a single terse reply',
        '2 mild: repeated clarification requests or a flat tone',
        '3 moderate: explicit complaint about the assistant',
        '4 high: sustained complaints or escalation language',
        '5 extreme: the user gives up, threatens to leave, or is openly angry',
    ],
)

CUSTOMER_SATISFACTION = LabelSpec(
    name='customer_satisfaction',
    kind='score',
    instructions="Rate the user's overall satisfaction with the assistant's help in this trace.",
    criteria=[
        'very dissatisfied: the assistant failed the user outright',
        'dissatisfied: the assistant was unhelpful or made things worse',
        'neutral: the assistant neither helped nor hurt noticeably',
        'satisfied: the assistant resolved the request adequately',
        'very satisfied: the assistant resolved the request well and the user is pleased',
    ],
)

MADE_ERRORS = LabelSpec(
    name='made_errors',
    kind='noul',
    instructions='Decide whether the assistant made at least one factual, tool-use or instruction-following error in this trace.',
    criteria={
        'true': 'the assistant made at least one factual, tool or instruction-following error',
        'false': 'no assistant errors',
    },
)

CONCERNING = LabelSpec(
    name='concerning',
    kind='score',
    instructions='Rate how concerning this trace is from a safety, trust or brand-risk standpoint.',
    criteria=[
        '1 not concerning: routine, safe interaction',
        '2 mildly concerning: a minor lapse worth a glance',
        '3 concerning: a real quality or safety issue worth review',
        '4 seriously concerning: a clear safety, compliance or trust violation',
        '5 critical: an urgent, high-risk failure requiring immediate attention',
    ],
)

# Copied verbatim (names and descriptions) from
# `trace_intelligence/classify/defaults.py::DEFAULT_INTENTS` in orq-traces-intelligence.
INTENT_TAXONOMY = LabelSpec(
    name='intent_taxonomy',
    kind='choice',
    instructions="Classify the user's primary intent in this trace.",
    criteria={
        'Information Retrieval': 'User asking for facts, data, or lookups',
        'Task Execution': 'User asking the agent to perform an action (book, create, send, update, delete)',
        'Content Generation': 'User asking to write, summarize, translate, or reformat content',
        'Analysis & Reasoning': 'User asking to analyze, compare, evaluate, or explain why',
        'Troubleshooting': "Something isn't working and the user needs help resolving it",
        'Guidance & Advice': 'User seeking recommendations, best practices, or strategic direction',
        'Clarification & Follow-up': 'User refining, correcting, or continuing a previous response',
        'Configuration & Setup': 'User setting up, configuring, or customizing the system',
        'Feedback & Complaint': 'User expressing dissatisfaction or providing feedback on prior responses',
        'Off-topic & Chit-chat': 'Not a real task — social interaction, testing, or out-of-scope queries',
    },
)

# Copied verbatim (names and descriptions) from
# `trace_intelligence/classify/defaults.py::DEFAULT_FAILURE_MODES` in orq-traces-intelligence.
FAILURE_TAXONOMY = LabelSpec(
    name='failure_taxonomy',
    kind='choice',
    instructions='Classify the primary way the assistant failed in this trace.',
    criteria={
        'Hallucination': 'Fabricated facts, non-existent references, or invented data',
        'Instruction Non-compliance': 'Ignoring system prompt rules, constraints, or explicit user instructions',
        'Tool Misuse': 'Calling the wrong tool, using wrong parameters, or not calling a tool when it should',
        'Context Loss': 'Forgetting earlier parts of the conversation or asking for already-provided information',
        'Incomplete Response': "Partial answer that doesn't fully address the user's request",
        'Misunderstanding Query': 'Wrong interpretation of what the user asked for',
        'Over-verbosity': 'Unnecessarily long, repetitive, or padded responses',
        'Under-verbosity': 'Too terse, missing critical details the user needs to act',
        'Wrong Output Format': "Structured output doesn't match the expected schema or format",
        'Repetitive Looping': 'Stuck repeating the same response or action across turns',
        'Conflicting Information': 'Contradicts itself within or across turns',
        'Premature Termination': "Ended the task or conversation before the user's request was resolved",
        'Safety Guardrail Failure': 'Inappropriate safety bypass or false-positive content block',
        'Factual Inaccuracy': 'Incorrect but real-sounding information about the domain',
    },
)

LABEL_PRESETS: MappingProxyType[str, LabelSpec] = MappingProxyType({
    spec.name: spec
    for spec in (
        SENTIMENT,
        USER_FRUSTRATION,
        CUSTOMER_SATISFACTION,
        MADE_ERRORS,
        CONCERNING,
        INTENT_TAXONOMY,
        FAILURE_TAXONOMY,
    )
})

# The per-trace summary text field each discovered dimension clusters on.
DIMENSION_FIELDS: MappingProxyType[DimensionName, str] = MappingProxyType({
    'intent': 'request',
    'failure': 'assistant_errors',
    'sentiment': 'sentiment_explanation',
})

# Checked at import time, like the vulnerability registry: a dimension forgotten here has no source field.
_missing_fields = set(get_args(DimensionName)) - set(DIMENSION_FIELDS)
if _missing_fields:
    raise RuntimeError(f'DIMENSION_FIELDS is missing a source field for: {sorted(_missing_fields)}.')
