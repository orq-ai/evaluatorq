"""Validate generated personas and scenarios against the generators' own quality guidelines.

The persona/scenario generators state quality guidelines in their prompts, but nothing checks that
the *output* honours them. `validate_personas` and `validate_scenarios` score a batch against the
guidelines that can be checked mechanically: specificity (no generic names, thin or filler text),
trait diversity across the batch (the "vary trait values" rule) and well-formedness (scenarios need a
goal, criteria and context). They return per-item scores plus batch diagnostics. A seed is only
matched as a substring, so a miss is reported and neither costs score nor fails the batch.

These are opt-in: no generator calls them, so a caller runs them on the batch it got back. They are
deterministic (no LLM), so a run costs nothing. They read surface form only: whether a trait
contradicts a background, or whether an item fits the agent description, is not checked.
"""

from __future__ import annotations

import re
from statistics import pvariance
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evaluatorq.simulation.contracts import Persona, Scenario

# Backgrounds/goals shorter than this (in words) are too thin to be "specific".
_MIN_BACKGROUND_WORDS = 12
_MIN_GOAL_WORDS = 5
# A numeric trait whose spread across the batch is below this is "all similar".
_MIN_TRAIT_SPREAD = 0.2
# Fewer personas than this cannot show whether trait values vary.
_MIN_DIVERSITY_BATCH = 3
# Filler phrases, matched as whole words. The prompts' DON'T list names "Customer with a problem". Words
# like "something" or "a customer who" are left out on purpose: they are common in specific text too.
_GENERIC_MARKER = re.compile(r'\b(?:customer with a problem|user with a problem|has a problem)\b', re.IGNORECASE)
_NUMERIC_TRAITS = ('patience', 'assertiveness', 'politeness', 'technical_level')

# `fail` fails the batch, `warn` only costs score, `info` reports something and costs nothing.
Severity = Literal['info', 'warn', 'fail']
IssueCode = Literal[
    'generic_name',
    'thin_background',
    'generic_background',
    'seed_not_reflected',
    'seed_unchecked',
    'low_diversity',
    'diversity_unchecked',
    'duplicate',
    'thin_goal',
    'generic_goal',
    'no_criteria',
    'no_context',
]


class Issue(BaseModel):
    """One thing wrong with a generated item or batch."""

    code: IssueCode
    message: str
    severity: Severity = 'warn'


class ItemValidation(BaseModel):
    """Score and issues for a single generated persona or scenario."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    issues: list[Issue] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Per-item scores plus batch-level diagnostics for one generated batch."""

    items: list[ItemValidation] = Field(default_factory=list)
    batch_issues: list[Issue] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    checks_skipped: list[str] = Field(
        default_factory=list,
        description='Checks that could not run, so a high score says nothing about them.',
    )


def _words(text: str) -> int:
    return len(text.split())


def _has_generic_marker(text: str) -> bool:
    return _GENERIC_MARKER.search(text) is not None


def _seed_tokens(seed: str) -> list[str]:
    """The seed's content words: those long enough for a substring match to mean anything.

    A hyphenated word also counts with spaces, so `first-time` matches a background that
    writes `first time`, but not on its parts, where `time` would match `sometimes`. Length
    counts letters, digits and the hyphens between them, after trimming hyphens and
    underscores from the ends: `3-4` and `9-5` are words, `---`, `___` and `a--` are not.
    """
    tokens: list[str] = []
    for word in re.findall(r'[\w-]+', seed.lower()):
        word = word.strip('-_')
        if len(re.sub(r'[\W_]', '', word)) + word.count('-') <= 2:
            continue
        for t in (word, word.replace('-', ' ')):
            if t not in tokens:
                tokens.append(t)
    return tokens


def _seed_reflected(tokens: list[str], *parts: str) -> bool:
    """Whether at least one content word of the seed appears in the item text.

    This is bare substring matching, and generators paraphrase: "Careful Retired Librarian" fits
    "first-time elderly shoppers" and shares no word with it. So a miss is a hint, never a failure.
    """
    hay = ' '.join(parts).lower()
    return any(t in hay for t in tokens)


def _seed_unchecked(seed: str) -> list[Issue]:
    """A seed with no content word (`AI`, `UX`, or punctuation such as `---`) cannot be checked,
    so say that rather than imply it held."""
    if seed.strip() and not _seed_tokens(seed):
        return [Issue(code='seed_unchecked', message=f'seed {seed!r} has no content word to check', severity='info')]
    return []


# Info codes that mean a check could not run, mapped to the check they stand for.
_SKIPPED_CHECKS: dict[IssueCode, str] = {'diversity_unchecked': 'trait_diversity', 'seed_unchecked': 'seed_adherence'}


def _finalize(items: list[ItemValidation], batch_issues: list[Issue], *, pass_threshold: float) -> ValidationReport:
    per_item = sum(i.score for i in items) / len(items) if items else 0.0
    # Each batch-level warning or failure is a flat penalty on the overall score.
    penalty = 0.15 * sum(i.severity != 'info' for i in batch_issues)
    overall = max(0.0, per_item - penalty)
    issues = [*batch_issues, *(issue for item in items for issue in item.issues)]
    passed = bool(items) and overall >= pass_threshold and not any(i.severity == 'fail' for i in issues)
    skipped = [_SKIPPED_CHECKS[i.code] for i in batch_issues if i.code in _SKIPPED_CHECKS]
    return ValidationReport(
        items=items, batch_issues=batch_issues, overall_score=round(overall, 4), passed=passed, checks_skipped=skipped
    )


def validate_personas(
    personas: Sequence[Persona],
    *,
    seed: str = '',
    pass_threshold: float = 0.7,
) -> ValidationReport:
    """Score a generated persona batch against the generator's quality guidelines (model-free).

    Per persona: a generic name and a thin or generic background each deduct from a 1.0 start.
    Batch-level: a numeric trait with almost no spread across the batch ("all similar") and duplicate
    backgrounds are flagged. ``passed`` needs the score to reach ``pass_threshold`` and no issue of
    severity ``fail``, so one thin background fails the batch. When ``seed`` is set, a persona whose
    name and background contain none of the seed's words as a substring is reported as
    ``seed_not_reflected``, which costs nothing. The trait check needs at least three personas; with
    fewer it is reported as ``diversity_unchecked`` and listed in ``checks_skipped``.
    """
    tokens = _seed_tokens(seed)
    items: list[ItemValidation] = []
    for p in personas:
        issues: list[Issue] = []
        score = 1.0
        if len(p.name.split()) < 2 or _has_generic_marker(p.name):
            issues.append(Issue(code='generic_name', message=f'name {p.name!r} is generic', severity='warn'))
            score -= 0.2
        if _words(p.background) < _MIN_BACKGROUND_WORDS:
            issues.append(
                Issue(code='thin_background', message='background is too short to be specific', severity='fail')
            )
            score -= 0.3
        elif _has_generic_marker(p.background):
            issues.append(Issue(code='generic_background', message='background uses generic filler', severity='warn'))
            score -= 0.2
        if tokens and not _seed_reflected(tokens, p.name, p.background):
            issues.append(
                Issue(
                    code='seed_not_reflected',
                    message=f'contains no word of seed {seed!r} as a substring',
                    severity='info',
                )
            )
        items.append(ItemValidation(name=p.name, score=max(0.0, round(score, 4)), issues=issues))

    batch_issues = [*_batch_persona_issues(personas), *_seed_unchecked(seed)]
    return _finalize(items, batch_issues, pass_threshold=pass_threshold)


def _batch_persona_issues(personas: Sequence[Persona]) -> list[Issue]:
    issues: list[Issue] = []
    if len(personas) < _MIN_DIVERSITY_BATCH:
        issues.append(
            Issue(
                code='diversity_unchecked',
                message=f'trait diversity needs at least {_MIN_DIVERSITY_BATCH} personas, got {len(personas)}',
                severity='info',
            )
        )
    else:
        for trait in _NUMERIC_TRAITS:
            values = [getattr(p, trait) for p in personas]
            if max(values) - min(values) < _MIN_TRAIT_SPREAD:
                issues.append(
                    Issue(
                        code='low_diversity', message=f'trait {trait!r} barely varies across the batch', severity='warn'
                    )
                )
    seen: set[str] = set()
    for p in personas:
        key = ' '.join(p.background.lower().split())
        if key in seen:
            issues.append(
                Issue(code='duplicate', message=f'persona {p.name!r} duplicates another background', severity='warn')
            )
        seen.add(key)
    return issues


def validate_scenarios(
    scenarios: Sequence[Scenario],
    *,
    seed: str = '',
    pass_threshold: float = 0.7,
) -> ValidationReport:
    """Score a generated scenario batch against the generator's quality guidelines (model-free).

    Per scenario: a thin or generic goal, a missing context, and missing success criteria each deduct.
    Batch-level: duplicate goals are flagged. ``passed`` needs the score to reach ``pass_threshold``
    and no issue of severity ``fail``. When ``seed`` is set, a scenario whose name and goal contain none
    of the seed's words as a substring is reported as ``seed_not_reflected``, which costs nothing.
    """
    tokens = _seed_tokens(seed)
    items: list[ItemValidation] = []
    for s in scenarios:
        issues: list[Issue] = []
        score = 1.0
        if _words(s.goal) < _MIN_GOAL_WORDS:
            issues.append(Issue(code='thin_goal', message='goal is too short to be specific', severity='fail'))
            score -= 0.3
        elif _has_generic_marker(s.goal):
            issues.append(Issue(code='generic_goal', message='goal uses generic filler', severity='warn'))
            score -= 0.2
        if not any(c.description.strip() for c in s.criteria or []):
            issues.append(Issue(code='no_criteria', message='scenario has no success criteria', severity='fail'))
            score -= 0.3
        if not (s.context or '').strip():
            issues.append(Issue(code='no_context', message='scenario has no context', severity='warn'))
            score -= 0.1
        if tokens and not _seed_reflected(tokens, s.name, s.goal):
            issues.append(
                Issue(
                    code='seed_not_reflected',
                    message=f'contains no word of seed {seed!r} as a substring',
                    severity='info',
                )
            )
        items.append(ItemValidation(name=s.name, score=max(0.0, round(score, 4)), issues=issues))

    batch_issues = _seed_unchecked(seed)
    seen: set[str] = set()
    for s in scenarios:
        key = ' '.join(s.goal.lower().split())
        if key in seen:
            batch_issues.append(
                Issue(code='duplicate', message=f'scenario {s.name!r} duplicates another goal', severity='warn')
            )
        seen.add(key)
    return _finalize(items, batch_issues, pass_threshold=pass_threshold)


def trait_spread(personas: Sequence[Persona]) -> dict[str, float]:
    """The max-min spread of each numeric persona trait across a batch, for reporting/tuning."""
    if not personas:
        return {t: 0.0 for t in _NUMERIC_TRAITS}
    return {
        t: round(max(getattr(p, t) for p in personas) - min(getattr(p, t) for p in personas), 4)
        for t in _NUMERIC_TRAITS
    }


def trait_variance(personas: Sequence[Persona]) -> dict[str, float]:
    """Population variance of each numeric persona trait across a batch."""
    if not personas:
        return {t: 0.0 for t in _NUMERIC_TRAITS}
    return {t: round(pvariance([getattr(p, t) for p in personas]), 6) for t in _NUMERIC_TRAITS}
