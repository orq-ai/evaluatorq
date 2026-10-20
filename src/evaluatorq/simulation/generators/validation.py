"""Validate generated personas and scenarios against the generators' own quality intent (RES-1086).

The persona/scenario generators embed quality guidelines in their prompts and coverage guards during
generation, but nothing checks that the *output* actually honours them. This adds a model-free
validation/scoring pass that scores a generated batch against the concrete, checkable guidelines the
generator prompts state - specificity (no generic names/backgrounds), trait diversity across the
batch (the "vary trait values, some low some high" rule), well-formedness (scenarios need a goal and
criteria), and seed adherence - and reports per-item scores plus batch diagnostics.

It is deliberately deterministic (no LLM): it runs on every generation at zero cost and catches the
exact failure modes the prompts warn against ("Generic", "Contradictory", "All similar"). The
subjective semantic check ("does this persona truly align with the agent description") is a natural
LLM-judge extension, not built here, so the shipped pass needs no model to run.
"""

from __future__ import annotations

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
# Phrases that signal a generic, non-specific persona/scenario (from the prompts' DON'T lists).
_GENERIC_MARKERS = (
    'a customer who',
    'customer with a problem',
    'has a problem',
    'something',
    'a user who',
    'generic',
)
_NUMERIC_TRAITS = ('patience', 'assertiveness', 'politeness', 'technical_level')

Severity = Literal['warn', 'fail']


class Issue(BaseModel):
    """One thing wrong with a generated item or batch."""

    code: str
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
    overall_score: float = Field(ge=0.0, le=1.0, default=1.0)
    passed: bool = True


def _words(text: str) -> int:
    return len(text.split())


def _has_generic_marker(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _GENERIC_MARKERS)


def _seed_reflected(seed: str, *parts: str) -> bool:
    """A loose check: at least one content word of the seed appears in the item text. Seeds are
    short archetypes ("angry customer"), so token presence is a reasonable deterministic proxy.

    This is bare substring matching, so it can over- or under-match ("cat" matches "category"). It
    errs toward reporting a seed as reflected (fewer false seed_not_reflected failures); the
    semantic "really on-seed" check belongs to the LLM-judge extension noted in the module docstring.
    """
    tokens = [t for t in seed.lower().split() if len(t) > 2]
    if not tokens:
        return True
    hay = ' '.join(parts).lower()
    return any(t in hay for t in tokens)


def _finalize(items: list[ItemValidation], batch_issues: list[Issue], *, pass_threshold: float) -> ValidationReport:
    per_item = sum(i.score for i in items) / len(items) if items else 0.0
    # Each batch-level issue is a flat penalty on the overall score; a failing batch issue caps it.
    penalty = 0.15 * len(batch_issues)
    overall = max(0.0, per_item - penalty)
    passed = bool(items) and overall >= pass_threshold and not any(i.severity == 'fail' for i in batch_issues)
    return ValidationReport(items=items, batch_issues=batch_issues, overall_score=round(overall, 4), passed=passed)


def validate_personas(
    personas: Sequence[Persona],
    *,
    seed: str = '',
    pass_threshold: float = 0.7,
) -> ValidationReport:
    """Score a generated persona batch against the generator's quality guidelines (model-free).

    Per persona: a generic name, a thin or generic background, and (when ``seed`` is set) a seed
    the persona does not reflect each deduct from a 1.0 start. Batch-level: a numeric trait with
    almost no spread across the batch ("all similar") and duplicate backgrounds are flagged.
    """
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
        if seed and not _seed_reflected(seed, p.name, p.background):
            issues.append(Issue(code='seed_not_reflected', message=f'does not reflect seed {seed!r}', severity='fail'))
            score -= 0.3
        items.append(ItemValidation(name=p.name, score=max(0.0, round(score, 4)), issues=issues))

    batch_issues = _batch_persona_issues(personas)
    return _finalize(items, batch_issues, pass_threshold=pass_threshold)


def _batch_persona_issues(personas: Sequence[Persona]) -> list[Issue]:
    issues: list[Issue] = []
    if len(personas) >= 3:
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

    Per scenario: a thin or generic goal, a missing context, and missing success criteria each deduct;
    a seed not reflected in the goal/name fails. Batch-level: duplicate goals are flagged.
    """
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
        if not s.criteria:
            issues.append(Issue(code='no_criteria', message='scenario has no success criteria', severity='fail'))
            score -= 0.3
        if not s.context:
            issues.append(Issue(code='no_context', message='scenario has no context', severity='warn'))
            score -= 0.1
        if seed and not _seed_reflected(seed, s.name, s.goal):
            issues.append(Issue(code='seed_not_reflected', message=f'does not reflect seed {seed!r}', severity='fail'))
            score -= 0.3
        items.append(ItemValidation(name=s.name, score=max(0.0, round(score, 4)), issues=issues))

    batch_issues: list[Issue] = []
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
