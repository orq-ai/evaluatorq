"""Renderer-agnostic section data layer for agent simulation reports.

``build_report_sections(results)`` converts a ``list[SimulationResult]`` into
a list of ``ReportSection`` objects consumed by the Markdown / HTML
renderers. Mirrors the structure of ``redteam.reports.sections`` so the same
shared dispatch in ``evaluatorq.common.reports`` can drive both flavours.

Section kinds:
    - ``summary``               aggregate goal-achieved / score statistics
    - ``persona_breakdown``     per-persona success rates and token usage
    - ``scenario_breakdown``    per-scenario success rates and judge stats
    - ``judge_verdicts``        terminated-by reasons, rules broken, top reasons
    - ``turn_metrics``          turn-count distribution + average per-turn scores
    - ``evaluator_scores``      mean per-evaluator scores, plus dropped/errored counts
    - ``token_usage``           prompt/completion/total + per-conversation summary
    - ``individual_results``    one entry per ``SimulationResult`` (transcript)
    - ``errors``                count by error type for error-terminated runs
    - ``recommendations``       LLM remediation suggestions (opt-in, when provided)
"""

from __future__ import annotations

import json
import operator
from collections import Counter, defaultdict
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from evaluatorq.common.messages import coerce_content_text
from evaluatorq.contracts import ReportSection, Usage
from evaluatorq.simulation.evaluators.scorers import read_criteria_meta
from evaluatorq.simulation.metrics import TURN_METRICS
from evaluatorq.simulation.types import CriteriaRow, SimulationEntry, TranscriptMessage, criterion_id_for

if TYPE_CHECKING:
    from evaluatorq.simulation.types import SimulationRecommendation, SimulationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _persona_name(result: SimulationResult) -> str:
    return str(result.metadata.get('persona', 'unknown'))


def _scenario_name(result: SimulationResult) -> str:
    return str(result.metadata.get('scenario', 'unknown'))


def _model_name(result: SimulationResult) -> str:
    return str(result.metadata.get('model', 'unknown'))


def _evaluator_scores(result: SimulationResult) -> dict[str, float]:
    raw = result.metadata.get('evaluator_scores')
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, int | float)}


def _evaluator_errors(result: SimulationResult) -> dict[str, str]:
    """Evaluator name → why its score was unusable, as stamped by ``_stamp_evaluator_scores``."""
    raw = result.metadata.get('evaluator_errors')
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _error_message(result: SimulationResult) -> str | None:
    err = result.metadata.get('error')
    if err:
        return str(err)
    errors = result.metadata.get('criteria_errors')
    if isinstance(errors, list) and errors:
        return str(errors[0])
    return None


def _is_errored(result: SimulationResult) -> bool:
    """A run is errored if it has an error metadata key or the judge terminated
    it due to an error. Shared between the summary and errors sections so the
    "errors" counts agree across the report.
    """
    return bool(_error_message(result)) or result.terminated_by.value == 'error'


def _criteria_meta(result: SimulationResult) -> list[dict[str, Any]]:
    raw = result.metadata.get('criteria_meta')
    if raw is not None:
        valid, _ = read_criteria_meta(result)
        return [c.model_dump(mode='json') for c in valid]
    # Fallback to lossy criteria_results (no ids/type). The safety classification
    # (must_not_happen) is unavailable here, so make the degradation visible.
    logger.debug('criteria_meta absent; safety classification unavailable, falling back to criteria_results')
    cr = result.criteria_results or {}
    return [
        {'id': criterion_id_for(i), 'description': desc, 'type': None, 'passed': bool(passed)}
        for i, (desc, passed) in enumerate(cr.items())
    ]


def _cohort_id(result: SimulationResult, kind: Literal['persona', 'scenario']) -> str:
    explicit = result.metadata.get(f'{kind}_id')
    if explicit:
        return f'{kind}:{explicit}'
    if kind == 'persona':
        snapshot = {
            'name': _persona_name(result),
            'traits': result.metadata.get('persona_traits') or {},
        }
    else:
        snapshot = {
            'name': _scenario_name(result),
            'goal': result.metadata.get('scenario_goal'),
            'context': result.metadata.get('scenario_context'),
            'criteria': [
                {
                    'id': criterion.get('id'),
                    'description': criterion.get('description'),
                    'type': criterion.get('type'),
                }
                for criterion in _criteria_meta(result)
            ],
        }
    payload = json.dumps(snapshot, sort_keys=True, separators=(',', ':'), default=str)
    return f'{kind}:{sha256(payload.encode()).hexdigest()[:16]}'


def _persona_cohort_id(result: SimulationResult) -> str:
    return _cohort_id(result, 'persona')


def _scenario_cohort_id(result: SimulationResult) -> str:
    return _cohort_id(result, 'scenario')


def _criteria_rows(result: SimulationResult) -> list[dict[str, Any]]:
    """One row per criterion, carrying the audit provenance every renderer needs.

    ``audited`` and ``evidence`` come straight from ``criteria_meta`` (``None`` on
    runs saved before those keys existed). ``state`` — the rendering verdict no
    surface may recompute — is `CriteriaRow`'s computed field, which is why these
    rows are built as `CriteriaRow` objects and dumped rather than hand-assembled:
    the dict and the model then cannot disagree.

    A malformed ``audited`` or ``evidence`` degrades to ``None`` (unknown) **with a
    warning**: this is the field whose whole purpose is separating *unknown* from
    *checked*, so coercing it in silence would recreate the defect it exists to
    expose (CLAUDE.md: a degraded path announces itself).
    """
    rows = []
    for c in _criteria_meta(result):
        passed = c['passed']
        audited = c.get('audited')
        evidence = c.get('evidence')
        rows.append(
            CriteriaRow(
                id=c['id'],
                description=c.get('description', c['id']),
                type=c.get('type'),
                passed=passed,
                safety=(c.get('type') == 'must_not_happen') and not passed,
                audited=audited,
                evidence=str(evidence) if evidence else None,
            ).model_dump(mode='json')
        )
    return rows


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_summary_section(results: list[SimulationResult], executive_summary: str | None = None) -> ReportSection:
    total = len(results)
    achieved = sum(1 for r in results if r.goal_achieved and not _is_errored(r))
    errored = sum(1 for r in results if _is_errored(r))
    avg_score = sum(r.goal_completion_score for r in results) / total if total else 0.0
    avg_turns = sum(r.turn_count for r in results) / total if total else 0.0
    total_tokens = sum(r.token_usage.total_tokens for r in results)
    success_rate = (achieved / total) if total else 0.0
    verdict = 'pass' if success_rate >= 0.8 else ('warn' if success_rate >= 0.5 else 'fail')

    if success_rate >= 0.80:
        confidence = 'HIGH'
        confidence_note = f'{achieved}/{total} goals achieved'
    elif success_rate >= 0.50:
        confidence = 'MEDIUM'
        confidence_note = f'{achieved}/{total} goals achieved'
    else:
        confidence = 'LOW'
        confidence_note = f'only {achieved}/{total} goals achieved'

    return ReportSection(
        kind='summary',
        title='Executive Summary',
        data={
            'total_conversations': total,
            'goals_achieved': achieved,
            'goals_failed': total - achieved - errored,
            'errors': errored,
            'success_rate': success_rate,
            'avg_goal_completion_score': avg_score,
            'avg_turn_count': avg_turns,
            'total_tokens': total_tokens,
            'confidence': confidence,
            'confidence_note': confidence_note,
            'verdict': verdict,
            'narrative': executive_summary,
        },
    )


def _build_overview_section(results: list[SimulationResult]) -> ReportSection:
    """Introductory framing: which personas (with traits) and scenarios (with
    goals + criteria) were exercised. Traits/goals are read from metadata when
    persisted; older results fall back to names + recovered criteria only."""
    personas: dict[str, dict[str, Any]] = {}
    for r in results:
        cohort_id = _persona_cohort_id(r)
        name = _persona_name(r)
        if cohort_id not in personas:
            traits = r.metadata.get('persona_traits')
            personas[cohort_id] = {
                'id': cohort_id,
                'name': name,
                'conversations': 0,
                'traits': traits if isinstance(traits, dict) else None,
                'background': (traits or {}).get('background') if isinstance(traits, dict) else None,
            }
        personas[cohort_id]['conversations'] += 1

    scenarios: dict[str, dict[str, Any]] = {}
    scenario_results: dict[str, list[SimulationResult]] = {}
    for r in results:
        cohort_id = _scenario_cohort_id(r)
        name = _scenario_name(r)
        if cohort_id not in scenarios:
            scenarios[cohort_id] = {
                'id': cohort_id,
                'name': name,
                'goal': r.metadata.get('scenario_goal'),
                'context': r.metadata.get('scenario_context'),
                'criteria': [{'description': c['description'], 'type': c['type']} for c in _criteria_rows(r)],
            }
        scenario_results.setdefault(cohort_id, []).append(r)

    for cohort_id, scenario in scenarios.items():
        items = scenario_results.get(cohort_id, [])
        scenario['pass_rate'] = (sum(1 for r in items if r.goal_achieved) / len(items)) if items else None

    return ReportSection(
        kind='overview',
        title='Overview',
        data={
            'total_conversations': len(results),
            'personas': list(personas.values()),
            'scenarios': list(scenarios.values()),
        },
    )


def _build_failures_first_section(results: list[SimulationResult]) -> ReportSection:
    rows = []
    for idx, r in enumerate(results):
        if r.goal_achieved or _is_errored(r):
            continue
        rows_c = _criteria_rows(r)
        violated = [c['description'] for c in rows_c if not c['passed']]
        rows.append({
            'index': idx + 1,
            'persona': _persona_name(r),
            'scenario': _scenario_name(r),
            'violated': violated,
            # All criteria (pass + fail) for the collapsible dot view; ``violated``
            # is kept for the markdown renderer.
            'criteria': [
                {
                    'description': c['description'],
                    'passed': c['passed'],
                    'safety': c['safety'],
                    # Carried so the dot view can paint an unaudited criterion
                    # neutral instead of green (RES-1308).
                    'state': c['state'],
                }
                for c in rows_c
            ],
            'has_safety': any(c['safety'] for c in rows_c),
            'terminated_by': r.terminated_by.value,
            'reason': r.reason or '',
            'score': r.goal_completion_score,
            'anchor': f'conv-{idx + 1}',
        })
    return ReportSection(kind='failures_first', title='Failures', data={'rows': rows})


def _build_persona_breakdown_section(results: list[SimulationResult]) -> ReportSection:
    by_persona: dict[str, list[SimulationResult]] = defaultdict(list)
    for r in results:
        by_persona[_persona_cohort_id(r)].append(r)

    rows: list[dict[str, Any]] = []
    for cohort_id, items in by_persona.items():
        total = len(items)
        achieved = sum(1 for r in items if r.goal_achieved and not _is_errored(r))
        avg_score = sum(r.goal_completion_score for r in items) / total
        tokens = sum(r.token_usage.total_tokens for r in items)
        rows.append({
            'id': cohort_id,
            'persona': _persona_name(items[0]),
            'conversations': total,
            'goals_achieved': achieved,
            'success_rate': achieved / total,
            'avg_goal_completion_score': avg_score,
            'total_tokens': tokens,
        })
    rows.sort(key=operator.itemgetter('success_rate'))
    return ReportSection(
        kind='persona_breakdown',
        title='Per-Persona Breakdown',
        data={'rows': rows},
    )


def _build_scenario_breakdown_section(results: list[SimulationResult]) -> ReportSection:
    by_scenario: dict[str, list[SimulationResult]] = defaultdict(list)
    for r in results:
        by_scenario[_scenario_cohort_id(r)].append(r)

    rows: list[dict[str, Any]] = []
    for cohort_id, items in by_scenario.items():
        total = len(items)
        achieved = sum(1 for r in items if r.goal_achieved and not _is_errored(r))
        avg_score = sum(r.goal_completion_score for r in items) / total
        avg_turns = sum(r.turn_count for r in items) / total
        rows.append({
            'id': cohort_id,
            'scenario': _scenario_name(items[0]),
            'conversations': total,
            'goals_achieved': achieved,
            'success_rate': achieved / total,
            'avg_goal_completion_score': avg_score,
            'avg_turn_count': avg_turns,
            'total_tokens': sum(r.token_usage.total_tokens for r in items),
        })
    rows.sort(key=operator.itemgetter('success_rate'))
    return ReportSection(
        kind='scenario_breakdown',
        title='Per-Scenario Breakdown',
        data={'rows': rows},
    )


def _build_judge_verdicts_section(results: list[SimulationResult]) -> ReportSection:
    by_terminated_by: Counter[str] = Counter(r.terminated_by.value for r in results)
    all_rules_broken: Counter[str] = Counter()
    for r in results:
        all_rules_broken.update(r.rules_broken)

    return ReportSection(
        kind='judge_verdicts',
        title='Judge Verdicts',
        data={
            'terminated_by': dict(by_terminated_by),
            'rules_broken': dict(all_rules_broken.most_common(15)),
            'total_rules_broken_instances': sum(all_rules_broken.values()),
        },
    )


def _build_turn_metrics_section(results: list[SimulationResult]) -> ReportSection:
    turn_counts: Counter[int] = Counter(r.turn_count for r in results)
    # Aggregate per-turn quality metrics (averages across all runs that reported them).
    # Use direct attribute access on TurnMetrics so a future rename surfaces
    # as AttributeError instead of silently dropping the field from the report.
    qualities: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for tm in r.turn_metrics:
            for metric in TURN_METRICS:
                value = getattr(tm, metric.key)
                if value is not None:
                    qualities[metric.key].append(float(value))
    avg_qualities = {k: sum(v) / len(v) for k, v in qualities.items() if v}

    # Per-conversation turn counts, longest first, for the horizontal bar.
    per_conversation = [
        {
            'index': idx + 1,
            'label': f'#{idx + 1} {_persona_name(r)} · {_scenario_name(r)}',
            'turns': r.turn_count,
        }
        for idx, r in enumerate(results)
    ]
    per_conversation.sort(key=operator.itemgetter('turns'), reverse=True)

    return ReportSection(
        kind='turn_metrics',
        title='Turn Metrics',
        data={
            'per_conversation': per_conversation,
            'turn_count_distribution': dict(sorted(turn_counts.items())),
            'avg_quality_metrics': avg_qualities,
        },
    )


def _build_evaluator_scores_section(results: list[SimulationResult]) -> ReportSection | None:
    """Per-evaluator score statistics, with the scores that never arrived counted.

    ``runs`` counts only the numeric scores the statistics are computed over, so
    on its own it cannot distinguish an evaluator that ran twice from one that
    ran ten times and died eight. ``dropped`` carries that missing count, and an
    evaluator whose every run failed still gets a row — with ``None`` statistics
    rather than a fabricated ``0.00``.
    """
    by_evaluator: dict[str, list[float]] = defaultdict(list)
    dropped: Counter[str] = Counter()
    reasons: dict[str, str] = {}
    for r in results:
        for name, score in _evaluator_scores(r).items():
            by_evaluator[name].append(score)
        for name, reason in _evaluator_errors(r).items():
            dropped[name] += 1
            reasons.setdefault(name, reason)
    if not by_evaluator and not dropped:
        return None
    rows: list[dict[str, Any]] = []
    for name in sorted(set(by_evaluator) | set(dropped)):
        values = by_evaluator.get(name) or []
        rows.append({
            'evaluator': name,
            'runs': len(values),
            'dropped': dropped.get(name, 0),
            'mean_score': (sum(values) / len(values)) if values else None,
            'min_score': min(values) if values else None,
            'max_score': max(values) if values else None,
            'first_error': reasons.get(name),
        })
    return ReportSection(
        kind='evaluator_scores',
        title='Evaluator Scores',
        data={'rows': rows},
    )


def _build_token_usage_section(
    results: list[SimulationResult],
    *,
    run_token_usage_total: Usage | None = None,
) -> ReportSection:
    # Usage.__add__ aggregates cost alongside tokens, keeping None ("provider
    # did not report cost") distinct from 0.0 ("provider reported free").
    usage_total: Usage = sum((r.token_usage for r in results), Usage())
    input_tokens = usage_total.input_tokens
    output_tokens = usage_total.output_tokens
    total = usage_total.total_tokens
    cached = usage_total.cached_tokens
    cache_creation = usage_total.cache_creation_tokens
    reasoning = usage_total.reasoning_tokens
    n = len(results) or 1
    data: dict[str, Any] = {
        # Legacy prompt_/completion_ keys retained for downstream consumers. These are
        # the simulation-only total, labelled 'simulation_*' so a renderer holding
        # run_token_usage_total too cannot conflate the two.
        'prompt_tokens': input_tokens,
        'completion_tokens': output_tokens,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total,
        'cached_tokens': cached,
        'cache_creation_tokens': cache_creation,
        'reasoning_tokens': reasoning,
        'avg_total_per_conversation': total / n,
        'avg_input_per_conversation': input_tokens / n,
        'avg_output_per_conversation': output_tokens / n,
        'avg_prompt_per_conversation': input_tokens / n,
        'avg_completion_per_conversation': output_tokens / n,
        'input_cost': usage_total.input_cost,
        'output_cost': usage_total.output_cost,
        'total_cost': usage_total.total_cost,
        # Without these cost_coverage() always sees 0 and drops the "(N of M calls)"
        # qualifier, so a partial total reads as authoritative.
        'calls': usage_total.calls,
        'priced_calls': usage_total.priced_calls,
        'unknown_usage_conversations': sum(1 for r in results if not r.token_usage_known),
    }
    # Whole-run figure: simulation + generation + executive summary. Omitted when the
    # caller has no figure that fairly represents the run (e.g. a filtered view).
    if run_token_usage_total is not None:
        data['run_total_input_tokens'] = run_token_usage_total.input_tokens
        data['run_total_output_tokens'] = run_token_usage_total.output_tokens
        data['run_total_total_tokens'] = run_token_usage_total.total_tokens
        data['run_total_cost'] = run_token_usage_total.total_cost
        data['run_total_calls'] = run_token_usage_total.calls
        data['run_total_priced_calls'] = run_token_usage_total.priced_calls
    return ReportSection(
        kind='token_usage',
        title='Token Usage',
        data=data,
    )


def individual_entries(results: list[SimulationResult]) -> list[SimulationEntry]:
    """Build typed ``SimulationEntry`` objects for every result.

    This is the single source of truth for the per-result field extraction that
    used to live inline inside ``_build_individual_results_section``.  The
    wrapper delegates here and dumps via ``model_dump(mode='json')``, so both
    static exporters and any future consumer get byte-identical output.
    """
    entries: list[SimulationEntry] = []
    for idx, r in enumerate(results):
        target_model = r.metadata.get('target_model')
        entries.append(
            SimulationEntry(
                index=idx,
                persona=_persona_name(r),
                scenario=_scenario_name(r),
                model=_model_name(r),
                target_model=str(target_model) if target_model else None,
                terminated_by=r.terminated_by.value,
                goal_achieved=r.goal_achieved,
                goal_completion_score=r.goal_completion_score,
                rules_broken=list(r.rules_broken),
                criteria=[CriteriaRow(**row) for row in _criteria_rows(r)],
                criteria_verified=r.criteria_verified,
                turn_count=r.turn_count,
                total_tokens=r.token_usage.total_tokens,
                judge_reason=r.reason,
                error=_error_message(r),
                evaluator_scores=_evaluator_scores(r),
                transcript=[TranscriptMessage(role=m.role, content=coerce_content_text(m.content)) for m in r.messages],
                thread_id=r.thread_id,
                last_trace_id=r.last_trace_id,
            )
        )
    return entries


def _build_individual_results_section(results: list[SimulationResult]) -> ReportSection:
    return ReportSection(
        kind='individual_results',
        title='Individual Conversations',
        data={'entries': [e.model_dump(mode='json') for e in individual_entries(results)]},
    )


def _build_errors_section(results: list[SimulationResult]) -> ReportSection | None:
    errored = [r for r in results if _is_errored(r)]
    if not errored:
        return None
    err_messages = [(_error_message(r) or r.reason or 'unknown') for r in errored]
    by_message: Counter[str] = Counter(err_messages)
    return ReportSection(
        kind='errors',
        title='Errors',
        data={
            'total_errored': len(errored),
            'by_message': dict(by_message.most_common(10)),
        },
    )


def _build_persona_scenario_heatmap_section(results: list[SimulationResult]) -> ReportSection:
    persona_names: dict[str, str] = {}
    scenario_names: dict[str, str] = {}
    agg: dict[tuple[str, str], list[bool]] = defaultdict(list)
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in results:
        persona_id, scenario_id = _persona_cohort_id(r), _scenario_cohort_id(r)
        persona_names.setdefault(persona_id, _persona_name(r))
        scenario_names.setdefault(scenario_id, _scenario_name(r))
        agg[persona_id, scenario_id].append(r.goal_achieved)
        scores[persona_id, scenario_id].append(r.goal_completion_score)

    def labels(names: dict[str, str]) -> dict[str, str]:
        raw_names = set(names.values())
        used: set[str] = set()
        disambiguated: dict[str, str] = {}
        for cohort_id, name in names.items():
            label = name
            if label in used:
                suffix = 2
                label = f'{name} ({suffix})'
                suffix += 1
                while label in raw_names or label in used:
                    label = f'{name} ({suffix})'
                    suffix += 1
            disambiguated[cohort_id] = label
            used.add(label)
        return disambiguated

    persona_labels = labels(persona_names)
    scenario_labels = labels(scenario_names)
    cells = [
        {
            'persona_id': persona_id,
            'scenario_id': scenario_id,
            'persona': persona_labels[persona_id],
            'scenario': scenario_labels[scenario_id],
            'success_rate': (sum(v) / len(v)) if v else 0.0,
            # Continuous avg goal-completion score — the heatmap renders this so
            # single-conversation cells show a gradient, not a 0/100 binary.
            'avg_score': (sum(sc) / len(sc)) if (sc := scores[persona_id, scenario_id]) else 0.0,
            'n': len(v),
        }
        for (persona_id, scenario_id), v in agg.items()
    ]
    return ReportSection(
        kind='persona_scenario_heatmap',
        title='Persona x Scenario Success',
        data={'personas': list(persona_labels.values()), 'scenarios': list(scenario_labels.values()), 'cells': cells},
    )


def _build_score_distribution_section(results: list[SimulationResult]) -> ReportSection:
    return ReportSection(
        kind='score_distribution',
        title='Goal Score Distribution',
        data={'scores': [r.goal_completion_score for r in results]},
    )


def _build_turn_quality_timeline_section(results: list[SimulationResult]) -> ReportSection:
    by_turn: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for tm in r.turn_metrics:
            for metric in TURN_METRICS:
                val = getattr(tm, metric.key)
                if val is not None:
                    by_turn[tm.turn_number][metric.key].append(val)
    turns = sorted(by_turn)
    # None (not 0.0) for turns with no measurement — e.g. factual_accuracy is
    # only scored when ground truth exists, so unmeasured turns must read as a
    # gap, not a zero score. Series with no data at all are dropped entirely.
    series = {
        metric.key: [(sum(vals) / len(vals)) if (vals := by_turn[t][metric.key]) else None for t in turns]
        for metric in TURN_METRICS
    }
    series = {m: vals for m, vals in series.items() if any(v is not None for v in vals)}
    return ReportSection(
        kind='turn_quality_timeline',
        title='Turn Quality Timeline',
        data={'turns': turns, 'series': series},
    )


def _pretty_trigger(trigger: str) -> str:
    """Humanize a 'kind: detail' trigger label for report display."""
    kind, _, detail = trigger.partition(':')
    label = {
        'rule_broken': 'Rule broken',
        'criterion_failed': 'Criterion failed',
        'low_factual_accuracy': 'Low factual accuracy',
        'high_hallucination_risk': 'High hallucination risk',
        'poor_tone': 'Poor tone',
    }.get(kind.strip(), kind.strip().replace('_', ' ').capitalize())
    detail = detail.strip()
    return f'{label}: {detail}' if detail else label


def _build_recommendations_section(
    recommendations: list[SimulationRecommendation],
) -> ReportSection:
    rows = [
        {
            'index': rec.result_index + 1,
            'datapoint_id': rec.datapoint_id,
            'persona': rec.persona,
            'scenario': rec.scenario,
            'triggers': [_pretty_trigger(t) for t in rec.triggers],
            'suggestions': list(rec.suggestions),
            'anchor': f'conv-{rec.result_index + 1}',
        }
        for rec in recommendations
    ]
    return ReportSection(
        kind='recommendations',
        title='Remediation Suggestions',
        data={'rows': rows},
    )


def _build_failure_mode_section(results: list[SimulationResult]) -> ReportSection:
    counts: Counter[str] = Counter()
    for r in results:
        if r.goal_achieved:
            continue
        scen = _scenario_name(r)
        for c in _criteria_rows(r):
            # `state`, not `passed`: a criterion nobody audited is unknown, and a
            # cross-run failure-mode tally is exactly where an unknown must not be
            # counted as a recurring failure.
            if c['state'] == 'fail':
                counts[f'{scen}: {c["description"]}'] += 1
    return ReportSection(
        kind='failure_mode',
        title='Failure Modes',
        data={'rows': counts.most_common(15)},
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_report_sections(
    results: list[SimulationResult],
    *,
    executive_summary: str | None = None,
    recommendations: list[SimulationRecommendation] | None = None,
    run_token_usage_total: Usage | None = None,
) -> list[ReportSection]:
    """Produce the ordered list of report sections from simulation results.

    ``recommendations`` are pre-generated (LLM calls happen at run time, not
    render time); when absent or empty the section is omitted entirely.

    ``run_token_usage_total`` is ``SimulationRun.token_usage_total`` — the
    whole-run figure (simulation + generation + executive summary). Pass it
    when *results* is the run's full, unfiltered result set so the rendered
    token-usage section can show both the simulation-only total (recomputed
    from *results*, so it stays correct even when *results* is filtered) and
    the run total side by side, labelled distinctly. Omit it (the default)
    when *results* is a subset — a filtered dashboard view, for instance — for
    which the whole-run figure would be misleading.
    """
    sections: list[ReportSection] = []
    # Order tells the story worst-first: verdict -> what failed -> how it failed
    # -> where (heatmaps) -> distributions/trends -> breakdowns -> diagnostics.
    # Token usage is operational trivia, so it sinks near the transcripts.
    sections.extend((
        _build_summary_section(results, executive_summary),
        _build_overview_section(results),
        _build_failures_first_section(results),
        _build_failure_mode_section(results),
    ))
    if recommendations:
        sections.append(_build_recommendations_section(recommendations))
    sections.extend((
        _build_persona_scenario_heatmap_section(results),
        _build_score_distribution_section(results),
        _build_turn_quality_timeline_section(results),
        _build_persona_breakdown_section(results),
        _build_scenario_breakdown_section(results),
        _build_judge_verdicts_section(results),
        _build_turn_metrics_section(results),
    ))
    evaluator = _build_evaluator_scores_section(results)
    if evaluator is not None:
        sections.append(evaluator)
    errors = _build_errors_section(results)
    if errors is not None:
        sections.append(errors)
    sections.extend((
        _build_token_usage_section(results, run_token_usage_total=run_token_usage_total),
        _build_individual_results_section(results),
    ))
    return sections
