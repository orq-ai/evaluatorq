"""Simulation conversation evidence shared by dashboard and HTML exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluatorq.common.messages import coerce_content_text
from evaluatorq.common.reports import esc
from evaluatorq.common.reports.messages import render_message_list

if TYPE_CHECKING:
    from evaluatorq.simulation.contracts import SimulationEntry


_ROLE_LABELS = {'user': 'USR', 'assistant': 'AGT', 'system': 'SYS', 'developer': 'SYS', 'tool': 'TOOL'}


def _render_criteria_column(entry: SimulationEntry) -> str:
    """Render the CRITERIA section: verdict header + polarity-explicit rows.

    Outcome polarity is sacred here (DESIGN.md): a green check on a
    ``must_not_happen`` criterion reads backwards unless the requirement is
    stated *next to* the check. So each row leads with the icon (result:
    pass/fail) immediately followed by a ``Required`` / ``Prohibited`` chip
    (the requirement that gives the check its meaning) — no far-right orphan
    label to hunt for. A verdict line above the list answers the reader's
    first question ("did it pass, how many met?") before they scan the rows.

    A criterion the judge never audited renders as ``unknown`` (a neutral ``?``),
    never as met, and is counted separately in the tally — otherwise this column
    reads "2/2 criteria met" beside an evaluator column reading
    ``criteria_met 0.0`` (RES-1308). ``entry.criteria_verified is False`` says the
    whole audit is missing and gets its own line.
    """
    criteria = entry.criteria or []
    items: list[str] = []
    for c in criteria:
        state_class = f'sim-criterion-{c.state}'
        # ✓ / ✗ / ? — "?" is the honest glyph for a criterion nobody checked.
        icon = {'pass': '&#x2713;', 'fail': '&#x2717;'}.get(c.state, '?')
        prohibited = c.type == 'must_not_happen'
        # The chip carries the requirement's polarity right beside the result
        # icon; keep the unsafe hook so `must_not_happen` reads red.
        chip_html = ''
        if c.type:
            chip_label = 'Prohibited' if prohibited else 'Required'
            chip_class = 'sim-ctype sim-ctype-unsafe' if prohibited else 'sim-ctype'
            chip_html = f'<span class="{chip_class}">{chip_label}</span>'
        desc = esc(c.description)
        # Evidence is the judge's own quote for why this criterion is marked
        # occurred; showing the outcome without it makes the verdict unfalsifiable.
        evidence_html = (
            f'<span class="sim-criterion-evidence">&ldquo;{esc(c.evidence)}&rdquo;</span>' if c.evidence else ''
        )
        unaudited_html = '<span class="sim-criterion-unaudited">not audited</span>' if c.state == 'unknown' else ''
        items.append(
            f'<li class="sim-criterion {state_class}">'
            f'<span class="sim-criterion-icon">{icon}</span>'
            f'{chip_html}'
            f'<span class="sim-criterion-desc">{desc}</span>'
            f'{unaudited_html}'
            f'{evidence_html}'
            f'</li>'
        )

    # Goal outcome and criteria-met count are two different facts: an agent can
    # miss the goal while still satisfying every criterion (and vice versa), so
    # "FAIL · 4/4 met" reads as a contradiction. Show them as separate spans —
    # the coloured goal verdict, then a neutral criteria tally.
    verdict_html = ''
    if criteria:
        met = sum(1 for c in criteria if c.state == 'pass')
        unknown = sum(1 for c in criteria if c.state == 'unknown')
        passed = entry.goal_achieved
        verdict_class = 'sim-criteria-verdict--pass' if passed else 'sim-criteria-verdict--fail'
        verdict_word = 'Goal met' if passed else 'Goal missed'
        if entry.error or entry.terminated_by == 'error':
            verdict_word = 'Execution error'
            verdict_class = 'sim-criteria-verdict--error'
        unknown_html = f'<span class="sim-criteria-unknown">{unknown} not audited</span>' if unknown else ''
        verdict_html = (
            f'<span class="sim-criteria-verdict {verdict_class}">{verdict_word}</span>'
            f'<span class="sim-criteria-count">{met}/{len(criteria)} criteria met</span>'
            f'{unknown_html}'
        )

    # Run-level: no audit arrived at all, so every row above is a default and
    # `criteria_met` scored this run 0.0. Say it here rather than let the tally
    # imply otherwise.
    unverified_html = ''
    if entry.criteria_verified is False:
        unverified_html = (
            '<p class="sim-criteria-unverified">Criteria unverified — the judge returned no '
            'per-criterion audit, so these verdicts are defaults and <code>criteria_met</code> '
            'scored this run 0.0.</p>'
        )

    # Empty state: a criteria block that vanishes on zero rows is
    # indistinguishable from a bug (CODING_STANDARDS.md).
    list_html = (
        f'<ul class="sim-criteria-list">{"".join(items)}</ul>'
        if items
        else '<p class="sim-criteria-empty">No criteria defined for this scenario.</p>'
    )

    # The judge rationale is folded into the criteria block (it explains the
    # verdict above it), not a separate callout — one "outcome" section.
    judge_reason = esc(entry.judge_reason or '')
    judge_html = ''
    if judge_reason:
        judge_html = (
            f'<div class="sim-judge">'
            f'<span class="sim-judge-label">Judge</span>'
            f'<p class="sim-judge-reason">{judge_reason}</p>'
            f'</div>'
        )

    return (
        f'<div class="sim-criteria">'
        f'<div class="sim-criteria-head">'
        f'<span class="sim-criteria-header">CRITERIA</span>{verdict_html}'
        f'</div>'
        f'{unverified_html}'
        f'{list_html}'
        f'{judge_html}'
        f'</div>'
    )


_PERSONA_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.5-6 8-6s8 2 8 6"/></svg>'
)

_SCENARIO_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M4 5h16M4 12h16M4 19h10"/></svg>'
)


def _sim_conv_value(text: str, kind: str, entity_id: str | None) -> str:
    """A persona/scenario value; a click-through to its cohort card when we know
    the entity's DOM id, otherwise plain text.

    The trigger reuses the same ``data-sim-entity-trigger`` contract the cohort
    cards use, so clicking the value inside the conversation drawer opens the
    persona/scenario card (with a Back button, via the shared drawer JS).
    """
    safe = esc(text)
    if not entity_id:
        return f'<span class="sim-conv-value">{safe}</span>'
    return (
        f'<button type="button" class="sim-conv-value sim-conv-value--link" '
        f'data-sim-entity-trigger data-entity-kind="{esc(kind)}" data-entity-id="{esc(entity_id)}">'
        f'{safe}</button>'
    )


def _render_conversation_summary(
    entry: SimulationEntry,
    persona_entity_id: str | None = None,
    scenario_entity_id: str | None = None,
) -> str:
    """Persona + scenario recap and a turn-count chip for the drawer header.

    The collapsed row shows only "#N · persona · scenario" truncated; here we
    give the full text so the reader has the setup in view while reading the
    transcript. When the persona/scenario resolve to a cohort card, the value is
    a click-through trigger. All fields are user-supplied — esc() guards the
    stored-XSS vector.
    """
    turns = entry.turn_count
    turn_word = 'turn' if turns == 1 else 'turns'
    persona_val = _sim_conv_value(entry.persona, 'persona', persona_entity_id)
    scenario_val = _sim_conv_value(entry.scenario, 'scenario', scenario_entity_id)

    def field(icon: str, label: str, value_html: str) -> str:
        return (
            f'<div class="sim-conv-field">'
            f'<span class="sim-conv-ico">{icon}</span>'
            f'<div class="sim-conv-field-text">'
            f'<span class="sim-conv-label">{label}</span>{value_html}'
            f'</div></div>'
        )

    return (
        f'<div class="sim-conv-summary">'
        f'<div class="sim-conv-head">'
        f'<span class="sim-conv-index">#{entry.index}</span>'
        f'<span class="sim-conv-turns-pill">{turns} {turn_word}</span>'
        f'</div>'
        f'<div class="sim-conv-meta">'
        f'{field(_PERSONA_ICON, "Persona", persona_val)}'
        f'{field(_SCENARIO_ICON, "Scenario", scenario_val)}'
        f'</div>'
        f'</div>'
    )


def render_transcript_fragment(
    entry: SimulationEntry,
    persona_entity_id: str | None = None,
    scenario_entity_id: str | None = None,
) -> str:
    """Render the drill-down transcript fragment for a single sim result entry.

    Design-aligned layout (spec §Transcripts): a Judge callout (sunken
    background, teal-600 left border, "Judge" mono-label + ``judge_reason``)
    followed by a two-column grid — chat bubbles (via ``render_message_list``)
    on the left, the CRITERIA column on the right. Turn-count / score /
    terminated-by metrics now live in the conversation card's ``<summary>``
    header (``render_sim_row_list``), so this fragment no longer duplicates
    them.

    Error entries retain the recorded transcript and criteria alongside the
    error, because a partial conversation remains evidence.

    All user-supplied content (persona, scenario, judge_reason, criteria
    description, message content) goes through ``esc()`` — stored-XSS vector.

    Args:
        entry: A typed ``SimulationEntry`` from ``individual_entries``.

    Returns:
        An HTML fragment (no full-page shell).
    """
    error = entry.error

    criteria_col = _render_criteria_column(entry)
    summary_html = _render_conversation_summary(entry, persona_entity_id, scenario_entity_id)

    error_html = f'<div class="sim-transcript-error"><strong>Error:</strong> {esc(str(error))}</div>' if error else ''

    # Transcript. Normalise content via coerce_content_text (handles OpenAI
    # content blocks) before handing off to the shared renderer. The '(empty)' fallback is sim-specific so we
    # apply it here rather than inside render_message_list.
    transcript = entry.transcript or []
    normalised_msgs: list[dict[str, Any]] = []
    for msg in transcript:
        content_text = coerce_content_text(msg.content) or '(empty)'
        normalised_msgs.append({'role': msg.role, 'content': content_text})

    bubbles_html = (
        f'<div class="sim-transcript-bubbles">'
        f'{render_message_list(normalised_msgs, role_labels=_ROLE_LABELS, class_prefix="sim")}'
        f'</div>'
    )
    # Criteria above the conversation: the outcome verdict frames the
    # transcript the reader is about to scroll through, not a footnote after it.
    grid_html = f'<div class="sim-transcript-grid">{criteria_col}{error_html}{bubbles_html}</div>'

    return f'<div class="sim-transcript-detail">{summary_html}{grid_html}</div>'
