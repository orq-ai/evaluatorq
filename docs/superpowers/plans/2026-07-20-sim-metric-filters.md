# Agent Simulation Metric Filters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Agent Simulation filters for rule violations, incomplete outcomes, raw run counts, and all four per-turn metrics.

**Architecture:** Reuse the HTMX `FilterDef` list-of-strings protocol. Complete-run bounds remain raw. A shared descriptor owns the metric key, label, and direction for report rendering and filtering.

**Tech Stack:** Python 3.13, FastHTML, HTMX, pytest, Ruff.

## Constraints

- Never filter positional criterion IDs: they mean different things across scenarios.
- `rule_broken=yes` is opt-in; absent means all results.
- Goal/quality thresholds are ceilings; turns/tokens/hallucination-risk thresholds are floors.
- Turns/tokens use raw integer maxima and `step="1"`; they are never normalized or compressed.
- Unscored results remain visible; partial results aggregate numeric turns only.
- Hide unavailable metric and no-op count controls. Reuse `filter-dd-more` and existing JavaScript state persistence.

### Task 1: Canonical metric descriptor

**Files:** Create `src/evaluatorq/simulation/metrics.py`; modify `src/evaluatorq/simulation/reports/sections.py` and `src/evaluatorq/dashboard/report_tabs.py`; test `tests/dashboard/test_report_tabs.py`.

- [ ] **Step 1: Write the failing test**

```python
def test_turn_metric_descriptor_has_keys_and_directions():
    from evaluatorq.simulation.metrics import TURN_METRICS
    assert [(m.key, m.high_is_risky) for m in TURN_METRICS] == [
        ('response_quality', False), ('hallucination_risk', True),
        ('tone_appropriateness', False), ('factual_accuracy', False),
    ]
```

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/dashboard/test_report_tabs.py::test_turn_metric_descriptor_has_keys_and_directions -q`

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement and use the descriptor**

```python
@dataclass(frozen=True)
class TurnMetric:
    key: str
    label: str
    high_is_risky: bool = False
```

Define ordered `TURN_METRICS` for response quality, hallucination risk (risky), tone appropriateness, and factual accuracy. Replace local tuples/constants in `sections.py` and `report_tabs.py` with `TURN_METRICS` plus `getattr(turn, metric.key)`.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/dashboard/test_report_tabs.py -q`

Run: `git add src/evaluatorq/simulation/metrics.py src/evaluatorq/simulation/reports/sections.py src/evaluatorq/dashboard/report_tabs.py tests/dashboard/test_report_tabs.py && git commit -m "refactor(sim): centralize turn metric metadata"`

### Task 2: Pure filter predicates

**Files:** Modify `src/evaluatorq/dashboard/filters.py`; test `tests/dashboard/test_filter.py`.

- [ ] **Step 1: Write failing tests**

```python
def test_rule_broken_is_opt_in(sim_run):
    assert [r.rules_broken for r in _sim_apply(sim_run, {'rule_broken': ['yes']})] == [['criteria_0']]
    assert len(_sim_apply(sim_run, {})) == len(sim_run.results)

def test_risk_uses_worst_turn_and_keeps_unscored(sim_run):
    filtered = _sim_apply(sim_run, {'min_hallucination_risk': ['0.70']})
    assert {r.metadata['persona'] for r in filtered} == {'risky', 'unscored'}
```

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/dashboard/test_filter.py -q`

Expected: FAIL because the new dimensions are absent.

- [ ] **Step 3: Implement**

Add `rule_broken`, `max_goal_score`, `min_turns`, `min_total_tokens`, and one threshold key per descriptor to `_SIM_DIMS`. `_sim_full_options` exposes raw `max_turns`, raw `max_total_tokens`, and available metric keys. Parse/clamp score values to `0..1` and counts to their complete-run maxima. Use `max()` for risk and `min()` for the three quality metrics; retain no-score results.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/dashboard/test_filter.py -q && uv run ruff check src/evaluatorq/dashboard/filters.py`

Run: `git add src/evaluatorq/dashboard/filters.py tests/dashboard/test_filter.py && git commit -m "feat(dashboard): filter sim runs by rule and metrics"`

### Task 3: Rail markup and HTMX integration

**Files:** Modify `src/evaluatorq/dashboard/view.py`, `src/evaluatorq/dashboard/styles.py`, `tests/dashboard/test_filter.py`, and `docs/dashboard.md`.

- [ ] **Step 1: Write failing markup/round-trip tests**

```python
def test_sim_rail_uses_raw_count_maxima():
    html = render_filter_form('rid', 'sim', {
        'persona': [], 'scenario': [], 'terminated_by': [], 'goal_outcome': ['Achieved', 'Not achieved'],
        'max_turns': ['8'], 'max_total_tokens': ['2500'], 'turn_metrics': ['hallucination_risk'],
    }, {})
    assert 'name="min_turns" min="1" max="8" step="1"' in html
    assert 'name="min_total_tokens" min="0" max="2500" step="1"' in html
    assert 'name="rule_broken" value="yes"' in html
```

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/dashboard/test_filter.py -q`

Expected: FAIL because the rail has only its current four controls.

- [ ] **Step 3: Implement and document**

Render the rule chip, goal-score ceiling, and min-turns control directly. Use the existing `filter-dd-more` element for min total tokens and all available descriptor metrics. A reusable range helper renders `all` or the `≤`/`≥` threshold. Score steps are `0.05`; raw counts use `1`. Do not render an empty More expander. Document all controls and missing-score behavior in `docs/dashboard.md`.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/dashboard/test_filter.py tests/dashboard/test_report_tabs.py -q && uv run ruff check src/evaluatorq/simulation/metrics.py src/evaluatorq/simulation/reports/sections.py src/evaluatorq/dashboard/report_tabs.py src/evaluatorq/dashboard/filters.py src/evaluatorq/dashboard/view.py`

Run: `git add src/evaluatorq/dashboard/view.py src/evaluatorq/dashboard/styles.py tests/dashboard/test_filter.py docs/dashboard.md && git commit -m "feat(dashboard): add sim metric filter controls"`

## Self-Review

- The tasks cover every approved filter, its direction, raw-unit constraint, missing-score policy, state reuse, tests, and documentation.
- Metric keys have one source of truth and all existing state machinery is reused.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-20-sim-metric-filters.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
