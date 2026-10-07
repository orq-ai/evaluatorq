"""The Insights run form: one server-rendered markup shown as the new-run page and as the run-page dialog."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, get_args
from urllib.parse import urlencode

from pydantic import ValidationError

from evaluatorq.common.model_roles import role_model
from evaluatorq.common.reports import esc
from evaluatorq.dashboard.facet_picker import render_facet_chips, render_facet_menu, window_count_note
from evaluatorq.dashboard.insights_launch import InsightsLaunchSpec, Source, get_finder_exports_dir
from evaluatorq.dashboard.insights_uploads import is_uploaded_source
from evaluatorq.dashboard.insights_views import _back_to_runs
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.view import model_control
from evaluatorq.insights.models import DimensionName, LabelSpec, reads_snapshot
from evaluatorq.insights.presets import CODING_LABELS, LABEL_PRESETS
from evaluatorq.trace_finder.models import FACET_NAMES, FacetCatalogue, FacetSelection

if TYPE_CHECKING:
    from collections.abc import Mapping

    from evaluatorq.common.model_catalogue import ModelKind
    from evaluatorq.insights.models import InsightsRun

Mount = Literal['page', 'dialog']


@dataclass(frozen=True)
class RunPreset:
    """A starting point for step 2: the groupings and questions it ticks, nothing else."""

    id: str
    title: str
    description: str
    dimensions: tuple[DimensionName, ...]
    labels: tuple[str, ...]
    coding_labels: tuple[str, ...]


RUN_PRESETS: tuple[RunPreset, ...] = (
    RunPreset(
        id='failures',
        title='Find failures',
        description='Where does the agent go wrong?',
        dimensions=('failure', 'intent'),
        labels=('made_errors', 'user_frustration'),
        coding_labels=(),
    ),
    RunPreset(
        id='intents',
        title='Understand intents',
        description='What do users come for?',
        dimensions=('intent',),
        labels=(),
        coding_labels=(),
    ),
    RunPreset(
        id='coding',
        title='Coding agent',
        description='Frustration, outcome, risky actions',
        dimensions=('intent', 'failure'),
        labels=('user_frustration',),
        coding_labels=('task_type', 'outcome', 'unfixed_error', 'risky_action'),
    ),
)

OFFERED_LABELS: tuple[str, ...] = ('sentiment', 'customer_satisfaction', 'made_errors', 'user_frustration')
OFFERED_DIMENSIONS: tuple[DimensionName, ...] = ('intent', 'failure', 'sentiment')
_CODING_NAMES: tuple[str, ...] = tuple(spec.name for spec in CODING_LABELS[1:])

QUESTION_TITLES: MappingProxyType[str, str] = MappingProxyType({
    'sentiment': 'Sentiment',
    'customer_satisfaction': 'Customer satisfaction',
    'made_errors': 'Assistant mistakes',
    'user_frustration': 'User frustration',
    'task_type': 'Task type',
    'outcome': 'Outcome',
    'verified': 'Verified',
    'scope_creep': 'Scope creep',
    'user_corrections': 'User corrections',
    'unfixed_error': 'Unfixed error',
    'risky_action': 'Risky action',
})

_DIMENSION_TEXT: MappingProxyType[DimensionName, tuple[str, str]] = MappingProxyType({
    'intent': ('Intent', 'what users asked'),
    'failure': ('Failure', 'what went wrong'),
    'sentiment': ('Sentiment', 'how users felt'),
})

_SOURCES: tuple[tuple[Source, str, str], ...] = (
    ('recent', 'Recent traces', 'Analyze a recent window.'),
    ('query', 'Search by question', 'Include only traces that match a question.'),
    ('finder', 'Finder export', 'Use a saved set of Finder matches.'),
    ('snapshot', 'Local trace file', 'Analyze a saved trace snapshot.'),
)


def _check_registries() -> None:
    """Fail at import when a preset or offered name drifts from the registries it names."""
    problems: list[str] = []
    if set(get_args(DimensionName)) != set(OFFERED_DIMENSIONS) or set(_DIMENSION_TEXT) != set(OFFERED_DIMENSIONS):
        problems.append('OFFERED_DIMENSIONS and _DIMENSION_TEXT must list every DimensionName')
    problems.extend(
        f'offered label {name!r} is not in LABEL_PRESETS' for name in OFFERED_LABELS if name not in LABEL_PRESETS
    )
    problems.extend(
        f'{name!r} has no entry in QUESTION_TITLES'
        for name in (*OFFERED_LABELS, *_CODING_NAMES)
        if name not in QUESTION_TITLES
    )
    for preset in RUN_PRESETS:
        problems.extend(
            f'preset {preset.id!r}: label {name!r} is not in LABEL_PRESETS'
            for name in preset.labels
            if name not in LABEL_PRESETS
        )
        problems.extend(
            f'preset {preset.id!r}: coding label {name!r} is not in CODING_LABELS[1:]'
            for name in preset.coding_labels
            if name not in _CODING_NAMES
        )
        problems.extend(
            f'preset {preset.id!r}: dimension {name!r} is not a DimensionName'
            for name in preset.dimensions
            if name not in get_args(DimensionName)
        )
    if len({preset.id for preset in RUN_PRESETS}) != len(RUN_PRESETS):
        problems.append('RUN_PRESETS ids must be unique')
    if problems:
        raise RuntimeError('; '.join(problems))


_check_registries()


def _getlist(form: Mapping[str, object], key: str) -> list[str]:
    getlist = getattr(form, 'getlist', None)
    if getlist is not None:
        return [str(value) for value in getlist(key)]
    raw = form.get(key)
    if raw is None:
        return []
    return [str(value) for value in raw] if isinstance(raw, (list, tuple)) else [str(raw)]


def _whole_number(form: Mapping[str, object], key: str, default: int, label: str) -> int:
    raw = str(form.get(key) or '').strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f'{label} must be a whole number.') from None


def _custom_labels(raw: str) -> tuple[LabelSpec, ...]:
    try:
        loaded = json.loads(raw or '[]')
    except json.JSONDecodeError:
        raise ValueError('Custom questions must be valid JSON.') from None
    if not isinstance(loaded, list):
        raise ValueError('Custom questions must be a JSON list.')  # noqa: TRY004 — bad input, not a bad call
    try:
        return tuple(LabelSpec.model_validate(item) for item in loaded)
    except ValidationError as exc:
        raise ValueError('; '.join(error['msg'] for error in exc.errors())) from exc


def _saved_source(population: Mapping[str, object]) -> Source:
    if reads_snapshot(population):
        return 'snapshot'
    if population.get('mode') == 'finder' or population.get('finder_export'):
        return 'finder'
    if population.get('mode') == 'query' or population.get('query'):
        return 'query'
    return 'recent'


def _saved_file_usable(source: Source, path: Path, runs_dir: Path) -> bool:
    """Whether a saved file source can still be read and would be accepted by a new launch."""
    if not path.is_file():
        return False
    if source == 'snapshot':
        return True
    return is_uploaded_source(runs_dir, path) or path.resolve().parent == get_finder_exports_dir().resolve()


INSIGHTS_MODEL_FIELDS: MappingProxyType[str, tuple[ModelKind, str]] = MappingProxyType({
    'summary_model': ('chat', 'Summary model'),
    'classifier_model': ('classify', 'Classifier model'),
    'embedding_model': ('embedding', 'Embedding model'),
    'compiler_model': ('chat', 'Question compiler'),
})


def _model_defaults() -> dict[str, str]:
    """What a run uses when the form does not choose: each Insights task's model role."""
    return {
        'summary_model': role_model('smart', task='insights.summary'),
        'classifier_model': role_model('classifier', task='insights.labels'),
        'embedding_model': role_model('embedding', task='insights.embedding'),
        'compiler_model': role_model('fast', task='finder.compiler'),
    }


@dataclass(frozen=True)
class RunFormValues:
    """Everything the run form shows, so a rejected submission re-renders exactly what the user entered."""

    source: Source = 'recent'
    query: str = ''
    window_days: int = 7
    limit: int = 100
    facets: FacetSelection = field(default_factory=FacetSelection)
    dimensions: tuple[DimensionName, ...] = ()
    labels: tuple[str, ...] = ()
    coding_labels: tuple[str, ...] = ()
    custom_labels: tuple[LabelSpec, ...] = ()
    name: str = ''
    parallelism: int = 20
    finder_export: str = ''
    snapshot_path: str = ''
    source_name: str = ''
    preset: str | None = None
    mount: Mount = 'page'
    error: str | None = None
    summary_model: str = ''
    classifier_model: str = ''
    embedding_model: str = ''
    compiler_model: str = ''

    @classmethod
    def defaults(cls) -> RunFormValues:
        """The first preset applied to Recent traces: 7 days, 200 traces."""
        preset = RUN_PRESETS[0]
        return cls(
            window_days=7,
            limit=200,
            dimensions=preset.dimensions,
            labels=preset.labels,
            coding_labels=preset.coding_labels,
            preset=preset.id,
            **_model_defaults(),
        )

    @classmethod
    def from_run(cls, run: InsightsRun, runs_dir: Path) -> RunFormValues:
        """Prefill the form from a saved run; a file source that is gone is cleared and a fresh one requested."""
        population = dict(run.population)
        source = _saved_source(population)
        saved_path = str(population.get('snapshot_path') or population.get('finder_export') or '')
        file_backed = source in ('finder', 'snapshot')
        usable = bool(saved_path) and _saved_file_usable(source, Path(saved_path).expanduser(), runs_dir)
        coding_names = {spec.name for spec in CODING_LABELS}
        configured = {spec.name for spec in run.config.labels}
        coding = [spec.name for spec in run.config.coding_labels]
        if run.config.coding_analysis and not coding:
            coding = list(_CODING_NAMES)
        custom = tuple(
            spec for spec in run.config.labels if spec.name not in LABEL_PRESETS and spec.name not in coding_names
        )
        kept_path = saved_path if file_backed and usable else ''
        return cls(
            source=source,
            query=str(population.get('query') or ''),
            window_days=int(population.get('window_days', 7)),
            limit=int(population.get('limit', 500)),
            facets=FacetSelection.model_validate(population.get('facets') or {}),
            dimensions=tuple(run.config.dimensions),
            labels=tuple(name for name in LABEL_PRESETS if name in configured),
            coding_labels=tuple(coding),
            custom_labels=custom,
            name=run.run_name,
            summary_model=run.config.summary_model,
            classifier_model=run.config.classifier_model,
            embedding_model=run.config.embedding_model,
            compiler_model=run.config.compiler_model or _model_defaults()['compiler_model'],
            finder_export=kept_path if source == 'finder' else '',
            snapshot_path=kept_path if source == 'snapshot' else '',
            source_name=str(population.get('source_name') or '') if kept_path else '',
            error=(
                'The original local file is no longer available. Browse for a fresh file to continue.'
                if file_backed and not usable
                else None
            ),
        )

    @classmethod
    def from_form(cls, form: Mapping[str, object]) -> RunFormValues:
        """Parse the form fields from `FormData` or query parameters; raises `ValueError` on malformed numbers or JSON."""
        mount: Mount = 'dialog' if form.get('mount') == 'dialog' else 'page'
        return cls(
            source=str(form.get('source') or 'recent'),
            query=str(form.get('query') or ''),
            window_days=_whole_number(form, 'window_days', 7, 'Window'),
            limit=_whole_number(form, 'limit', 100, 'Trace limit'),
            facets=FacetSelection.model_validate({name: _getlist(form, f'facet_{name}') for name in FACET_NAMES}),
            dimensions=tuple(_getlist(form, 'dimensions')),
            labels=tuple(_getlist(form, 'labels')),
            coding_labels=tuple(_getlist(form, 'coding_labels')),
            custom_labels=_custom_labels(str(form.get('custom_labels_json') or '[]')),
            name=str(form.get('name') or ''),
            parallelism=_whole_number(form, 'parallelism', 20, 'Parallel requests'),
            finder_export=str(form.get('finder_export') or ''),
            snapshot_path=str(form.get('snapshot_path') or ''),
            source_name=str(form.get('source_name') or ''),
            preset=str(form.get('preset') or '') or None,
            mount=mount,
            **{name: str(form.get(name) or '').strip() or default for name, default in _model_defaults().items()},
        )

    def launch_fields(self) -> dict[str, object]:
        """The fields `InsightsLaunchSpec.model_validate` reads."""
        return {
            'name': self.name,
            'source': self.source,
            'query': self.query,
            'finder_export': self.finder_export,
            'snapshot_path': self.snapshot_path,
            'source_name': self.source_name,
            'window_days': self.window_days,
            'limit': self.limit,
            'facets': {name: sorted(getattr(self.facets, name)) for name in FACET_NAMES},
            'parallelism': self.parallelism,
            'labels': list(self.labels),
            'coding_labels': list(self.coding_labels),
            'custom_labels': [spec.model_dump(mode='json') for spec in self.custom_labels],
            'dimensions': list(self.dimensions),
            **{name: getattr(self, name) for name in INSIGHTS_MODEL_FIELDS},
        }


def facet_options(
    catalogue: FacetCatalogue | None,
    selection: FacetSelection,
    *,
    profile_name: str | None = None,
    credential_rejected: bool = False,
    window_days: int | None = None,
) -> str:
    """Render the filter picker for an Insights window, with Orq's per-value counts and the selected values as chips."""
    if catalogue is None and credential_rejected:
        if profile_name == 'CLI OAuth':
            credential = 'Orq rejected the CLI OAuth sign-in. Run <code>orq auth login</code> and try again.'
        elif profile_name == 'Saved API key':
            credential = 'Orq rejected the API key entered in Settings. Enter a key with trace access and Save.'
        elif profile_name == 'ORQ_API_KEY' or not profile_name:
            credential = (
                'Orq rejected the dashboard Environment credentials for this workspace. Check '
                '<code>ORQ_API_KEY</code> and <code>ORQ_BASE_URL</code>.'
            )
        else:
            credential = (
                f'Orq rejected the key for profile <code>{esc(profile_name)}</code>. Check that this profile '
                'has trace access to the workspace selected in Settings.'
            )
        unavailable = (
            f'<p class="insights-facet-unavailable" role="status">{credential} In '
            '<a href="/settings" target="_blank" rel="noopener">Settings → Authentication</a>, choose a '
            'matching profile or workspace, Save, then click '
            '<button type="button" data-retry-facets>Retry</button>. Your selected filters are kept.</p>'
        )
    elif catalogue is None:
        unavailable = (
            '<p class="insights-facet-unavailable" role="status">Could not load filter choices from Orq. '
            'In <a href="/settings" target="_blank" rel="noopener">Settings → Authentication</a>, select an Orq '
            'profile with trace access and Save. If you use the environment API key, set <code>ORQ_API_KEY</code> for the '
            'dashboard and restart it. Then click <button type="button" data-retry-facets>Retry</button>. '
            'Your selected filters are kept.</p>'
        )
    else:
        unavailable = ''
    if catalogue is None and not any(getattr(selection, name) for name in FACET_NAMES):
        return unavailable
    menu = render_facet_menu(
        catalogue,
        form_id='insights-new-form',
        selection=selection,
        include_numeric=False,
        counts=catalogue.value_counts if catalogue is not None else None,
        count_note=window_count_note(window_days),
    )
    return (
        f'{unavailable}<div class="finder-controls insights-filter-picker">'
        '<span class="addwrap"><button class="add" type="button" aria-haspopup="true" '
        'aria-label="Add a trace filter">+ Filter</button>'
        f'{menu}</span><div class="insights-selected-facets" aria-live="polite">'
        f'{render_facet_chips(selection, removable=True)}</div></div>'
    )


def _kind_hint(spec: LabelSpec) -> str:
    if spec.kind == 'noul':
        return 'yes / no'
    if spec.kind == 'score':
        return 'score'
    return ', '.join(spec.criteria) if isinstance(spec.criteria, dict) else 'one of a list'


_LABEL_NOTES: MappingProxyType[str, str] = MappingProxyType({
    'customer_satisfaction': 'the priority matrix needs this',
})


def _label_hint(name: str) -> str:
    hint = _kind_hint(LABEL_PRESETS[name])
    return f'{hint} · {_LABEL_NOTES[name]}' if name in _LABEL_NOTES else hint


def _toggle(name: str, spec: LabelSpec | None, value: str, title: str, hint: str, *, checked: bool) -> str:
    tooltip = f'{spec.instructions} ({hint})' if spec is not None else hint
    return (
        f'<label class="tg" title="{esc(tooltip)}"><input type="checkbox" name="{name}" value="{esc(value)}"'
        f'{" checked" if checked else ""}><span class="tg-title">{esc(title)}</span><small>{esc(hint)}</small></label>'
    )


def _question_title(name: str) -> str:
    return QUESTION_TITLES.get(name) or name.replace('_', ' ').capitalize()


def _custom_chips(specs: tuple[LabelSpec, ...]) -> str:
    return ''.join(
        f'<span class="tg on" data-custom="{esc(spec.name)}"><span class="tg-title">{esc(spec.name)}</span>'
        f'<small>custom</small><button type="button" data-cq-remove="{esc(spec.name)}" '
        f'aria-label="Remove question {esc(spec.name)}">&times;</button></span>'
        for spec in specs
    )


def _preset_buttons(values: RunFormValues) -> str:
    return ''.join(
        f'<button type="button" class="preset{" on" if values.preset == preset.id else ""}" data-preset="{esc(preset.id)}" '
        f'data-dimensions="{esc(" ".join(preset.dimensions))}" data-labels="{esc(" ".join(preset.labels))}" '
        f'data-coding-labels="{esc(" ".join(preset.coding_labels))}" aria-pressed="{str(values.preset == preset.id).lower()}">'
        f'<b>{esc(preset.title)}</b><span>{esc(preset.description)}</span></button>'
        for preset in RUN_PRESETS
    )


def _file_source(kind: Literal['finder', 'snapshot'], values: RunFormValues) -> str:
    chosen = values.finder_export if kind == 'finder' else values.snapshot_path
    shown = values.source_name if values.source == kind and chosen else ''
    shown = shown or (Path(chosen).name if chosen else '')
    title = 'Finder export' if kind == 'finder' else 'Trace snapshot'
    hint = 'Choose a Finder export JSON file.' if kind == 'finder' else 'Choose a trace snapshot JSON file.'
    status = f'Selected {title.lower()} is ready.' if chosen else hint
    return (
        f'<div class="irf-field" data-source="{kind}"><span class="irf-label">{title}</span>'
        f'<div class="irf-file"><input type="text" readonly data-file-name="{kind}" aria-label="Selected {esc(title.lower())}" '
        f'placeholder="No file chosen" value="{esc(shown)}">'
        f'<button type="button" class="irf-btn" data-browse="{kind}">Browse…</button>'
        f'<input type="file" accept="application/json,.json" data-file="{kind}" hidden></div>'
        f'<p class="irf-hint" role="status" data-file-status="{kind}">{esc(status)}</p></div>'
    )


def _facet_picker(values: RunFormValues) -> str:
    if any(getattr(values.facets, name) for name in FACET_NAMES):
        return facet_options(FacetCatalogue(), values.facets, window_days=values.window_days)
    return '<p class="insights-muted" role="status">Loading filter values…</p>'


def _step_one(values: RunFormValues) -> str:
    sources = ''.join(
        f'<label title="{esc(hint)}"><input type="radio" name="source" value="{source}"'
        f'{" checked" if values.source == source else ""}><span>{esc(title)}</span></label>'
        for source, title, hint in _SOURCES
    )
    return (
        '<section class="irf-step" data-step="1"><h3>Which traces?</h3>'
        f'<div class="seg" role="radiogroup" aria-label="Where the traces come from">{sources}</div>'
        '<label class="irf-field" data-source="query"><span class="irf-label">Question</span>'
        '<textarea name="query" rows="3" maxlength="500" placeholder="Which conversations are about refunds?">'
        f'{esc(values.query)}</textarea>'
        '<span class="irf-hint">The classifier reads each trace in the window and keeps only the ones that match.</span></label>'
        f'{_file_source("finder", values)}{_file_source("snapshot", values)}'
        '<div id="insights-snapshot-preview" data-source="snapshot" aria-live="polite"></div>'
        '<div class="irf-row" data-source="recent query">'
        f'<label>Last <input name="window_days" type="number" min="1" max="90" value="{values.window_days}"> days, up to</label>'
        f'<label><input name="limit" type="number" min="1" max="5000" value="{values.limit}"> traces</label></div>'
        '<div class="irf-facets" data-source="recent query"><span class="irf-label">Only traces with</span>'
        f'<div id="insights-facet-options" aria-live="polite">{_facet_picker(values)}</div></div>'
        '</section>'
    )


_QUESTION_EDITOR = (
    '<div class="card irf-editor" data-cq-editor hidden>'
    '<label class="irf-field"><span class="irf-label">Name</span>'
    '<input data-cq="name" maxlength="64" placeholder="needs_human_follow_up"></label>'
    '<label class="irf-field"><span class="irf-label">Answer type</span><select data-cq="kind">'
    '<option value="noul">Yes / no</option><option value="choice">One of a list</option>'
    '<option value="score">Score 1&ndash;5</option></select></label>'
    '<label class="irf-field"><span class="irf-label">Question</span>'
    '<textarea data-cq="text" rows="2" placeholder="Should a human follow up on this conversation?"></textarea></label>'
    '<label class="irf-field" data-cq-criteria hidden><span class="irf-label">Answer options</span>'
    '<textarea data-cq="criteria" rows="5" aria-label="Answer options, one per line"></textarea></label>'
    '<p class="insights-error" role="alert" data-cq-error hidden></p>'
    '<div class="irf-editor-actions"><button type="button" class="irf-btn primary" data-cq-add>Add question</button>'
    '<button type="button" class="irf-btn" data-cq-cancel>Cancel</button></div></div>'
)


def _step_two(values: RunFormValues) -> str:
    dimensions = ''.join(
        _toggle(
            'dimensions',
            None,
            name,
            _DIMENSION_TEXT[name][0],
            _DIMENSION_TEXT[name][1],
            checked=name in values.dimensions,
        )
        for name in OFFERED_DIMENSIONS
    )
    extra_labels = tuple(name for name in values.labels if name not in OFFERED_LABELS)
    labels = ''.join(
        _toggle(
            'labels',
            LABEL_PRESETS[name],
            name,
            _question_title(name),
            _label_hint(name),
            checked=name in values.labels,
        )
        for name in (*OFFERED_LABELS, *extra_labels)
        if name in LABEL_PRESETS
    )
    coding = ''.join(
        _toggle(
            'coding_labels',
            spec,
            spec.name,
            _question_title(spec.name),
            _kind_hint(spec),
            checked=spec.name in values.coding_labels,
        )
        for spec in CODING_LABELS[1:]
    )
    hidden_hint = ' hidden' if 'made_errors' in values.labels else ''
    return (
        '<section class="irf-step" data-step="2"><h3>What should Insights find?</h3>'
        f'<h4>Presets</h4><div class="presets">{_preset_buttons(values)}</div>'
        f'<h4>Group traces by</h4><div class="toggles">{dimensions}</div>'
        f'<h4>Ask about every trace</h4><div class="toggles">{labels}'
        f'<span class="irf-custom-list" data-custom-list>{_custom_chips(values.custom_labels)}</span>'
        '<button type="button" class="tg tg-add" data-cq-open>+ Write your own question</button></div>'
        f'{_QUESTION_EDITOR}'
        f'<p class="irf-hint" data-hint-made-errors{hidden_hint}>Without “Assistant mistakes”, the error share is estimated '
        'from the summaries instead of asked directly.</p>'
        f'<h4>Coding agent questions</h4><div class="toggles">{coding}</div>'
        '</section>'
    )


def _model_pickers(values: RunFormValues) -> str:
    rows = []
    for name, (_, label) in INSIGHTS_MODEL_FIELDS.items():
        value = getattr(values, name)
        fallback = model_control(name, value, {}, label=label)
        scope = ' data-source="query"' if name == 'compiler_model' else ''
        hidden = ' hidden' if name == 'compiler_model' and values.source != 'query' else ''
        rows.append(
            f'<div class="irf-field"{scope}{hidden}><label class="irf-label" for="{name}">{esc(label)}</label>'
            f'<span hx-get="/insights/models?{esc(urlencode({"field": name, name: value}))}" hx-trigger="load" '
            f'hx-include="find input" hx-swap="outerHTML">{fallback}</span></div>'
        )
    return ''.join(rows)


def _step_three(values: RunFormValues) -> str:
    return (
        '<section class="irf-step" data-step="3"><h3>Review and start</h3>'
        '<label class="irf-field"><span class="irf-label">Run name <i>(optional)</i></span>'
        f'<input name="name" type="text" maxlength="80" placeholder="Weekly support review" value="{esc(values.name)}"></label>'
        f'<div id="insights-run-models">{_model_pickers(values)}</div>'
        '<label class="irf-field"><span class="irf-label">Parallel requests</span>'
        f'<input name="parallelism" type="number" min="1" max="200" value="{values.parallelism}"></label>'
        '<div id="insights-run-estimate"></div>'
        '<div id="insights-run-plan" class="irf-plan" role="status" aria-live="polite"></div>'
        '<p class="irf-hint">The run reads the selected traces and makes model requests. Progress appears on its run page.</p>'
        '</section>'
    )


def render_run_form(values: RunFormValues, *, csrf: str, error: str | None = None) -> str:
    """Render the three-step form; `insights-run-form.js` only controls presentation."""
    message = error or values.error
    custom_json = json.dumps([spec.model_dump(mode='json', exclude_none=True) for spec in values.custom_labels])
    steps = ''.join(
        '<li class="active" aria-current="step">' + f'{index} · {title}</li>'
        if index == 1
        else f'<li>{index} · {title}</li>'
        for index, title in enumerate(('Traces', 'Analysis', 'Review'), start=1)
    )
    return (
        f'<form id="insights-new-form" class="insights-run-form" method="post" action="/insights/runs" '
        f'data-mount="{values.mount}">'
        f'<input type="hidden" name="csrf" value="{esc(csrf)}">'
        f'<input type="hidden" name="mount" value="{values.mount}">'
        f'<input type="hidden" name="preset" value="{esc(values.preset or "")}">'
        f'<input type="hidden" name="custom_labels_json" value="{esc(custom_json)}">'
        f'<input type="hidden" name="finder_export" value="{esc(values.finder_export)}">'
        f'<input type="hidden" name="snapshot_path" value="{esc(values.snapshot_path)}">'
        f'<ol class="irf-steps" aria-label="Steps">{steps}</ol>'
        '<p id="insights-run-compact" class="irf-compact" role="status" aria-live="polite">Estimating traces, cost and time…</p>'
        f'<p id="insights-run-error" class="insights-error" role="alert"{"" if message else " hidden"}>{esc(message or "")}</p>'
        f'{_step_one(values)}{_step_two(values)}{_step_three(values)}'
        '<div class="irf-actions"><button type="button" class="irf-btn" data-irf-back>Back</button>'
        '<button type="button" class="irf-btn primary" data-irf-next>Continue</button>'
        '<button type="submit" class="irf-btn start" data-irf-start>Start run</button></div>'
        '</form>'
    )


def run_form_script_tag() -> str:
    """The deferred `insights-run-form.js` tag, versioned by content so a stale cached copy is never used."""
    script = Path(__file__).parent / 'static' / 'insights-run-form.js'
    try:
        version = hashlib.sha256(script.read_bytes()).hexdigest()[:12]
    except OSError:
        version = 'missing'
    return f'<script src="/static/insights-run-form.js?v={version}" defer></script>'


def render_run_page(values: RunFormValues, *, csrf: str, error: str | None = None) -> str:
    """Render the full new-run page around the form."""
    body = (
        '<div class="insights-layout"><div class="insights-main insights-run-page">'
        '<header class="insights-run-head"><h2>New Insights run</h2>'
        '<p>Choose the traces and what to learn from them.</p></header>'
        f'{render_run_form(values, csrf=csrf, error=error)}</div></div>'
        f'{run_form_script_tag()}'
    )
    return page('New Insights run', body, active_nav='insights', back_html=_back_to_runs())


def _source_summary(spec: InsightsLaunchSpec) -> str:
    days = f'{spec.window_days} day{"" if spec.window_days == 1 else "s"}'
    filtered = sum(len(getattr(spec.facets, name)) for name in FACET_NAMES)
    filters = f', {filtered} filter value{"" if filtered == 1 else "s"}' if filtered else ''
    if spec.source == 'query':
        return f'Traces matching “{spec.query.strip()}” from the last {days}, up to {spec.limit}{filters}'
    if spec.source == 'finder':
        return f'Finder export {spec.source_name or Path(spec.finder_export).name}'
    if spec.source == 'snapshot':
        return f'Local trace file {spec.source_name or Path(spec.snapshot_path).name}'
    return f'Recent traces from the last {days}, up to {spec.limit}{filters}'


def render_plan(spec: InsightsLaunchSpec, stages: list[tuple[str, str]]) -> str:
    """Summarize what the run will do and list its expected stages, as the worker will report them."""
    questions = len(spec.labels) + len(spec.custom_labels) + len(spec.coding_labels)
    groupings = len(spec.dimensions)
    summary = (
        f'{_source_summary(spec)}. {questions} question{"" if questions == 1 else "s"}, '
        f'{groupings} grouping{"" if groupings == 1 else "s"}.'
    )
    items = ''.join(f'<li>{esc(title)}</li>' for _, title in stages)
    return f'<p class="irf-summary">{esc(summary)}</p><h4>Expected stages</h4><ol class="irf-stages">{items}</ol>'
