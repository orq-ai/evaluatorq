"""Render the compression inspector's knobs form: every limit and include switch, with its default."""

from __future__ import annotations

from evaluatorq.dashboard.compression_inspector_data import (
    JEV_EDGE_FIELDS,
    JEV_FLAGS,
    MAX_KNOB_CHARS,
    VIEW_CAP_FIELDS,
    CompressionKnobs,
)
from evaluatorq.dashboard.security import csrf_field
from evaluatorq.insights.transcript import MESSAGE_KINDS

PREVIEW_ROUTE = '/compression-inspector/preview'
_RESET_SCRIPT = (
    'var f=this.form;'
    "f.querySelectorAll('[data-default]').forEach(function(e){"
    "if(e.type==='checkbox'){e.checked=e.dataset.default==='1';}else{e.value=e.dataset.default;}});"
    "f.dispatchEvent(new Event('change',{bubbles:true}));"
)

_JEV_EDGE_LABELS = {
    'assistant': 'Assistant messages',
    'user': 'User messages',
    'system': 'System and developer messages',
    'tool_input': 'Tool call input',
    'tool_result': 'Tool results',
}
_JEV_FLAG_LABELS = {
    'system': 'System',
    'developer': 'Developer',
    'user': 'User',
    'assistant': 'Assistant',
    'tool_calls': 'Tool calls',
    'tool_results': 'Tool results',
}
_VIEW_KIND_LABELS = {
    'system': 'System',
    'user': 'User',
    'assistant': 'Assistant',
    'reasoning': 'Reasoning',
    'tool_input': 'Tool inputs',
    'tool_result': 'Tool results',
}
_VIEW_CAP_LABELS = {
    'view_system_chars': 'System message limit',
    'view_user_chars': 'User message limit',
    'view_assistant_edge': 'Assistant edge kept',
    'view_reasoning_chars': 'Reasoning limit',
    'view_tool_input_chars': 'Tool input limit',
    'view_tool_max_chars': 'Tool result limit',
}
_TOOLS_OPTIONS = (
    ('per_call', 'One line per call'),
    ('collapsed', 'Collapsed, no results'),
)

Values = dict[str, int | str | bool]


def _shown(value: object) -> str:
    return 'whole (-1)' if value == -1 else f'{value:,}' if isinstance(value, int) else str(value)


def _number(name: str, label: str, values: Values, defaults: Values, *, low: int = 1) -> str:
    return (
        '<label class="compression-knob">'
        f'<span>{label}</span>'
        f'<input type="number" name="{name}" value="{values[name]}" min="{low}" max="{MAX_KNOB_CHARS}" '
        f'step="1" required data-default="{defaults[name]}">'
        f'<small>Default {_shown(defaults[name])}</small></label>'
    )


def _check(name: str, label: str, values: Values, defaults: Values) -> str:
    checked = ' checked' if values[name] else ''
    return (
        '<label class="compression-check">'
        f'<input type="checkbox" name="{name}" value="1"{checked} data-default="{int(bool(defaults[name]))}">'
        f'<span>{label}</span></label>'
    )


def _jev_group(values: Values, defaults: Values) -> str:
    limits = ''.join(
        '<div class="compression-knob-pair">'
        f'{_number(total, _JEV_EDGE_LABELS[part] + ", whole up to", values, defaults)}'
        f'{_number(edge, _JEV_EDGE_LABELS[part] + ", edge kept", values, defaults)}'
        '</div>'
        for part, (total, edge) in JEV_EDGE_FIELDS.items()
    )
    flags = ''.join(_check(f'jev_include_{flag}', _JEV_FLAG_LABELS[flag], values, defaults) for flag in JEV_FLAGS)
    return (
        '<fieldset class="compression-knob-group">'
        '<legend>Output (Jev state)</legend>'
        '<div class="compression-knob-grid">'
        f'{_number("global_char_cap", "Global character cap", values, defaults)}'
        f'{_number("jev_state_chars", "Jev state limit", values, defaults)}'
        '</div>'
        f'<div class="compression-knob-grid">{limits}</div>'
        '<p class="compression-knob-title">Include</p>'
        f'<div class="compression-checks">{flags}</div>'
        '<p class="compression-count-note">A text longer than its whole-up-to limit keeps that many characters at '
        'each edge, so an edge may be at most half of it. Turning off tool calls drops their results as well; '
        'turning off tool results keeps the calls without them.</p>'
        '</fieldset>'
    )


def _view_group(values: Values, defaults: Values) -> str:
    kinds = ''.join(
        _check(f'view_include_{kind}', _VIEW_KIND_LABELS[kind], values, defaults) for kind in sorted(MESSAGE_KINDS)
    )
    caps = ''.join(_number(name, _VIEW_CAP_LABELS[name], values, defaults, low=-1) for name in VIEW_CAP_FIELDS)
    options = ''.join(
        f'<option value="{value}"{" selected" if values["view_tools"] == value else ""}>{label}</option>'
        for value, label in _TOOLS_OPTIONS
    )
    return (
        '<fieldset class="compression-knob-group">'
        '<legend>Original (readable view)</legend>'
        '<div class="compression-knob-grid">'
        f'{_number("view_budget", "Whole view limit", values, defaults)}'
        '<label class="compression-knob"><span>Tool display</span>'
        f'<select name="view_tools" data-default="{defaults["view_tools"]}">{options}</select>'
        '<small>Default one line per call</small></label>'
        '</div>'
        '<p class="compression-knob-title">Include</p>'
        f'<div class="compression-checks">{kinds}</div>'
        f'<div class="compression-knob-grid">{caps}</div>'
        '<p class="compression-count-note">Use -1 to keep a part whole. A tool result limit of 0 leaves results out.</p>'
        '</fieldset>'
    )


def knobs_form_html(knobs: CompressionKnobs, defaults: CompressionKnobs, token: str) -> str:
    """Render the live-updating form; its fields post under the names ``CompressionKnobs.from_form`` reads."""
    values, default_values = knobs.as_form(), defaults.as_form()
    return (
        '<form class="compression-knobs" autocomplete="off" novalidate '
        f'hx-post="{PREVIEW_ROUTE}" hx-trigger="change, input delay:400ms" hx-target="#compression-output" '
        'hx-swap="outerHTML" hx-sync="this:replace">'
        f'{csrf_field()}<input type="hidden" name="token" value="{token}">'
        '<div class="compression-knobs-groups">'
        f'{_view_group(values, default_values)}{_jev_group(values, default_values)}'
        '</div>'
        '<div class="compression-knobs-footer">'
        f'<button type="button" class="compression-reset" onclick="{_RESET_SCRIPT}">Reset to defaults</button>'
        '<p class="compression-count-note">The Readable view uses the Insights full_conversation_view options and the '
        'Jev state uses the classifier Jev limits. Scrubbing of known secrets is always on. Changes update the '
        'preview as you make them.</p>'
        '</div></form>'
    )
