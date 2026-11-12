"""Small display helpers shared by the compare renderers."""

from __future__ import annotations

from evaluatorq.common.reports import esc, status_badge
from evaluatorq.dashboard.sim_compare.model import _FLAT


def _delta_pill(delta: float | None, text: str, *, higher_better: bool | None = True) -> str:
    """Signed change pill. Colour follows good/bad (green/red), not direction;
    ``higher_better=None`` keeps it neutral (e.g. conversation count, turns)."""
    if delta is None:
        return '<span class="cmp-delta cmp-delta--flat">n/a</span>'
    if abs(delta) < _FLAT:
        return '<span class="cmp-delta cmp-delta--flat">no change</span>'
    arrow = '&#x25B2;' if delta > 0 else '&#x25BC;'
    tone = 'flat' if higher_better is None else 'good' if (delta > 0) == higher_better else 'bad'
    return f'<span class="cmp-delta cmp-delta--{tone}">{arrow} {esc(text)}</span>'


def _run_key(which: str) -> str:
    """The A/B identity marker: a grey (A, baseline) or ink (B) square + letter."""
    return f'<span class="cmp-key cmp-key--{which.lower()}" aria-hidden="true"></span><b class="cmp-key-letter">{which}</b>'


_STATUS_BADGE = {
    'met': ('Goal met', 'pass'),
    'missed': ('Goal missed', 'fail'),
    'error': ('Error', 'warn'),
}


def _badge(status: str) -> str:
    text, tone = _STATUS_BADGE[status]
    return status_badge(text, tone)


_CHANGE_PILL = {
    'regressed': ('Regressed', 'bad'),
    'improved': ('Improved', 'good'),
    'error': ('Error', 'warn'),
    'same': ('Same', 'flat'),
}


def _change_pill(change: str) -> str:
    text, tone = _CHANGE_PILL[change]
    return f'<span class="cmp-change cmp-change--{tone}">{text}</span>'


def _tone(delta: float) -> str:
    """good / bad / flat for a higher-is-better delta."""
    return 'flat' if abs(delta) < _FLAT else 'good' if delta > 0 else 'bad'


def _fmt_rate(v: float | None) -> str:
    return 'n/a' if v is None else f'{v:.0%}'


def _fmt_num(v: float | None, digits: int = 2) -> str:
    return 'n/a' if v is None else f'{v:.{digits}f}'
