"""Small display helpers shared by the compare renderers."""

from __future__ import annotations

from evaluatorq.dashboard.report_kit import pill
from evaluatorq.dashboard.sim_compare.model import FLAT_DELTA


def delta_pill(delta: float | None, text: str, *, higher_better: bool | None = True) -> str:
    """Signed change pill. Colour follows good/bad (green/red), not direction;
    ``higher_better=None`` keeps it neutral (e.g. conversation count, turns)."""
    if delta is None:
        return pill('n/a', tone='neutral')
    if abs(delta) < FLAT_DELTA:
        return pill('no change', tone='neutral')
    arrow = '\u25b2' if delta > 0 else '\u25bc'
    tone = 'neutral' if higher_better is None else 'good' if (delta > 0) == higher_better else 'danger'
    return pill(f'{arrow} {text}', tone=tone)


def run_key(which: str) -> str:
    """The A/B identity marker: a grey (A, baseline) or ink (B) square + letter."""
    return f'<span class="cmp-key cmp-key--{which.lower()}" aria-hidden="true"></span><b class="cmp-key-letter">{which}</b>'


_STATUS_PILL = {
    'met': ('Goal met', 'good'),
    'missed': ('Goal missed', 'danger'),
    'error': ('Error', 'warn'),
}


def status_badge(status: str) -> str:
    text, tone = _STATUS_PILL[status]
    return pill(text, tone=tone)


_CHANGE_PILL = {
    'regressed': ('Regressed', 'danger'),
    'improved': ('Improved', 'good'),
    'error': ('Error', 'warn'),
    'same': ('Same', 'neutral'),
}


def change_pill(change: str) -> str:
    text, tone = _CHANGE_PILL[change]
    return pill(text, tone=tone, dot=True)


def delta_tone(delta: float) -> str:
    """good / bad / flat for a higher-is-better delta."""
    return 'flat' if abs(delta) < FLAT_DELTA else 'good' if delta > 0 else 'bad'


def fmt_num(v: float | None, digits: int = 2) -> str:
    return 'n/a' if v is None else f'{v:.{digits}f}'
