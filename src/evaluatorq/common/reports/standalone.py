"""Offline documents using the same report composition as dashboard tabs.

The dashboard presentation modules imported here are pure Python. Server routes,
FastHTML, and interactive apply flows are never needed to export a report.
"""

from __future__ import annotations

import base64
import re
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING

from evaluatorq.common.reports import esc, load_css, load_logo_svg

if TYPE_CHECKING:
    from evaluatorq.redteam.contracts import RedTeamReport
    from evaluatorq.simulation.contracts import SimulationRun


def expanded_sections(group: str, items: list[tuple[str, str] | tuple[str, str, str]], *, brief: str = '') -> str:
    """Turn the dashboard's ordered tab content into visible anchored sections."""
    links: list[str] = []
    sections: list[str] = []
    number = 0
    for item in items:
        label, body = item[:2]
        if not body.strip():
            continue
        number += 1
        anchor = f'{group}-{label.lower().replace(" ", "-")}'
        appendix = label in {'Config', 'Usage'}
        display_label = f'{label} · Appendix' if appendix else label
        links.append(f'<a href="#{esc(anchor)}">{number}. {esc(display_label)}</a>')
        note = _section_note(group, label)
        content = f'<p class="export-reading-note">{note}</p>{body}'
        if appendix:
            content = (
                f'<details class="export-appendix-detail"><summary>Show {esc(label.lower())} details</summary>'
                f'{content}</details>'
            )
        back_link = '<p class="export-back-brief"><a href="#export-brief">Back to brief</a></p>' if brief else ''
        sections.append(
            f'<section class="export-section" id="{esc(anchor)}" aria-labelledby="{esc(anchor)}-title">'
            f'<h2 class="export-section-title" id="{esc(anchor)}-title">{number}. {esc(display_label)}</h2>{content}'
            f'{back_link}</section>'
        )
    finding_count = len(re.findall(r'id="export-finding-\d+"', brief))
    brief_links = ''.join(f'<a href="#export-finding-{i}">Finding {i}</a>' for i in range(1, finding_count + 1))
    brief_link = '<a href="#export-brief">Brief</a>' if brief else ''
    nav = '<nav class="export-nav" aria-label="Report sections">' + brief_link + brief_links
    return brief + nav + ''.join(links) + '</nav>' + ''.join(sections)


def _section_note(group: str, label: str) -> str:
    notes = {
        ('simtab', 'Overview'): 'Outcome shares use all conversations, including execution errors.',
        ('simtab', 'Breakdown'): 'Heatmap average scores are higher when better; binary pass rates are separate.',
        (
            'simtab',
            'Turn quality',
        ): 'Higher is better for response, tone, and factual accuracy; lower is better for hallucination risk.',
        ('rttab', 'Overview'): 'Attack outcomes use the saved run totals; attacks without verdicts remain unknown.',
        ('rttab', 'Breakdowns'): 'Attack success rate divides vulnerabilities by evaluated attacks; lower is better.',
        (
            'simtab',
            'Recommendations',
        ): 'Check each saved suggestion against the conversation evidence that supports it.',
        ('simtab', 'Transcripts'): 'Open a conversation to inspect the recorded turns and its outcome.',
        ('simtab', 'Config'): 'Configuration describes the saved run and its full configured population.',
        ('simtab', 'Usage'): 'Usage totals cover the provider data recorded with this report.',
        ('rttab', 'Agents'): 'Agent counts and outcomes reflect the saved report population.',
        ('rttab', 'Focus areas'): 'Read each focus area with its confirmed and evaluated attack counts.',
        ('rttab', 'Attacks'): 'Open an attack to inspect its verdict and full recorded evidence.',
        ('rttab', 'Usage'): 'Usage totals cover the provider data recorded with this report.',
        ('rttab', 'Config'): 'Configuration describes the saved run and its tested agents.',
    }
    return notes.get((group, label), 'Read the saved counts and evidence together.')


class _ExpandDetails(HTMLParser):
    """Expand report disclosures while keeping individual evidence folded."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        raw = self.get_starttag_text() or ''
        classes = (dict(attrs).get('class') or '').split()
        if (
            tag == 'details'
            and 'export-evidence' not in classes
            and 'export-appendix-detail' not in classes
            and not any(key == 'open' for key, _ in attrs)
        ):
            raw = raw[:-1] + ' open>'
        self.parts.append(raw)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.parts.append(self.get_starttag_text() or '')

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(f'</{tag}>')

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f'&{name};')

    def handle_charref(self, name: str) -> None:
        self.parts.append(f'&#{name};')

    def handle_comment(self, data: str) -> None:
        self.parts.append(f'<!--{data}-->')


@lru_cache(maxsize=1)
def _styles() -> str:
    from evaluatorq.dashboard.styles import REPORT_CSS
    from evaluatorq.dashboard.theme import EDITORIAL_CSS

    font_root = Path(__file__).parents[2] / 'dashboard' / 'static' / 'fonts'

    def embed_font(match: re.Match[str]) -> str:
        encoded = base64.b64encode((font_root / match[1]).read_bytes()).decode('ascii')
        return f'url("data:font/woff2;base64,{encoded}")'

    theme = re.sub(r'url\("/static/fonts/([^"/]+)"\)', embed_font, EDITORIAL_CSS)
    overrides = Path(__file__).with_name('standalone.css').read_text(encoding='utf-8')
    base = re.sub(r'@import\s+[^;]+;', '', load_css())
    return f'{base}\n{theme}\n{REPORT_CSS}\n{overrides}'


def _document(title: str, body: str) -> str:
    expanded = _ExpandDetails()
    expanded.feed(body)
    expanded.close()
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{esc(title)}</title><style>{_styles()}</style></head>'
        '<body class="standalone-report"><div class="export-brand">'
        f'{load_logo_svg()}<span>evaluatorq · {esc(title)}</span></div>'
        '<p class="export-print-note">Collapsed conversations, attacks, and appendix details are omitted from print. '
        'Open the sections you want to include in this report before printing.</p>'
        f'<main>{"".join(expanded.parts)}</main>'
        '<footer class="export-footer">Generated by evaluatorq. This report renders the saved results; '
        'external trace and agent links require access to Orq.</footer></body></html>'
    )


def simulation_html(run: SimulationRun) -> str:
    """Render a saved simulation run with all dashboard tabs expanded."""
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    return _document('Agent Simulation Report', sim_report_tabs('', run, standalone=True))


def redteam_html(report: RedTeamReport) -> str:
    """Render a red-team report with all dashboard tabs expanded."""
    from evaluatorq.dashboard.report_tabs import redteam_report_tabs

    return _document('Red Team Security Report', redteam_report_tabs('', report, standalone=True))
