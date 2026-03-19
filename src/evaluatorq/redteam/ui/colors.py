"""ORQ Brand Color Palette for Red Team Visualizations.

Provides a consistent color palette aligned with ORQ brand guidelines.
Mirrors the palette defined in the research repository
(orq_shared.utils.orq_brand_colors) so dashboards share a unified look.

Usage:
    from evaluatorq.redteam.ui.colors import COLORS, SEVERITY_COLORS, ORQ_SCALE_HEAT
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core brand palette
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    # Primary Brand Colors
    'orange_300': '#ff8f34',  # Pulse Orange (Primary Accent)
    'orange_400': '#df5325',  # Darker Orange
    'teal_400': '#025558',    # Deep Teal (Primary Brand)
    'teal_500': '#01483d',    # Dark Teal
    'ink_700': '#25232e',     # Primary Text
    'ink_800': '#1a1921',     # Dark Background
    'sand_100': '#f9f8f6',    # Light Backgrounds
    'sand_400': '#e4e2df',    # Border/Dividers
    # Accent Colors
    'turquoise_400': '#00ffdd',  # Glowing Turquoise
    'blue_400': '#4fd2ff',       # Cyber Blue
    'success_400': '#2ebd85',    # Success Green
    'yellow_400': '#f2b600',     # Yellow
    'red_400': '#d92d20',        # Red
    'info_400': '#2f80ed',       # Info Blue
    'purple_400': '#7e22ce',     # Purple
}

# ---------------------------------------------------------------------------
# Semantic aliases
# ---------------------------------------------------------------------------

SEVERITY_COLORS: dict[str, str] = {
    'critical': COLORS['red_400'],
    'high': COLORS['orange_300'],
    'medium': COLORS['yellow_400'],
    'low': COLORS['success_400'],
}

SEVERITY_ORDER: list[str] = ['critical', 'high', 'medium', 'low']

STATUS_COLORS: dict[str, str] = {
    'vulnerable': COLORS['red_400'],
    'resistant': COLORS['success_400'],
    'error': COLORS['yellow_400'],
}

# ---------------------------------------------------------------------------
# Plotly colorscales (list-of-[position, color] pairs)
# ---------------------------------------------------------------------------

# Sequential heat: sand -> orange -> red  (replaces "Reds")
ORQ_SCALE_HEAT: list[list[float | str]] = [
    [0.0, COLORS['sand_100']],
    [0.5, COLORS['orange_300']],
    [1.0, COLORS['red_400']],
]

# Good-to-bad: green -> yellow -> red
ORQ_SCALE_GOOD_BAD: list[list[float | str]] = [
    [0.0, COLORS['success_400']],
    [0.5, COLORS['yellow_400']],
    [1.0, COLORS['red_400']],
]

# Agent-themed sequential: sand -> orange -> teal
ORQ_SCALE_AGENT: list[list[float | str]] = [
    [0.0, COLORS['sand_100']],
    [0.5, COLORS['orange_300']],
    [1.0, COLORS['teal_400']],
]

# ---------------------------------------------------------------------------
# Qualitative palette for multi-series charts
# ---------------------------------------------------------------------------

QUALITATIVE: list[str] = [
    COLORS['success_400'],
    COLORS['orange_300'],
    COLORS['purple_400'],
    COLORS['turquoise_400'],
    COLORS['teal_400'],
    COLORS['blue_400'],
    COLORS['orange_400'],
    COLORS['info_400'],
]
