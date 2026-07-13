"""Editorial v1 theme tokens for the dashboard chrome.

Ported verbatim from the research-team v1 design (the "Orq.ai Design System"
results-app, ``ui_kits/results-app/index.html`` ``:root`` override): the
*editorial skin* — sand canvas, serif headings, flat surfaces, and a teal-led
sanctioned palette.

Scope: these tokens drive the dashboard **chrome** only — the sidebar shell, the
combined landing, and the run lists (``styles.DASHBOARD_CSS``).  The embedded
report bodies keep their own ``common/reports`` tokens (they are shared with the
standalone HTML exports), so this block deliberately uses the design's own token
names (``--surface-app``, ``--text-strong``, ``--chart-1`` …) which do not
collide with the report palette.

Injected by ``shell.page()`` after ``load_css()`` so it resolves for chrome
rules.  To re-skin, edit the values here — nothing downstream changes.

    ORQ_TEAL        #025558  primary / chart bars / fills
    ORQ_TEAL_LIGHT  #299D8F  secondary / positive (resistant, pass)
    ORQ_ORANGE      #ff8f34  accent / highlight / warning
    ORQ_ORANGE_DARK #df5325  danger / critical / vulnerable
    ORQ_BRIGHT_CYAN #28FFE2  3rd chart series
    ORQ_INK         #25232e  text / primary action
    ORQ_SAND        #f9f8f6  backgrounds
"""

from __future__ import annotations

EDITORIAL_CSS = """
/* Orq brand fonts (self-hosted, same woff2 as the docs site). Served from the
   dashboard's /static/fonts/ route. Body = Avio Sans, headings = Kurrent,
   code = Kurrent Mono. */
@font-face { font-family: "Avio Sans"; src: url("/static/fonts/AvioSans-Regular.woff2") format("woff2"); font-weight: 400; font-style: normal; font-display: swap; }
@font-face { font-family: "Avio Sans"; src: url("/static/fonts/AvioSans-RegularItalic.woff2") format("woff2"); font-weight: 400; font-style: italic; font-display: swap; }
@font-face { font-family: "Avio Sans"; src: url("/static/fonts/AvioSans-Medium.woff2") format("woff2"); font-weight: 500; font-style: normal; font-display: swap; }
@font-face { font-family: "Avio Sans"; src: url("/static/fonts/AvioSans-SemiBold.woff2") format("woff2"); font-weight: 600; font-style: normal; font-display: swap; }
@font-face { font-family: "Avio Sans"; src: url("/static/fonts/AvioSans-Bold.woff2") format("woff2"); font-weight: 700; font-style: normal; font-display: swap; }
@font-face { font-family: "Kurrent"; src: url("/static/fonts/ESKlarheitKurrent-Md.woff2") format("woff2"); font-weight: 500; font-style: normal; font-display: swap; }
@font-face { font-family: "Kurrent"; src: url("/static/fonts/ESKlarheitKurrent-Smbd.woff2") format("woff2"); font-weight: 600; font-style: normal; font-display: swap; }
@font-face { font-family: "Kurrent Mono"; src: url("/static/fonts/ESKlarheitKurrentMono-Md.woff2") format("woff2"); font-weight: 500; font-style: normal; font-display: swap; }

:root {
    /* Surfaces — sand paper */
    --surface-app:    #f9f8f6;
    --surface-canvas: #f9f8f6;
    --surface-card:   #ffffff;
    --surface-sunken: #f1efec;
    --app-gray-50:    #f3f1ed;   /* row hover / table header */
    --app-gray-100:   #efece7;   /* sidebar */

    /* Text — ink + desaturated ink grays */
    --text-strong: #25232e;
    --text-body:   #4b4955;
    --text-muted:  #6f6d78;
    --text-faint:  #6b6974;
    /* Darkest ink, for on-color text (e.g. heatmap cells on a light fill) */
    --ink-900: #25232e;

    /* Borders — warm neutral hairlines */
    --border-subtle:  #e9e7e2;
    --border-default: #dad8d2;
    --border-strong:  #bdbbb5;

    /* Accent — ORQ_ORANGE */
    --orange-500: #ff8f34;
    --orange-600: #df5325;
    --orange-100: #ffe8d2;
    --orange-50:  #fff5ea;
    --accent:       #ff8f34;
    --accent-hover: #df5325;
    --text-link:    var(--red-700);
    --ring: 0 0 0 3px rgba(255,143,52,0.32);

    /* Primary action — ink */
    --action-bg:       #25232e;
    --action-bg-hover: #3a3845;

    /* Status — danger=orange-dark, warn=orange, pass=jade */
    --red-600:   #df5325;
    --red-700:    #b8341c;   /* AA-contrast red for small text on light bg */
    --orange-700: #c2410c;   /* AA-contrast orange for small text on light bg */
    --red-100:   #fbe1d6;
    --red-50:    #fdeee9;
    --amber-600: #ff8f34;
    --amber-100: #ffe8d2;
    --green-600: #299D8F;
    --green-100: #d6ece8;
    --green-50:  #eef6f4;

    /* Semantic red-team scale — one source of truth for severity + outcome
       colours. Repoint these to restyle every severity pill, outcome badge and
       the severity-definitions table at once. */
    --sev-critical: var(--red-600);
    --sev-high:     var(--amber-600);
    --sev-medium:   var(--text-muted);
    --sev-low:      var(--green-600);
    --outcome-vulnerable: var(--red-600);
    --outcome-resistant:  var(--green-600);
    --outcome-error:      var(--amber-600);

    /* Brand */
    --teal-600: #025558;
    --teal-100: #d4e7e6;
    --teal-50:  #eef5f5;

    /* Data-viz — teal leads, jade seconds, bright-cyan 3rd */
    --chart-1: #025558;
    --chart-2: #299D8F;
    --chart-3: #28FFE2;
    --chart-4: #ff8f34;
    --chart-5: #df5325;
    --chart-6: #25232e;
    --chart-grid:  #e8e6e1;
    --chart-track: #ece9e4;
    --chart-axis:  #9d9ba4;

    /* Type — Orq brand fonts (self-hosted above), system fallbacks. */
    --font-display: 'Kurrent', Georgia, 'Times New Roman', ui-serif, serif;
    --font-sans:    'Avio Sans', var(--sans, system-ui, -apple-system, sans-serif);
    --font-mono:    'Kurrent Mono', var(--mono, ui-monospace, 'SF Mono', Menlo, monospace);
    /* Line-height — body copy (matches the mockup's 16px/24px = 1.5 rhythm,
       overriding the report stylesheet's looser 1.65 default). */
    --leading-body: 1.5;

    /* Corners — flat surfaces (no card shadows); soft lift for overlays only */
    --radius-md: 8px;
    --radius-lg: 12px;
    --shadow-xs: none;
    --shadow-sm: none;
    --shadow-lg: 0 8px 28px rgba(37,35,46,0.12);
}
"""
