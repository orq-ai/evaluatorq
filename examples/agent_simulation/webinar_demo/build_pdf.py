#!/usr/bin/env python3
"""Render webinar-deck.html -> webinar-deck.pdf, one slide per page, styling preserved.

Injects a print-only stylesheet (page = 1280x720, page-break per .slide,
exact colours) then drives headless Chrome's --print-to-pdf. The #s4b block
works around a Chrome print bug where `aspect-ratio` inside a grid balloons the
card's green frame to fill the page.

Usage: python build_pdf.py [deck.html] [out.pdf]
Chrome path override: CHROME=/path/to/chrome python build_pdf.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PRINT_CSS = """
<style>
@media print {
  @page { size: 1280px 720px; margin: 0; }
  html, body { scroll-snap-type: none !important; overflow: visible !important; height: auto !important; }
  * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
  .slide { page-break-after: always; break-after: page; height: 720px !important; width: 1280px !important; overflow: hidden; }
  .slide:last-of-type { page-break-after: auto; break-after: auto; }
  #counter { display: none !important; }
  /* Chrome print path miscomputes aspect-ratio inside grid -> pin sizes, keep layout */
  #s4b .hero { grid-template-columns: 300px 1fr !important; }
  #s4b .face { aspect-ratio: auto !important; width: 300px !important; height: 300px !important; }
  #s4b .card { aspect-ratio: auto !important; height: 149px !important; }
  #s4b .card .c-chip { aspect-ratio: auto !important; }
}
</style>
"""

CHROME_CANDIDATES = [
    os.environ.get("CHROME"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome"),
    shutil.which("chromium"),
    shutil.which("chrome"),
]


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and Path(c).exists():
            return c
    sys.exit("Chrome not found. Set CHROME=/path/to/chrome")


def main() -> None:
    here = Path(__file__).parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "webinar-deck.html"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "webinar-deck.pdf"

    html = src.read_text(encoding="utf-8").replace("</head>", PRINT_CSS + "</head>", 1)
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=src.parent, delete=False, encoding="utf-8") as f:
        tmp = Path(f.name)
        f.write(html)
    try:
        subprocess.run(
            [
                find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                "--run-all-compositor-stages-before-draw", "--virtual-time-budget=8000",
                f"--print-to-pdf={out}", tmp.as_uri(),
            ],
            check=True,
        )
    finally:
        tmp.unlink(missing_ok=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
