"""Generate the orq.ai-branded social/OG preview card (docs/assets/og-image.png).

One hand-designed 1200x630 link-unfurl card (Slack / Twitter / OpenGraph) that mirrors
the orq.ai brand system: a warm off-white ground with the signature soft teal/cyan aurora
glow, the monochrome Orq mark + "orq.ai" lockup, near-black display type, capability pills,
and a `pip install` line. Uses the real self-hosted brand fonts (Kurrent display, Avio Sans
body, Kurrent Mono code) — the same palette/type as the orq.ai site and report.css.

We render a static image (not the mkdocs `social` plugin) because that plugin can only
source faces from Google Fonts, so it can't use the Orq brand fonts. This runs locally only;
the PNG is committed, so CI needs nothing extra.

Run:  uv run --with "fonttools[woff]" --with cairosvg --with pillow python scripts/gen_og_card.py
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "docs" / "stylesheets" / "fonts"
MARK_WHITE = ROOT / "docs" / "assets" / "orq-mark-white.svg"
OUT = ROOT / "docs" / "assets" / "og-image.png"

W, H = 1200, 630

# orq.ai palette
GROUND = (247, 246, 243)   # warm off-white
INK = (23, 23, 22)         # near-black display
BODY = (92, 90, 86)        # muted body gray
MUTE = (150, 147, 140)     # captions / prompt
TEAL = (114, 239, 227)
CYAN = (150, 214, 226)
BORDER = (222, 219, 212)

TITLE = "evaluatorq"
TAGLINE = "Run LLM evaluations, red-team agents, and simulate multi-turn conversations in Python."
PILLS = ["LLM Evaluation", "Red-Teaming", "Agent Simulation"]
INSTALL = "pip install evaluatorq"
LEFT = 96


def load_font(woff2: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a self-hosted .woff2 brand face as a Pillow font (woff2 -> ttf in memory)."""
    f = TTFont(FONTS / woff2)
    f.flavor = None
    buf = io.BytesIO()
    f.save(buf)
    buf.seek(0)
    return ImageFont.truetype(buf, size)


def svg_png(path: Path, width: int, recolor: tuple[int, int, int] | None = None) -> Image.Image:
    """Render an SVG to an RGBA image, optionally flat-recolored (keeps original alpha)."""
    png = cairosvg.svg2png(url=str(path), output_width=width)
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    if recolor is not None:
        solid = Image.new("RGBA", img.size, (*recolor, 0))
        solid.putalpha(img.split()[3])
        img = solid
    return img


def aurora() -> Image.Image:
    """The signature orq.ai teal/cyan glow, concentrated upper-centre and softly blurred."""
    card = Image.new("RGBA", (W, H), (*GROUND, 255))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([W // 2 - 760, -440, W // 2 + 760, 190], fill=(*TEAL, 70))
    d.ellipse([W // 2 + 40, -360, W // 2 + 900, 150], fill=(*CYAN, 55))
    card.alpha_composite(layer.filter(ImageFilter.GaussianBlur(150)))
    return card


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def pill(card: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font: ImageFont.FreeTypeFont) -> int:
    """A white outline capability pill with a teal dot; returns its width."""
    h, pad, gap, r = 56, 28, 16, 7
    tw = draw.textlength(text, font=font)
    w = int(pad * 2 + gap + r * 2 + tw)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=(255, 255, 255, 240), outline=BORDER, width=2)
    cy = y + h // 2
    draw.ellipse([x + pad, cy - r, x + pad + r * 2, cy + r], fill=TEAL, outline=(90, 200, 190), width=1)
    draw.text((x + pad + r * 2 + gap, cy), text, font=font, fill=INK, anchor="lm")
    return w


def main() -> None:
    card = aurora()
    draw = ImageDraw.Draw(card)

    # Lockup: monochrome Orq mark + "orq.ai".
    mark = svg_png(MARK_WHITE, 40, recolor=INK)
    card.alpha_composite(mark, (LEFT, 66))
    draw.text((LEFT + 56, 86), "orq.ai", font=load_font("ESKlarheitKurrent-Smbd.woff2", 30), fill=INK, anchor="lm")

    # Title in Kurrent SemiBold, near-black.
    draw.text((LEFT - 6, 176), TITLE, font=load_font("ESKlarheitKurrent-Smbd.woff2", 120), fill=INK)

    # Tagline in Avio Sans.
    tag_font = load_font("AvioSans-Regular.woff2", 34)
    y = 330
    for line in wrap(draw, TAGLINE, tag_font, 900):
        draw.text((LEFT, y), line, font=tag_font, fill=BODY)
        y += 46

    # Capability pills.
    pill_font = load_font("AvioSans-Medium.woff2", 25)
    x = LEFT
    for label in PILLS:
        x += pill(card, draw, x, 452, label, pill_font) + 16

    # Install line in Kurrent Mono, with a muted prompt.
    mono = load_font("ESKlarheitKurrentMono-Md.woff2", 27)
    draw.text((LEFT, H - 74), "$", font=mono, fill=MUTE)
    draw.text((LEFT + 26, H - 74), INSTALL, font=mono, fill=INK)

    card.convert("RGB").save(OUT, "PNG")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
