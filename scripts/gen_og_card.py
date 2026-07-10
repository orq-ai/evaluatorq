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
ORANGE = (245, 139, 78)    # orq accent
WHITE = (255, 255, 255)
BORDER = (222, 219, 212)

TITLE_HEAD, TITLE_Q = "evaluator", "q"  # the "q" is highlighted (orq tie-in)
TAGLINE = "Run LLM evaluations, red-team agents, and simulate multi-turn conversations in Python."
# What the library contains: (icon, tile colour, glyph colour, label).
FEATURES = [
    ("check", TEAL, INK, "LLM evaluation"),
    ("target", ORANGE, WHITE, "Agent red-teaming"),
    ("chat", TEAL, INK, "Conversation simulation"),
]
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
    """The signature orq.ai glow: teal/cyan upper-centre with a warm orange hint, softly blurred."""
    card = Image.new("RGBA", (W, H), (*GROUND, 255))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([W // 2 - 760, -440, W // 2 + 760, 190], fill=(*TEAL, 66))
    d.ellipse([W // 2 + 120, -360, W // 2 + 940, 150], fill=(*CYAN, 52))
    d.ellipse([W - 520, -320, W + 200, 210], fill=(*ORANGE, 34))  # warm orange hint, top-right
    card.alpha_composite(layer.filter(ImageFilter.GaussianBlur(150)))
    return card


def feature_icon(draw: ImageDraw.ImageDraw, x: int, y: int, kind: str, tile: tuple[int, int, int], glyph: tuple[int, int, int]) -> int:
    """Draw a rounded icon tile with a simple glyph; returns the tile size."""
    s = 48
    draw.rounded_rectangle([x, y, x + s, y + s], radius=13, fill=tile)
    cx, cy = x + s // 2, y + s // 2
    if kind == "check":  # evaluation / scoring
        draw.line([(x + 13, cy), (x + 21, cy + 8), (x + 35, y + 14)], fill=glyph, width=4, joint="curve")
    elif kind == "target":  # red-teaming / attack
        draw.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], outline=glyph, width=3)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=glyph)
    elif kind == "chat":  # multi-turn conversation
        draw.rounded_rectangle([x + 11, y + 12, x + 37, y + 30], radius=6, outline=glyph, width=3)
        draw.polygon([(x + 17, y + 29), (x + 17, y + 37), (x + 25, y + 29)], fill=glyph)
    return s


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


def main() -> None:
    card = aurora()
    draw = ImageDraw.Draw(card)

    # Lockup: monochrome Orq mark + "orq.ai".
    mark = svg_png(MARK_WHITE, 40, recolor=INK)
    card.alpha_composite(mark, (LEFT, 66))
    draw.text((LEFT + 56, 86), "orq.ai", font=load_font("ESKlarheitKurrent-Smbd.woff2", 30), fill=INK, anchor="lm")

    # Title in Kurrent SemiBold — near-black "evaluator" with an orange "q" (the orq tie-in).
    title_font = load_font("ESKlarheitKurrent-Smbd.woff2", 120)
    tx = LEFT - 6
    draw.text((tx, 176), TITLE_HEAD, font=title_font, fill=INK)
    draw.text((tx + draw.textlength(TITLE_HEAD, font=title_font), 176), TITLE_Q, font=title_font, fill=ORANGE)

    # Tagline in Avio Sans.
    tag_font = load_font("AvioSans-Regular.woff2", 34)
    y = 330
    for line in wrap(draw, TAGLINE, tag_font, 900):
        draw.text((LEFT, y), line, font=tag_font, fill=BODY)
        y += 46

    # What's inside: icon-led feature row (reads as capabilities, not tags).
    feat_font = load_font("AvioSans-Medium.woff2", 26)
    x, fy = LEFT, 448
    for kind, tile, glyph, label in FEATURES:
        s = feature_icon(draw, x, fy, kind, tile, glyph)
        lx = x + s + 15
        draw.text((lx, fy + s // 2), label, font=feat_font, fill=INK, anchor="lm")
        x = int(lx + draw.textlength(label, font=feat_font) + 46)

    # Install line in Kurrent Mono, with a muted prompt.
    mono = load_font("ESKlarheitKurrentMono-Md.woff2", 27)
    draw.text((LEFT, H - 72), "$", font=mono, fill=MUTE)
    draw.text((LEFT + 26, H - 72), INSTALL, font=mono, fill=INK)

    card.convert("RGB").save(OUT, "PNG")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
