"""Wrap every frame of an agg GIF in a terminal window and write PNGs for gifski.

Adds a title bar with the three window dots and rounds the corners, leaving them
transparent so the GIF sits cleanly on light and dark pages. agg emits one frame
per change with its own duration; gifski wants a fixed frame rate, so each frame
is repeated for its duration (gifski merges the identical copies again).

Usage: python recording/frame.py demo-raw.gif frames/ 20 path/to/JetBrainsMono-Regular.ttf
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence

TITLE = "vulnerable_support_agent — uv run python run.py"
BAR_HEIGHT = 34
RADIUS = 10
BAR_COLOUR = (33, 34, 44)
TITLE_COLOUR = (130, 140, 175)
DOT_COLOURS = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]


def main(src: str, outdir: str, fps: int, font_path: str) -> None:
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    gif = Image.open(src)
    width, height = gif.size
    size = (width, height + BAR_HEIGHT)

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], RADIUS, fill=255)
    bar = Image.new("RGB", (width, BAR_HEIGHT), BAR_COLOUR)
    draw = ImageDraw.Draw(bar)
    for i, colour in enumerate(DOT_COLOURS):
        cx, cy = 20 + i * 20, BAR_HEIGHT // 2
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=colour)
    font = ImageFont.truetype(font_path, 13)
    draw.text(
        ((width - draw.textlength(TITLE, font=font)) / 2, BAR_HEIGHT // 2),
        TITLE,
        font=font,
        fill=TITLE_COLOUR,
        anchor="lm",
    )

    count = 0
    for frame in ImageSequence.Iterator(gif):
        window = Image.new("RGB", size)
        window.paste(bar, (0, 0))
        window.paste(frame.convert("RGB"), (0, BAR_HEIGHT))
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        canvas.paste(window, (0, 0), mask)
        first = out / f"{count:05d}.png"
        canvas.save(first)
        count += 1
        for _ in range(max(1, round(frame.info.get("duration", 100) / 1000 * fps)) - 1):
            shutil.copy(first, out / f"{count:05d}.png")
            count += 1
    print(f"{count} frames at {fps} fps = {count / fps:.1f}s, {size[0]}x{size[1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
