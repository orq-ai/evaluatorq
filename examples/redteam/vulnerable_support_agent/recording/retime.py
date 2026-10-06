"""Retime a recorded asciicast so the output streams line by line.

Rich prints each table in one write, so a raw replay jumps between a few large
frames. This keeps the recorded output verbatim and only changes its timing: a
typed prompt first, one line every 35 ms, short pauses where the real run spends
its time, and a long hold on the final summary.

Usage: python recording/retime.py raw.cast demo.cast
"""

from __future__ import annotations

import json
import re
import sys

PROMPT = "\x1b[1;32m❯\x1b[0m \x1b[36mvulnerable_support_agent\x1b[0m "
COMMAND = "uv run python run.py"
SGR = re.compile(r"\x1b\[[0-9;]*m")

# (marker in a line, pause in seconds after it)
PAUSES = [
    ("Generating Attack Datapoints", 1.2),
    ("Executing Attacks", 2.2),
    ("Generating Report", 1.2),
    ("RED TEAM REPORT SUMMARY", 0.6),
    ("Run Complete", 0.6),
]


def main(src: str, dst: str) -> None:
    raw = open(src, encoding="utf-8").read().splitlines()
    header = json.loads(raw[0])
    text = "".join(event[2] for event in map(json.loads, raw[1:]) if event[1] == "o")

    events: list[tuple[float, str]] = [(0.4, PROMPT)]
    events += [(0.055, ch) for ch in COMMAND]
    events.append((0.5, "\r\n"))
    for line in text.split("\r\n"):
        plain = SGR.sub("", line)
        events.append((0.035 if plain.strip() else 0.02, line + "\r\n"))
        events += [(pause, "") for marker, pause in PAUSES if marker in plain]
    events += [(0.3, PROMPT), (6.0, "")]

    header.pop("timestamp", None)
    header["command"] = COMMAND
    with open(dst, "w", encoding="utf-8") as out:
        out.write(json.dumps(header) + "\n")
        for delay, data in events:
            out.write(json.dumps([round(delay, 3), "o", data]) + "\n")
    print(f"{len(events)} events, {sum(delay for delay, _ in events):.1f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
