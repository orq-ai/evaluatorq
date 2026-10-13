"""Opt-in layout and de-noising options for the readable conversation text (`full_conversation_view`).

Insights summaries and the trace-finder classifier read the legacy layout, so every option here is off by default and
`ReadableStyle()` renders exactly that layout. The options exist so a person can compare formats in the compression
inspector. The layout options (`blocks`, `pair_results`, `plain_tool_input`, `labelled_cuts`) change only how text is
arranged; the reduction options (`strip_terminal_noise`, `shorten_cwd`, `collapse_repeats`, `collapse_blobs`) leave
information out of tool inputs and results and are named as such in the coverage header.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from evaluatorq.common.redact import scrub_known_secrets

if TYPE_CHECKING:
    from collections.abc import Iterable

# A run of this many identical lines is shown once with a count.
MIN_REPEATED_LINES = 3
# A result shorter than this is repeated instead of replaced by a reference that would be longer than the text.
MIN_REFERENCED_RESULT_CHARS = 40
BASE64_RUN_CHARS = 200
HEX_RUN_CHARS = 64
# A run of one repeated character ("====", "0000") is not an encoded blob.
MIN_BLOB_DISTINCT_CHARS = 8

_ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]')
_BASE64_RUN = re.compile(rf'[A-Za-z0-9+/=]{{{BASE64_RUN_CHARS},}}')
_HEX_RUN = re.compile(rf'[0-9a-fA-F]{{{HEX_RUN_CHARS},}}')
_CWD_KEYS = ('cwd', 'workdir', 'working_directory')


@dataclass(frozen=True)
class ReadableStyle:
    """Which opt-in layouts and reductions `full_conversation_view` applies; the default is the legacy layout.

    `blocks` writes one header line per message (`[14] assistant`) and indents every body line two spaces, so a line
    inside a result can never read as a new turn; tool calls and results are labelled `c1`, `c2`, ... in call order.
    `pair_results` puts each result right after its call when `blocks` is enabled; otherwise pairing is disabled.
    `plain_tool_input` writes a JSON object argument as one `key: value` line per key. `labelled_cuts` names the
    message kind and exact omitted count while keeping the marker inside the part's character cap.

    The reductions run on scrubbed tool inputs and results before any cut: `strip_terminal_noise` removes ANSI
    sequences, keeps the last state of a carriage-return redraw and counts runs of three or more identical lines;
    `shorten_cwd` shows an explicitly recorded, unambiguous working directory as `./`; `collapse_repeats` folds
    consecutive identical calls into one `xN` call without dropping any results and references identical outputs;
    `collapse_blobs` replaces base64-looking and long hex runs with `[blob, N chars]`.
    """

    blocks: bool = False
    pair_results: bool = False
    plain_tool_input: bool = False
    labelled_cuts: bool = False
    strip_terminal_noise: bool = False
    shorten_cwd: bool = False
    collapse_repeats: bool = False
    collapse_blobs: bool = False

    def format_notes(self) -> list[str]:
        """The enabled layout options, worded for the coverage header."""
        return [
            note
            for note, enabled in (
                ('a header line per message with an indented body', self.blocks),
                ('results paired with calls', self.pair_results and self.blocks),
                ('tool inputs as key: value lines', self.plain_tool_input),
                ('cut markers name what was cut', self.labelled_cuts),
            )
            if enabled
        ]

    def reduction_notes(self) -> list[str]:
        """The enabled lossy options, worded for the coverage header."""
        return [
            note
            for note, enabled in (
                ('ANSI/progress noise', self.strip_terminal_noise),
                ('recorded working directory shown as ./ when unambiguous', self.shorten_cwd),
                ('repeated calls and results', self.collapse_repeats),
                ('encoded blobs', self.collapse_blobs),
            )
            if enabled
        ]


@dataclass(frozen=True)
class TextReducer:
    """The reductions applied to one tool input or result, split around secret scrubbing.

    Noise normalization runs between two secret-scrubbing passes, so neither a discarded redraw nor a credential
    whose characters straddle escape sequences can expose secret fragments in later reductions.
    """

    noise: bool = False
    blobs: bool = False
    cwd: str | None = None

    @classmethod
    def for_style(cls, style: ReadableStyle, cwd: str | None) -> TextReducer | None:
        reducer = cls(
            noise=style.strip_terminal_noise, blobs=style.collapse_blobs, cwd=cwd if style.shorten_cwd else None
        )
        return reducer if reducer.noise or reducer.blobs or reducer.cwd else None

    def before_scrub(self, text: str) -> str:
        return strip_ansi_and_redraws(text) if self.noise else text

    def after_scrub(self, text: str) -> str:
        if self.noise:
            text = collapse_repeated_lines(text)
        if self.blobs:
            text = collapse_blobs(text)
        if self.cwd:
            text = shorten_cwd(text, self.cwd)
        return text


def strip_ansi_and_redraws(text: str) -> str:
    """Remove ANSI escape sequences and keep only the last state of every line a carriage return redrew."""
    text = _ANSI.sub('', text).replace('\r\n', '\n')
    if '\r' not in text:
        return text
    return '\n'.join(
        next((part for part in reversed(line.split('\r')) if part), '') if '\r' in line else line
        for line in text.split('\n')
    )


def collapse_repeated_lines(text: str) -> str:
    """Show a run of identical non-blank lines once, followed by `(repeated N times)` with N the run length."""
    lines = text.split('\n')
    kept: list[str] = []
    position = 0
    while position < len(lines):
        end = position + 1
        while end < len(lines) and lines[end] == lines[position]:
            end += 1
        kept.append(lines[position])
        if lines[position].strip() and end - position >= MIN_REPEATED_LINES:
            kept.append(f'(repeated {end - position} times)')
        else:
            kept.extend(lines[position + 1 : end])
        position = end
    return '\n'.join(kept)


def collapse_blobs(text: str) -> str:
    """Replace base64-looking runs of 200 or more characters and hex runs of 64 or more with `[blob, N chars]`."""

    def replace(match: re.Match[str]) -> str:
        run = match[0]
        return f'[blob, {len(run)} chars]' if len(set(run)) >= MIN_BLOB_DISTINCT_CHARS else run

    return _HEX_RUN.sub(replace, _BASE64_RUN.sub(replace, text))


def detect_working_directory(arguments: Iterable[str], *, recorded: Iterable[str] = ()) -> str | None:
    """Return one explicit absolute cwd from capture metadata or JSON argument fields, never a guessed path prefix.

    Conflicting or non-absolute directory evidence disables abbreviation for the whole view.
    """
    evidence = list(recorded)
    for raw in arguments:
        try:
            parsed = json.loads(scrub_known_secrets(raw))
        except ValueError:
            continue
        if isinstance(parsed, dict):
            evidence.extend(parsed[key] for key in _CWD_KEYS if isinstance(parsed.get(key), str))
    directories: set[str] = set()
    for value in evidence:
        value = scrub_known_secrets(value)
        path = PurePosixPath(value)
        if (
            not value.startswith('/')
            or value.startswith('//')
            or str(path) == '/'
            or '..' in path.parts
            or any(character in value for character in '\r\n\t<>')
        ):
            return None
        directories.add(str(path))
    return next(iter(directories)) if len(directories) == 1 else None


def shorten_cwd(text: str, cwd: str) -> str:
    """Show `cwd/x` as `./x` and a bare `cwd` as `.`, only where `cwd` is a whole path prefix."""
    pattern = re.compile(rf'(?<![\w.~/@+-]){re.escape(cwd)}(/|(?![\w.@+-]))')
    return pattern.sub(lambda match: './' if match[1] else '.', text)


def plain_arguments(raw: str) -> str:
    """One `key: value` line per top-level key of a JSON object; anything else is returned as it came."""
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(parsed, dict):
        return raw
    return '\n'.join(
        f'{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))}'
        for key, value in parsed.items()
    )
