"""Tell a generator's real answer from the stub some models return under a JSON-schema response format.

Asked for a list through a JSON schema, Claude Sonnet 5.5 and Opus 5.5 often answer with one schema-valid item whose
text is empty, ``x`` or ``placeholder``. It parses, so the generators hand `generate_structured` an ``accept`` check
built from these rules; it then asks once more through a tool call.
"""

from __future__ import annotations

MIN_TEXT_CHARS = 3


def is_stub_item(name: str, body: str) -> bool:
    """Whether an item's two required texts say nothing: either is blank, or both are under `MIN_TEXT_CHARS`.

    ``body`` is a persona's background or a scenario's goal. A short name beside a real body is a real item
    (a persona called "Li", a scenario called "QA"), so one short text alone is not enough.
    """
    name, body = name.strip(), body.strip()
    return not name or not body or (len(name) < MIN_TEXT_CHARS and len(body) < MIN_TEXT_CHARS)


def too_few(returned: int, expected: int) -> bool:
    """Whether a list is too short to be an honest answer: some items, but fewer than half of those asked for.

    One short of the request is a model miscounting and is kept. One item for five is the stub. An empty list is not
    the stub and is left to the caller, which already warns about it. ``expected`` of 0 means the caller did not say.
    """
    return 0 < returned < (expected + 1) // 2
