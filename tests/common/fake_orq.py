"""A fake Orq client whose `pii.redact_async` redacts a few known credential shapes, with no network.

It numbers placeholders the way the real endpoint does (`<API_KEY_1>`, `<UUID_1>`, `<EMAIL_ADDRESS_1>`; the same value
keeps its number inside one request) and records every request text, so a test can assert what left the process.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from orq_ai_sdk import Orq

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ('API_KEY', re.compile(r'sk-[A-Za-z0-9_-]{8,}|AKIA[A-Z0-9]{16}|ghp_[A-Za-z0-9]{20,}')),
    ('UUID', re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b')),
    ('EMAIL_ADDRESS', re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b')),
)


def redact(text: str) -> str:
    """Replace each credential, UUID and email in `text` with a numbered placeholder."""
    for label, pattern in _RULES:
        numbers: dict[str, int] = {}

        def swap(match: re.Match[str], label: str = label, numbers: dict[str, int] = numbers) -> str:
            return f'<{label}_{numbers.setdefault(match.group(0), len(numbers) + 1)}>'

        text = pattern.sub(swap, text)
    return text


class FakeOrq:
    """Stands in for `orq_ai_sdk.Orq`; `orq.pii.redact_async(text=...)` returns the redacted text.

    Attributes:
        requests: The `text` of every request, in order.
        error: Raised by every request when set, or only by those `error_when` accepts.
        reply: Returned as `redacted_text` instead of the real redaction when set, for a malformed reply.
    """

    def __init__(
        self, *, error: Exception | None = None, error_when: Callable[[str], bool] | None = None, reply: Any = None
    ) -> None:
        self.requests: list[str] = []
        self.error = error
        self.error_when = error_when
        self.reply = reply
        self.pii = SimpleNamespace(redact_async=self._redact_async)

    @property
    def client(self) -> Orq:
        """This fake typed as the SDK client, for the `orq=` parameters."""
        return cast('Orq', self)

    async def _redact_async(self, *, text: str, **_: Any) -> Any:
        self.requests.append(text)
        if self.error is not None and (self.error_when is None or self.error_when(text)):
            raise self.error
        return SimpleNamespace(redacted_text=redact(text) if self.reply is None else self.reply, mappings={})
