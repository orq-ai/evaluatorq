"""Tell a generator's real answer from the stub some models return under a JSON-schema response format.

Asked for a list through a JSON schema, Claude Sonnet 5.5 and Opus 5.5 often answer with one schema-valid item whose
text is empty, ``x`` or ``placeholder``. It parses, so the list response models check for it and raise, and
``generate_structured(tool_fallback_on_invalid=True)`` then asks again through a tool call.
"""

from __future__ import annotations

from typing import TypeVar, cast

from pydantic import BaseModel

ListResponse = TypeVar('ListResponse', bound=BaseModel)

MIN_TEXT_CHARS = 3


def is_stub_text(*values: str) -> bool:
    """Whether any of these required texts is too short to be real: under `MIN_TEXT_CHARS` characters."""
    return any(len(value.strip()) < MIN_TEXT_CHARS for value in values)


def too_few(returned: int, expected: int) -> bool:
    """Whether a list is too short to be an honest answer: some items, but fewer than half of those asked for.

    One short of the request is a model miscounting and is kept. One item for five is the stub. An empty list is not
    the stub and is left to the caller, which already warns about it. ``expected`` of 0 means the caller did not say.
    """
    return 0 < returned < (expected + 1) // 2


def expecting(response_model: type[ListResponse], count: int) -> type[ListResponse]:
    """Return ``response_model`` with its ``expected`` class variable set to ``count``.

    The subclass keeps the name and the docstring, so the schema sent to the provider is unchanged.
    """
    namespace = {'expected': count, '__doc__': response_model.__doc__}
    return cast('type[ListResponse]', type(response_model.__name__, (response_model,), namespace))
