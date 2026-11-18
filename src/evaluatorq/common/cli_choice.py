"""A ``click.Choice`` that also accepts unlisted old spellings of its choices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class AliasedChoice(click.Choice[str]):
    """Offer only ``choices`` in help and errors, and read each key of ``aliases`` as its value.

    ``on_alias`` runs with the old spelling before it is replaced, so a caller can warn about it.
    """

    def __init__(
        self,
        choices: Sequence[str],
        aliases: Mapping[str, str],
        *,
        on_alias: Callable[[str], object] | None = None,
    ) -> None:
        super().__init__(list(choices))
        self.aliases = dict(aliases)
        self.on_alias = on_alias

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> Any:
        if isinstance(value, str) and value in self.aliases:
            if self.on_alias is not None:
                self.on_alias(value)
            value = self.aliases[value]
        return super().convert(value, param, ctx)
