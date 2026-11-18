"""Keep renamed public names and values working for a while, with one kind of warning."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_PACKAGE_DIR = str(Path(__file__).resolve().parent.parent)
# Frames in these packages are skipped when a warning picks the line it points at, so it names the caller's code.
_INTERNAL_PACKAGES = ('pydantic', 'pydantic_core', 'typer', 'click')
_logged: set[str] = set()


def renamed_attributes(module: str, namespace: Mapping[str, Any], renamed: Mapping[str, str]) -> Callable[[str], Any]:
    """Build a module ``__getattr__`` that resolves each old name in ``renamed`` to its new one.

    ``namespace`` is the module's ``globals()``, read at lookup time, so the new names may be defined
    after this call. Any other missing name raises ``AttributeError`` as usual.
    """

    def __getattr__(name: str) -> Any:  # noqa: N807 - installed as a module __getattr__
        new_name = renamed.get(name)
        if new_name is None:
            raise AttributeError(f'module {module!r} has no attribute {name!r}')
        warnings.warn(f'{name} is deprecated; use {new_name}.', DeprecationWarning, stacklevel=2)
        return namespace[new_name]

    return __getattr__


def warn_deprecated_value(message: str) -> None:
    """Warn that a caller used a deprecated value, the same way wherever the value arrives.

    Logs ``message`` once per process, so a command-line or dashboard user sees it, and raises a
    ``DeprecationWarning`` every time. The warning points at the first frame outside evaluatorq and
    the validation and CLI libraries, so Python shows it for the caller's own code instead of
    attributing it to pydantic or typer, where the default filters would hide it.
    """
    if message not in _logged:
        _logged.add(message)
        logger.warning(message)
    warnings.warn(message, DeprecationWarning, stacklevel=_caller_stacklevel())


def _caller_stacklevel() -> int:
    """The ``stacklevel`` that makes a warning raised in `warn_deprecated_value` name the first outside frame."""
    level = 2
    frame = sys._getframe(2)  # noqa: SLF001 - frame walking is the point
    while frame.f_back is not None and _is_internal(frame.f_code.co_filename):
        frame = frame.f_back
        level += 1
    return level


def _is_internal(filename: str) -> bool:
    if filename.startswith((_PACKAGE_DIR, '<')):
        return True
    parts = Path(filename).parts
    return any(package in parts for package in _INTERNAL_PACKAGES)
