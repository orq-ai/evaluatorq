"""Run-scoped registries (RES-1922).

Every mutable registry in the red-team package is a `ScopedDict` or `ScopedSet`: a view
over storage owned by the current scope. The global scope holds the built-ins and
anything registered with the global `register_*` calls. `run_scope()` opens a scope
that starts as a copy of the current one, so a run sees the built-ins, the global
registrations and its own definitions, and nothing it registers reaches the global
scope or another run, including runs on other asyncio tasks. The scope lives in a
`ContextVar`, which `asyncio` tasks and `asyncio.to_thread` copy.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, MutableMapping, MutableSet
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generic, TypeVar

_K = TypeVar('_K')
_V = TypeVar('_V')

_Scope = dict[str, Any]

_GLOBAL: _Scope = {}
_CURRENT: ContextVar[_Scope] = ContextVar('evaluatorq_redteam_registry_scope', default=_GLOBAL)

# name -> True when the store is derived (rebuilt from the others, so a new scope starts it empty)
_STORES: dict[str, bool] = {}
_REBUILDERS: list[Callable[[], None]] = []


def _copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return list(value)
    if isinstance(value, set):
        return set(value)
    return value


def _register_store(name: str, initial: Any, *, derived: bool) -> None:
    if name in _STORES:
        msg = f'Registry store {name!r} is defined twice.'
        raise RuntimeError(msg)
    _STORES[name] = derived
    _GLOBAL[name] = initial


def _data(name: str) -> Any:
    return _CURRENT.get()[name]


class ScopedDict(MutableMapping[_K, _V], Generic[_K, _V]):
    """A dict whose contents belong to the current registry scope."""

    def __init__(self, name: str, initial: dict[_K, _V] | None = None, *, derived: bool = False) -> None:
        self._name = name
        _register_store(name, dict(initial or {}), derived=derived)

    def __getitem__(self, key: _K) -> _V:
        return _data(self._name)[key]

    def __setitem__(self, key: _K, value: _V) -> None:
        _data(self._name)[key] = value

    def __delitem__(self, key: _K) -> None:
        del _data(self._name)[key]

    def __iter__(self) -> Iterator[_K]:
        return iter(_data(self._name))

    def __len__(self) -> int:
        return len(_data(self._name))

    def __contains__(self, key: object) -> bool:
        return key in _data(self._name)

    def __repr__(self) -> str:
        return f'ScopedDict({_data(self._name)!r})'


class ScopedSet(MutableSet[_K], Generic[_K]):
    """A set whose contents belong to the current registry scope."""

    def __init__(self, name: str) -> None:
        self._name = name
        _register_store(name, set(), derived=False)

    def __contains__(self, value: object) -> bool:
        return value in _data(self._name)

    def __iter__(self) -> Iterator[_K]:
        return iter(_data(self._name))

    def __len__(self) -> int:
        return len(_data(self._name))

    def add(self, value: _K) -> None:
        _data(self._name).add(value)

    def discard(self, value: _K) -> None:
        _data(self._name).discard(value)


def register_rebuilder(rebuild: Callable[[], None]) -> None:
    """Run *rebuild* whenever a scope's derived stores must be refilled, in registration order."""
    _REBUILDERS.append(rebuild)


def _rebuild() -> None:
    for rebuild in _REBUILDERS:
        rebuild()


def _copy_sources(scope: _Scope) -> _Scope:
    return {name: (type(scope[name])() if derived else _copy(scope[name])) for name, derived in _STORES.items()}


@contextmanager
def run_scope() -> Generator[None]:
    """Registrations inside the block live in a fresh copy of the current scope and end with it."""
    token = _CURRENT.set(_copy_sources(_CURRENT.get()))
    try:
        _rebuild()
        yield
    finally:
        _CURRENT.reset(token)


def snapshot() -> Callable[[], None]:
    """Copy the current scope; the returned function puts that copy back."""
    scope = _CURRENT.get()
    saved = _copy_sources(scope)

    def restore() -> None:
        for name, derived in _STORES.items():
            if not derived:
                live = scope[name]
                live.clear()
                live.update(saved[name])
        _rebuild()

    return restore
