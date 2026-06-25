"""Starlette 1.3.x compatibility shim for FastHTML 0.12.x.

FastHTML 0.12.x passes ``on_startup`` and ``on_shutdown`` positional arguments
to ``Starlette.__init__``, which Starlette 1.3.x removed (it now requires a
single ``lifespan`` context-manager).

This module applies a **process-global** class patch to ``Starlette.__init__``
that translates those removed kwargs into a lifespan so they still execute.
The shim is semantics-preserving: it does NOT silently drop the handlers.

Import order matters
--------------------
This module must be imported BEFORE ``fasthtml`` is imported anywhere in the
same interpreter session.  ``app.py`` imports it at the very top (before the
``fasthtml.core`` import) so that the patch is in place when ``build_app()``
constructs the ``FastHTML`` instance.

Why not in ``serve()``?
-----------------------
Dashboard tests construct the app via ``build_app() + TestClient`` without
ever calling ``serve()``.  Moving the patch into ``serve()`` would break those
tests because the ``FastHTML`` constructor runs at ``build_app()`` time.
"""

from __future__ import annotations

import contextlib

import starlette.applications as _starlette_app

_orig_starlette_init = _starlette_app.Starlette.__init__


def _lifespan_from_handlers(on_startup, on_shutdown):
    """Fold Starlette's removed on_startup/on_shutdown lists into a lifespan."""

    @contextlib.asynccontextmanager
    async def _lifespan(_app):
        for handler in on_startup or ():
            result = handler()
            if result is not None:
                await result
        try:
            yield
        finally:
            for handler in on_shutdown or ():
                result = handler()
                if result is not None:
                    await result

    return _lifespan


def _starlette_compat_init(  # type: ignore[override]
    self: _starlette_app.Starlette,
    debug: object = False,  # noqa: FBT002
    routes: object = None,
    middleware: object = None,
    exception_handlers: object = None,
    lifespan: object = None,
    on_startup: object = None,
    on_shutdown: object = None,
    **kw: object,
) -> None:
    # Preserve startup/shutdown handlers by folding them into a lifespan,
    # instead of discarding them. Only synthesize when no explicit lifespan
    # was given (Starlette forbids passing both).
    if lifespan is None and (on_startup or on_shutdown):
        lifespan = _lifespan_from_handlers(on_startup, on_shutdown)  # type: ignore[arg-type]
    _orig_starlette_init(
        self,
        debug=debug,  # type: ignore[arg-type]
        routes=routes,  # type: ignore[arg-type]
        middleware=middleware,  # type: ignore[arg-type]
        exception_handlers=exception_handlers,  # type: ignore[arg-type]
        lifespan=lifespan,  # type: ignore[arg-type]
    )


_starlette_app.Starlette.__init__ = _starlette_compat_init  # type: ignore[method-assign]
