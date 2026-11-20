"""Resolve the ``save=`` argument that `red_team`, `simulate`, `generate_and_simulate` and `evaluatorq` share."""

from __future__ import annotations

import warnings

from evaluatorq.contracts import SaveMode


def resolve_save_mode(*, save: SaveMode | str | bool, allow_detail: bool = True, stacklevel: int = 3) -> SaveMode:
    """The `SaveMode` that *save* names.

    ``True`` and ``False`` still work as ``SaveMode.FINAL`` and ``SaveMode.NONE``, with a
    ``DeprecationWarning``. *stacklevel* is counted from this function: the default 3
    blames the caller of the public entry point that calls this directly.

    Raises:
        ValueError: *save* is not a `SaveMode` value, or is ``SaveMode.DETAIL`` when
            *allow_detail* is False (only red teaming has per-stage artifacts to write).
    """
    if isinstance(save, bool):
        warnings.warn(
            'Passing save=True/False is deprecated. Use SaveMode.FINAL / SaveMode.NONE instead.',
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        return SaveMode.FINAL if save else SaveMode.NONE
    try:
        mode = SaveMode(save)
    except ValueError:
        msg = f'Invalid save value {save!r}. Must be one of: {", ".join(SaveMode.__members__.values())}.'
        raise ValueError(msg) from None
    if mode == SaveMode.DETAIL and not allow_detail:
        raise ValueError(
            'save=SaveMode.DETAIL only applies to red_team(), which writes per-stage artifacts. Use SaveMode.FINAL.'
        )
    return mode
