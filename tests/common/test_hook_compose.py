"""Structural validation for public lifecycle-hook composition."""

from __future__ import annotations

import pytest

from evaluatorq.common.hook_compose import compose_run_hooks


class _ConfirmOnlyHook:
    async def on_confirm(self, _payload: object) -> bool:
        return True


def test_composition_rejects_hook_missing_required_lifecycle_method() -> None:
    with pytest.raises(TypeError, match='on_stage_start'):
        compose_run_hooks(
            _ConfirmOnlyHook(),
            method_names=('on_confirm', 'on_stage_start'),
            composite_cls=lambda children: children,
            default_hooks_factory=object,
            manifest_factory=None,
            manifest_hook_factory=lambda writer: writer,
        )
