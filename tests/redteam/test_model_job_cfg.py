"""``create_deployment_job`` must consume the ``LLMConfig`` it is handed, not the defaults.

``_create_job_for_target`` threads a per-run ``cfg`` into ``create_deployment_job``
(``redteam/runtime/jobs.py``). Both legs read it:

* router leg — ``target_agent_timeout_ms`` (per-call timeout),
  ``target_reasoning_effort``, and ``max_target_retries`` (the SDK client's own
  retry budget, only when this factory builds the client);
* deployment leg — ``target_agent_timeout_ms`` and ``max_target_retries``
  (``with_retry`` attempts), and a warning that ``target_reasoning_effort`` is
  inert there.

These assert on the outgoing call, so a leg that stops reading ``cfg`` and falls
back to ``PIPELINE_CONFIG`` fails here.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq import DataPoint
from evaluatorq.redteam.contracts import LLMConfig

_JOBS = 'evaluatorq.redteam.runtime.jobs'


def _datapoint() -> DataPoint:
    return DataPoint(
        inputs={'id': 'cfg-1', 'category': 'ASI01', 'messages': [{'role': 'user', 'content': 'hello'}]}
    )


def _tuned_cfg(**overrides: Any) -> LLMConfig:
    """An LLMConfig whose every asserted field differs from the field default."""
    values: dict[str, Any] = {
        'target_agent_timeout_ms': 4321,
        'max_target_retries': 5,
        'target_reasoning_effort': 'high',
    }
    for field, value in values.items():
        assert LLMConfig.model_fields[field].default != value, f'{field} must differ from its default'
    values.update(overrides)
    return LLMConfig(**values)


class TestDeploymentLegReadsCfg:
    """The deployment leg reads cfg for retries/timeout and warns on reasoning effort."""

    @staticmethod
    def _install_sdk(monkeypatch: pytest.MonkeyPatch, deployments: MagicMock) -> None:
        module = ModuleType('orq_ai_sdk')
        module.Orq = MagicMock(return_value=MagicMock(deployments=deployments))  # pyright: ignore[reportAttributeAccessIssue]
        monkeypatch.setitem(sys.modules, 'orq_ai_sdk', module)
        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    @pytest.mark.asyncio
    async def test_with_retry_attempts_come_from_cfg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sole retry layer on this leg is with_retry, budgeted from cfg.

        Asserted on the ``max_attempts`` the leg passes rather than by counting
        real attempts, so the test neither depends on which exception classes
        ``with_retry`` considers retryable nor pays its exponential backoff.
        """
        from evaluatorq.redteam.runtime.jobs import create_deployment_job

        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = 'deployment response'
        completion.model = 'gpt-4o-mini'
        completion.usage = None
        deployments = MagicMock()
        deployments.invoke_async = AsyncMock(return_value=completion)
        self._install_sdk(monkeypatch, deployments)

        captured: dict[str, Any] = {}

        async def fake_with_retry(fn: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return await fn()

        monkeypatch.setattr(f'{_JOBS}.with_retry', fake_with_retry)

        cfg = _tuned_cfg(target_reasoning_effort=None)
        job_fn = create_deployment_job(deployment_key='dep', cfg=cfg)
        await job_fn(_datapoint(), 0)

        assert captured['max_attempts'] == cfg.max_target_retries + 1
        assert captured['max_attempts'] != LLMConfig.model_fields['max_target_retries'].default + 1

    @pytest.mark.asyncio
    async def test_timeout_comes_from_cfg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deployment slower than cfg.target_agent_timeout_ms is abandoned, not awaited."""
        import asyncio

        from evaluatorq.redteam.runtime.jobs import create_deployment_job

        async def _never_returns(**_kwargs: Any) -> Any:
            await asyncio.sleep(30)

        deployments = MagicMock()
        deployments.invoke_async = AsyncMock(side_effect=_never_returns)
        self._install_sdk(monkeypatch, deployments)

        from evaluatorq.job_helper import JobError

        cfg = LLMConfig(target_agent_timeout_ms=10, max_target_retries=0)
        job_fn = create_deployment_job(deployment_key='dep', cfg=cfg)
        # The @job decorator rewraps whatever the body raises, so the timeout is
        # asserted on the cause rather than the surfaced type.
        with pytest.raises(JobError) as excinfo:
            await asyncio.wait_for(job_fn(_datapoint(), 0), timeout=5)
        assert isinstance(excinfo.value.__cause__, asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_reasoning_effort_is_announced_as_inert(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A degraded path announces itself: the deployment leg cannot forward it."""
        from loguru import logger

        from evaluatorq.redteam.runtime.jobs import create_deployment_job

        self._install_sdk(monkeypatch, MagicMock())
        records: list[str] = []
        sink_id = logger.add(lambda msg: records.append(msg), level='WARNING')
        try:
            create_deployment_job(deployment_key='dep', cfg=_tuned_cfg())
        finally:
            logger.remove(sink_id)

        assert any('target_reasoning_effort' in r and 'not forwarded' in r for r in records), records

    @pytest.mark.asyncio
    async def test_no_warning_when_reasoning_effort_is_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from loguru import logger

        from evaluatorq.redteam.runtime.jobs import create_deployment_job

        self._install_sdk(monkeypatch, MagicMock())
        records: list[str] = []
        sink_id = logger.add(lambda msg: records.append(msg), level='WARNING')
        try:
            create_deployment_job(deployment_key='dep', cfg=_tuned_cfg(target_reasoning_effort=None))
        finally:
            logger.remove(sink_id)

        assert not any('target_reasoning_effort' in r for r in records), records
