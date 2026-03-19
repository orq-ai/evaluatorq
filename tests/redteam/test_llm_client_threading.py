"""Tests for llm_client parameter threading through all red teaming call sites.

Verifies that when a custom llm_client is provided, it is used instead of
calling create_async_llm_client(), and that the env var priority is correct.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI


def _make_report():
    """Create a minimal RedTeamReport for use in tests."""
    from evaluatorq.redteam.contracts import Pipeline, RedTeamReport, ReportSummary

    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        description="Test report",
        pipeline=Pipeline.DYNAMIC,
        framework=None,
        categories_tested=["ASI01"],
        tested_agents=["agent:test"],
        total_results=0,
        results=[],
        summary=ReportSummary(),
    )


# ---------------------------------------------------------------------------
# 1. create_async_llm_client — env var priority (OPENAI first, then ORQ)
# ---------------------------------------------------------------------------

class TestCreateAsyncLlmClientPriority:
    """Verify OPENAI_* env vars take priority over ORQ_*."""

    @patch.dict(
        'os.environ',
        {'OPENAI_API_KEY': 'sk-openai', 'ORQ_API_KEY': 'orq-key'},
        clear=True,
    )
    @patch('openai.AsyncOpenAI')
    def test_openai_takes_priority_over_orq(self, mock_cls):
        from evaluatorq.redteam.backends.registry import create_async_llm_client

        create_async_llm_client()
        mock_cls.assert_called_once_with(api_key='sk-openai')

    @patch.dict(
        'os.environ',
        {'OPENAI_API_KEY': 'sk-openai', 'OPENAI_BASE_URL': 'http://local:8000'},
        clear=True,
    )
    @patch('openai.AsyncOpenAI')
    def test_openai_with_base_url(self, mock_cls):
        from evaluatorq.redteam.backends.registry import create_async_llm_client

        create_async_llm_client()
        mock_cls.assert_called_once_with(api_key='sk-openai', base_url='http://local:8000')

    @patch.dict(
        'os.environ',
        {'ORQ_API_KEY': 'orq-key'},
        clear=True,
    )
    @patch('openai.AsyncOpenAI')
    def test_orq_fallback_when_no_openai(self, mock_cls):
        from evaluatorq.redteam.backends.registry import create_async_llm_client

        create_async_llm_client()
        mock_cls.assert_called_once_with(
            api_key='orq-key',
            base_url='https://my.orq.ai/v2/router',
        )

    @patch.dict('os.environ', {}, clear=True)
    def test_no_keys_raises(self):
        from evaluatorq.redteam.backends.registry import create_async_llm_client
        from evaluatorq.redteam.exceptions import CredentialError

        with pytest.raises(CredentialError, match='Missing LLM credentials'):
            create_async_llm_client()


# ---------------------------------------------------------------------------
# 2. resolve_backend — uses provided llm_client for openai backend
# ---------------------------------------------------------------------------

class TestResolveBackendLlmClient:
    """Verify resolve_backend passes llm_client to OpenAI backend."""

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test'}, clear=True)
    def test_openai_backend_uses_provided_client(self):
        from evaluatorq.redteam.backends.registry import resolve_backend

        custom_client = MagicMock(spec=AsyncOpenAI)
        bundle = resolve_backend('openai', llm_client=custom_client)
        # The target factory should have been created with our custom client
        assert bundle.target_factory._client is custom_client  # pyright: ignore[reportAttributeAccessIssue]

    @patch('evaluatorq.redteam.backends.registry.create_async_llm_client')
    def test_openai_backend_falls_back_to_create(self, mock_create):
        from evaluatorq.redteam.backends.registry import resolve_backend

        mock_create.return_value = MagicMock(spec=AsyncOpenAI)
        bundle = resolve_backend('openai', llm_client=None)
        mock_create.assert_called_once()
        assert bundle.target_factory._client is mock_create.return_value  # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# 3. OWASPEvaluator — uses provided llm_client
# ---------------------------------------------------------------------------

class TestOWASPEvaluatorLlmClient:
    """Verify OWASPEvaluator uses provided llm_client."""

    def test_uses_custom_client(self):
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator

        custom_client = MagicMock(spec=AsyncOpenAI)
        evaluator = OWASPEvaluator(llm_client=custom_client)
        assert evaluator.client is custom_client

    @patch('evaluatorq.redteam.adaptive.evaluator.create_async_llm_client')
    def test_falls_back_to_create(self, mock_create):
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator

        mock_create.return_value = MagicMock(spec=AsyncOpenAI)
        evaluator = OWASPEvaluator()
        mock_create.assert_called_once()
        assert evaluator.client is mock_create.return_value


# ---------------------------------------------------------------------------
# 4. create_dynamic_evaluator — threads llm_client to OWASPEvaluator
# ---------------------------------------------------------------------------

class TestCreateDynamicEvaluatorLlmClient:
    """Verify create_dynamic_evaluator passes llm_client to OWASPEvaluator."""

    @patch('evaluatorq.redteam.adaptive.pipeline.OWASPEvaluator')
    def test_passes_client_to_evaluator(self, mock_cls):
        from evaluatorq.redteam.adaptive.pipeline import create_dynamic_evaluator

        custom_client = MagicMock(spec=AsyncOpenAI)
        create_dynamic_evaluator(llm_client=custom_client)
        mock_cls.assert_called_once_with(evaluator_model='gpt-5-mini', llm_client=custom_client, llm_kwargs=None)

    @patch('evaluatorq.redteam.adaptive.pipeline.OWASPEvaluator')
    def test_passes_none_when_no_client(self, mock_cls):
        from evaluatorq.redteam.adaptive.pipeline import create_dynamic_evaluator

        create_dynamic_evaluator()
        mock_cls.assert_called_once_with(evaluator_model='gpt-5-mini', llm_client=None, llm_kwargs=None)


# ---------------------------------------------------------------------------
# 5. create_owasp_evaluator — threads llm_client into scorer closure
# ---------------------------------------------------------------------------

class TestCreateOWASPEvaluatorLlmClient:
    """Verify create_owasp_evaluator uses provided llm_client in scorer."""

    @pytest.mark.asyncio
    @patch('evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.get_evaluator_for_category')
    @patch('evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.create_async_llm_client')
    async def test_scorer_uses_custom_client(self, mock_create, mock_get_eval):
        from evaluatorq import DataPoint, EvaluationResult
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import create_owasp_evaluator

        custom_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"value": true, "explanation": "Resistant"}'
        custom_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_eval = MagicMock()
        mock_eval.prompt = 'Evaluate: {{output.response}} with {{input.all_messages}}'
        mock_get_eval.return_value = mock_eval

        evaluator_config = create_owasp_evaluator(llm_client=custom_client)
        result = await evaluator_config['scorer']({
            'data': DataPoint(inputs={'category': 'ASI01', 'messages': []}),
            'output': {'response': 'test response'},
        })

        custom_client.chat.completions.create.assert_awaited_once()
        mock_create.assert_not_called()
        assert isinstance(result, EvaluationResult)

    @pytest.mark.asyncio
    @patch('evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.get_evaluator_for_category')
    @patch('evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.create_async_llm_client')
    async def test_scorer_falls_back_when_no_client(self, mock_create, mock_get_eval):
        from evaluatorq import DataPoint
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import create_owasp_evaluator

        fallback_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"value": true, "explanation": "OK"}'
        fallback_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_create.return_value = fallback_client

        mock_eval = MagicMock()
        mock_eval.prompt = 'Evaluate: {{output.response}} with {{input.all_messages}}'
        mock_get_eval.return_value = mock_eval

        evaluator_config = create_owasp_evaluator(llm_client=None)
        await evaluator_config['scorer']({
            'data': DataPoint(inputs={'category': 'ASI01', 'messages': []}),
            'output': {'response': 'test'},
        })

        mock_create.assert_called_once()
        fallback_client.chat.completions.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. create_model_job — threads llm_client into router_job closure
# ---------------------------------------------------------------------------

class TestCreateModelJobLlmClient:
    """Verify create_model_job uses provided llm_client in router_job."""

    @pytest.mark.asyncio
    @patch('evaluatorq.redteam.runtime.jobs.create_async_llm_client')
    async def test_router_job_uses_custom_client(self, mock_create):
        from evaluatorq import DataPoint
        from evaluatorq.redteam.runtime.jobs import create_model_job

        custom_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'response text'
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = None
        custom_client.chat.completions.create = AsyncMock(return_value=mock_response)

        job_fn = create_model_job(model='gpt-4o', llm_client=custom_client)
        data = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'hello'}]})
        result = await job_fn(data, 0)

        custom_client.chat.completions.create.assert_awaited_once()
        mock_create.assert_not_called()
        assert result['output']['response'] == 'response text'

    @pytest.mark.asyncio
    @patch('evaluatorq.redteam.runtime.jobs.create_async_llm_client')
    async def test_router_job_falls_back_when_no_client(self, mock_create):
        from evaluatorq import DataPoint
        from evaluatorq.redteam.runtime.jobs import create_model_job

        fallback_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'fallback response'
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = None
        fallback_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_create.return_value = fallback_client

        job_fn = create_model_job(model='gpt-4o', llm_client=None)
        data = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'hello'}]})
        result = await job_fn(data, 0)

        mock_create.assert_called_once()
        assert result['output']['response'] == 'fallback response'


# ---------------------------------------------------------------------------
# 7. red_team() → _run_static — llm_client is forwarded
# ---------------------------------------------------------------------------

class TestRedTeamStaticLlmClientForwarding:
    """Verify red_team() passes llm_client to _run_static."""

    @pytest.mark.asyncio
    async def test_static_mode_forwards_llm_client(self):
        sentinel = MagicMock(spec=AsyncOpenAI)

        with patch(
            'evaluatorq.redteam.runner._run_static',
            new_callable=AsyncMock,
            return_value=_make_report(),
        ) as mock_static:
            await red_team('agent:test', mode='static', llm_client=sentinel)
            call_kwargs = mock_static.call_args.kwargs
            assert call_kwargs['llm_client'] is sentinel

    @pytest.mark.asyncio
    async def test_dynamic_mode_forwards_llm_client(self):
        sentinel = MagicMock(spec=AsyncOpenAI)

        with patch(
            'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
            new_callable=AsyncMock,
            return_value=_make_report(),
        ) as mock_dynamic:
            await red_team('agent:test', mode='dynamic', llm_client=sentinel)
            call_kwargs = mock_dynamic.call_args.kwargs
            assert call_kwargs['llm_client'] is sentinel


# Import red_team at module level (after all patches are defined)
from evaluatorq.redteam import red_team  # noqa: E402
