"""Unit tests for runtime/jobs.py."""

# ruff: noqa: S101

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.common import model_catalogue
from evaluatorq.redteam.contracts import Message, TokenUsage

# ===========================================================================
# runtime/jobs.py — _sanitize_job_name
# ===========================================================================


class TestSanitizeJobName:
    """Tests for _sanitize_job_name()."""

    def test_alphanumeric_passthrough(self):
        from evaluatorq.redteam.runtime.jobs import _sanitize_job_name

        assert _sanitize_job_name('abc123') == 'abc123'

    def test_dash_and_underscore_preserved(self):
        from evaluatorq.redteam.runtime.jobs import _sanitize_job_name

        assert _sanitize_job_name('my-agent_key') == 'my-agent_key'

    def test_special_chars_replaced_with_dash(self):
        from evaluatorq.redteam.runtime.jobs import _sanitize_job_name

        result = _sanitize_job_name('my.agent/key@v2')
        assert result == 'my-agent-key-v2'

    def test_leading_trailing_dashes_stripped(self):
        from evaluatorq.redteam.runtime.jobs import _sanitize_job_name

        result = _sanitize_job_name('.leading-trailing.')
        assert not result.startswith('-')
        assert not result.endswith('-')

    def test_empty_string_returns_unknown(self):
        from evaluatorq.redteam.runtime.jobs import _sanitize_job_name

        assert _sanitize_job_name('') == 'unknown'

    def test_all_special_chars_becomes_unknown(self):
        from evaluatorq.redteam.runtime.jobs import _sanitize_job_name

        # All special chars → all dashes → strip → empty → 'unknown'
        result = _sanitize_job_name('@@@')
        assert result == 'unknown'

    def test_spaces_replaced(self):
        from evaluatorq.redteam.runtime.jobs import _sanitize_job_name

        result = _sanitize_job_name('my agent key')
        assert ' ' not in result

    def test_model_with_slash(self):
        from evaluatorq.redteam.runtime.jobs import _sanitize_job_name

        result = _sanitize_job_name('azure/gpt-4o-mini')
        assert '/' not in result
        assert 'azure' in result
        assert 'gpt' in result


# ===========================================================================
# runtime/jobs.py — _build_messages
# ===========================================================================


class TestBuildMessages:
    """Tests for _build_messages()."""

    def _make_datapoint(self, messages: list[Any]) -> MagicMock:
        """Create a mock DataPoint with an inputs['messages'] list."""
        dp = MagicMock()
        dp.inputs = {'messages': messages}
        return dp

    def test_from_message_objects(self):
        from evaluatorq.redteam.runtime.jobs import _build_messages

        dp = self._make_datapoint([
            Message(role='user', content='Hello'),
            Message(role='assistant', content='Hi there'),
        ])
        result = _build_messages(dp)
        assert len(result) == 2
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == 'Hello'
        assert result[1]['role'] == 'assistant'

    def test_from_valid_dicts(self):
        from evaluatorq.redteam.runtime.jobs import _build_messages

        dp = self._make_datapoint([
            {'role': 'user', 'content': 'What is the weather?'},
        ])
        result = _build_messages(dp)
        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == 'What is the weather?'

    def test_from_invalid_dicts_kept_as_raw(self):
        from evaluatorq.redteam.runtime.jobs import _build_messages

        # Invalid role → validation fails → raw dict kept
        dp = self._make_datapoint([
            {'role': 'invalid_role', 'content': 'Something'},
        ])
        result = _build_messages(dp)
        assert len(result) == 1
        assert result[0]['role'] == 'invalid_role'

    def test_from_plain_strings(self):
        from evaluatorq.redteam.runtime.jobs import _build_messages

        dp = self._make_datapoint(['Just a plain string'])
        result = _build_messages(dp)
        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == 'Just a plain string'

    def test_mixed_types(self):
        from evaluatorq.redteam.runtime.jobs import _build_messages

        dp = self._make_datapoint([
            Message(role='system', content='System'),
            {'role': 'user', 'content': 'Dict user'},
            'plain string',
        ])
        result = _build_messages(dp)
        assert len(result) == 3


# ===========================================================================
# runtime/jobs.py — _extract_deployment_content
# ===========================================================================


class TestExtractDeploymentContent:
    """Tests for _extract_deployment_content()."""

    def _make_completion(self, content) -> MagicMock:
        """Create a mock deployment completion with choices[0].message.content."""
        completion = MagicMock()
        choice = MagicMock()
        message = MagicMock()
        message.content = content
        choice.message = message
        completion.choices = [choice]
        return completion

    def test_string_content_returned_directly(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        completion = self._make_completion('Hello world')
        assert _extract_deployment_content(completion) == 'Hello world'

    def test_list_content_text_parts_joined(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        part1 = MagicMock()
        part1.type = 'text'
        part1.text = 'First'
        part2 = MagicMock()
        part2.type = 'text'
        part2.text = 'Second'

        completion = self._make_completion([part1, part2])
        result = _extract_deployment_content(completion)
        assert 'First' in result
        assert 'Second' in result

    def test_list_content_non_text_parts_filtered(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        text_part = MagicMock()
        text_part.type = 'text'
        text_part.text = 'Visible'
        image_part = MagicMock()
        image_part.type = 'image'
        image_part.text = 'Hidden'

        completion = self._make_completion([image_part, text_part])
        result = _extract_deployment_content(completion)
        assert 'Visible' in result
        assert 'Hidden' not in result

    def test_empty_choices_returns_empty_string(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        completion = MagicMock()
        completion.choices = []
        assert _extract_deployment_content(completion) == ''

    def test_none_choices_returns_empty_string(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        completion = MagicMock()
        completion.choices = None
        assert _extract_deployment_content(completion) == ''

    def test_none_content_returns_empty_string(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        completion = self._make_completion(None)
        assert _extract_deployment_content(completion) == ''


# ===========================================================================
# runtime/jobs.py — _normalize_usage
# ===========================================================================


class TestNormalizeUsage:
    """Tests for _normalize_usage()."""

    def test_token_usage_passthrough(self):
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        result = _normalize_usage(usage)
        assert result is usage

    def test_dict_with_standard_keys(self):
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        raw = {'prompt_tokens': 5, 'completion_tokens': 15, 'total_tokens': 20}
        result = _normalize_usage(raw)
        assert result is not None
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 15
        assert result.total_tokens == 20

    def test_dict_with_short_keys(self):
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        raw = {'prompt': 3, 'completion': 7, 'total': 10}
        result = _normalize_usage(raw)
        assert result is not None
        assert result.prompt_tokens == 3
        assert result.completion_tokens == 7
        assert result.total_tokens == 10

    def test_none_returns_none(self):
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        assert _normalize_usage(None) is None

    def test_non_dict_non_token_usage_returns_none(self):
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        assert _normalize_usage('not a dict') is None
        assert _normalize_usage(42) is None

    def test_empty_dict_returns_none(self):
        """An empty usage dict carries no signal: extract returns None rather than a
        fabricated billed-zero record (see TokenUsage.extract gate)."""
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        assert _normalize_usage({}) is None


# ===========================================================================
# runtime/jobs.py — create_model_job
# ===========================================================================


class TestCreateModelJob:
    """Tests for create_model_job() factory function."""

    def test_raises_value_error_when_no_params(self):
        from evaluatorq.redteam.runtime.jobs import create_model_job

        with pytest.raises(ValueError, match="Provide one of: 'model' or 'deployment_key'"):
            create_model_job()

    @pytest.mark.asyncio
    async def test_deployment_job_prices_usage_from_catalogue(self, monkeypatch: pytest.MonkeyPatch):
        """RES-1295: deployments.invoke_async usage comes back priced, keyed on
        the model the deployment actually ran, with no AsyncOpenAI client
        available to resolve the catalogue against (falls back to ORQ_BASE_URL,
        which is the host the deployment client itself was built with)."""
        from evaluatorq import DataPoint
        from evaluatorq.redteam.runtime.jobs import create_model_job

        model_catalogue.reset_catalogue_cache()

        async def fake_load(client=None):  # noqa: ANN001, ARG001
            assert client is None
            return {'gpt-4o-mini': model_catalogue.ModelInfo(0.00025, 0.002, 'openai', supports_responses=True)}

        monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)

        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = 'mock target response'
        completion.model = 'gpt-4o-mini'
        completion.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        deployments = MagicMock()
        deployments.invoke_async = AsyncMock(return_value=completion)
        module = ModuleType('orq_ai_sdk')
        module.Orq = MagicMock(return_value=MagicMock(deployments=deployments))  # pyright: ignore[reportAttributeAccessIssue]
        monkeypatch.setitem(sys.modules, 'orq_ai_sdk', module)
        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        job_fn = create_model_job(deployment_key='test-deployment', run_id='static-run')
        result = await job_fn(
            DataPoint(
                inputs={
                    'id': 'deployment-1',
                    'category': 'ASI01',
                    'messages': [{'role': 'user', 'content': 'hello'}],
                }
            ),
            0,
        )

        usage = result['output']['token_usage']
        assert usage is not None
        assert usage.calls == usage.priced_calls == 1
        assert usage.total_cost is not None
        assert usage.total_cost > 0

        model_catalogue.reset_catalogue_cache()

    @pytest.mark.asyncio
    async def test_deployment_client_targets_the_configured_host(self, monkeypatch: pytest.MonkeyPatch):
        """The deployment client must be built with ORQ_BASE_URL, not the SDK's prod default.

        Without server_url the call goes to my.orq.ai while price_usage looks the model
        up in the ORQ_BASE_URL host's catalogue — a staging run priced off prod (RES-1295).
        """
        from evaluatorq.redteam.runtime.jobs import create_model_job

        module = ModuleType('orq_ai_sdk')
        module.Orq = MagicMock(return_value=MagicMock(deployments=MagicMock()))  # pyright: ignore[reportAttributeAccessIssue]
        monkeypatch.setitem(sys.modules, 'orq_ai_sdk', module)
        monkeypatch.setenv('ORQ_API_KEY', 'test-key')
        monkeypatch.setenv('ORQ_BASE_URL', 'https://staging.orq.ai')

        create_model_job(deployment_key='test-deployment')

        module.Orq.assert_called_once_with(api_key='test-key', server_url='https://staging.orq.ai')  # pyright: ignore[reportAttributeAccessIssue]

    @pytest.mark.asyncio
    async def test_router_job_returns_priced_usage(self, monkeypatch: pytest.MonkeyPatch):
        """RES-1295: router_job must return the priced Usage execute_chat_completion
        already computed, not re-derive an unpriced one from the raw response —
        that re-derivation was the exact counted-but-unpriced defect this task closes."""
        from evaluatorq import DataPoint
        from evaluatorq.redteam.runtime.jobs import create_model_job

        model_catalogue.reset_catalogue_cache()

        async def fake_load(client=None):  # noqa: ANN001, ARG001
            return {'gpt-4o-mini': model_catalogue.ModelInfo(0.00025, 0.002, 'openai', supports_responses=True)}

        monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = 'mock target response'
        response.choices[0].finish_reason = 'stop'
        response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        client = AsyncMock()
        client.base_url = 'https://api.openai.com/v1'
        client.chat.completions.create = AsyncMock(return_value=response)

        job_fn = create_model_job(model='gpt-4o-mini', llm_client=client, run_id='static-run')
        result = await job_fn(
            DataPoint(
                inputs={
                    'id': 'router-1',
                    'category': 'ASI01',
                    'messages': [{'role': 'user', 'content': 'hello'}],
                }
            ),
            0,
        )

        usage = result['output']['token_usage']
        assert usage is not None
        assert usage.calls == usage.priced_calls == 1
        assert usage.total_cost is not None
        assert usage.total_cost > 0

        model_catalogue.reset_catalogue_cache()
