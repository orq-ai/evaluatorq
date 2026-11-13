"""Regression tests for red-team gripe fixes.

Covers: full ASI evaluator coverage, zero-category dataset warnings, clean
dataset-download errors, the ASI07/08 multi-agent gate, and attacker
self-censor / content-filter handling.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from loguru import logger


def _capture_logs(level: str = 'WARNING'):
    """Return (messages_list, remove_fn) capturing loguru records at ``level``."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record['message']), level=level)
    return messages, lambda: logger.remove(sink_id)


class TestEvaluatorCoverage:
    """All ten OWASP ASI categories resolve a registered evaluator (gripe #2)."""

    @pytest.mark.parametrize(
        'category',
        ['ASI01', 'ASI02', 'ASI03', 'ASI04', 'ASI05', 'ASI06', 'ASI07', 'ASI08', 'ASI09', 'ASI10'],
    )
    def test_every_asi_category_has_evaluator(self, category: str):
        from evaluatorq.redteam.frameworks.owasp.evaluators import get_evaluator_for_category

        assert get_evaluator_for_category(category) is not None

    def test_every_builtin_vulnerability_is_evaluable(self):
        from evaluatorq.redteam.contracts import Vulnerability
        from evaluatorq.redteam.frameworks.owasp.evaluators import VULNERABILITY_EVALUATOR_REGISTRY

        missing = [v.value for v in Vulnerability if v not in VULNERABILITY_EVALUATOR_REGISTRY]
        assert missing == [], f'vulnerabilities with no evaluator: {missing}'


def _category(dp: Any) -> str:
    return dp.inputs.get('category', '')


def _id(dp: Any) -> str | None:
    return dp.inputs.get('id')


class TestZeroCategoryWarning:
    """A requested category with zero datapoints warns instead of silently vanishing (gripe #3)."""

    def test_warns_for_empty_category(self):
        from evaluatorq import DataPoint
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import _filter_by_categories

        dps = [DataPoint(inputs={'category': 'ASI01'})]
        messages, stop = _capture_logs('WARNING')
        try:
            kept = _filter_by_categories(dps, ['ASI01', 'ASI03'], category_of=_category, id_of=_id)
        finally:
            stop()

        assert len(kept) == 1
        joined = '\n'.join(messages)
        assert 'ASI03' in joined
        assert 'ASI01' not in joined  # present category is not warned

    def test_no_warning_when_all_present(self):
        from evaluatorq import DataPoint
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import _filter_by_categories

        dps = [DataPoint(inputs={'category': 'ASI01'}), DataPoint(inputs={'category': 'OWASP-ASI02'})]
        messages, stop = _capture_logs('WARNING')
        try:
            kept = _filter_by_categories(dps, ['ASI01', 'ASI02'], category_of=_category, id_of=_id)
        finally:
            stop()

        assert len(kept) == 2
        assert messages == []

    @pytest.mark.parametrize('category', ['ASI1', 'LLM03'])
    def test_warns_with_ids_for_unrequestable_category(self, category: str):
        from evaluatorq import DataPoint
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import _filter_by_categories

        dps = [
            DataPoint(inputs={'id': 'ok', 'category': 'ASI01'}),
            DataPoint(inputs={'id': 'r1', 'category': category}),
            DataPoint(inputs={'id': 'r2', 'category': category}),
        ]
        messages, stop = _capture_logs('WARNING')
        try:
            kept = _filter_by_categories(dps, ['ASI01', 'ASI04'], category_of=_category, id_of=_id)
        finally:
            stop()

        assert len(kept) == 1
        assert (
            f'Skipped 2 static datapoint(s) with category {category!r}, which is not a requested '
            'category code (ids: r1, r2).'
        ) in '\n'.join(messages)

    def test_no_skip_warning_for_unrequested_valid_category(self):
        from evaluatorq import DataPoint
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import _filter_by_categories

        dps = [DataPoint(inputs={'id': 'a', 'category': 'ASI01'}), DataPoint(inputs={'id': 'b', 'category': 'LLM03'})]
        messages, stop = _capture_logs('WARNING')
        try:
            kept = _filter_by_categories(dps, ['ASI01'], category_of=_category, id_of=_id)
        finally:
            stop()

        assert len(kept) == 1
        assert messages == []


class TestDatasetDownloadError:
    """A HuggingFace download failure surfaces a clean DatasetError, not a raw traceback (gripe #1)."""

    def test_download_failure_raises_dataset_error(self, monkeypatch):
        import huggingface_hub

        from evaluatorq.redteam.exceptions import DatasetError
        from evaluatorq.redteam.frameworks.owasp import evaluatorq_bridge

        def _boom(*args, **kwargs):
            raise OSError('network is unreachable')

        monkeypatch.setattr(huggingface_hub, 'hf_hub_download', _boom)

        with pytest.raises(DatasetError, match='Failed to download'):
            evaluatorq_bridge._fetch_from_huggingface()

    def test_default_run_dataset_is_pinned(self, monkeypatch, tmp_path):
        import huggingface_hub

        from evaluatorq.redteam.frameworks.owasp import evaluatorq_bridge

        dataset = tmp_path / 'd.json'
        row = {
            'input': {
                'id': 'r1',
                'category': 'ASI01',
                'severity': 'high',
                'vulnerability_domain': 'agent',
                'turn_type': 'single',
                'source': 't',
            },
            'messages': [{'role': 'user', 'content': 'x'}],
        }
        dataset.write_text(json.dumps({'samples': [row]}), encoding='utf-8')
        calls = []

        def _record(**kwargs):
            calls.append(kwargs)
            return str(dataset)

        monkeypatch.setattr(huggingface_hub, 'hf_hub_download', _record)

        evaluatorq_bridge.load_owasp_agentic_dataset(None)

        assert calls[0]['revision'] == evaluatorq_bridge.DEFAULT_HF_REVISION

    def test_dataset_error_is_redteam_error(self):
        from evaluatorq.redteam.exceptions import DatasetError, RedTeamError

        assert issubclass(DatasetError, RedTeamError)


class TestOrqDatasetFetchError:
    """A wrong orq:<dataset-id> surfaces a clean DatasetError naming the id (RES-1870)."""

    @staticmethod
    def _load_with_transport(monkeypatch, handler):
        import httpx
        from orq_ai_sdk import Orq

        from evaluatorq.redteam.frameworks.owasp import evaluatorq_bridge

        client = Orq(
            api_key='test-key',
            server_url='http://orq.test',
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        monkeypatch.setenv('ORQ_API_KEY', 'test-key')
        monkeypatch.setattr(evaluatorq_bridge, 'resolve_orq_client', lambda _key: client)
        return evaluatorq_bridge.load_owasp_agentic_dataset('orq:not-a-real-id')

    def test_unknown_id_raises_dataset_error_with_id_and_status(self, monkeypatch):
        import httpx
        from orq_ai_sdk.models import ResponseValidationError

        from evaluatorq.redteam.exceptions import DatasetError

        def _not_found(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={'code': 404, 'error': 'Dataset not found'})

        with pytest.raises(DatasetError, match=r'orq:not-a-real-id.*HTTP 404: Dataset not found\.') as ei:
            self._load_with_transport(monkeypatch, _not_found)
        assert isinstance(ei.value.__cause__, ResponseValidationError)
        assert 'valid string' not in str(ei.value)

    def test_missing_api_key_raises_credential_error(self, monkeypatch):
        from evaluatorq.redteam.exceptions import CredentialError
        from evaluatorq.redteam.frameworks.owasp import evaluatorq_bridge

        monkeypatch.delenv('ORQ_API_KEY', raising=False)
        with pytest.raises(CredentialError, match='ORQ_API_KEY'):
            evaluatorq_bridge.load_owasp_agentic_dataset('orq:some-id')

    def test_network_error_raises_dataset_error(self, monkeypatch):
        import httpx

        from evaluatorq.redteam.exceptions import DatasetError

        def _refused(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('connection refused', request=request)

        with pytest.raises(DatasetError, match='orq:not-a-real-id') as ei:
            self._load_with_transport(monkeypatch, _refused)
        assert isinstance(ei.value.__cause__, httpx.ConnectError)

    def test_later_page_failure_names_the_page(self, monkeypatch):
        import httpx

        from evaluatorq.redteam.exceptions import DatasetError

        page_one = {
            'object': 'list',
            'data': [{'_id': 'dp-1', 'workspace_id': 'ws', 'dataset_id': 'ds', 'inputs': {}}],
            'has_more': True,
        }

        def _second_page_fails(request: httpx.Request) -> httpx.Response:
            if 'starting_after' in request.url.params:
                return httpx.Response(500, json={'message': 'internal error'})
            return httpx.Response(200, json=page_one)

        with pytest.raises(DatasetError, match=r'\(page after datapoint dp-1\): HTTP 500: internal error'):
            self._load_with_transport(monkeypatch, _second_page_fails)


@pytest.mark.parametrize(
    ('body', 'expected'),
    [
        ('{"message": "dataset not found", "error": "other"}', 'dataset not found'),
        ('{"error": "unauthorized"}', 'unauthorized'),
        ('[1, 2]', '[1, 2]'),
        ('<html>bad gateway</html>', '<html>bad gateway</html>'),
        ('x' * 400, 'x' * 300 + '...'),
    ],
)
def test_orq_error_message(body: str, expected: str):
    from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import _orq_error_message

    assert _orq_error_message(body) == expected


class TestLoadFromFileError:
    """_load_from_file's JSON parse/validation surfaces a clean DatasetError (gripe #4)."""

    def test_malformed_json_raises_dataset_error(self, tmp_path):
        from evaluatorq.redteam.exceptions import DatasetError
        from evaluatorq.redteam.frameworks.owasp import evaluatorq_bridge

        bad = tmp_path / 'dataset.json'
        bad.write_text('{not valid json')

        with pytest.raises(DatasetError, match='Failed to load red team dataset') as ei:
            evaluatorq_bridge._load_from_file(bad)
        # __cause__ pins that this distinct except arm was the one hit (not a broadened catch).
        assert isinstance(ei.value.__cause__, json.JSONDecodeError)

    def test_missing_file_raises_dataset_error(self, tmp_path):
        from evaluatorq.redteam.exceptions import DatasetError
        from evaluatorq.redteam.frameworks.owasp import evaluatorq_bridge

        with pytest.raises(DatasetError, match='Failed to load red team dataset') as ei:
            evaluatorq_bridge._load_from_file(tmp_path / 'does_not_exist.json')
        assert isinstance(ei.value.__cause__, OSError)

    def test_wrong_shape_raises_dataset_error(self, tmp_path):
        from pydantic import ValidationError

        from evaluatorq.redteam.exceptions import DatasetError
        from evaluatorq.redteam.frameworks.owasp import evaluatorq_bridge

        wrong = tmp_path / 'dataset.json'
        wrong.write_text('{"not_samples": []}')

        with pytest.raises(DatasetError, match='Failed to load red team dataset') as ei:
            evaluatorq_bridge._load_from_file(wrong)
        assert isinstance(ei.value.__cause__, ValidationError)

    def test_unknown_vulnerability_raises_dataset_error(self, tmp_path):
        from evaluatorq.redteam.exceptions import DatasetError
        from evaluatorq.redteam.frameworks.owasp import evaluatorq_bridge

        row = {
            'input': {
                'id': 'typo-001',
                'vulnerability': 'goal_hijack',
                'severity': 'low',
                'vulnerability_domain': 'agent',
                'turn_type': 'single',
                'source': 'test',
            },
            'messages': [{'role': 'user', 'content': 'x'}],
        }
        path = tmp_path / 'dataset.json'
        path.write_text(json.dumps({'samples': [row]}))

        with pytest.raises(DatasetError, match="Unknown vulnerability 'goal_hijack'"):
            evaluatorq_bridge._load_from_file(path)


class TestMultiAgentGate:
    """ASI07/08 are skipped for single-agent targets and kept for multi-agent ones (gripe #5)."""

    def _plan(self, *, is_multi_agent: bool, vulns):
        from evaluatorq.contracts import AgentContext
        from evaluatorq.redteam.adaptive.strategy_planner import plan_strategies_for_vulnerabilities

        ctx = AgentContext(key='t', is_multi_agent=is_multi_agent)

        async def run():
            plans, _, _ = await plan_strategies_for_vulnerabilities(
                vulnerabilities=vulns,
                agent_context=ctx,
                attack_model='x',
                llm_client=None,
                max_turns=1,
                max_per_category=None,
                generate_additional_strategies=False,
                generated_strategy_count=0,
            )
            return plans

        return asyncio.run(run())

    def test_single_agent_drops_multi_agent_vulns(self):
        from evaluatorq.redteam.contracts import Vulnerability

        plans = self._plan(
            is_multi_agent=False,
            vulns=[
                Vulnerability.INTER_AGENT_COMMS,
                Vulnerability.CASCADING_FAILURES,
                Vulnerability.GOAL_HIJACKING,
            ],
        )
        assert Vulnerability.INTER_AGENT_COMMS not in plans
        assert Vulnerability.CASCADING_FAILURES not in plans
        assert Vulnerability.GOAL_HIJACKING in plans

    def test_multi_agent_keeps_multi_agent_vulns(self):
        from evaluatorq.redteam.contracts import Vulnerability

        plans = self._plan(is_multi_agent=True, vulns=[Vulnerability.INTER_AGENT_COMMS])
        assert Vulnerability.INTER_AGENT_COMMS in plans


class TestAttackerSelfCensor:
    """Only the structural content_filter signal marks an attacker turn unusable.

    Natural-language self-censorship is NOT detected (it is indistinguishable from a real
    attack at the protocol level) and is forwarded to the target — no keyword guessing.
    """

    def test_classify_content_filter(self):
        from evaluatorq.common.content_filter import classify_finish_reason

        assert classify_finish_reason('content_filter') == 'content_filter'

    @pytest.mark.parametrize(
        'finish_reason',
        ['stop', 'length', None, ''],
    )
    def test_non_filter_reasons_are_usable(self, finish_reason: str | None):
        # A prose refusal arrives as finish_reason='stop' and must NOT be flagged —
        # it is forwarded to the target rather than killing the attack.
        from evaluatorq.common.content_filter import classify_finish_reason

        assert classify_finish_reason(finish_reason) is None

    def test_retry_config_default(self):
        from evaluatorq.redteam.contracts import LLMConfig

        assert LLMConfig().max_content_filter_retries == 2
