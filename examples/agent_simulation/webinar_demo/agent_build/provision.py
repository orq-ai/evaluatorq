#!/usr/bin/env python3
"""Provision the Bank of Holland credit-card demo agent + its tools + knowledge base in Orq.

Reads the definitions exported from the platform under ``orq_export/`` and the FAQ
under ``assets/`` and (re)creates them via the Orq Python SDK. Idempotent: any
existing entity with our demo key is deleted first, so re-running gives a clean
copy. Prints the provisioned agent key for the runbook to target.

    export ORQ_API_KEY=...        # the workspace that hosts the demo
    uv run python provision.py    # or: make provision
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import config
import httpx
from dotenv import load_dotenv
from loguru import logger
from orq_ai_sdk import Orq
from orq_ai_sdk.models import CodeExecutionTool, CreateAgentRequestSettings, QueryKnowledgeBaseTool

HERE = Path(__file__).parent
EXPORT = HERE / 'orq_export'


def _load_json(rel: str) -> dict[str, Any]:
    return json.loads((EXPORT / rel).read_text(encoding='utf-8'))


def _ids_by_key(base: str, api_key: str, resource: str) -> dict[str, str]:
    """Map key -> id for a resource collection.

    The SDK's ``list`` responses omit the object id for tools/knowledge, so we
    hit the REST endpoint directly to resolve ids for idempotent deletes.
    """
    resp = httpx.get(
        f'{base}/v2/{resource}',
        headers={'Authorization': f'Bearer {api_key}'},
        params={'limit': 100},
        timeout=30.0,
    )
    resp.raise_for_status()
    rows = resp.json().get('data', [])
    return {r['key']: (r.get('_id') or r.get('id')) for r in rows if r.get('key')}


def provision_tools(client: Orq, tool_ids: dict[str, str]) -> list[str]:
    provisioned: list[str] = []
    for demo_key, export_file in config.TOOL_KEYS.items():
        if demo_key in tool_ids:
            client.tools.delete(tool_id=tool_ids[demo_key])
            logger.info(f'Deleted existing tool {demo_key} ({tool_ids[demo_key]})')
        src = _load_json(f'tools/{export_file}')
        client.tools.create(
            request={
                'type': 'code',
                'key': demo_key,
                'path': config.ORQ_PATH,
                'display_name': src.get('display_name') or demo_key,
                'description': src.get('description') or '',
                'code_tool': src['code_tool'],  # {language, code, parameters}
            }
        )
        logger.info(f'Created tool {demo_key}')
        provisioned.append(demo_key)
    return provisioned


def provision_kb(client: Orq, kb_ids: dict[str, str]) -> str:
    if config.KB_KEY in kb_ids:
        client.knowledge.delete(knowledge_id=kb_ids[config.KB_KEY])
        logger.info(f'Deleted existing knowledge base {config.KB_KEY} ({kb_ids[config.KB_KEY]})')
    kb_meta = _load_json('kb/knowledge.json')
    kb = client.knowledge.create(
        request={
            'key': config.KB_KEY,
            'path': config.ORQ_PATH,
            'embedding_model': config.EMBEDDING_MODEL,
            'retrieval_settings': kb_meta.get('retrieval_settings')
            or {'retrieval_type': 'hybrid_search', 'top_k': 5, 'threshold': 0},
        }
    )
    logger.info(f'Created knowledge base {config.KB_KEY} ({kb.id})')

    datasource = client.knowledge.create_datasource(knowledge_id=kb.id, display_name='boh_faq')
    faq = (HERE / config.FAQ_FILE).read_text(encoding='utf-8')
    # Chunk on the FAQ's own "---" separators (one Q&A / section per chunk).
    chunks = [c.strip() for c in faq.split('\n---\n') if c.strip()]
    for i in range(0, len(chunks), 100):  # API caps create_chunks at 100 items/call
        client.knowledge.create_chunks(
            knowledge_id=kb.id,
            datasource_id=datasource.id,
            request_body=[{'text': c} for c in chunks[i : i + 100]],
        )
    logger.info(f'Ingested {len(chunks)} chunk(s) from {config.FAQ_FILE}')
    return kb.id


def provision_agent(client: Orq, tool_keys: list[str], kb_id: str) -> str:
    try:
        client.agents.delete(agent_key=config.AGENT_KEY)
        logger.info(f'Deleted existing agent {config.AGENT_KEY}')
    except Exception:
        logger.debug(f'No existing agent {config.AGENT_KEY} to delete')

    agent = _load_json('agent_creditcard_full.json')
    settings = CreateAgentRequestSettings(
        tools=[
            *[CodeExecutionTool(key=k, type='code') for k in tool_keys],
            QueryKnowledgeBaseTool(type='query_knowledge_base'),
        ]
    )
    client.agents.create(
        key=config.AGENT_KEY,
        role=agent.get('role') or 'Bank of Holland Creditcard Support Bot',
        description=agent.get('description') or 'Bank of Holland credit-card support demo agent.',
        instructions=agent['instructions'],
        path=config.ORQ_PATH,
        model={'id': (agent.get('model') or {}).get('id', 'azure/gpt-5-mini')},
        settings=settings,
        knowledge_bases=[{'knowledge_id': kb_id}],
    )
    logger.info(f'Created agent {config.AGENT_KEY}')
    return config.AGENT_KEY


def main() -> None:
    load_dotenv()
    api_key = os.environ.get('ORQ_API_KEY')
    if not api_key:
        raise SystemExit('ORQ_API_KEY is required (export it or put it in .env).')

    base = os.environ.get('ORQ_BASE_URL', 'https://my.orq.ai').rstrip('/')
    client = Orq(api_key=api_key, server_url=base)

    tool_keys = provision_tools(client, _ids_by_key(base, api_key, 'tools'))
    kb_id = provision_kb(client, _ids_by_key(base, api_key, 'knowledge'))
    agent_key = provision_agent(client, tool_keys, kb_id)

    logger.success(f'Provisioned. Target it with: --target agent:{agent_key}')


if __name__ == '__main__':
    main()
