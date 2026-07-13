"""Provisioning config for the Bank of Holland credit-card demo agent.

Keys are deliberately distinct from the customer's live entities so `provision.py`
can delete-then-create idempotently without ever touching the originals. Override
the path/keys via env if you want to provision into a different folder.
"""

from __future__ import annotations

import os

# Folder path in the Orq workspace for all provisioned entities.
ORQ_PATH = os.environ.get('BOH_DEMO_PATH', 'AgentSimulation/webinar-demo')

# Entity keys (workspace-unique). Suffixed to avoid clobbering the source agent.
AGENT_KEY = 'boh_creditcard_agent_demo'
KB_KEY = 'boh_faq_kb_demo'
TOOL_KEYS = {
    # our demo key -> exported definition file under orq_export/tools/
    'boh_get_card_info': 'GetCardInfo.json',
    'boh_get_transaction_details': 'get_transaction_details2.json',
}

EMBEDDING_MODEL = 'cohere/embed-multilingual-v3.0'
FAQ_FILE = 'assets/boh_faq.txt'
