from evaluatorq.dashboard.orq_links import orq_studio_url


def test_agent_url():
    url = orq_studio_url(
        target_kind='agent', entity_id='abc123', workspace_id='ws1',
        base_url='https://my.orq.ai',
    )
    assert url == 'https://my.orq.ai/ws1/agents/abc123'


def test_deployment_url():
    url = orq_studio_url(
        target_kind='deployment', entity_id='dep9', workspace_id='ws1',
        base_url='https://my.orq.ai/',  # trailing slash tolerated
    )
    assert url == 'https://my.orq.ai/ws1/deployments/dep9'


def test_non_orq_target_returns_none():
    assert orq_studio_url(
        target_kind='openai', entity_id='gpt', workspace_id='ws1',
        base_url='https://my.orq.ai',
    ) is None


def test_missing_field_returns_none():
    assert orq_studio_url(
        target_kind='agent', entity_id=None, workspace_id='ws1',
        base_url='https://my.orq.ai',
    ) is None
