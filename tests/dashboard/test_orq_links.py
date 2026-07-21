import pytest

from evaluatorq.dashboard.orq_links import orq_studio_url, parse_experiment_url


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Clear workspace/host env so the fallback path is deterministic."""
    for var in ('ORQ_WORKSPACE', 'ORQ_WORKSPACE_SLUG', 'ORQ_BASE_URL'):
        monkeypatch.delenv(var, raising=False)


# --- experiment_url parsing -------------------------------------------------


def test_parse_experiment_url_extracts_host_and_workspace():
    host, ws = parse_experiment_url('https://my.orq.ai/orq-research/experiments/01KX?runId=01YY')
    assert host == 'https://my.orq.ai'
    assert ws == 'orq-research'


@pytest.mark.parametrize(
    'url',
    [None, '', 'not-a-url', 'https://my.orq.ai/orq-research/agents/x', 'https://my.orq.ai/experiments/x'],
)
def test_parse_experiment_url_rejects_non_experiment(url):
    assert parse_experiment_url(url) == (None, None)


def test_experiment_url_drives_entity_link():
    # host + workspace come from the run's own experiment_url, ignoring env.
    url = orq_studio_url(
        target_kind='agent',
        entity_id='abc123',
        experiment_url='https://staging.orq.ai/team-x/experiments/01KX',
    )
    assert url == 'https://staging.orq.ai/team-x/agents/abc123'


def test_agent_url(monkeypatch):
    monkeypatch.delenv('ORQ_WORKSPACE', raising=False)
    url = orq_studio_url(
        target_kind='agent',
        entity_id='abc123',
        workspace_id='ws1',
        base_url='https://my.orq.ai',
    )
    assert url == 'https://my.orq.ai/ws1/agents/abc123'


def test_deployment_url(monkeypatch):
    monkeypatch.delenv('ORQ_WORKSPACE', raising=False)
    url = orq_studio_url(
        target_kind='deployment',
        entity_id='dep9',
        workspace_id='ws1',
        base_url='https://my.orq.ai/',  # trailing slash tolerated
    )
    assert url == 'https://my.orq.ai/ws1/deployments/dep9'


def test_non_orq_target_returns_none(monkeypatch):
    monkeypatch.delenv('ORQ_WORKSPACE', raising=False)
    assert (
        orq_studio_url(
            target_kind='openai',
            entity_id='gpt',
            workspace_id='ws1',
            base_url='https://my.orq.ai',
        )
        is None
    )


def test_missing_field_returns_none(monkeypatch):
    monkeypatch.delenv('ORQ_WORKSPACE', raising=False)
    assert (
        orq_studio_url(
            target_kind='agent',
            entity_id=None,
            workspace_id='ws1',
            base_url='https://my.orq.ai',
        )
        is None
    )


def test_workspace_uuid_is_not_a_studio_route_key(monkeypatch):
    monkeypatch.delenv('ORQ_WORKSPACE', raising=False)
    assert (
        orq_studio_url(
            target_kind='agent',
            entity_id='abc123',
            workspace_id='624ccbbd-a482-40e2-b3d9-3621e09da1f8',
            base_url='https://my.orq.ai',
        )
        is None
    )


def test_configured_workspace_key_overrides_entity_workspace_id(monkeypatch):
    monkeypatch.setenv('ORQ_WORKSPACE', 'research')

    assert (
        orq_studio_url(
            target_kind='agent',
            entity_id='abc123',
            workspace_id='624ccbbd-a482-40e2-b3d9-3621e09da1f8',
            base_url='https://my.orq.ai',
        )
        == 'https://my.orq.ai/research/agents/abc123'
    )
