from types import SimpleNamespace

from evaluatorq.common.orq_client import create_orq_dataset


def test_create_orq_dataset_uses_request_dict_on_old_sdk():
    seen = {}

    def create(request, retries=None):
        seen['request'] = request
        return 'ds'

    client = SimpleNamespace(datasets=SimpleNamespace(create=create))
    assert create_orq_dataset(client, display_name='n', path='p') == 'ds'
    assert seen['request'] == {'display_name': 'n', 'path': 'p'}


def test_create_orq_dataset_uses_keywords_on_new_sdk():
    seen = {}

    def create(*, display_name, path=None, retries=None):
        seen.update(display_name=display_name, path=path)
        return 'ds'

    client = SimpleNamespace(datasets=SimpleNamespace(create=create))
    assert create_orq_dataset(client, display_name='n', path='p') == 'ds'
    assert seen == {'display_name': 'n', 'path': 'p'}
