"""Filesystem and protected-route boundaries for the local project folder browser."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import insights_routes
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.project_browser import list_project_directories


@pytest.fixture
def client() -> TestClient:
    return TestClient(build_app())


def _token(client: TestClient) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', client.get('/insights/new').text)
    assert match is not None
    return match.group(1)


def test_lists_hidden_and_symlink_directories_but_not_files(tmp_path: Path) -> None:
    (tmp_path / 'visible').mkdir()
    (tmp_path / '.hidden').mkdir()
    (tmp_path / 'file.txt').write_text('not read')
    target = tmp_path / 'target'
    target.mkdir()
    try:
        (tmp_path / 'linked').symlink_to(target, target_is_directory=True)
        (tmp_path / 'broken').symlink_to(tmp_path / 'missing', target_is_directory=True)
    except OSError as exc:
        pytest.skip(f'directory symlinks are unavailable: {exc}')

    result = list_project_directories(str(tmp_path))

    assert [entry['name'] for entry in result['directories']] == ['.hidden', 'linked', 'target', 'visible']
    assert all(entry['name'] != 'file.txt' and entry['name'] != 'broken' for entry in result['directories'])
    assert next(entry for entry in result['directories'] if entry['name'] == 'linked')['path'] == str(target.resolve())


def test_empty_directory_is_distinct_from_missing_directory(tmp_path: Path) -> None:
    empty = tmp_path / 'empty'
    empty.mkdir()

    assert list_project_directories(str(empty))['directories'] == []
    with pytest.raises(ValueError, match='could not be opened'):
        list_project_directories(str(tmp_path / 'missing'))


def test_blank_and_tilde_paths_resolve_to_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'child').mkdir()
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))

    assert list_project_directories()['path'] == str(tmp_path.resolve())
    assert list_project_directories('~/')['path'] == str(tmp_path.resolve())


def test_parent_and_breadcrumbs_follow_resolved_path(tmp_path: Path) -> None:
    nested = tmp_path / 'one' / 'two'
    nested.mkdir(parents=True)

    result = list_project_directories(str(nested))

    assert result['path'] == str(nested.resolve())
    assert result['parent'] == str(nested.parent.resolve())
    breadcrumb_paths = [*reversed(nested.resolve().parents), nested.resolve()]
    assert result['breadcrumbs'] == [
        {'name': path.name or str(path), 'path': str(path)} for path in breadcrumb_paths
    ]
    root = list_project_directories(nested.resolve().anchor)
    assert root['parent'] is None


def test_page_offsets_cover_sorted_directories_without_overlap(tmp_path: Path) -> None:
    names = [f'Project {index:03}' for index in range(405)]
    for name in names:
        (tmp_path / name).mkdir()

    first = list_project_directories(str(tmp_path))
    second = list_project_directories(str(tmp_path), offset=200)
    third = list_project_directories(str(tmp_path), offset=400)
    combined = [entry['name'] for page in (first, second, third) for entry in page['directories']]

    assert [len(page['directories']) for page in (first, second, third)] == [200, 200, 5]
    assert first['has_more'] is True and second['has_more'] is True and third['has_more'] is False
    assert combined == sorted(names, key=lambda name: (name.casefold(), name))
    assert len(combined) == len(set(combined))


def test_invalid_paths_and_offsets_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='absolute directory path'):
        list_project_directories('relative/path')
    file_path = tmp_path / 'file'
    file_path.write_text('')
    with pytest.raises(ValueError, match='could not be opened'):
        list_project_directories(str(file_path))
    with pytest.raises(ValueError, match='valid directory path'):
        list_project_directories('\x00')
    invalid_offsets: tuple[Any, ...] = (-1, True, 1.5)
    for invalid_offset in invalid_offsets:
        with pytest.raises(ValueError, match='nonnegative integer'):
            list_project_directories(str(tmp_path), offset=invalid_offset)


@pytest.mark.parametrize('name', [' leading', 'trailing '])
def test_browsing_preserves_directory_name_whitespace(
    client: TestClient, tmp_path: Path, name: str
) -> None:
    if sys.platform == 'win32' and name.endswith(' '):
        pytest.skip('Windows normalizes trailing spaces in ordinary filesystem paths.')
    directory = tmp_path / name
    directory.mkdir()
    (directory / 'child').mkdir()

    response = client.post(
        '/insights/sessions/directories',
        data={'csrf': _token(client), 'path': str(directory)},
    )

    assert response.status_code == 200
    assert response.json()['path'] == str(directory.resolve())
    assert response.json()['directories'] == [
        {'name': 'child', 'path': str((directory / 'child').resolve())},
    ]


def test_directory_results_keep_untrusted_names_as_plain_data(tmp_path: Path) -> None:
    name = "folder & 'quoted'"
    (tmp_path / name).mkdir()

    result = list_project_directories(str(tmp_path))

    assert result['directories'] == [{'name': name, 'path': str((tmp_path / name).resolve())}]


def test_directory_route_rejects_unprotected_requests_before_filesystem_access(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError('filesystem helper must not be called for an unprotected request')

    monkeypatch.setattr(insights_routes, 'list_project_directories', unexpected_call)

    missing = client.post('/insights/sessions/directories', data={'path': '/'})
    stale = client.post('/insights/sessions/directories', data={'csrf': 'stale', 'path': '/'})
    cross_origin = client.post(
        '/insights/sessions/directories',
        data={'csrf': _token(client), 'path': '/'},
        headers={'sec-fetch-site': 'cross-site'},
    )

    assert missing.status_code == stale.status_code == cross_origin.status_code == 403


def test_directory_route_returns_the_requested_listing(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / 'folder').mkdir()
    (tmp_path / "folder & 'quoted'").mkdir()
    (tmp_path / 'private.txt').write_text('not a directory')

    response = client.post(
        '/insights/sessions/directories',
        data={'csrf': _token(client), 'path': str(tmp_path), 'offset': '0'},
    )

    assert response.status_code == 200
    assert response.json()['path'] == str(tmp_path.resolve())
    assert response.json()['directories'] == [
        {'name': 'folder', 'path': str((tmp_path / 'folder').resolve())},
        {'name': "folder & 'quoted'", 'path': str((tmp_path / "folder & 'quoted'").resolve())},
    ]


def test_directory_route_returns_clean_invalid_input_errors(client: TestClient, tmp_path: Path) -> None:
    token = _token(client)

    invalid_path = client.post(
        '/insights/sessions/directories', data={'csrf': token, 'path': 'relative'}
    )
    invalid_offset = client.post(
        '/insights/sessions/directories', data={'csrf': token, 'path': str(tmp_path), 'offset': '-1'}
    )

    assert invalid_path.status_code == invalid_offset.status_code == 422
    assert set(invalid_path.json()) == set(invalid_offset.json()) == {'error'}
