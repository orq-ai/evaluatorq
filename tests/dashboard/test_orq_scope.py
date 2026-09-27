"""Credential-scoped Orq workspace and project discovery."""

from __future__ import annotations

import subprocess

from evaluatorq.dashboard import orq_scope


def test_project_key_discovers_only_its_project_and_matching_workspace(monkeypatch) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'project-key')
    calls: list[list[str]] = []

    def cli(args: list[str], *, profile: str | None, timeout: float) -> dict:
        calls.append(args)
        assert profile is None
        if args[:2] == ['projects', 'list']:
            return {'data': [
                {'project_id': 'project-bauke', 'name': 'Bauke', 'workspace_id': 'research-id'},
            ], 'has_more': False}
        return {
            'credential': {'scope': 'project', 'workspace_id': 'research-id'},
            'workspaces': [{'id': 'research-id', 'key': 'orq-research'}],
        }

    monkeypatch.setattr(orq_scope, '_cli_json', cli)
    scope = orq_scope.discover_orq_scope()

    assert scope.workspace_key == 'orq-research'
    assert scope.projects == (orq_scope.OrqProject('project-bauke', 'Bauke', 'research-id'),)
    assert len(calls) == 2


def test_profile_project_list_is_paged_and_workspace_comes_from_profile(monkeypatch) -> None:
    calls: list[list[str]] = []

    def cli(args: list[str], *, profile: str | None, timeout: float) -> dict:
        assert profile == 'research-management'
        calls.append(args)
        if args[:2] == ['workspaces', 'list']:
            return {'data': [{'id': 'research-id', 'key': 'orq-research'}], 'has_more': False}
        if '--starting-after' in args:
            return {'data': [{'project_id': 'project-b', 'name': 'B', 'workspace_id': 'research-id'}], 'has_more': False}
        return {'data': [{'project_id': 'project-a', 'name': 'A', 'workspace_id': 'research-id'}], 'has_more': True}

    monkeypatch.setattr(orq_scope, '_cli_json', cli)
    scope = orq_scope.discover_orq_scope('research-management')

    assert [project.id for project in scope.projects] == ['project-a', 'project-b']
    assert scope.workspace_key == 'orq-research'
    assert any('--starting-after' in args for args in calls)


def test_scope_rejects_session_workspace_mismatch(monkeypatch) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'project-key')

    def cli(args: list[str], *, profile: str | None, timeout: float) -> dict:
        if args[:2] == ['projects', 'list']:
            return {'data': [{'project_id': 'project-a', 'name': 'A', 'workspace_id': 'workspace-a'}]}
        return {'credential': {'workspace_id': 'workspace-b'}, 'workspaces': []}

    monkeypatch.setattr(orq_scope, '_cli_json', cli)
    scope = orq_scope.discover_orq_scope()

    assert scope.error is not None
    assert scope.projects == ()


def test_profile_cli_call_drops_unrelated_environment_credential(monkeypatch) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'other-key')
    monkeypatch.setenv('ORQ_BASE_URL', 'https://other.orq.ai')
    monkeypatch.setenv('ORQ_SERVER', 'https://other.orq.ai')
    monkeypatch.setattr(orq_scope.shutil, 'which', lambda _: '/usr/bin/orq')

    def run(command, **kwargs):
        assert command[:3] == ['/usr/bin/orq', '--profile', 'research']
        assert 'ORQ_API_KEY' not in kwargs['env']
        assert 'ORQ_SERVER' not in kwargs['env']
        return subprocess.CompletedProcess(command, 0, '{"data": []}', '')

    monkeypatch.setattr(orq_scope.subprocess, 'run', run)

    assert orq_scope._cli_json(['projects', 'list'], profile='research', timeout=5) == {'data': []}


def test_direct_key_uses_evaluatorq_host_and_clears_scope_overrides(monkeypatch) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'project-key')
    monkeypatch.delenv('ORQ_BASE_URL', raising=False)
    monkeypatch.setenv('ORQ_SERVER', 'https://staging.orq.ai')
    monkeypatch.setenv('ORQ_WORKSPACE', 'wrong-workspace')
    monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'wrong-slug')
    monkeypatch.setenv('ORQ_PROJECT', 'wrong-project')
    monkeypatch.setattr(orq_scope.shutil, 'which', lambda _: '/usr/bin/orq')

    def run(command, **kwargs):
        environment = kwargs['env']
        assert environment['ORQ_SERVER'] == 'https://my.orq.ai'
        assert all(name not in environment for name in ('ORQ_WORKSPACE', 'ORQ_WORKSPACE_SLUG', 'ORQ_PROJECT'))
        return subprocess.CompletedProcess(command, 0, '{"data": []}', '')

    monkeypatch.setattr(orq_scope.subprocess, 'run', run)

    assert orq_scope._cli_json(['projects', 'list'], profile=None, timeout=5) == {'data': []}


def test_profile_workspace_lookup_reads_later_pages(monkeypatch) -> None:
    calls: list[list[str]] = []

    def cli(args: list[str], *, profile: str | None, timeout: float) -> dict:
        calls.append(args)
        if args[:2] == ['projects', 'list']:
            return {'data': [{'project_id': 'project-a', 'name': 'A', 'workspace_id': 'workspace-b'}]}
        if '--starting-after' in args:
            return {'data': [{'id': 'workspace-b', 'key': 'target'}], 'has_more': False}
        return {'data': [{'id': 'workspace-a', 'key': 'other'}], 'has_more': True}

    monkeypatch.setattr(orq_scope, '_cli_json', cli)

    scope = orq_scope.discover_orq_scope('research')

    assert scope.workspace_key == 'target'
    assert any('--starting-after' in args for args in calls)


def test_scope_discovery_has_one_deadline_across_project_and_workspace_pages(monkeypatch) -> None:
    times = iter((0.0, 0.0, 5.0, 11.0, 16.0))
    monkeypatch.setattr(orq_scope, 'monotonic', lambda: next(times))
    calls: list[tuple[list[str], float]] = []

    def cli(args: list[str], *, profile: str | None, timeout: float) -> dict:
        calls.append((args, timeout))
        if args[:2] == ['projects', 'list']:
            return {'data': [{'project_id': 'project-a', 'name': 'A', 'workspace_id': 'workspace-target'}]}
        return {'data': [{'id': f'workspace-{len(calls)}', 'key': 'other'}], 'has_more': True}

    monkeypatch.setattr(orq_scope, '_cli_json', cli)
    scope = orq_scope.discover_orq_scope('research', timeout=5)

    assert scope.error == 'Orq scope discovery timed out.'
    assert len(calls) == 3
    assert [timeout for _, timeout in calls] == [5, 5, 4]
