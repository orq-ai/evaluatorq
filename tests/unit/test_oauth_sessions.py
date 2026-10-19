"""The dashboard's CLI OAuth session picker lists saved logins and checks the expired ones through the orq CLI."""

from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import patch

import pytest

from evaluatorq.common.cli_oauth import OAuthSession, list_oauth_sessions


def _completed(stdout: str = '', returncode: int = 0, stderr: str = '') -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _row(host: str, status: str, **extra: Any) -> dict[str, Any]:
    return {'host': host, 'server': f'https://{host}', 'status': status, **extra}


def _fake_cli(listing: dict[str, Any], whoami: dict[str, subprocess.CompletedProcess[str] | BaseException]):
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert 'ORQ_API_KEY' not in kwargs['env']
        if command[-1] == 'sessions':
            return _completed(json.dumps(listing))
        outcome = whoami[command[command.index('--server') + 1]]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return run, calls


_SIGNED_IN = _completed(json.dumps({'authenticated': True, 'user_id': 'u1'}))
_REVOKED = _completed(
    returncode=1,
    stderr="Error: Invalid refresh token!\nYour login for https://x has expired or was revoked — run 'orq auth login'.",
)


def test_only_expired_logins_are_checked_and_each_gets_its_own_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'would-hide-the-oauth-login')
    listing = {
        'sessions': [
            _row('my.orq.ai', 'ok', user='ada@orq.ai', workspace='research', active=True),
            _row('refreshed.orq.ai', 'needs-refresh'),
            _row('revoked.orq.ai', 'needs-refresh'),
            _row('offline.orq.ai', 'needs-refresh'),
            _row('slow.orq.ai', 'needs-refresh'),
            {'host': 'no-server.orq.ai', 'status': 'ok'},
            {'host': 'broken.orq.ai', 'status': 'invalid', 'active': False},
            {'host': 'locked.orq.ai', 'status': 'unreadable', 'active': True},
            'not a row',
        ]
    }
    run, calls = _fake_cli(listing, {
        'https://refreshed.orq.ai': _SIGNED_IN,
        'https://revoked.orq.ai': _REVOKED,
        'https://offline.orq.ai': _completed(returncode=1, stderr='Error: dial tcp: lookup offline.orq.ai: no such host'),
        'https://slow.orq.ai': subprocess.TimeoutExpired(cmd='orq', timeout=5),
    })
    with (
        patch('evaluatorq.common.cli_oauth.shutil.which', return_value='/usr/bin/orq'),
        patch('evaluatorq.common.cli_oauth.subprocess.run', side_effect=run),
    ):
        sessions = list_oauth_sessions()

    assert sessions == (
        OAuthSession('https://my.orq.ai', 'my.orq.ai', 'ada@orq.ai', 'research', 'valid', True),
        OAuthSession('https://refreshed.orq.ai', 'refreshed.orq.ai', None, None, 'valid', False),
        OAuthSession('https://revoked.orq.ai', 'revoked.orq.ai', None, None, 'signed-out', False),
        OAuthSession('https://offline.orq.ai', 'offline.orq.ai', None, None, 'unknown', False),
        OAuthSession('https://slow.orq.ai', 'slow.orq.ai', None, None, 'unknown', False),
        OAuthSession('', 'broken.orq.ai', None, None, 'unreadable', False),
        OAuthSession('', 'locked.orq.ai', None, None, 'unreadable', True),
    )
    checked = sorted(command[command.index('--server') + 1] for command in calls if command[-1] == 'whoami')
    assert 'https://my.orq.ai' not in checked
    assert len(checked) == 4


def test_no_cli_means_no_sessions() -> None:
    with patch('evaluatorq.common.cli_oauth.shutil.which', return_value=None):
        assert list_oauth_sessions() == ()


@pytest.mark.parametrize(
    'outcome',
    [
        _completed('', returncode=1),
        _completed('not json'),
        _completed(json.dumps({'unexpected': 'shape'})),
        subprocess.TimeoutExpired(cmd='orq', timeout=5),
        OSError('boom'),
    ],
)
def test_a_failing_cli_listing_means_no_sessions(outcome: subprocess.CompletedProcess[str] | BaseException) -> None:
    with (
        patch('evaluatorq.common.cli_oauth.shutil.which', return_value='/usr/bin/orq'),
        patch('evaluatorq.common.cli_oauth.subprocess.run') as run,
    ):
        if isinstance(outcome, BaseException):
            run.side_effect = outcome
        else:
            run.return_value = outcome
        assert list_oauth_sessions() == ()
