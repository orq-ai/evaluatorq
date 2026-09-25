"""The dashboard's profile selector reads the orq CLI's credential profiles through the CLI itself."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from evaluatorq.common.orq_client import OrqProfile, list_orq_profiles

_LISTING = {
    'message': '',
    'profiles': [
        {'active': False, 'api_key': 'key-a', 'name': 'alpha', 'server': 'https://a.orq.ai', 'type': ''},
        {'active': True, 'api_key': 'key-b', 'name': 'beta', 'type': ''},
        {'active': False, 'api_key': '', 'name': 'device-login', 'type': 'oauth'},
        'not a row',
    ],
}


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr='')


def test_profiles_come_from_the_cli_listing_and_skip_rows_without_a_key() -> None:
    with (
        patch('evaluatorq.common.orq_client.shutil.which', return_value='/usr/bin/orq'),
        patch('evaluatorq.common.orq_client.subprocess.run', return_value=_completed(json.dumps(_LISTING))) as run,
    ):
        profiles = list_orq_profiles()

    assert profiles == (
        OrqProfile('alpha', 'key-a', 'https://a.orq.ai', False),
        OrqProfile('beta', 'key-b', None, True),
    )
    assert run.call_args.args[0] == ['/usr/bin/orq', 'auth', 'profile', 'list', '-o', 'json', '--no-input']


def test_no_cli_means_no_profiles() -> None:
    with patch('evaluatorq.common.orq_client.shutil.which', return_value=None):
        assert list_orq_profiles() == ()


def test_masked_profile_without_stored_key_is_skipped() -> None:
    listing = {'profiles': [{'name': 'research', 'api_key': 'key-****', 'active': True}]}
    with (
        patch('evaluatorq.common.orq_client.shutil.which', return_value='/usr/bin/orq'),
        patch('evaluatorq.common.orq_client.subprocess.run', return_value=_completed(json.dumps(listing))),
        patch('evaluatorq.common.orq_client._stored_profile_keys', return_value={}),
    ):
        assert list_orq_profiles() == ()


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
def test_a_failing_cli_means_no_profiles(
    outcome: subprocess.CompletedProcess[str] | BaseException,
) -> None:
    with (
        patch('evaluatorq.common.orq_client.shutil.which', return_value='/usr/bin/orq'),
        patch('evaluatorq.common.orq_client.subprocess.run') as run,
    ):
        if isinstance(outcome, BaseException):
            run.side_effect = outcome
        else:
            run.return_value = outcome
        assert list_orq_profiles() == ()
