"""The CLI masks profile keys, while dashboard SDK calls need the real key."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from evaluatorq.common import orq_client


def test_masked_cli_profile_uses_private_local_key(tmp_path: Path, monkeypatch) -> None:
    credential_file = tmp_path / '.orq' / 'credentials.json'
    credential_file.parent.mkdir()
    credential_file.write_text(json.dumps({'profiles': {'research': {'api_key': 'key-research'}}}))
    credential_file.chmod(0o600)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(orq_client.shutil, 'which', lambda _: '/usr/bin/orq')
    monkeypatch.setattr(orq_client.subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 0, json.dumps({'profiles': [{'name': 'research', 'api_key': 'key-****', 'active': False}]}), ''
    ))

    profiles = orq_client.list_orq_profiles()

    assert [(profile.name, profile.api_key) for profile in profiles] == [('research', 'key-research')]


def test_loose_credential_permissions_never_expose_a_key(tmp_path: Path) -> None:
    credential_file = tmp_path / 'credentials.json'
    credential_file.write_text(json.dumps({'profiles': {'research': {'api_key': 'key-research'}}}))
    credential_file.chmod(0o644)

    assert orq_client._stored_profile_keys(credential_file) == {}
