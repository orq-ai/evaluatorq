from __future__ import annotations

import stat
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from evaluatorq.backends.coding_agent_cli import BUILD_DIR, app
from evaluatorq.backends.container import DEFAULT_CODING_AGENT_IMAGE


def test_build_image_argv(tmp_path: Path) -> None:
    log = tmp_path / 'log'
    fake = tmp_path / 'docker'
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" > {log}\nexit 3\n')
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    result = CliRunner().invoke(
        app, ['build-image', '--binary', str(fake), '--context', 'orbstack', '--build-arg', 'CODEX_VERSION=1.2.3']
    )
    assert result.exit_code == 3
    assert log.read_text().strip() == (
        f'--context orbstack build -t {DEFAULT_CODING_AGENT_IMAGE} --build-arg CODEX_VERSION=1.2.3 {BUILD_DIR}'
    )


def test_build_dir_ships_both_files() -> None:
    assert (BUILD_DIR / 'Dockerfile').is_file() and (BUILD_DIR / 'entrypoint.sh').is_file()


def test_entrypoint_is_idempotent_and_execs(tmp_path: Path) -> None:
    passwd = tmp_path / 'passwd'
    passwd.write_text('root:x:0:0::/root:/bin/sh\n')
    script = (BUILD_DIR / 'entrypoint.sh').read_text().replace('/etc/passwd', str(passwd))
    for _ in range(2):
        out = subprocess.run(
            ['sh', '-c', script, 'evq-entrypoint', 'echo', 'ran'],
            capture_output=True,
            text=True,
            env={'HOME': '/evq-home', 'PATH': '/usr/bin:/bin'},
        )
        assert out.stdout.strip() == 'ran'
    assert passwd.read_text().count('\n') == 1
