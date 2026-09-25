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
        app,
        [
            'build-image',
            '--binary',
            str(fake),
            '--context',
            'orbstack',
            '--tag',
            'custom:tag',
            '--build-arg',
            'CODEX_VERSION=1.2.3',
            '--build-arg',
            'CLAUDE_CODE_VERSION=2.1.282',
        ],
    )
    assert result.exit_code == 3
    assert log.read_text().strip() == (
        f'--context orbstack build -t custom:tag --build-arg CODEX_VERSION=1.2.3 '
        f'--build-arg CLAUDE_CODE_VERSION=2.1.282 {BUILD_DIR}'
    )
    default_result = CliRunner().invoke(app, ['build-image', '--binary', str(fake)])
    assert default_result.exit_code == 3
    assert log.read_text().strip() == f'build -t {DEFAULT_CODING_AGENT_IMAGE} {BUILD_DIR}'


def test_build_image_missing_binary(tmp_path: Path) -> None:
    missing = str(tmp_path / 'missing-docker')
    result = CliRunner().invoke(app, ['build-image', '--binary', missing])
    assert result.exit_code == 2
    output = ' '.join(result.output.split())
    assert 'Invalid value for --binary' in output
    assert 'not' in output and 'found on PATH' in output


def test_build_dir_ships_both_files() -> None:
    assert (BUILD_DIR / 'Dockerfile').is_file() and (BUILD_DIR / 'entrypoint.sh').is_file()


def test_entrypoint_is_idempotent_and_execs(tmp_path: Path) -> None:
    passwd = tmp_path / 'passwd'
    passwd.write_text('root:x:0:0::/root:/bin/sh\n')
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_id = fake_bin / 'id'
    fake_id.write_text(
        '#!/bin/sh\n'
        'case "$1" in\n'
        f"  -un) if grep -q '^evq:x:4242:4242:' '{passwd}'; then echo evq; else exit 1; fi ;;\n"
        '  -u) echo 4242 ;;\n'
        '  -g) echo 4242 ;;\n'
        'esac\n'
    )
    fake_id.chmod(fake_id.stat().st_mode | stat.S_IXUSR)
    script = (BUILD_DIR / 'entrypoint.sh').read_text().replace('/etc/passwd', str(passwd))
    for _ in range(2):
        out = subprocess.run(
            ['sh', '-c', script, 'evq-entrypoint', 'echo', 'ran'],
            capture_output=True,
            text=True,
            env={'HOME': '/evq-home', 'PATH': f'{fake_bin}:/usr/bin:/bin'},
        )
        assert out.stdout.strip() == 'ran'
    assert passwd.read_text().count('evq:x:4242:4242:') == 1
    assert passwd.read_text().count('\n') == 2
