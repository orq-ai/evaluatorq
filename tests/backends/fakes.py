"""Fake CLIs written in Python, so the backend tests spawn the same stand-in on Linux, macOS and Windows."""

from __future__ import annotations

import stat
import sys
from pathlib import Path


def install_fake(directory: Path, name: str, source: str) -> Path:
    """Write ``source`` as a runnable fake named ``name`` in ``directory`` and return its launcher path.

    POSIX gets an executable ``name`` that execs the current interpreter; Windows gets ``name.cmd``, which
    is what a PATH lookup finds there for an npm-installed CLI too.
    """
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / f'{name}.py'
    script.write_text(source, encoding='utf-8')
    if sys.platform == 'win32':
        launcher = directory / f'{name}.cmd'
        launcher.write_text(f'@"{sys.executable}" "%~dp0{name}.py" %*\r\n', encoding='utf-8')
    else:
        launcher = directory / name
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding='utf-8')
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    return launcher


# Echoes $FAKE_STDOUT and exits with $FAKE_EXIT, after draining the prompt from stdin.
ECHO = """\
import os, sys
sys.stdin.read()
if os.environ.get('FAKE_STDOUT'):
    with open(os.environ['FAKE_STDOUT'], 'rb') as f:
        sys.stdout.buffer.write(f.read())
sys.exit(int(os.environ.get('FAKE_EXIT', '0')))
"""

# Starts a long-lived child, publishes its pid in $FAKE_PIDFILE, and waits on it.
SLEEPER = """\
import os, subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
tmp = os.environ['FAKE_PIDFILE'] + '.tmp'
with open(tmp, 'w') as f:
    f.write(str(child.pid))
os.replace(tmp, os.environ['FAKE_PIDFILE'])
child.wait()
"""
