"""Terminal-level tests for interrupting shared confirmation prompts."""
from __future__ import annotations

import os
import select
import signal
import sys
import time

import pytest


@pytest.mark.skipif(os.name != 'posix', reason='requires a POSIX pseudo-terminal')
def test_ctrl_c_exits_confirmation_without_waiting_for_enter() -> None:
    """Ctrl-C at the shared run-plan prompt exits while its stdin read is pending."""
    import pty

    program = """
import asyncio
import signal
import sys
from rich.console import Console
from evaluatorq.common.reports import confirm_run_plan

async def main():
    await confirm_run_plan(
        Console(file=sys.stderr, force_terminal=False),
        title='Run Plan',
        rows=[('Target', 'example')],
        prompt='Proceed?',
        skip_confirm=False,
    )

signal.signal(signal.SIGINT, signal.default_int_handler)
try:
    asyncio.run(main())
except KeyboardInterrupt:
    raise SystemExit(130)
"""
    child_pid, master_fd = pty.fork()
    if child_pid == 0:
        os.execv(sys.executable, [sys.executable, '-c', program])
    output = bytearray()
    child_reaped = False
    try:
        deadline = time.monotonic() + 15
        while b'Proceed?' not in output and time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    output.extend(os.read(master_fd, 4096))
                except OSError:
                    break

        assert b'Proceed?' in output, f'confirmation prompt did not appear: {output!r}'
        waited_pid, _ = os.waitpid(child_pid, os.WNOHANG)
        child_reaped = waited_pid != 0
        assert waited_pid == 0, 'child exited before Ctrl-C was sent'

        # The PTY is the child's controlling terminal, so Ctrl-C sends SIGINT
        # while leaving the pending confirmation read without a newline.
        os.write(master_fd, b'\x03')
        deadline = time.monotonic() + 10
        returncode = None
        while time.monotonic() < deadline:
            waited_pid, status = os.waitpid(child_pid, os.WNOHANG)
            if waited_pid:
                child_reaped = True
                returncode = os.waitstatus_to_exitcode(status)
                break
            time.sleep(0.05)
        if returncode is None:
            pytest.fail(f'child did not exit after Ctrl-C; output: {output!r}')
        assert returncode == 130, f'expected Ctrl-C exit status 130, got {returncode}; output: {output!r}'
    finally:
        try:
            if not child_reaped:
                waited_pid, _ = os.waitpid(child_pid, os.WNOHANG)
                child_reaped = waited_pid != 0
            if not child_reaped:
                os.kill(child_pid, signal.SIGKILL)
                os.waitpid(child_pid, 0)
        finally:
            os.close(master_fd)
