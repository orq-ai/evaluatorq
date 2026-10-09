"""Terminal-level tests for interrupting shared confirmation prompts."""
from __future__ import annotations

import os
import select
import signal
import sys
import time

import pytest


@pytest.mark.skipif(os.name != 'posix', reason='requires a POSIX pseudo-terminal')
@pytest.mark.parametrize('interrupt_thread', ['main', 'worker'])
def test_ctrl_c_exits_confirmation_without_waiting_for_enter(interrupt_thread: str) -> None:
    """Ctrl-C at the shared run-plan prompt exits while its stdin read is pending."""
    import pty

    program = r"""
import asyncio
import os
import signal
import sys
import termios
from rich.console import Console
from evaluatorq.common.reports import confirm_run_plan

terminal = termios.tcgetattr(0)
terminal[3] |= termios.ISIG
terminal[6][termios.VINTR] = b'\x03'
termios.tcsetattr(0, termios.TCSANOW, terminal)
assert os.tcgetpgrp(0) == os.getpgrp()
signal.signal(signal.SIGINT, signal.default_int_handler)
if sys.argv[1] == 'worker':
    import typer

    real_confirm = typer.confirm

    def worker_confirm(*args, **kwargs):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
        return real_confirm(*args, **kwargs)

    typer.confirm = worker_confirm
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
else:
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})

async def main():
    await confirm_run_plan(
        Console(file=sys.stderr, force_terminal=False),
        title='Run Plan',
        rows=[('Target', 'example')],
        prompt='Proceed?',
        skip_confirm=False,
    )

try:
    asyncio.run(main())
except KeyboardInterrupt:
    raise SystemExit(130)
"""
    child_pid, master_fd = pty.fork()
    if child_pid == 0:
        os.execv(sys.executable, [sys.executable, '-c', program, interrupt_thread])
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
        master_open = True
        while time.monotonic() < deadline:
            if master_open:
                ready, _, _ = select.select([master_fd], [], [], 0.05)
                if ready:
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        master_open = False
                    else:
                        if chunk:
                            output.extend(chunk)
                        else:
                            master_open = False
            waited_pid, status = os.waitpid(child_pid, os.WNOHANG)
            if waited_pid:
                child_reaped = True
                returncode = os.waitstatus_to_exitcode(status)
                break
            if not master_open:
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
