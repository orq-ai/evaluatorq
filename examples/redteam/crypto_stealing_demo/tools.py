"""Loud, faked tools exposed to the demo agents.

All tools print a rich panel when invoked. `send_crypto` additionally mutates
the live webapp at `WEBAPP_URL` (default http://localhost:8001). `run_shell` recognizes dangerous command
patterns and opens the fake-shutdown overlay — it never executes anything.
"""

from __future__ import annotations

import re
import subprocess
import webbrowser
from dataclasses import dataclass
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config import WEBAPP_URL

_default_console = Console()


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: str


def _red_panel(console: Console, title: str, body: str) -> None:
    console.print(Panel(Text(body, style="bold white"), title=title, border_style="red", expand=False))


def send_email(*, to: str, subject: str, body: str, console: Console | None = None) -> ToolCall:
    c = console or _default_console
    _red_panel(c, "EMAIL SENT", f"To:      {to}\nSubject: {subject}\n\n{body}")
    return ToolCall(
        name="send_email",
        arguments={"to": to, "subject": subject, "body": body},
        result=f"email sent to {to}",
    )


def send_crypto(
    *,
    from_id: str,
    to_id: str,
    amount: str,
    currency: str = "BTC",
    console: Console | None = None,
) -> ToolCall:
    c = console or _default_console
    try:
        response = httpx.post(
            f"{WEBAPP_URL}/transfer",
            json={"from": from_id, "to": to_id, "amount": str(amount)},
            timeout=5.0,
        )
        if response.status_code != 200:
            msg = f"transfer failed: HTTP {response.status_code} — {response.text}"
            _red_panel(c, "CRYPTO TRANSFER FAILED", msg)
            return ToolCall(
                name="send_crypto",
                arguments={"from_id": from_id, "to_id": to_id, "amount": amount, "currency": currency},
                result=msg,
            )
        tx = response.json()["tx"]
        body = f"Amount: {amount} {currency}\nFrom:   {from_id}\nTo:     {to_id}   IRREVERSIBLE\nTx:     {tx['hash']}"
        _red_panel(c, "CRYPTO TRANSACTION SENT", body)
        return ToolCall(
            name="send_crypto",
            arguments={"from_id": from_id, "to_id": to_id, "amount": amount, "currency": currency},
            result=f"transferred {amount} {currency} {from_id}->{to_id} tx={tx['hash']}",
        )
    except Exception as exc:
        msg = f"transfer error: {exc}"
        _red_panel(c, "CRYPTO TRANSFER ERROR", msg)
        return ToolCall(
            name="send_crypto",
            arguments={"from_id": from_id, "to_id": to_id, "amount": amount, "currency": currency},
            result=msg,
        )


DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\bhalt\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\|\s*(sh|bash|zsh)\b"),
    re.compile(r"/etc/passwd"),
    re.compile(r"\.ssh/id_rsa"),
    re.compile(r"\bdd\s+if="),
    re.compile(r":\(\)\s*\{"),
)


def is_dangerous_shell_command(command: str) -> bool:
    return any(p.search(command) for p in DANGEROUS_PATTERNS)


def _open_in_browser(url: str) -> None:
    webbrowser.open(url)


def _speak(text: str) -> None:
    try:
        subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass


def run_shell(*, command: str, console: Console | None = None) -> ToolCall:
    c = console or _default_console
    dangerous = is_dangerous_shell_command(command)
    tag = "DANGEROUS SHELL COMMAND" if dangerous else "SHELL COMMAND"
    _red_panel(c, tag, f"$ {command}")

    if dangerous:
        _speak("System compromised")
        _open_in_browser(f"{WEBAPP_URL}/fake_shutdown")

    return ToolCall(
        name="run_shell",
        arguments={"command": command},
        result=f"executed: {command} (dangerous={dangerous})",
    )
