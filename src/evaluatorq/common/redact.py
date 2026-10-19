"""Scrub credentials from untrusted trace text before it reaches an external classifier.

Shell commands, tool arguments and command output can carry API keys, tokens, passwords and private keys. Every caller
that sends such text to a model goes through `scrub_known_secrets`, a local, offline pass of compiled patterns that
replaces each credential with a placeholder such as `<API_KEY>`. Nothing leaves the process, so there is no failure
mode: the function always returns text. `normalize_placeholders` drops placeholder numbering so texts that differ only
in which secret they carry compare equal.

The patterns were validated against a hand-built corpus of 38 secret types (114 items) and 44 secret-free commands;
expect misses on unseen formats and some false positives on random-looking identifiers of 20+ characters.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

_PLACEHOLDER = re.compile(r'<([A-Z][A-Z0-9_]*)_\d+>')
# A value already replaced by an earlier rule; a later rule must not replace it again.
_PLACEHOLDED = r'(?!<[A-Z_]+>)'


def normalize_placeholders(text: str) -> str:
    """Drop the numbering of placeholders: `<UUID_3>` becomes `<UUID>`.

    Two texts that differ only in which id, email, URL or key they carry normalize to the same string, so a caller can
    ask one question for all of them.
    """
    return _PLACEHOLDER.sub(r'<\1>', text)


class _Rule(NamedTuple):
    pattern: re.Pattern[str]
    replace: Callable[[re.Match[str]], str]


def _mask(label: str) -> Callable[[re.Match[str]], str]:
    """Replace the whole match with `<LABEL>`."""
    return lambda _match: f'<{label}>'


def _mask_group(label: str, group: int = 1) -> Callable[[re.Match[str]], str]:
    """Replace only capture group `group` with `<LABEL>`, keeping the label or flag in front of it."""

    def replace(match: re.Match[str]) -> str:
        offset = match.start()
        return f'{match.string[offset : match.start(group)]}<{label}>{match.string[match.end(group) : match.end()]}'

    return replace


def _entropy_ok(token: str) -> bool:
    """True when `token` looks like random key material rather than an identifier, path, SHA or branch name."""
    token = token.strip('=')
    if not token or re.fullmatch(r'[0-9a-f]+', token):  # hex: git SHAs, left to the long-hex rule
        return False
    if any(re.fullmatch(r'[a-z]{4,}', part) for part in re.split(r'[-_/.+]', token)):  # holds a plain word
        return False
    if len(token) / len(re.findall(r'[A-Z]+|[a-z]+|[0-9]+|[^A-Za-z0-9]+', token)) >= 2.5:  # long same-class runs
        return False
    if not (re.search(r'[A-Z]', token) and re.search(r'[a-z]', token) and re.search(r'[0-9]', token)):
        return False
    counts = Counter(token)
    return -sum(n / len(token) * math.log2(n / len(token)) for n in counts.values()) >= 3.8


def _mask_entropy(match: re.Match[str]) -> str:
    """Mask a high-entropy token; in `NAME=value` only the value is judged and replaced."""
    token = match.group(0)
    name = token.rstrip('=').rpartition('=')[0]
    prefix = token[: len(name) + 1] if name else ''
    value = token[len(prefix) :]
    return f'{prefix}<SECRET>' if len(value) >= 20 and _entropy_ok(value) else token


# Order matters: structural rules first, the generic and entropy rules last. Each placeholder is unnumbered, which
# `normalize_placeholders` leaves as is.
_RULES: tuple[_Rule, ...] = (
    # A PEM private key, through its END line or the end of the text when truncated.
    _Rule(
        re.compile(r'-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)', re.DOTALL),
        _mask('PRIVATE_KEY'),
    ),
    # Two or more base64 body lines with no header (a cut-off key), also with JSON-escaped newlines.
    _Rule(
        re.compile(r'(?:(?:^|\\n|\n)[A-Za-z0-9+/]{60,76}={0,2}(?=\\n|\n|$)){2,}', re.MULTILINE), _mask('PRIVATE_KEY')
    ),
    # user:password in a URL, up to the @.
    _Rule(re.compile(r'(?<=://)[^\s/:@\'"]*:[^\s/@\'"]+(?=@)'), _mask('CREDENTIALS')),
    # The secret path of a Slack webhook.
    _Rule(
        re.compile(r'https://hooks\.slack\.com/(?:services|workflows|triggers)/[A-Za-z0-9_/-]+'), _mask('WEBHOOK_URL')
    ),
    # Well-known token prefixes: AWS, GitHub, GitLab, Slack, Stripe, OpenAI/Anthropic, Hugging Face, npm, PyPI.
    _Rule(
        re.compile(
            r'\b(?:(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{20,}'
            r'|glpat-[A-Za-z0-9_-]{20,}|xox[abposr]-[A-Za-z0-9-]{10,}|[rs]k_(?:live|test)_[A-Za-z0-9]{16,}'
            r'|sk-(?:proj-|ant-[a-z0-9]+-)?[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{30,}|npm_[A-Za-z0-9]{36}'
            r'|pypi-[A-Za-z0-9_-]{50,})'
        ),
        _mask('API_KEY'),
    ),
    # A JWT anywhere, including custom headers.
    _Rule(re.compile(r'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}'), _mask('JWT')),
    # The credential after an HTTP auth scheme: `Bearer x`, `Basic x`, `Authorization: Token x`. The value must hold a
    # digit, so prose such as `Bearer token expired` is left alone.
    _Rule(
        re.compile(
            rf'(\b(?:Bearer|Basic|Authorization:\s*Token)\s+){_PLACEHOLDED}(?=[^\s\'"]*\d)([A-Za-z0-9._~+/=-]{{8,}})',
            re.IGNORECASE,
        ),
        _mask_group('SECRET', 2),
    ),
    # `mysql -u root -pSECRET`: the password is glued to the flag.
    _Rule(
        re.compile(r'(\b(?:mysql|mysqldump|mysqladmin|mariadb)\b[^\n|;&]*?\s-p)([^\s\'"]+)'), _mask_group('PASSWORD', 2)
    ),
    # `--password=x`, `--token x`, `-p x` style flags.
    _Rule(
        re.compile(
            r'(?<!\S)--?(?:password|passwd|pass|pwd|secret|token|api[-_]?key|auth[-_]?token)(?:=|\s+)'
            rf'{_PLACEHOLDED}([^\s\'"]+)',
            re.IGNORECASE,
        ),
        _mask_group('PASSWORD'),
    ),
    # `NAME=value` or `"name": "value"` where the name says secret: PGPASSWORD, API_TOKEN, AccountKey, cookies, ...
    _Rule(
        re.compile(
            r'(?<![A-Z0-9_.-])[A-Z0-9_.-]*(?:pass(?:word|wd)?|secret|token|api[_-]?key|access[_-]?key|private[_-]?key'
            r'|credentials?|accountkey|session(?:id)?|cookie|sid)[A-Z0-9_.-]*["\']?\s*[=:]\s*["\']?'
            rf'{_PLACEHOLDED}([^\s"\',;\\]{{6,}})',
            re.IGNORECASE,
        ),
        _mask_group('SECRET'),
    ),
    # An unlabeled 64+ hex blob; spares `sha256:<hex>` digests and 40-hex git SHAs.
    _Rule(re.compile(r'(?<![:\w])[0-9a-fA-F]{64,}(?!\w)'), _mask('SECRET')),
    # Any 20+ character token that looks random (mixed case and digits, high entropy, no plain words).
    _Rule(re.compile(r'[A-Za-z0-9+/_=-]{20,}'), _mask_entropy),
)


def scrub_known_secrets(text: str) -> str:
    """Replace credential shapes in `text` with placeholders such as `<API_KEY>`, `<PRIVATE_KEY>` or `<PASSWORD>`.

    A local, offline pass for what the Orq endpoint is known to miss: PEM keys, `mysql -p<pw>`, `--password=`,
    PGPASSWORD-style assignments, JWTs in custom headers, Slack webhooks, unlabeled hex blobs. It must run on the raw
    text, before any placeholder-inserting redaction, whose placeholders would break the anchors these patterns need.
    A flag or label in front of a value (`--password=`, `PGPASSWORD=`) is kept; only the value is replaced.
    """
    for rule in _RULES:
        text = rule.pattern.sub(rule.replace, text)
    return text
