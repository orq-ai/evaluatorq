"""Encryption helpers for secrets stored in local dashboard settings."""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache

_KEY_ENV = 'EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY'
_KEYCHAIN_SERVICE = 'evaluatorq-dashboard-api-key-encryption'


def _fernet_type():
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise RuntimeError(
            'Encrypted API-key storage requires the dashboard extra, which includes cryptography.'
        ) from exc
    return Fernet


@lru_cache(maxsize=1)
def _encryption_key() -> bytes:
    """Load the encryption key from Keychain or the explicit portable env var."""
    Fernet = _fernet_type()
    configured = os.environ.get(_KEY_ENV, '').strip()
    if configured:
        key = configured.encode('ascii')
        try:
            Fernet(key)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f'{_KEY_ENV} must contain a valid Fernet key.') from exc
        return key

    if sys.platform != 'darwin':
        raise RuntimeError(f'Encrypted API-key storage needs macOS Keychain or {_KEY_ENV} on this platform.')

    account = os.environ.get('USER', 'evaluatorq')
    read = subprocess.run(
        ['security', 'find-generic-password', '-a', account, '-s', _KEYCHAIN_SERVICE, '-w'],
        check=False,
        capture_output=True,
        text=True,
    )
    if read.returncode == 0 and read.stdout.strip():
        key = read.stdout.strip().encode('ascii')
        try:
            Fernet(key)
        except (ValueError, TypeError) as exc:
            raise RuntimeError('The dashboard encryption key in Keychain is invalid.') from exc
        return key

    key = Fernet.generate_key()
    # This key protects the API key ciphertext in the config file. The actual API
    # key is never passed to a subprocess or written outside the encrypted config.
    write = subprocess.run(
        [
            'security',
            'add-generic-password',
            '-U',
            '-a',
            account,
            '-s',
            _KEYCHAIN_SERVICE,
            '-w',
            key.decode('ascii'),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if write.returncode != 0:
        raise RuntimeError('Could not save the dashboard encryption key in macOS Keychain.')
    return key


def encrypt_api_key(key: str) -> str:
    """Return a Fernet authenticated-encryption token for an API key."""
    return _fernet_type()(_encryption_key()).encrypt(key.encode('utf-8')).decode('ascii')


def decrypt_api_key(ciphertext: str) -> str:
    """Decrypt a saved API key or raise a safe error when its key is unavailable."""
    from cryptography.fernet import InvalidToken

    try:
        return _fernet_type()(_encryption_key()).decrypt(ciphertext.encode('ascii')).decode('utf-8')
    except (InvalidToken, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(
            'The saved API key cannot be decrypted. Restore its original encryption key or enter it again.'
        ) from exc
