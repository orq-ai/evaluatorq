"""`common.redact`: the local credential scrub and placeholder normalization."""

from __future__ import annotations

import pytest

from evaluatorq.common.redact import normalize_placeholders, scrub_known_secrets

PRIVATE_KEY = (
    '-----BEGIN RSA PRIVATE KEY-----\n'
    + '\n'.join('MIIEowIBAAKCAQEA' + 'x7Qz9' * 9 + 'Qw' for _ in range(3))
    + '\n-----END RSA PRIVATE KEY-----'
)
HEX_64 = '692d8f1c3a5b7e9d0f2a4c6e8b1d3f5a7c9e0b2d4f6a8c1e3b5d7f9a0c2e4b6d'
LEAKS = [
    ('pem', f'cat > k <<EOF\n{PRIVATE_KEY}\nEOF', 'x7Qz9x7Qz9'),
    ('mysql', 'mysql -u root -pS3cr3tPw -h db prod', 'S3cr3tPw'),
    ('password-flag', 'psql --host=db --password=pTp0Zk81mQ -c "select 1"', 'pTp0Zk81mQ'),
    ('pgpassword', '+ PGPASSWORD=pLYQr8Tn2Wv9\n+ psql -h db', 'pLYQr8Tn2Wv9'),
    ('pypi', 'twine upload -u __token__ -p pypi-' + 'AgEIcHlwaS5vcmc' * 5 + ' dist/*', 'AgEIcHlwaS5vcmc'),
    ('hex', f'python sign.py {HEX_64} payload.json', HEX_64[:16]),
    ('url-userinfo', 'git clone https://bot:hunter2hunter2@github.com/acme/x.git', 'hunter2hunter2'),
    ('slack-webhook', 'curl https://hooks.slack.com/services/TG84K60BE/BGIPLOUXOKY/lRg4Zq9Xw2 -d x', 'lRg4Zq9Xw2'),
    (
        'jwt',
        "curl -H 'X-Auth: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhb'",
        'dBjftJeZ4CVP',
    ),
    ('known-prefix', 'export TOK=ghp_' + 'a1B2c3D4e5' * 3, 'a1B2c3D4e5'),
    ('secret-assignment', 'API_TOKEN=gXeRk29vLq0Pz8Wd', 'gXeRk29vLq0Pz8Wd'),
    ('high-entropy', 'deploy --key Zq8Rk2LxV7mN4pT9wYb3HcD5', 'Zq8Rk2LxV7mN4pT9wYb3HcD5'),
    ('flag-at-start', '--password hunter2', 'hunter2'),
    ('bearer', 'curl -H "Authorization: Bearer abcdefghijklmnop123456" https://api', 'abcdefghijklmnop123456'),
    ('basic', 'Authorization: Basic dXNlcjpwYXNzd29yZDEy', 'dXNlcjpwYXNzd29yZDEy'),
]


@pytest.mark.parametrize(('_name', 'text', 'secret'), LEAKS, ids=[leak[0] for leak in LEAKS])
def test_scrub_known_secrets_removes_each_credential_shape(_name: str, text: str, secret: str) -> None:
    assert secret not in scrub_known_secrets(text)


def test_scrub_keeps_the_label_and_uses_unnumbered_placeholders() -> None:
    assert scrub_known_secrets('psql --password=pTp0Zk81mQ db') == 'psql --password=<PASSWORD> db'
    assert scrub_known_secrets('PGPASSWORD=pLYQr8Tn2Wv9 psql') == 'PGPASSWORD=<SECRET> psql'
    assert scrub_known_secrets(PRIVATE_KEY) == '<PRIVATE_KEY>'
    assert normalize_placeholders(scrub_known_secrets('x AKIAABCDEFGHIJKLMNOP')) == 'x <API_KEY>'


def test_scrub_is_idempotent() -> None:
    once = scrub_known_secrets('psql --password=pTp0Zk81mQ\nPGPASSWORD=abcdef12 x\n' + PRIVATE_KEY)
    assert scrub_known_secrets(once) == once


@pytest.mark.parametrize(
    'text',
    [
        'git push --force origin main',
        'rm -rf build/',
        'kubectl delete deployment api -n prod',
        'terraform destroy -auto-approve',
        'terraform destroy -target=aws_db_instance.main',
        'aws s3 rm s3://acme-prod-backups/2026/ --recursive',
        'dd if=/dev/zero of=/dev/nvme1n1 bs=1M',
        'docker login --password-stdin ghcr.io',
        'export MAX_TOKENS=4096 GITHUB_SHA=0123456789abcdef0123456789abcdef01234567',
        'git checkout feature/RES-412-redact',
        'FAILED tests/test_a.py::test_x - AssertionError: assert 1 == 2\n=== 1 failed, 214 passed in 12.34s ===',
        'docker pull acme/api@sha256:' + HEX_64,
        'ERROR: Bearer token expired, please log in again',
    ],
)
def test_scrub_leaves_secret_free_commands_unchanged(text: str) -> None:
    assert scrub_known_secrets(text) == text


def test_normalize_placeholders_drops_the_numbering() -> None:
    assert normalize_placeholders('<UUID_3> and <EMAIL_ADDRESS_12> via <IP_ADDRESS_1>') == (
        '<UUID> and <EMAIL_ADDRESS> via <IP_ADDRESS>'
    )
    assert normalize_placeholders('a < b_2 > c <not_a_placeholder_1>') == 'a < b_2 > c <not_a_placeholder_1>'
