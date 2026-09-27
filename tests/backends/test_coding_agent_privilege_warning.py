"""Security warnings for opt-in container privilege escalation."""

from loguru import logger

from evaluatorq.backends import DockerOptions
from evaluatorq.backends.coding_agent import CodingAgentTarget


def test_privilege_escalation_opt_in_warns_about_root() -> None:
    seen: list[str] = []
    handler_id = logger.add(lambda message: seen.append(str(message)), level='WARNING')
    try:
        CodingAgentTarget(agent='claude', container=DockerOptions(allow_privilege_escalation=True))
    finally:
        logger.remove(handler_id)

    warning = '\n'.join(seen)
    assert 'no-new-privileges' in warning
    assert '/etc/passwd' in warning
    assert 'root inside the container' in warning
