"""Per-clone temp workdir copied from the source, skills symlinked, removed on close()."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from evaluatorq.backends.coding_agent import CodingAgentError, CodingAgentTarget, OrqLaunchOptions
from evaluatorq.contracts import AgentTarget


def _skill(tmp_path: Path, name: str) -> Path:
    d = tmp_path / 'skills' / name
    d.mkdir(parents=True)
    (d / 'SKILL.md').write_text(f'# {name}\n')
    return d


def test_workdir_is_lazy_then_copied_per_clone(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'README.md').write_text('hello')
    target = CodingAgentTarget('claude', workdir=src)
    assert target.workdir is None
    a, b = target.new(), target.new()
    wa, wb = a._ensure_workdir(), b._ensure_workdir()
    assert wa != wb and wa != src and wb != src
    assert (wa / 'README.md').read_text() == 'hello'
    assert (wb / 'README.md').read_text() == 'hello'
    (wa / 'README.md').write_text('mutated by a live agent')
    c = a.new()
    assert (c._ensure_workdir() / 'README.md').read_text() == 'hello'


def test_skills_symlinked_into_agent_dir(tmp_path: Path) -> None:
    skill = _skill(tmp_path, 'grill-me')
    for agent, rel in (('claude', '.claude/skills'), ('codex', '.agents/skills'), ('opencode', '.agents/skills')):
        target = CodingAgentTarget(agent, skills=[skill])  # pyright: ignore[reportArgumentType]
        link = target._ensure_workdir() / rel / 'grill-me'
        assert link.is_symlink() and link.resolve() == skill.resolve()
        assert (link / 'SKILL.md').exists()


def test_skill_name_collision_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / 'src'
    (src / '.claude' / 'skills' / 'grill-me').mkdir(parents=True)
    skill = _skill(tmp_path, 'grill-me')
    target = CodingAgentTarget('claude', workdir=src, skills=[skill])
    allocated = tmp_path / 'allocated-workdir'
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda prefix: str(allocated))
    with pytest.raises(FileExistsError, match='grill-me'):
        target._ensure_workdir()
    assert not allocated.exists()
    assert target.workdir is None


def test_symlinked_skills_dir_outside_source_is_rejected_and_cleaned(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    outside = tmp_path / 'outside'
    src.mkdir()
    outside.mkdir()
    (src / '.claude').mkdir()
    (src / '.claude' / 'skills').symlink_to(outside, target_is_directory=True)
    skill = _skill(tmp_path, 'grill-me')
    target = CodingAgentTarget('claude', workdir=src, skills=[skill])

    with pytest.raises(ValueError, match='symlink that leaves the private copy'):
        target._ensure_workdir()

    assert not any(outside.iterdir())
    assert target.workdir is None


@pytest.mark.asyncio
async def test_close_removes_workdir_and_is_idempotent() -> None:
    target = CodingAgentTarget('claude')
    wd = target._ensure_workdir()
    assert wd.exists()
    await target.close()
    assert not wd.exists()
    await target.close()


@pytest.mark.asyncio
async def test_keep_workdir_retains(tmp_path: Path) -> None:
    target = CodingAgentTarget('claude', keep_workdir=True)
    wd = target._ensure_workdir()
    await target.close()
    assert wd.exists()
    import shutil

    shutil.rmtree(wd)


def test_new_uses_type_self_and_same_kwargs(tmp_path: Path) -> None:
    class Sub(CodingAgentTarget):
        pass

    target = Sub('codex', model='gpt-5.6-luna', permission_mode='read-only', keep_workdir=False)
    clone = target.new()
    assert type(clone) is Sub
    assert clone._kwargs == target._kwargs
    assert isinstance(clone, AgentTarget)


def test_permission_mode_on_opencode_raises_at_construction() -> None:
    with pytest.raises(ValueError, match='opencode'):
        CodingAgentTarget('opencode', permission_mode='x')


def test_orq_options_under_direct_warn() -> None:
    from loguru import logger

    sink = []
    handle = logger.add(sink.append, level='WARNING')
    try:
        CodingAgentTarget('claude', launcher='direct', orq=OrqLaunchOptions(mcp=False))
    finally:
        logger.remove(handle)
    assert any('orq' in str(m) and 'direct' in str(m) for m in sink)


def test_unknown_agent_raises() -> None:
    with pytest.raises(ValueError, match='gemini'):
        CodingAgentTarget('gemini')  # pyright: ignore[reportArgumentType]


@pytest.mark.asyncio
async def test_agent_context_lists_tools_and_skills(tmp_path: Path) -> None:
    skill = _skill(tmp_path, 'grill-me')
    ctx = await CodingAgentTarget('codex', skills=[skill], system_prompt='be terse').get_agent_context()
    names = [t.name for t in ctx.tools]
    assert names[:2] == ['shell', 'apply_patch']
    assert 'grill-me' in names
    assert ctx.key == 'coding-agent:codex'
    assert ctx.system_prompt == 'be terse'


def test_map_error_reports_cli_codes_only() -> None:
    target = CodingAgentTarget('claude')
    assert target.map_error(CodingAgentError('cli.exit.2', 'boom')) == ('cli.exit.2', 'boom')
    assert target.map_error(RuntimeError('x')) is None
