"""Read-only directory listings for the local-session project picker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

_DIRECTORY_PAGE_SIZE = 200


def _breadcrumbs(path: Path) -> list[dict[str, str]]:
    parts = reversed((path, *path.parents))
    return [{'name': part.name or str(part), 'path': str(part)} for part in parts]


def list_project_directories(path: str = '', *, offset: int = 0) -> dict[str, Any]:
    """Return one page of child directories without reading file contents."""
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError('Directory page offset must be a nonnegative integer.')
    if not isinstance(path, str) or '\x00' in path:
        raise ValueError('Enter a valid directory path.')

    try:
        candidate = Path(path).expanduser() if path else Path.home()
        if not candidate.is_absolute():
            raise ValueError('Enter an absolute directory path, or leave it blank to browse your home folder.')
        current = candidate.resolve(strict=True)
        if not current.is_dir():
            raise NotADirectoryError
    except ValueError:
        raise
    except (OSError, RuntimeError) as exc:
        category = 'directory_missing' if isinstance(exc, FileNotFoundError) else 'directory_inaccessible'
        logger.warning('Project directory browser rejected path ({})', category)
        raise ValueError('That folder could not be opened. Check the path and your access, then try again.') from None

    try:
        names = [child.name for child in current.iterdir() if child.is_dir()]
    except (OSError, RuntimeError):
        logger.warning('Project directory browser could not list folder (directory_inaccessible)')
        raise ValueError('That folder could not be listed. Check your access, then try again.') from None

    names.sort(key=lambda name: (name.casefold(), name))
    page_names = names[offset : offset + _DIRECTORY_PAGE_SIZE]
    directories = []
    for name in page_names:
        try:
            directory_path = (current / name).resolve(strict=True)
        except (OSError, RuntimeError):
            # A directory may disappear between enumeration and path resolution.
            continue
        directories.append({'name': name, 'path': str(directory_path)})

    parent = current.parent if current.parent != current else None
    return {
        'path': str(current),
        'parent': str(parent) if parent is not None else None,
        'breadcrumbs': _breadcrumbs(current),
        'directories': directories,
        'offset': offset,
        'limit': _DIRECTORY_PAGE_SIZE,
        'has_more': offset + _DIRECTORY_PAGE_SIZE < len(names),
    }
