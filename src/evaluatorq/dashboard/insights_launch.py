"""Validate dashboard Insights requests and launch an independent run process."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing_extensions import Self

from evaluatorq.common.orq_client import DEFAULT_ORQ_BASE_URL, OrqProfile
from evaluatorq.common.run_manifest import ManifestWriter, start_manifest
from evaluatorq.common.run_store_dir import get_store_dir
from evaluatorq.contracts import ManifestStatus, RunManifest
from evaluatorq.insights.models import DimensionName, InsightsPopulation, LabelSpec
from evaluatorq.insights.presets import LABEL_PRESETS
from evaluatorq.insights.progress import stage_plan
from evaluatorq.trace_finder.export import RunExport
from evaluatorq.trace_finder.models import FacetSelection

Source = Literal['recent', 'query', 'finder']
Preset = Literal['sentiment', 'customer_satisfaction']
_REQUEST_ENV = 'EVALUATORQ_INSIGHTS_LAUNCH_REQUEST'
_MANIFEST_ENV = 'EVALUATORQ_INSIGHTS_MANIFEST'
_SNAPSHOT_ENV = 'EVALUATORQ_INSIGHTS_FINDER_SNAPSHOT'
MAX_FINDER_EXPORT_BYTES = 10 * 1024 * 1024
FINDER_EXPORT_REFERENCE_DIR = '.finder-export-leases'
INSIGHTS_WORKER_STATE_DIR = '.insights-workers'
INSIGHTS_WORKER_STALE_SECONDS = 90
INSIGHTS_WORKER_HEARTBEAT_SECONDS = 10


def worker_state_path(runs_dir: Path, run_id: str) -> Path:
    if re.fullmatch(r'[A-Za-z0-9_-]+', run_id) is None:
        raise ValueError('Invalid Insights run ID for worker state')
    return runs_dir / INSIGHTS_WORKER_STATE_DIR / f'{run_id}.json'


def _write_worker_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_info = path.parent.lstat()
    if (
        path.parent.is_symlink()
        or not stat.S_ISDIR(directory_info.st_mode)
        or (hasattr(os, 'getuid') and directory_info.st_uid != os.getuid())
    ):
        raise OSError(f'Insights worker state directory is not a private directory: {path.parent}')
    Path(path.parent).chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.stem}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def start_worker_heartbeat(manifest_path: Path) -> tuple[threading.Event, threading.Thread]:
    """Refresh the detached worker lease independently of its async workload."""
    run_id = manifest_path.stem
    path = worker_state_path(manifest_path.parent.parent, run_id)
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    state.update(pid=os.getpid(), process_identity=_worker_process_identity(os.getpid()), heartbeat_at=time.time())
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.is_set():
            try:
                _write_worker_state(path, state)
            except OSError as exc:
                logger.warning('Could not refresh Insights worker heartbeat {}: {}', path, exc)
            if stop.wait(INSIGHTS_WORKER_HEARTBEAT_SECONDS):
                return
            state['heartbeat_at'] = time.time()

    _write_worker_state(path, state)
    thread = threading.Thread(target=heartbeat, name=f'insights-heartbeat-{run_id}', daemon=True)
    thread.start()
    return stop, thread


def reconcile_stale_worker(runs_dir: Path, run_id: str) -> bool:
    """Fail a dashboard worker whose startup or heartbeat lease expired."""
    state_path = worker_state_path(runs_dir, run_id)
    manifest_path = runs_dir / '.manifests' / f'{run_id}.json'
    try:
        directory_info = state_path.parent.lstat()
        state_info = state_path.lstat()
        if (
            state_path.parent.is_symlink()
            or not stat.S_ISDIR(directory_info.st_mode)
            or (os.name != 'nt' and stat.S_IMODE(directory_info.st_mode) != 0o700)
            or (hasattr(os, 'getuid') and directory_info.st_uid != os.getuid())
            or not stat.S_ISREG(state_info.st_mode)
            or (os.name != 'nt' and stat.S_IMODE(state_info.st_mode) != 0o600)
            or (hasattr(os, 'getuid') and state_info.st_uid != os.getuid())
        ):
            return False
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if not isinstance(state, dict):
            return False
        heartbeat_at = state.get('heartbeat_at')
        stale = isinstance(heartbeat_at, (int, float)) and time.time() - heartbeat_at > INSIGHTS_WORKER_STALE_SECONDS
        pid = state.get('pid')
        process_identity = state.get('process_identity')
    except (OSError, ValueError, TypeError):
        return False
    if manifest.status != ManifestStatus.RUNNING or not stale:
        return False
    # A worker can be paused by the OS or debugger long enough for its lease
    # to age. A live PID is stronger evidence than an old heartbeat, so leave
    # its manifest and private Finder artifacts alone until it exits.
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or _worker_process_is_alive(pid, process_identity):
        return False
    # Re-read the lease immediately before failing. The worker may have
    # resumed and refreshed it while the manifest was being parsed.
    try:
        latest_state = json.loads(state_path.read_text(encoding='utf-8'))
        latest_manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return False
    if (
        latest_manifest.status != ManifestStatus.RUNNING
        or not isinstance(latest_state, dict)
        or latest_state.get('pid') != pid
        or latest_state.get('process_identity') != process_identity
        or not isinstance(latest_state.get('heartbeat_at'), (int, float))
        or time.time() - latest_state['heartbeat_at'] <= INSIGHTS_WORKER_STALE_SECONDS
        or _worker_process_is_alive(pid, process_identity)
    ):
        return False
    state = latest_state
    manifest = latest_manifest
    ManifestWriter(manifest, manifest_path).fail(
        'Insights worker stopped before completing the run; start a new run to retry.', stage='worker'
    )
    try:
        persisted = RunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if persisted.status == ManifestStatus.RUNNING:
        return False
    snapshot = state.get('snapshot_path')
    if isinstance(snapshot, str):
        # Reuse the worker's strict private-temp ownership and mode checks.
        from evaluatorq.dashboard.insights_worker import _cleanup_snapshot

        _cleanup_snapshot(Path(snapshot))
    try:
        reference_path = finder_export_reference_path(runs_dir, run_id)
        if not reference_path.exists():
            pass
        elif reference_path.parent.is_symlink():
            logger.warning('Leaving untrusted Finder export reference directory in place: {}', reference_path.parent)
        else:
            validate_private_finder_reference(reference_path)
            reference_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning('Could not release Finder export reference for stale run {}: {}', run_id, exc)
    try:
        state_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning('Could not remove stale Insights worker state {}: {}', state_path, exc)
    return True


def _worker_process_is_alive(pid: object, expected_identity: object = None) -> bool:
    """Return whether the recorded worker still exists with its original identity."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or not hasattr(os, 'kill'):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # Only ESRCH proves death; permission or platform errors are uncertain.
        return exc.errno != errno.ESRCH
    except OverflowError:
        return True
    if isinstance(expected_identity, str):
        observed = _read_worker_process_identity(pid)
        if observed is None:
            logger.warning('Could not verify Insights worker process identity for PID {}; leaving its run active', pid)
            return True
        identity, zombie = observed
        return identity == expected_identity and not zombie
    logger.warning('Insights worker PID {} has no recorded process identity; leaving its run active', pid)
    return True


def _read_worker_process_identity(pid: int) -> tuple[str, bool] | None:
    """Read a process start marker and zombie flag where the OS exposes them."""
    if sys.platform.startswith('linux'):
        try:
            contents = Path(f'/proc/{pid}/stat').read_text(encoding='ascii')
        except OSError:
            return None
        # The command name is parenthesized and can itself contain spaces or ')'.
        closing = contents.rfind(')')
        if closing < 0:
            return None
        fields = contents[closing + 1 :].split()
        if len(fields) <= 19:
            return None
        try:
            start_ticks = int(fields[19])
        except ValueError:
            return None
        return f'linux:{start_ticks}', fields[0] == 'Z'
    if sys.platform == 'darwin':
        try:
            result = subprocess.run(
                ['/bin/ps', '-o', 'lstart=', '-o', 'stat=', '-p', str(pid)],
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        fields = result.stdout.split()
        if len(fields) < 6:
            return None
        return f'darwin:{" ".join(fields[:5])}', fields[5].startswith('Z')
    return None


def _worker_process_identity(pid: int) -> str | None:
    """Capture the worker's start marker, if this OS exposes one."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    details = _read_worker_process_identity(pid)
    return details[0] if details is not None else None


# The child must record import-time failures too. This tiny stdlib-only wrapper
# runs before importing the package worker, then edits the already-created
# manifest if Python cannot import the worker or its dependencies.
_WORKER_BOOTSTRAP = """
import datetime, json, os, runpy, stat, tempfile
manifest = os.environ.get("EVALUATORQ_INSIGHTS_MANIFEST")
request = os.environ.get("EVALUATORQ_INSIGHTS_LAUNCH_REQUEST")
def cleanup_snapshot():
    try:
        snapshot = os.environ.get("EVALUATORQ_INSIGHTS_FINDER_SNAPSHOT")
        if not snapshot:
            snapshot = json.loads(request or "{}").get("finder_export_snapshot")
        if not isinstance(snapshot, str) or os.path.basename(snapshot) != "finder-export.json":
            return
        directory = os.path.dirname(snapshot)
        if not os.path.exists(directory):
            return
        info = os.stat(directory, follow_symlinks=False)
        if (os.path.islink(directory) or not os.path.basename(directory).startswith("evaluatorq-finder-snapshot-")
                or os.path.dirname(os.path.realpath(directory)) != os.path.realpath(tempfile.gettempdir())
                or (os.name != "nt" and stat.S_IMODE(info.st_mode) != 0o700)
                or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
            return
        if os.path.islink(snapshot) or not os.path.isfile(snapshot):
            return
        os.unlink(snapshot)
        os.rmdir(directory)
    except (OSError, ValueError, TypeError) as cleanup_error:
        print(f"Could not remove validated Finder snapshot: {cleanup_error}", file=__import__("sys").stderr)
def cleanup_reference():
    try:
        directory = os.path.dirname(manifest or "")
        name = os.path.basename(manifest or "")
        run_id = name[:-5] if name.endswith(".json") else ""
        if os.path.basename(directory) != ".manifests" or not run_id or not all(ch.isalnum() or ch in "-_" for ch in run_id):
            return
        lease_dir = os.path.join(os.path.dirname(directory), ".finder-export-leases")
        lease_parent = os.path.dirname(lease_dir)
        parent_info = os.stat(lease_parent, follow_symlinks=False)
        directory_info = os.stat(lease_dir, follow_symlinks=False)
        reference = os.path.join(lease_dir, run_id + ".json")
        reference_info = os.stat(reference, follow_symlinks=False)
        if (os.path.islink(lease_parent) or not stat.S_ISDIR(parent_info.st_mode)
                or (hasattr(os, "getuid") and parent_info.st_uid != os.getuid())
                or (os.name != "nt" and stat.S_IMODE(parent_info.st_mode) & 0o022)
                or os.path.islink(lease_dir) or not stat.S_ISDIR(directory_info.st_mode)
                or (os.name != "nt" and stat.S_IMODE(directory_info.st_mode) != 0o700)
                or (hasattr(os, "getuid") and directory_info.st_uid != os.getuid())
                or not stat.S_ISREG(reference_info.st_mode)
                or (os.name != "nt" and stat.S_IMODE(reference_info.st_mode) != 0o600)
                or (hasattr(os, "getuid") and reference_info.st_uid != os.getuid())):
            return
        os.unlink(reference)
    except FileNotFoundError:
        pass
    except OSError as cleanup_error:
        print(f"Could not remove Finder export reference: {cleanup_error}", file=__import__("sys").stderr)
def cleanup_worker_state():
    try:
        directory = os.path.dirname(manifest or "")
        name = os.path.basename(manifest or "")
        run_id = name[:-5] if name.endswith(".json") else ""
        if os.path.basename(directory) != ".manifests" or not run_id or not all(ch.isalnum() or ch in "-_" for ch in run_id):
            return
        state_dir = os.path.join(os.path.dirname(directory), ".insights-workers")
        if os.path.islink(state_dir):
            return
        os.unlink(os.path.join(state_dir, run_id + ".json"))
    except FileNotFoundError:
        pass
    except OSError as cleanup_error:
        print(f"Could not remove Insights worker state: {cleanup_error}", file=__import__("sys").stderr)
try:
    runpy.run_module("evaluatorq.dashboard.insights_worker", run_name="__main__")
except BaseException as exc:
    if isinstance(exc, SystemExit) and exc.code in (None, 0):
        raise
    cleanup_snapshot()
    cleanup_reference()
    cleanup_worker_state()
    if manifest:
        try:
            with open(manifest, encoding="utf-8") as source:
                data = json.load(source)
            if data.get("status") == "running":
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                data.update(status="error", stage="setup", error=f"Could not start Insights worker: {exc}", ended_at=now, updated_at=now)
                directory = os.path.dirname(manifest)
                fd, temporary = tempfile.mkstemp(prefix=".insights-failure-", dir=directory)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as target:
                        json.dump(data, target)
                    os.replace(temporary, manifest)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        except BaseException as recovery_error:
            print(f"Could not mark Insights manifest failed: {recovery_error}", file=__import__("sys").stderr)
    raise
"""


def get_finder_exports_dir() -> Path:
    """Return the only directory from which dashboard runs may read Finder exports."""
    return get_store_dir('finder-exports')


def finder_export_reference_path(runs_dir: Path, run_id: str) -> Path:
    """Return the private lease path for a running Finder-sourced Insights run."""
    if re.fullmatch(r'[A-Za-z0-9_-]+', run_id) is None:
        raise ValueError('Invalid Insights run ID for Finder export reference')
    return runs_dir / FINDER_EXPORT_REFERENCE_DIR / f'{run_id}.json'


def ensure_private_finder_reference_dir(path: Path) -> None:
    """Create or validate the private Finder lease directory."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    validate_private_finder_reference_dir(path)


def validate_private_finder_reference_dir(path: Path) -> None:
    """Check an existing Finder lease directory without creating it."""
    parent_info = path.parent.lstat()
    if (
        path.parent.is_symlink()
        or not stat.S_ISDIR(parent_info.st_mode)
        or (hasattr(os, 'getuid') and parent_info.st_uid != os.getuid())
        or (os.name != 'nt' and stat.S_IMODE(parent_info.st_mode) & 0o022)
    ):
        raise OSError(f'Finder export lease parent is not private from other users: {path.parent}')
    info = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISDIR(info.st_mode)
        or (os.name != 'nt' and stat.S_IMODE(info.st_mode) != 0o700)
        or (hasattr(os, 'getuid') and info.st_uid != os.getuid())
    ):
        raise OSError(f'Finder export lease directory is not private: {path}')


def validate_private_finder_reference(path: Path) -> None:
    """Reject a Finder lease unless its directory and file are private and owned here."""
    validate_private_finder_reference_dir(path.parent)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or (os.name != 'nt' and stat.S_IMODE(info.st_mode) != 0o600)
        or (hasattr(os, 'getuid') and info.st_uid != os.getuid())
    ):
        raise OSError(f'Finder export lease file is not private: {path}')


def read_private_finder_reference(path: Path) -> dict[str, object]:
    """Read a lease only after checking owner, mode, and regular-file type."""
    validate_private_finder_reference(path)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or (os.name != 'nt' and stat.S_IMODE(info.st_mode) != 0o600)
            or (hasattr(os, 'getuid') and info.st_uid != os.getuid())
        ):
            raise OSError(f'Finder export lease file changed while opening: {path}')
        with os.fdopen(descriptor, 'r', encoding='utf-8') as reference_file:
            descriptor = -1
            data = json.load(reference_file)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(data, dict):
        raise TypeError(f'Finder export lease is not an object: {path}')
    return data


def _initial_worker_state(path: Path, snapshot: Path | None) -> None:
    _write_worker_state(
        path,
        {
            'pid': None,
            'heartbeat_at': time.time(),
            'snapshot_path': str(snapshot) if snapshot is not None else None,
        },
    )


class InsightsLaunchSpec(BaseModel):
    """Small, allow-listed subset of the Insights CLI for the dashboard wizard."""

    name: str = Field(default='', max_length=80)
    source: Source = 'recent'
    query: str = Field(default='', max_length=500)
    finder_export: str = Field(default='', max_length=4096)
    window_days: int = Field(default=7, ge=1, le=90)
    limit: int = Field(default=100, ge=1, le=5000)
    facets: FacetSelection = FacetSelection()
    parallelism: int = Field(default=20, ge=1, le=200)
    labels: list[Preset] = Field(default_factory=list)
    dimensions: list[DimensionName] = Field(default_factory=lambda: ['intent'])
    _finder_export_snapshot: str | None = PrivateAttr(default=None)

    def validated_finder_export_snapshot(self) -> str | None:
        """Return the bounded Finder JSON captured during source validation."""
        return self._finder_export_snapshot

    @model_validator(mode='after')
    def validate_source(self) -> Self:
        if self.source == 'query' and not self.query.strip():
            raise ValueError('Enter a question to find matching traces.')
        if self.source == 'finder':
            if self.facets != FacetSelection():
                raise ValueError('A Finder export already fixes the trace population; remove the facet filters.')
            if not self.finder_export.strip():
                raise ValueError('Enter the path to a Finder JSON export.')
            try:
                root = get_finder_exports_dir().resolve()
                requested = Path(self.finder_export).expanduser()
                path = (requested if requested.is_absolute() else root / requested).resolve()
            except (OSError, RuntimeError) as exc:
                raise ValueError(f'Could not resolve Finder export path: {exc}') from exc
            if path.parent != root:
                raise ValueError(f'Finder exports must be in {root}.')
            try:
                if not path.is_file():
                    raise ValueError('Finder export must be a regular file.')
                with path.open('rb') as export_file:
                    raw = export_file.read(MAX_FINDER_EXPORT_BYTES + 1)
                if len(raw) > MAX_FINDER_EXPORT_BYTES:
                    raise ValueError(
                        f'Finder export exceeds the {MAX_FINDER_EXPORT_BYTES // (1024 * 1024)} MiB size limit.'
                    )
                RunExport.model_validate_json(raw)
                self._finder_export_snapshot = raw.decode('utf-8')
            except (OSError, ValueError) as exc:
                raise ValueError(f'Could not read a valid Finder export: {exc}') from exc
            self.finder_export = str(path)
        if not self.labels and not self.dimensions:
            raise ValueError('Select at least one label or dimension.')
        if len(set(self.labels)) != len(self.labels) or len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError('Select each label and dimension once.')
        return self

    def population(self) -> InsightsPopulation:
        if self.source == 'finder':
            return InsightsPopulation.from_finder_export(Path(self.finder_export).expanduser())
        return InsightsPopulation(
            query=self.query.strip() if self.source == 'query' else None,
            facets=self.facets,
            window_days=self.window_days,
            limit=self.limit,
        )

    def label_specs(self) -> list[LabelSpec]:
        return [LABEL_PRESETS[name] for name in self.labels]

    def dimension_names(self) -> list[DimensionName]:
        return list(self.dimensions)


class InsightsLaunchPayload(BaseModel):
    run_id: str
    run_name: str
    runs_dir: Path
    spec: InsightsLaunchSpec
    finder_export_snapshot: Path | None = None


def launch_insights(spec: InsightsLaunchSpec, runs_dir: Path, *, profile: OrqProfile | None = None) -> str:
    """Create a visible manifest, then spawn a worker that survives dashboard reloads."""
    run_id = str(uuid.uuid4())
    run_name = spec.name.strip() or f'Insights {datetime.now().astimezone():%Y-%m-%d %H:%M}'
    plan = stage_plan(spec.population(), spec.label_specs(), spec.dimension_names())
    writer = start_manifest(
        run_id=run_id,
        surface='insights',
        run_name=run_name,
        runs_dir=runs_dir,
        planned_stages=[name for name, _ in plan],
        stage_labels=dict(plan),
    )
    snapshot_path: Path | None = None
    reference_path: Path | None = None
    reference_temporary: Path | None = None
    state_path = worker_state_path(runs_dir, run_id)
    try:
        finder_snapshot = spec.validated_finder_export_snapshot()
        if finder_snapshot is not None:
            reference_path = finder_export_reference_path(runs_dir, run_id)
            ensure_private_finder_reference_dir(reference_path.parent)
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=reference_path.parent, prefix=f'.{run_id}.', suffix='.tmp', delete=False
            ) as reference_file:
                reference_temporary = Path(reference_file.name)
                os.fchmod(reference_file.fileno(), 0o600)
                reference_file.write(json.dumps({'finder_export': spec.finder_export}))
                reference_file.flush()
                os.fsync(reference_file.fileno())
            reference_temporary.replace(reference_path)
            reference_temporary = None
            validate_private_finder_reference(reference_path)
            snapshot_directory = Path(tempfile.mkdtemp(prefix='evaluatorq-finder-snapshot-'))
            snapshot_path = snapshot_directory / 'finder-export.json'
            descriptor = os.open(snapshot_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w', encoding='utf-8') as snapshot_file:
                snapshot_file.write(finder_snapshot)
        # Serialize the already validated spec without asking Pydantic to validate
        # the mutable Finder path a second time.
        worker_request = json.dumps({
            'run_id': run_id,
            'run_name': run_name,
            'runs_dir': str(runs_dir),
            'spec': spec.model_dump(mode='json'),
            'finder_export_snapshot': str(snapshot_path) if snapshot_path is not None else None,
        })
        worker_env = {
            **os.environ,
            _REQUEST_ENV: worker_request,
            _MANIFEST_ENV: str(writer.path),
        }
        if snapshot_path is not None:
            worker_env[_SNAPSHOT_ENV] = str(snapshot_path)
        else:
            worker_env.pop(_SNAPSHOT_ENV, None)
        _initial_worker_state(state_path, snapshot_path)
        if profile is not None:
            worker_env.update(ORQ_API_KEY=profile.api_key, ORQ_BASE_URL=profile.server or DEFAULT_ORQ_BASE_URL)
        log_dir = runs_dir / '.logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f'{run_id}.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(
                [sys.executable, '-c', _WORKER_BOOTSTRAP],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=worker_env,
                start_new_session=True,
            )
        try:
            state = json.loads(state_path.read_text(encoding='utf-8'))
            state['pid'] = process.pid
            state['process_identity'] = _worker_process_identity(process.pid)
            _write_worker_state(state_path, state)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning('Could not record Insights worker process {}: {}', run_id, exc)
    except Exception as exc:  # noqa: BLE001 — every ordinary setup failure must terminate this run.
        try:
            writer.fail(f'Could not start Insights worker: {exc}', stage='start')
        finally:
            for path in (reference_path, reference_temporary):
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError as cleanup_error:
                        logger.warning('Could not remove Finder export reference {}: {}', path, cleanup_error)
            try:
                state_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                logger.warning('Could not remove Insights worker state {}: {}', state_path, cleanup_error)
            if snapshot_path is not None:
                try:
                    shutil.rmtree(snapshot_path.parent)
                except OSError as cleanup_error:
                    logger.warning(
                        'Could not remove Finder snapshot directory {}: {}', snapshot_path.parent, cleanup_error
                    )
    return run_id


def read_launch_payload() -> InsightsLaunchPayload:
    raw = os.environ.pop(_REQUEST_ENV, None)
    if raw is None:
        raise ValueError('Missing Insights launch request')
    data = json.loads(raw)
    spec_data = dict(data['spec'])
    finder_source = spec_data.get('source') == 'finder'
    finder_path = spec_data.get('finder_export', '')
    if finder_source:
        # Validate all user-selected options while neutralizing the original
        # pathname, which may have changed since launch validation.
        spec_data['source'] = 'recent'
        spec_data['finder_export'] = ''
    data['spec'] = spec_data
    payload = InsightsLaunchPayload.model_validate(data)
    if finder_source:
        payload.spec = payload.spec.model_copy(update={'source': 'finder', 'finder_export': finder_path})
    return payload
