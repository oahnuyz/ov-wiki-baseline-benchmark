"""Out-of-project snapshots for retrying an ingest batch from clean state."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimedSnapshotOperation:
    duration_seconds: float


class ProjectSnapshotManager:
    def __init__(self, *, project_path: Path, snapshot_root: Path) -> None:
        self.project_path = project_path.resolve()
        self.snapshot_root = snapshot_root.resolve()
        if (
            self.snapshot_root == self.project_path
            or self.snapshot_root in self.project_path.parents
            or self.project_path in self.snapshot_root.parents
        ):
            raise ValueError("snapshot_root and project_path must not contain each other")

    def batch_path(self, corpus_id: str, batch_index: int) -> Path:
        digest = hashlib.sha256(corpus_id.encode("utf-8")).hexdigest()[:20]
        return self.snapshot_root / f"corpus-{digest}" / f"batch-{batch_index:04d}"

    def create(self, corpus_id: str, batch_index: int) -> TimedSnapshotOperation:
        started = time.monotonic()
        destination = self.batch_path(corpus_id, batch_index)
        temporary = destination.with_name(destination.name + ".creating")
        self._remove(temporary)
        self._remove(destination)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.project_path, temporary, copy_function=shutil.copy2)
        os.replace(temporary, destination)
        return TimedSnapshotOperation(time.monotonic() - started)

    def restore(self, corpus_id: str, batch_index: int) -> TimedSnapshotOperation:
        started = time.monotonic()
        source = self.batch_path(corpus_id, batch_index)
        if not source.is_dir():
            raise RuntimeError(f"Batch snapshot does not exist: {source}")
        discarded = source.parent / f"{source.name}.discarded-project"
        self._remove(discarded)
        if self.project_path.exists():
            os.replace(self.project_path, discarded)
        try:
            shutil.copytree(source, self.project_path, copy_function=shutil.copy2)
        except BaseException:
            self._remove(self.project_path)
            if discarded.exists():
                os.replace(discarded, self.project_path)
            raise
        self._remove(discarded)
        return TimedSnapshotOperation(time.monotonic() - started)

    def delete_batch(self, corpus_id: str, batch_index: int) -> TimedSnapshotOperation:
        started = time.monotonic()
        destination = self.batch_path(corpus_id, batch_index)
        self._remove(destination)
        parent = destination.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
        return TimedSnapshotOperation(time.monotonic() - started)

    def cleanup_all(self) -> TimedSnapshotOperation:
        started = time.monotonic()
        if self.snapshot_root.is_dir():
            for child in self.snapshot_root.iterdir():
                self._remove(child)
        return TimedSnapshotOperation(time.monotonic() - started)

    def _remove(self, path: Path) -> None:
        resolved_parent = path.parent.resolve()
        if resolved_parent != self.snapshot_root and self.snapshot_root not in resolved_parent.parents:
            raise RuntimeError(f"Refusing to remove path outside snapshot_root: {path}")
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
