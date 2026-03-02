"""
ECT_IncomingFileEventHandler.py

A general-purpose watchdog event handler that:

1) Filters file events (optional patterns + predicate).
2) Debounces duplicate triggers.
3) Delegates actual work to a callback (typically a thread-pooled submit function).

This file is intentionally task-agnostic: no knowledge of file types, products,
SFTP, GDAL, etc. That all belongs in user-provided callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
import fnmatch
import threading
import time

try:
    from watchdog.events import FileSystemEventHandler, FileSystemEvent, FileMovedEvent
except ModuleNotFoundError:
    class FileSystemEvent:
        def __init__(self, src_path: str, is_directory: bool = False) -> None:
            self.src_path = src_path
            self.is_directory = is_directory

    class FileMovedEvent(FileSystemEvent):
        def __init__(self, src_path: str, dest_path: str, is_directory: bool = False) -> None:
            super().__init__(src_path=src_path, is_directory=is_directory)
            self.dest_path = dest_path

    class FileSystemEventHandler:
        pass


@dataclass(frozen=True)
class FileEventFilter:
    """
    Filtering rules for which filesystem paths should be processed.

    Attributes
    ----------
    patterns:
        Optional glob patterns (e.g., ["*.gz", "*.grb2", "*.png"]). If provided,
        the file must match at least one.
    predicate:
        Optional callable that returns True if the path should be processed.
        Use this for advanced checks (e.g., ignore temp names, size thresholds,
        subfolder rules, etc.).
    ignore_directories:
        If True, directory events are ignored (recommended).
    """
    patterns: Optional[Iterable[str]] = None
    predicate: Optional[Callable[[Path], bool]] = None
    ignore_directories: bool = True

    def matches(self, path: Path, is_directory: bool) -> bool:
        """Return True if a given path should be processed."""
        if self.ignore_directories and is_directory:
            return False

        if self.patterns:
            name = path.name
            if not any(fnmatch.fnmatch(name, pat) for pat in self.patterns):
                return False

        if self.predicate and not self.predicate(path):
            return False

        return True


class IncomingFileEventHandler(FileSystemEventHandler):
    """
    A general-purpose watchdog handler.

    Parameters
    ----------
    on_file_detected:
        Callback invoked with the filesystem path whenever a file event
        qualifies for processing. In practice, this is usually something
        like `watcher.submit(path)` which hands work to a thread pool.
    file_filter:
        Optional FileEventFilter that controls which files are processed.
    debounce_seconds:
        Many environments can emit multiple events for the same file. This
        prevents re-triggering within a short time window.
    """

    def __init__(
        self,
        on_file_detected: Callable[[Path], None],
        file_filter: Optional[FileEventFilter] = None,
        debounce_seconds: float = 1.0,
    ) -> None:
        super().__init__()
        self._on_file_detected = on_file_detected
        self._filter = file_filter or FileEventFilter()
        self._debounce_seconds = debounce_seconds

        # Track recently-seen paths to reduce duplicate triggers.
        self._recent: dict[Path, float] = {}
        self._lock = threading.Lock()

    def on_created(self, event: FileSystemEvent) -> None:
        self._maybe_emit(Path(event.src_path), event.is_directory)

    def on_moved(self, event: FileMovedEvent) -> None:
        # Many “atomic write” workflows create temp files then rename/move into place.
        self._maybe_emit(Path(event.dest_path), event.is_directory)

    def _maybe_emit(self, path: Path, is_directory: bool) -> None:
        """Apply filters + debounce; if accepted, invoke callback."""
        if not self._filter.matches(path, is_directory):
            return

        now = time.time()
        with self._lock:
            last = self._recent.get(path)
            if last is not None and (now - last) < self._debounce_seconds:
                return
            self._recent[path] = now

        self._on_file_detected(path)
