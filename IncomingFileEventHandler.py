"""
ECT_ImagesWatcher.py

General-purpose folder watcher + multithreaded processor.

This module wires together:
- watchdog.Observer for filesystem notifications
- IncomingFileEventHandler for filtering and event handling
- ThreadPoolExecutor for bounded concurrency
- A file "stabilization" wait (size + mtime unchanged) to avoid processing
  partially-uploaded files (common with SFTP drops)

Usage example
-------------
from pathlib import Path
from ECT_ImagesWatcher import FolderWatcher
from ECT_IncomingFileEventHandler import FileEventFilter

def process_file(path: Path) -> None:
    # Your real logic here
    print(f"Processing: {path} ({path.stat().st_size} bytes)")

watcher = FolderWatcher(
    watch_path="/srv/sftp/incoming",
    processor=process_file,
    patterns=["*.gz", "*.grb2", "*.png"],   # optional
    max_workers=8,
)
watcher.run_forever()
"""

from __future__ import annotations

import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Callable, Iterable, Optional
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future

try:
    from watchdog.observers import Observer
except ModuleNotFoundError:
    class Observer:
        """Lightweight polling observer fallback when watchdog is unavailable."""

        def __init__(self) -> None:
            self._watches: list[tuple[object, Path, bool]] = []
            self._state: dict[tuple[object, Path], dict[Path, tuple[int, int]]] = {}
            self._stop_event = threading.Event()
            self._thread: Optional[threading.Thread] = None

        def schedule(self, handler: object, path: str, recursive: bool = True) -> None:
            root = Path(path)
            key = (handler, root)
            self._watches.append((handler, root, recursive))
            self._state[key] = {}

        def start(self) -> None:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def stop(self) -> None:
            self._stop_event.set()

        def join(self, timeout: Optional[float] = None) -> None:
            if self._thread is not None:
                self._thread.join(timeout)

        def _run(self) -> None:
            while not self._stop_event.is_set():
                for handler, root, recursive in self._watches:
                    key = (handler, root)
                    prev = self._state.get(key, {})
                    curr: dict[Path, tuple[int, int]] = {}

                    if root.exists():
                        iterator = root.rglob("*") if recursive else root.glob("*")
                        for p in iterator:
                            try:
                                if p.is_dir():
                                    continue
                                st = p.stat()
                                sig = (st.st_size, st.st_mtime_ns)
                            except OSError:
                                continue

                            curr[p] = sig
                            if p not in prev or prev[p] != sig:
                                if hasattr(handler, "_maybe_emit"):
                                    handler._maybe_emit(p, False)  # type: ignore[attr-defined]

                    self._state[key] = curr

                self._stop_event.wait(1.0)

from ImagesWatcher import IncomingFileEventHandler, FileEventFilter


def build_rotating_logger(
    name: str,
    log_path: str = "file_watcher.log",
    level: int = logging.INFO,
    when: str = "midnight",
    backup_count: int = 14,
) -> logging.Logger:
    """
    Create a simple rotating file logger.

    Parameters
    ----------
    name:
        Logger name.
    log_path:
        Log file path.
    level:
        Logging level.
    when:
        Rotation interval passed to TimedRotatingFileHandler (e.g., "midnight").
    backup_count:
        Number of rotated logs to keep.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

        handler = TimedRotatingFileHandler(
            log_path,
            when=when,
            backupCount=backup_count,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Also log to console by default for dev convenience.
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


class FolderWatcher:
    """
    Watch a folder and process files using a bounded pool of worker threads.

    Parameters
    ----------
    watch_path:
        Folder to watch.
    processor:
        Callable invoked for each file (after it's stable). Signature: (Path) -> None
    patterns:
        Optional glob patterns for files to watch (e.g., ["*.gz", "*.png"]).
    predicate:
        Optional advanced filter predicate: (Path) -> bool
    recursive:
        If True, watch all subfolders.
    max_workers:
        Maximum number of concurrent processing threads.
    stable_checks:
        Number of consecutive "unchanged" checks required to consider a file stable.
    check_interval_seconds:
        Seconds between stability checks.
    stable_timeout_seconds:
        Maximum time to wait for a file to become stable.
    on_success:
        Optional callback invoked after successful processing. Signature: (Path) -> None
    on_error:
        Optional callback invoked on processing error. Signature: (Path, Exception) -> None
    logger:
        Optional logger. If None, a rotating logger is created.
    """

    def __init__(
        self,
        watch_path: str | Path,
        processor: Callable[[Path], None],
        *,
        patterns: Optional[Iterable[str]] = None,
        predicate: Optional[Callable[[Path], bool]] = None,
        recursive: bool = True,
        max_workers: int = 4,
        stable_checks: int = 3,
        check_interval_seconds: float = 1.0,
        stable_timeout_seconds: float = 10 * 60,
        on_success: Optional[Callable[[Path], None]] = None,
        on_error: Optional[Callable[[Path, Exception], None]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.watch_path = Path(watch_path)
        self.processor = processor
        self.recursive = recursive

        self.max_workers = max_workers
        self.stable_checks = stable_checks
        self.check_interval_seconds = check_interval_seconds
        self.stable_timeout_seconds = stable_timeout_seconds

        self.on_success = on_success
        self.on_error = on_error

        self.logger = logger or build_rotating_logger(self.__class__.__name__)

        self._stop_event = threading.Event()
        self._observer = Observer()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Avoid duplicate submissions for the same path.
        self._inflight: set[Path] = set()
        self._inflight_lock = threading.Lock()

        file_filter = FileEventFilter(patterns=patterns, predicate=predicate)
        self._handler = IncomingFileEventHandler(
            on_file_detected=self.submit,
            file_filter=file_filter,
            debounce_seconds=1.0,
        )

    # -------------------------
    # Lifecycle
    # -------------------------

    def start(self) -> None:
        """Start observing filesystem events."""
        self.watch_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Starting watcher: path=%s recursive=%s max_workers=%s",
            str(self.watch_path),
            self.recursive,
            self.max_workers,
        )
        self._observer.schedule(self._handler, str(self.watch_path), recursive=self.recursive)
        self._observer.start()

    def stop(self) -> None:
        """Stop observing and shut down worker threads."""
        if self._stop_event.is_set():
            return

        self.logger.info("Stopping watcher...")
        self._stop_event.set()

        try:
            self._observer.stop()
            self._observer.join()
        finally:
            # Cancel pending futures (Python 3.9+ supports cancel_futures)
            self._executor.shutdown(wait=True, cancel_futures=True)

        self.logger.info("Watcher stopped.")

    def run_forever(self, poll_seconds: float = 1.0) -> None:
        """
        Convenience main loop.

        Ctrl+C will stop cleanly.
        """
        self.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received.")
        finally:
            self.stop()

    # -------------------------
    # Submission + processing
    # -------------------------

    def submit(self, path: Path) -> None:
        """
        Submit a file for processing.

        This is typically called by the event handler. It is safe to call directly
        if you want to enqueue existing files on startup, etc.
        """
        # Basic sanity checks
        try:
            if not path.exists() or path.is_dir():
                return
        except OSError:
            return

        # Deduplicate inflight work
        with self._inflight_lock:
            if path in self._inflight:
                return
            self._inflight.add(path)

        self.logger.info("Queued: %s", str(path))
        future = self._executor.submit(self._process_one_file, path)
        future.add_done_callback(lambda f: self._on_done(path, f))

    def _process_one_file(self, path: Path) -> None:
        """
        Worker function:

        1) Wait for file upload/copy to finish (stable size + mtime).
        2) Call user processor(path).
        """
        self._wait_until_stable(path)
        self.processor(path)

    def _on_done(self, path: Path, future: Future) -> None:
        """Handle completion, remove inflight tracking, and fire callbacks."""
        with self._inflight_lock:
            self._inflight.discard(path)

        exc = future.exception()
        if exc is None:
            self.logger.info("Processed OK: %s", str(path))
            if self.on_success:
                try:
                    self.on_success(path)
                except Exception as callback_exc:
                    self.logger.exception("on_success callback failed: %s", callback_exc)
            return

        self.logger.exception("Processing FAILED: %s (%s)", str(path), exc)
        if self.on_error:
            try:
                self.on_error(path, exc)
            except Exception as callback_exc:
                self.logger.exception("on_error callback failed: %s", callback_exc)

    # -------------------------
    # File stabilization
    # -------------------------

    def _wait_until_stable(self, path: Path) -> None:
        """
        Wait until a file is stable before processing.

        Stability definition:
            The pair (size, mtime_ns) is unchanged for `stable_checks` consecutive checks.

        Raises
        ------
        TimeoutError:
            If stability is not achieved within stable_timeout_seconds.
        FileNotFoundError:
            If the file disappears during the wait and never returns.
        """
        deadline = time.time() + self.stable_timeout_seconds

        last_state: tuple[int, int] | None = None
        stable_count = 0

        self.logger.debug("Waiting for stable file: %s", str(path))

        while time.time() < deadline:
            if self._stop_event.is_set():
                return

            try:
                st = path.stat()
                state = (st.st_size, st.st_mtime_ns)
            except FileNotFoundError:
                stable_count = 0
                last_state = None
                time.sleep(self.check_interval_seconds)
                continue

            if last_state == state:
                stable_count += 1
                if stable_count >= self.stable_checks:
                    self.logger.debug("File is stable: %s", str(path))
                    return
            else:
                stable_count = 0
                last_state = state

            time.sleep(self.check_interval_seconds)

        raise TimeoutError(f"File did not stabilize in time: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Watch a folder and run capstone.py on each incoming FITS file."
    )
    ap.add_argument("--watch-path", default="FITSfileDropFolder", help="Folder to watch.")
    ap.add_argument("--outdir", default="outputs", help="Output directory for capstone.py.")
    ap.add_argument(
        "--scan-existing",
        action="store_true",
        help="Process existing FITS files already in the watch folder on startup.",
    )
    ap.add_argument(
        "--no-star-streak",
        action="store_true",
        help="Disable star streak detection in capstone.",
    )
    ap.add_argument(
        "--patterns",
        nargs="*",
        default=["*.fit", "*.fits", "*.fts"],
        help="File patterns to process.",
    )
    ap.add_argument("--max-workers", type=int, default=4, help="Parallel file workers.")
    args = ap.parse_args()

    def _capstone_processor(p: Path) -> None:
        from argparse import Namespace
        from capstone import process_one_file

        cap_args = Namespace(
            fits=str(p),
            outdir=args.outdir,
            ext=None,
            batch=False,
            data_dir="data",
            bg_tile=64,
            smooth=1.0,
            zeroth_box_w=100,
            zeroth_box_h=100,
            zeroth_step=4,
            zeroth_score_mode="compact_flux",
            first_fixed_w=400,
            first_fixed_h=100,
            first_pad=5,
            first_inner_w=21,
            first_inner_h=21,
            pre=30.0,
            profile_on="bgsub",
            width=5,
            reducer="mean",
            step=1.0,
            no_star_streak=args.no_star_streak,
        )

        rc = process_one_file(p, cap_args)
        if rc != 0:
            raise RuntimeError(f"capstone processing failed for {p} (exit code {rc})")
        print(f"[DB Pipeline] processed {p.name}")

    watcher = FolderWatcher(
        watch_path=args.watch_path,
        processor=_capstone_processor,
        patterns=args.patterns,
        max_workers=args.max_workers,
    )
    if args.scan_existing:
        base_dir = Path(args.watch_path)
        for existing in sorted(base_dir.glob("*")):
            if existing.is_file() and any(existing.match(pat) for pat in args.patterns):
                watcher.submit(existing)
    watcher.run_forever()
