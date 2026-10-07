"""Filesystem policy for researcher-selected and dataset-derived read paths.

Regular-file symlinks are intentional: the resolved target is pinned and
validated as a regular file. Special files are opened nonblocking and rejected
before a consumer can block on them.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import BinaryIO, Iterator


class ReadPathError(ValueError):
    """A selected path does not identify a stable readable regular file."""


@dataclass(frozen=True)
class FileIdentity:
    """Stable metadata used to notice ordinary local replacement or mutation."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, details: os.stat_result) -> "FileIdentity":
        return cls(
            details.st_dev,
            details.st_ino,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
        )


@dataclass(frozen=True)
class ResolvedFile:
    """Canonical pathname and identity of an opened regular file."""

    path: Path
    identity: FileIdentity


@contextmanager
def open_regular_file(
    path: os.PathLike[str] | str,
    description: str,
) -> Iterator[tuple[BinaryIO, ResolvedFile]]:
    """Open once, resolve the target, and validate the object through its fd.

    O_NONBLOCK prevents FIFO/device replacements from hanging before fstat.
    The resolved pathname is compared with the opened descriptor to catch a
    symlink or regular-file replacement during resolution. Consumers that
    later reopen by pathname must call :func:`require_unchanged` immediately
    before handing the path to that library.
    """

    selected = Path(path)
    descriptor: int | None = None
    source: BinaryIO | None = None
    canonical: Path | None = None
    try:
        # Resolve first and inspect the target before opening it. This rejects
        # known devices and sockets without asking their drivers to open.
        canonical = selected.resolve(strict=True)
        preflight_details = os.stat(canonical)
        if not stat.S_ISREG(preflight_details.st_mode):
            raise ReadPathError(f"{description} is not a regular file: {selected}.")
        preflight_identity = FileIdentity.from_stat(preflight_details)
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_BINARY", 0)
        descriptor = os.open(canonical, flags)
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise ReadPathError(f"{description} is not a regular file: {selected}.")
        identity = FileIdentity.from_stat(details)
        target_details = os.stat(canonical)
        if not stat.S_ISREG(target_details.st_mode):
            raise ReadPathError(f"{description} is not a regular file: {selected}.")
        if preflight_identity != identity or FileIdentity.from_stat(target_details) != identity:
            raise ReadPathError(
                f"{description} changed while its path was being validated: {selected}."
            )
        source = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = None
    except ReadPathError:
        if descriptor is not None:
            os.close(descriptor)
        if source is not None:
            source.close()
        raise
    except (OSError, RuntimeError, ValueError) as error:
        if descriptor is not None:
            os.close(descriptor)
        if source is not None:
            source.close()
        try:
            target_details = os.stat(selected)
        except OSError:
            target_details = None
        if target_details is not None and not stat.S_ISREG(target_details.st_mode):
            raise ReadPathError(f"{description} is not a regular file: {selected}.") from error
        if isinstance(error, (RuntimeError, ValueError)):
            raise ReadPathError(f"Cannot resolve {description} {selected}: {error}.") from error
        raise ReadPathError(f"Cannot open {description} {selected}: {error}.") from error
    try:
        assert source is not None
        with source:
            yield source, ResolvedFile(canonical, identity)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def require_unchanged(resolved: ResolvedFile, description: str) -> None:
    """Reject replacement or ordinary in-place mutation since validation."""

    try:
        details = os.stat(resolved.path)
    except OSError as error:
        raise ReadPathError(
            f"{description} changed after validation: {resolved.path}."
        ) from error
    if (
        not stat.S_ISREG(details.st_mode)
        or FileIdentity.from_stat(details) != resolved.identity
    ):
        raise ReadPathError(
            f"{description} changed after validation: {resolved.path}."
        )


def resolve_regular_file(
    path: os.PathLike[str] | str,
    description: str,
) -> ResolvedFile:
    """Capture a resolved regular-file target without retaining its handle."""

    with open_regular_file(path, description) as (_source, resolved):
        return resolved
