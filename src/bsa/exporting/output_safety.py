"""Portable output planning and no-replacement publication."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import unicodedata
from typing import Callable, Iterable, Sequence


class OutputCollisionError(ValueError):
    """Two outputs collide, or a destination is already occupied."""


@dataclass(frozen=True)
class PlannedOutput:
    source: str
    kind: str
    path: Path


def _portable_name(name: str) -> str:
    """Conservatively compare Windows/macOS spellings on every platform."""

    return unicodedata.normalize("NFC", name.rstrip(" .").casefold())


def _destination_key(path: Path) -> tuple[str, str]:
    parent = path.parent.resolve(strict=False)
    return _portable_name(str(parent)), _portable_name(path.name)


def validate_output_plan(
    outputs: Iterable[PlannedOutput], *, check_existing: bool = True,
) -> tuple[PlannedOutput, ...]:
    """Reject duplicate planned names and existing portable aliases."""

    plan = tuple(outputs)
    used: dict[tuple[str, str], PlannedOutput] = {}
    for item in plan:
        key = _destination_key(item.path)
        previous = used.get(key)
        if previous is not None:
            raise OutputCollisionError(
                f"Output collision: {previous.kind} {previous.source!r} and "
                f"{item.kind} {item.source!r} both target {item.path.name!r}."
            )
        used[key] = item

    if not check_existing:
        return plan

    existing: dict[Path, set[str]] = {}
    for item in plan:
        parent = item.path.parent.resolve(strict=False)
        name = _portable_name(item.path.name)
        if parent not in existing:
            if parent.exists() and not parent.is_dir():
                raise OutputCollisionError(f"Output folder is not a directory: {parent}.")
            existing[parent] = (
                {_portable_name(entry.name) for entry in parent.iterdir()}
                if parent.is_dir() else set()
            )
        if name in existing[parent]:
            raise OutputCollisionError(
                f"Output already exists: {item.path.name!r} in {item.path.parent}."
            )
    return plan


def available_output_plan(outputs: Iterable[PlannedOutput]) -> tuple[PlannedOutput, ...]:
    """Give automatic outputs unused numbered names, retaining no-clobber checks."""

    original = validate_output_plan(outputs, check_existing=False)
    existing: dict[Path, set[str]] = {}
    reserved = {_destination_key(item.path) for item in original}
    selected: list[PlannedOutput] = []
    for item in original:
        parent = item.path.parent.resolve(strict=False)
        if parent not in existing:
            if parent.exists() and not parent.is_dir():
                raise OutputCollisionError(f"Output folder is not a directory: {parent}.")
            existing[parent] = (
                {_portable_name(entry.name) for entry in parent.iterdir()}
                if parent.is_dir() else set()
            )
        path = item.path
        number = 0
        while _portable_name(path.name) in existing[parent] or (
            number and _destination_key(path) in reserved
        ):
            number += 1
            path = item.path.with_name(f"{item.path.stem}_{number:03d}{item.path.suffix}")
        reserved.add(_destination_key(path))
        existing[parent].add(_portable_name(path.name))
        selected.append(PlannedOutput(item.source, item.kind, path))
    return validate_output_plan(selected)


def publish_staged_outputs(
    staged: Sequence[tuple[PlannedOutput, Path]],
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> None:
    """Publish complete staged files without replacing a destination."""

    validate_output_plan(item for item, _ in staged)
    published: list[Path] = []
    try:
        for item, source in staged:
            if check_cancelled is not None:
                check_cancelled()
            # Recheck portable aliases before each publication. The final
            # creation is atomic for the exact name on the current filesystem.
            validate_output_plan((item,))
            try:
                if os.name == "nt":
                    os.rename(source, item.path)
                else:
                    os.link(source, item.path)
                published.append(item.path)
            except FileExistsError as error:
                raise OutputCollisionError(
                    f"Output already exists: {item.path.name!r} in {item.path.parent}."
                ) from error
        if check_cancelled is not None:
            check_cancelled()
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise


def write_output_bundle(
    outputs: Sequence[tuple[PlannedOutput, Callable[[Path], None]]],
) -> list[Path]:
    """Stage a same-folder bundle, then publish it as a no-clobber group."""

    plan = validate_output_plan(item for item, _ in outputs)
    if not plan:
        return []
    parents = {item.path.parent.resolve(strict=False) for item in plan}
    if len(parents) != 1:
        raise ValueError("An output bundle must use one destination directory.")
    parent = plan[0].path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bsa-output-", dir=parent) as temporary:
        staged = []
        for index, (item, writer) in enumerate(outputs):
            stage_path = Path(temporary) / f"output-{index}{item.path.suffix}"
            writer(stage_path)
            if not stage_path.is_file():
                raise OSError(f"Output writer did not create {item.path.name!r}.")
            staged.append((item, stage_path))
        publish_staged_outputs(staged)
    return [item.path for item in plan]


def write_output(
    item: PlannedOutput,
    writer: Callable[[Path], None],
) -> Path:
    return write_output_bundle(((item, writer),))[0]
