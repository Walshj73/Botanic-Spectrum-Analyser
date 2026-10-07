"""Runtime paths and lightweight application support shared by both GUIs."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, TypeVar

from bsa.utils.read_paths import ReadPathError, open_regular_file


class ModelFileError(ValueError):
    """Raised when a segmentation model cannot be used."""


ModelType = TypeVar("ModelType")
_ASSET_NAMES = {"BSA_logo.ico", "BSA_logo.png"}


def application_directory() -> Path:
    """Return the external-resource directory for source or frozen execution."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # resources.py -> utils -> bsa -> src -> repository root
    return Path(__file__).resolve().parents[3]


def assets_directory() -> Path:
    """Return the package asset directory, including PyInstaller extraction."""

    if getattr(sys, "frozen", False):
        extraction_root = Path(getattr(sys, "_MEIPASS", application_directory()))
        candidates = (
            extraction_root / "bsa" / "assets",
            extraction_root / "assets",
            application_directory() / "assets",
        )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return candidates[0]
    return Path(__file__).resolve().parents[1] / "assets"


def resource_path(*parts: str, base_directory: Path | None = None) -> Path:
    """Resolve external resources or bundled assets from any working directory."""

    if base_directory is not None:
        return Path(base_directory).joinpath(*parts)
    if len(parts) == 1 and parts[0] in _ASSET_NAMES:
        return assets_directory() / parts[0]
    return application_directory().joinpath(*parts)


def validate_model_file(model_path: os.PathLike[str] | str) -> Path:
    """Validate and return the resolved ordinary-file target of a model path."""

    try:
        with open_regular_file(model_path, "Model file") as (source, resolved):
            if os.fstat(source.fileno()).st_size == 0:
                raise ModelFileError(
                    f"Model file is empty: {resolved.path}. Install a trained model or choose another file."
                )
            return resolved.path
    except ReadPathError as error:
        raise ModelFileError(
            f"Model file not found or not a readable regular file: {model_path}. {error}"
        ) from error


def load_model_file(
    model_path: os.PathLike[str] | str,
    loader: Callable[[str], ModelType],
) -> ModelType:
    """Validate and load a model, translating loader failures into a clear error."""

    path = validate_model_file(model_path)
    try:
        return loader(str(path))
    except ModelFileError:
        raise
    except Exception as exc:
        raise ModelFileError(
            f"Model file is invalid or incompatible and could not be loaded: {path}"
        ) from exc


def open_in_file_manager(path: os.PathLike[str] | str) -> bool:
    """Open a directory with the platform file manager.

    Failure to find a desktop opener is non-fatal because the requested output
    has already been written when this helper is normally called.
    """

    target = str(Path(path).resolve())
    try:
        if sys.platform == "win32":
            os.startfile(target)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
    except OSError:
        return False
    return True
