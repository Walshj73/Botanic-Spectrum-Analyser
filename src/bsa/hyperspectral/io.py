"""ENVI header parsing and cube loading for the BSA analyzer."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

from bsa.hyperspectral.envi_validation import (
    EnviMetadata, band_count, read_header_fields, require_file_unchanged,
    validate_envi_binary,
    validate_reference_triplet, wavelength_values,
)

EnviOpen = Callable[[str, str], Any]


def extract_wavelengths(envi_header_path: os.PathLike[str] | str) -> list[float]:
    """Extract one validated wavelength grid, preserving its stored order."""

    fields = read_header_fields(envi_header_path)
    return list(wavelength_values(fields, band_count(fields)) or ())


def extract_band_count(envi_header_path: os.PathLike[str] | str) -> int | None:
    """Read the ENVI bands field for selectors when wavelengths are absent."""

    return band_count(read_header_fields(envi_header_path))


def _default_envi_open() -> EnviOpen:
    from spectral.io import envi

    return envi.open


def open_envi_image(
    header_path: os.PathLike[str] | str,
    image_path: os.PathLike[str] | str,
    envi_open: EnviOpen | None = None,
) -> Any:
    """Preflight one ENVI binary before Spectral opens or allocates it."""

    image, _ = _open_validated_envi_image(
        header_path, image_path, envi_open=envi_open
    )
    return image


def _open_validated_envi_image(
    header_path: os.PathLike[str] | str,
    image_path: os.PathLike[str] | str,
    *,
    envi_open: EnviOpen | None,
    metadata: EnviMetadata | None = None,
) -> tuple[Any, EnviMetadata]:
    metadata = metadata or validate_envi_binary(header_path, image_path)
    if metadata.header_file is None or metadata.binary_file is None:
        raise ValueError("ENVI preflight did not retain resolved input file identities.")
    require_file_unchanged(metadata.header_file, "ENVI header")
    require_file_unchanged(metadata.binary_file, "ENVI binary")
    opener = envi_open if envi_open is not None else _default_envi_open()
    image = opener(str(metadata.header_file.path), str(metadata.binary_file.path))
    # Spectral receives canonical target paths, so changing the original
    # symlink cannot redirect it after validation.
    require_file_unchanged(metadata.header_file, "ENVI header")
    require_file_unchanged(metadata.binary_file, "ENVI binary")
    return image, metadata


def _load_pinned_image(image: Any, metadata: EnviMetadata) -> Any:
    assert metadata.header_file is not None and metadata.binary_file is not None
    require_file_unchanged(metadata.header_file, "ENVI header")
    require_file_unchanged(metadata.binary_file, "ENVI binary")
    loaded = image.load()
    # Spectral reopens binary data by pathname. Detect a replacement or an
    # ordinary in-place write during that read before returning its array.
    require_file_unchanged(metadata.header_file, "ENVI header")
    require_file_unchanged(metadata.binary_file, "ENVI binary")
    return loaded


def load_envi_cube(
    header_path: os.PathLike[str] | str,
    image_path: os.PathLike[str] | str,
    envi_open: EnviOpen | None = None,
) -> Any:
    """Open an ENVI image and return Spectral Python's unchanged ``load()`` result."""

    image, metadata = _open_validated_envi_image(
        header_path, image_path, envi_open=envi_open
    )
    return _load_pinned_image(image, metadata)


@dataclass(frozen=True)
class LoadedReferenceCubes:
    """The dark, data, and white arrays loaded for one sample set."""

    dark: Any
    data: Any
    white: Any


def load_reference_cubes(
    dark_header_path: os.PathLike[str] | str,
    data_header_path: os.PathLike[str] | str,
    white_header_path: os.PathLike[str] | str,
    dark_image_path: os.PathLike[str] | str,
    data_image_path: os.PathLike[str] | str,
    white_image_path: os.PathLike[str] | str,
    envi_open: EnviOpen | None = None,
) -> LoadedReferenceCubes:
    """Validate the entire triplet before opening/loading in baseline order."""

    dark_metadata = validate_envi_binary(dark_header_path, dark_image_path)
    data_metadata = validate_envi_binary(data_header_path, data_image_path)
    white_metadata = validate_envi_binary(white_header_path, white_image_path)
    validate_reference_triplet(dark_metadata, data_metadata, white_metadata)
    opener = envi_open if envi_open is not None else _default_envi_open()
    dark_image, dark_metadata = _open_validated_envi_image(
        dark_header_path, dark_image_path, envi_open=opener, metadata=dark_metadata
    )
    data_image, data_metadata = _open_validated_envi_image(
        data_header_path, data_image_path, envi_open=opener, metadata=data_metadata
    )
    white_image, white_metadata = _open_validated_envi_image(
        white_header_path, white_image_path, envi_open=opener, metadata=white_metadata
    )

    data_array = _load_pinned_image(data_image, data_metadata)
    dark_array = _load_pinned_image(dark_image, dark_metadata)
    white_array = _load_pinned_image(white_image, white_metadata)
    return LoadedReferenceCubes(dark=dark_array, data=data_array, white=white_array)
