"""Numerical calibration and mask operations for the BSA analyzer."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from bsa.utils.image_input import ImageInputError, decode_raster_image


MaskReader = Callable[[str, int], Any]


def calibrate_cube(
    data_array: np.ndarray,
    dark_array: np.ndarray,
    white_array: np.ndarray,
) -> np.ndarray:
    """Apply BSA's PlantScreen PSI-validated per-band calibration calculation.

    For every band the calculation is::

        (data[:, :, band] - dark[0, :, band]) / white[0, :, band]

    This is validated application behavior, not the conventional
    ``(data - dark) / (white - dark)`` alternative. ``zeros_like`` and
    per-band assignment deliberately preserve the input data array's dtype
    and the baseline's broadcasting/casting behavior.
    """

    _, _, band_count = data_array.shape
    calibrated_data = np.zeros_like(data_array)
    for band in range(band_count):
        calibrated_data[:, :, band] = (
            data_array[:, :, band] - dark_array[0, :, band]
        ) / white_array[0, :, band]
    return np.array(calibrated_data)


def load_grayscale_mask(
    mask_path: os.PathLike[str] | str,
    reader: MaskReader | None = None,
    grayscale_flag: int | None = None,
    expected_shape: tuple[int, int] | None = None,
) -> Any:
    """Decode a mask once from a pinned regular file, or use an injected reader."""

    if reader is None:
        image = decode_raster_image(
            mask_path, mask=True, expected_shape=expected_shape
        )
        try:
            if image.mode == "1":
                converted = image.convert("L")
                try:
                    result = np.asarray(converted, dtype=np.uint8).copy()
                finally:
                    converted.close()
            else:
                result = np.asarray(image, dtype=np.uint8).copy()
        finally:
            image.close()
        if result.ndim != 2:
            raise ImageInputError("Mask must decode to grayscale pixels.")
        return result
    try:
        result = reader(str(mask_path), grayscale_flag)
    except Exception as error:
        raise ImageInputError(f"Cannot decode mask {mask_path}: {error}.") from error
    return result


def validate_mask_array(mask: Any, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Require an unambiguous grayscale mask with at least one selected pixel."""

    if mask is None:
        raise ImageInputError("Mask could not be decoded.")
    mask = np.asarray(mask)
    if mask.ndim != 2 or not all(mask.shape):
        raise ImageInputError("Mask must have nonempty two-dimensional grayscale pixels.")
    if expected_shape is not None and mask.shape != tuple(expected_shape):
        raise ImageInputError(
            f"Mask size {mask.shape[0]}x{mask.shape[1]} does not match acquisition "
            f"{expected_shape[0]}x{expected_shape[1]}."
        )
    if np.issubdtype(mask.dtype, np.floating) and not np.isfinite(mask).all():
        raise ImageInputError("Mask contains a non-finite pixel value.")
    if mask.dtype != np.uint8:
        raise ImageInputError(f"Mask must decode to grayscale uint8, not {mask.dtype}.")
    if not np.any(mask >= 128):
        raise ImageInputError("Mask selects no pixels at the >=128 boundary.")
    return mask


def mask_to_nan(mask: Any) -> np.ndarray:
    """Select the plant half of an 8-bit sigmoid probability mask.

    Mask Creator stores ``probability * 255`` as uint8. The supplied BARLEY
    SWIR model was trained with binary Precision and Recall at a 0.5 boundary;
    128 is the nearest integer boundary in the stored 0..255 image. Binary
    0/255 masks retain their original interpretation.
    """

    mask = validate_mask_array(mask)
    return np.where(mask >= 128, 1.0, np.nan)


def apply_nan_mask(calibrated_data: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
    """Apply the spatial mask independently to every spectral band."""

    _, _, band_count = calibrated_data.shape
    data_masked = np.zeros_like(calibrated_data)
    for band in range(band_count):
        data_masked[:, :, band] = calibrated_data[:, :, band] * nan_mask
    return data_masked


def load_and_apply_mask(
    calibrated_data: np.ndarray,
    mask_path: os.PathLike[str] | str,
    reader: MaskReader | None = None,
    grayscale_flag: int | None = None,
) -> np.ndarray:
    """Load and apply one mask at the stored probability boundary, without resizing."""

    mask = load_grayscale_mask(mask_path, reader=reader, grayscale_flag=grayscale_flag,
                               expected_shape=calibrated_data.shape[:2])
    validate_mask_array(mask, calibrated_data.shape[:2])
    return apply_nan_mask(calibrated_data, mask_to_nan(mask))
