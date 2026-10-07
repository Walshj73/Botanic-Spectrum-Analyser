"""Bound untrusted raster decoding before OpenCV or Tk consumes an image."""

from __future__ import annotations

import os
from pathlib import Path
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from bsa.utils.read_paths import (
    FileIdentity, ReadPathError, ResolvedFile, open_regular_file,
)


class ImageInputError(ValueError):
    """An image is unreadable, unexpectedly encoded, or unsafe to decode."""


_SUFFIX_FORMATS = {
    ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
    ".tif": "TIFF", ".tiff": "TIFF", ".bmp": "BMP",
    ".webp": "WEBP", ".gif": "GIF",
}
_MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS or 89_478_485


def _checked_raster(
    path: os.PathLike[str] | str, *,
    mask: bool = False,
    expected_shape: tuple[int, int] | None = None,
    copy_pixels: bool,
    expected_file: ResolvedFile | None = None,
) -> Image.Image | tuple[int, int]:
    """Fully decode one regular raster while enforcing Pillow's safety limit.

    Pillow's own decompression-bomb threshold is retained. Its warning is an
    error here because OpenCV would otherwise decode the same large image.
    """

    source = Path(path)
    try:
        with open_regular_file(source, "Image") as (file_source, resolved):
            if expected_file is not None and resolved != expected_file:
                raise ImageInputError(f"Image changed after validation: {source}.")
            if resolved.identity.size == 0:
                raise ImageInputError(f"Image is empty: {source}.")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                # Pillow's structural verifier catches missing image trailer
                # chunks before its permissive pixel decoder can accept them.
                with Image.open(file_source) as verification:
                    verification.verify()
                file_source.seek(0)
                with Image.open(file_source) as opened_image:
                    if getattr(opened_image, "n_frames", 1) != 1:
                        raise ImageInputError(f"Multi-frame images are not supported: {source}.")
                    expected_format = _SUFFIX_FORMATS.get(source.suffix.lower())
                    if expected_format is not None and opened_image.format != expected_format:
                        raise ImageInputError(
                            f"Image content does not match its {source.suffix} filename: {source}."
                        )
                    if mask and opened_image.mode not in ("L", "1"):
                        raise ImageInputError(
                            f"Mask must be grayscale 8-bit or 1-bit, not {opened_image.mode}: {source}."
                        )
                    # Match OpenCV's normal imread behavior for EXIF-oriented
                    # photographs while retaining the already-open file object.
                    oriented_image = ImageOps.exif_transpose(opened_image)
                    try:
                        width, height = oriented_image.size
                        if width <= 0 or height <= 0:
                            raise ImageInputError(f"Image has empty dimensions: {source}.")
                        if width * height > _MAX_IMAGE_PIXELS:
                            raise ImageInputError(
                                f"Image exceeds Pillow's {_MAX_IMAGE_PIXELS}-pixel safety limit: {source}."
                            )
                        if expected_shape is not None and (height, width) != tuple(expected_shape):
                            raise ImageInputError(
                                f"Mask size {height}x{width} does not match acquisition "
                                f"{expected_shape[0]}x{expected_shape[1]}: {source}."
                            )
                        oriented_image.load()  # Catch truncated pixel streams before returning decoded pixels.
                        if FileIdentity.from_stat(os.fstat(file_source.fileno())) != resolved.identity:
                            raise ImageInputError(f"Image changed while it was being decoded: {source}.")
                        return oriented_image.copy() if copy_pixels else (height, width)
                    finally:
                        if oriented_image is not opened_image:
                            oriented_image.close()
    except ImageInputError:
        raise
    except ReadPathError as error:
        raise ImageInputError(str(error)) from error
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError,
            Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise ImageInputError(f"Cannot decode image {source}: {error}.") from error


def decode_raster_image(
    path: os.PathLike[str] | str, *,
    mask: bool = False,
    expected_shape: tuple[int, int] | None = None,
    expected_file: ResolvedFile | None = None,
) -> Image.Image:
    """Return a fully decoded Pillow copy after format and size checks."""

    result = _checked_raster(
        path, mask=mask, expected_shape=expected_shape, copy_pixels=True,
        expected_file=expected_file,
    )
    assert isinstance(result, Image.Image)
    return result


def inspect_raster_image(
    path: os.PathLike[str] | str, *,
    mask: bool = False,
    expected_shape: tuple[int, int] | None = None,
    expected_file: ResolvedFile | None = None,
) -> tuple[int, int]:
    """Validate and decode one raster while returning its oriented dimensions."""

    result = _checked_raster(
        path, mask=mask, expected_shape=expected_shape, copy_pixels=False,
        expected_file=expected_file,
    )
    assert isinstance(result, tuple)
    return result
