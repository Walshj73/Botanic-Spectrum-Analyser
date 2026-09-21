"""Classical colour/texture features migrated from the manuscript scripts."""

from __future__ import annotations

import numpy as np

from bsa_benchmark.methods.base import OptionalDependencyError


def _rgb01(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError("classical colour features require at least three channels")
    rgb = array[..., :3].astype(np.float32)
    if np.issubdtype(array.dtype, np.integer) or float(np.max(rgb, initial=0.0)) > 1.0:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def extract_colour_texture_features(image: np.ndarray) -> np.ndarray:
    """Return the legacy 15-channel RGB/HSV/blur/edge/LBP/Gabor stack."""

    try:
        from skimage.color import rgb2gray, rgb2hsv
        from skimage.feature import local_binary_pattern
        from skimage.filters import gabor, gaussian, sobel
    except ImportError as exc:
        raise OptionalDependencyError(
            "classical feature extraction requires: pip install 'bsa-benchmark[classical]'"
        ) from exc

    rgb = _rgb01(image)
    gray = rgb2gray(rgb)
    features = [rgb, rgb2hsv(rgb)]
    for sigma in (1, 2, 4):
        features.append(gaussian(gray, sigma=sigma)[..., None])
    features.append(sobel(gray)[..., None])
    gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, P=8, R=1, method="uniform")
    features.append((lbp / (float(lbp.max()) + 1e-6))[..., None])
    for frequency in (0.1, 0.25):
        for theta in (0.0, np.pi / 2.0):
            real, imaginary = gabor(gray, frequency=frequency, theta=theta)
            features.append(np.hypot(real, imaginary)[..., None])
    return np.concatenate(features, axis=-1).astype(np.float32)


def sample_pixel_matrix(
    samples: list[tuple[np.ndarray, np.ndarray]],
    *,
    pixels_per_image: int | None,
    maximum_pixels: int | None,
    ignore_index: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("at least one training sample is required")
    rng = np.random.default_rng(seed)
    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for image, mask in samples:
        feature_volume = extract_colour_texture_features(image)
        flat_features = feature_volume.reshape(-1, feature_volume.shape[-1])
        flat_labels = np.asarray(mask).reshape(-1)
        valid = np.arange(flat_labels.size)
        if ignore_index is not None:
            valid = valid[flat_labels != ignore_index]
        if pixels_per_image and valid.size > pixels_per_image:
            valid = rng.choice(valid, pixels_per_image, replace=False)
        feature_rows.append(flat_features[valid])
        labels.append(flat_labels[valid])
    matrix = np.concatenate(feature_rows)
    target = np.concatenate(labels).astype(np.int32)
    if maximum_pixels and len(matrix) > maximum_pixels:
        selected = rng.choice(len(matrix), maximum_pixels, replace=False)
        matrix, target = matrix[selected], target[selected]
    return matrix, target
