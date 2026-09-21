"""Reproducible paired image/mask augmentation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from bsa_benchmark.config.schemas import AugmentationConfig


@dataclass(frozen=True)
class AugmentationDecision:
    seed: int
    horizontal_flip: bool
    vertical_flip: bool
    rotation_degrees: int
    brightness_offset: float
    contrast_factor: float


def _derived_seed(seed: int, sample_id: str, epoch: int) -> int:
    digest = hashlib.sha256(f"{seed}|{sample_id}|{epoch}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class AugmentationPipeline:
    def __init__(self, config: AugmentationConfig, seed: int) -> None:
        if seed < 0:
            raise ValueError("augmentation seed must be non-negative")
        self.config = config
        self.seed = seed

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        sample_id: str,
        epoch: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, AugmentationDecision]:
        local_seed = _derived_seed(self.seed, sample_id, epoch)
        if not self.config.enabled:
            decision = AugmentationDecision(local_seed, False, False, 0, 0.0, 1.0)
            return image.copy(), mask.copy(), decision
        rng = np.random.default_rng(local_seed)
        horizontal = bool(rng.random() < self.config.horizontal_flip_probability)
        vertical = bool(rng.random() < self.config.vertical_flip_probability)
        rotation = int(rng.choice(self.config.rotations_degrees))
        brightness = (
            float(rng.uniform(-self.config.brightness_delta, self.config.brightness_delta))
            if self.config.brightness_delta
            else 0.0
        )
        contrast = (
            float(rng.uniform(*self.config.contrast_range))
            if self.config.contrast_range is not None
            else 1.0
        )

        transformed_image = image
        transformed_mask = mask
        if horizontal:
            transformed_image = np.flip(transformed_image, axis=1)
            transformed_mask = np.flip(transformed_mask, axis=1)
        if vertical:
            transformed_image = np.flip(transformed_image, axis=0)
            transformed_mask = np.flip(transformed_mask, axis=0)
        turns = (rotation // 90) % 4
        if turns:
            transformed_image = np.rot90(transformed_image, k=turns, axes=(0, 1))
            transformed_mask = np.rot90(transformed_mask, k=turns, axes=(0, 1))

        if brightness or contrast != 1.0:
            original_dtype = transformed_image.dtype
            working = transformed_image.astype(np.float32)
            if np.issubdtype(original_dtype, np.integer):
                scale = float(np.iinfo(original_dtype).max)
                center = scale / 2.0
                working = (working - center) * contrast + center + brightness * scale
                working = np.clip(working, 0.0, scale).astype(original_dtype)
            else:
                center = float(np.mean(working))
                working = (working - center) * contrast + center + brightness
                if (
                    float(np.min(transformed_image)) >= 0.0
                    and float(np.max(transformed_image)) <= 1.0
                ):
                    working = np.clip(working, 0.0, 1.0)
                working = working.astype(original_dtype)
            transformed_image = working

        decision = AugmentationDecision(
            seed=local_seed,
            horizontal_flip=horizontal,
            vertical_flip=vertical,
            rotation_degrees=rotation,
            brightness_offset=brightness,
            contrast_factor=contrast,
        )
        return (
            np.ascontiguousarray(transformed_image),
            np.ascontiguousarray(transformed_mask),
            decision,
        )
