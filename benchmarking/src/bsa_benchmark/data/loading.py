"""Image, mask, and preprocessing implementation shared by all datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from bsa_benchmark.config.schemas import DatasetConfig, MaskFormat, Normalization
from bsa_benchmark.core.interfaces import SampleReference
from bsa_benchmark.data.augmentation import AugmentationDecision, AugmentationPipeline
from bsa_benchmark.data.index import DatasetIndex, DatasetValidationError, discover_dataset


@dataclass(frozen=True)
class LoadedSample:
    sample_id: str
    source_group: str
    image: np.ndarray
    mask: np.ndarray
    augmentation: AugmentationDecision | None = None


@dataclass(frozen=True)
class DatasetContentReport:
    dataset_id: str
    sample_count: int
    fingerprint: str
    image_shapes: tuple[tuple[int, ...], ...]
    mask_shapes: tuple[tuple[int, ...], ...]
    image_dtypes: tuple[str, ...]
    mask_values: tuple[int, ...]


def _resampling(name: str) -> Image.Resampling:
    choices = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    try:
        return choices[name.casefold()]
    except KeyError as exc:
        raise DatasetValidationError(f"unsupported interpolation mode: {name}") from exc


class ImageMaskLoader:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def _read(self, path: Any) -> np.ndarray:
        try:
            with Image.open(path) as opened:
                opened.load()
                return np.asarray(opened).copy()
        except (OSError, UnidentifiedImageError) as exc:
            raise DatasetValidationError(f"could not decode image file {path}: {exc}") from exc

    def _resize(self, array: np.ndarray, *, is_mask: bool) -> np.ndarray:
        dimensions = self.config.preprocessing.resize
        if dimensions is None:
            return array
        expected_shape = (dimensions.height, dimensions.width)
        if array.shape[:2] == expected_shape:
            return array
        interpolation = (
            self.config.preprocessing.mask_interpolation
            if is_mask
            else self.config.preprocessing.interpolation
        )
        mode = _resampling(interpolation)
        resized = Image.fromarray(array).resize(
            (dimensions.width, dimensions.height), resample=mode
        )
        return np.asarray(resized).copy()

    def _prepare_image(self, array: np.ndarray, reference: SampleReference) -> np.ndarray:
        expected = self.config.images.dimensions
        if array.shape[:2] != (expected.height, expected.width):
            raise DatasetValidationError(
                f"{reference.image_path} has shape {array.shape[:2]}, expected "
                f"{(expected.height, expected.width)}"
            )
        source_channels = 1 if array.ndim == 2 else array.shape[2]
        if source_channels != self.config.images.source_channels:
            raise DatasetValidationError(
                f"{reference.image_path} has {source_channels} channels, expected "
                f"{self.config.images.source_channels}"
            )
        if self.config.images.alpha_handling.value == "drop":
            array = array[..., :3]
        array = self._resize(array, is_mask=False)

        base_for_auxiliary = array.astype(np.float32) / 255.0
        normalization = self.config.preprocessing.normalization
        if normalization == Normalization.NONE:
            processed: np.ndarray = array
        elif normalization == Normalization.ZERO_ONE:
            processed = base_for_auxiliary
        elif normalization == Normalization.MINUS_ONE_ONE:
            processed = base_for_auxiliary * 2.0 - 1.0
        elif normalization == Normalization.IMAGENET:
            if base_for_auxiliary.ndim != 3 or base_for_auxiliary.shape[2] != 3:
                raise DatasetValidationError("ImageNet normalization requires RGB input")
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            processed = (base_for_auxiliary - mean) / std
        else:  # pragma: no cover - protected by schema enum
            raise DatasetValidationError(f"unsupported normalization: {normalization}")

        extra_channels: list[np.ndarray] = []
        for channel in self.config.preprocessing.auxiliary_channels:
            if not channel.enabled:
                continue
            if base_for_auxiliary.ndim != 3 or base_for_auxiliary.shape[2] < 3:
                raise DatasetValidationError(
                    "colour-derived auxiliary channels require at least three image channels"
                )
            red, green, blue = (
                base_for_auxiliary[..., 0],
                base_for_auxiliary[..., 1],
                base_for_auxiliary[..., 2],
            )
            transform = channel.transform.casefold()
            if transform == "exg":
                auxiliary = 2.0 * green - red - blue
            elif transform == "exm":
                auxiliary = 0.5 * (red + blue) - green
            elif transform == "nexm":
                epsilon = float(channel.parameters.get("epsilon", 1e-6))
                auxiliary = (red + blue - 2.0 * green) / (red + blue + 2.0 * green + epsilon)
            elif transform == "strict_purple":
                auxiliary = np.minimum(red, blue) - green
            else:
                raise DatasetValidationError(
                    f"unsupported auxiliary transform: {channel.transform}"
                )
            if bool(channel.parameters.get("signed_to_zero_one", False)):
                auxiliary = (np.clip(auxiliary, -1.0, 1.0) + 1.0) * 0.5
            elif channel.parameters.get("clip", False):
                limits = channel.parameters.get("clip_range", (-1.0, 2.0))
                auxiliary = np.clip(auxiliary, float(limits[0]), float(limits[1]))
            extra_channels.append(auxiliary.astype(np.float32)[..., None])
        if extra_channels:
            processed = np.concatenate([processed.astype(np.float32), *extra_channels], axis=-1)
        return np.ascontiguousarray(processed)

    def _prepare_mask(self, array: np.ndarray, reference: SampleReference) -> np.ndarray:
        expected = self.config.masks.dimensions
        if array.shape[:2] != (expected.height, expected.width):
            raise DatasetValidationError(
                f"{reference.mask_path} has shape {array.shape[:2]}, expected "
                f"{(expected.height, expected.width)}"
            )
        mask_format = self.config.masks.format
        if mask_format == MaskFormat.BINARY_RGB:
            if array.ndim != 3 or array.shape[2] < 3:
                raise DatasetValidationError(f"binary RGB mask is not RGB: {reference.mask_path}")
            if not (
                np.array_equal(array[..., 0], array[..., 1])
                and np.array_equal(array[..., 0], array[..., 2])
            ):
                raise DatasetValidationError(
                    f"binary RGB mask channels differ: {reference.mask_path}"
                )
            array = array[..., 0]
        elif mask_format == MaskFormat.BINARY_GRAYSCALE:
            if array.ndim == 3:
                if array.shape[2] == 1:
                    array = array[..., 0]
                else:
                    raise DatasetValidationError(
                        f"binary grayscale mask has multiple channels: {reference.mask_path}"
                    )
        elif mask_format in {MaskFormat.CLASS_INDEX, MaskFormat.PALETTE}:
            if array.ndim == 3:
                raise DatasetValidationError(
                    f"class-index mask has multiple channels: {reference.mask_path}"
                )
            return np.ascontiguousarray(self._resize(array, is_mask=True).astype(np.int32))

        array = self._resize(array, is_mask=True)
        threshold = self.config.masks.threshold
        if threshold is not None:
            binary = array > threshold
        else:
            binary = np.isin(array, self.config.masks.foreground_values)
        return np.ascontiguousarray(binary.astype(np.uint8))

    def load(
        self,
        reference: SampleReference,
        *,
        augmentation: AugmentationPipeline | None = None,
        epoch: int = 0,
    ) -> LoadedSample:
        image = self._prepare_image(self._read(reference.image_path), reference)
        mask = self._prepare_mask(self._read(reference.mask_path), reference)
        if image.shape[:2] != mask.shape[:2]:
            raise DatasetValidationError(
                f"processed image/mask shapes differ for {reference.sample_id}: "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )
        decision = None
        if augmentation is not None:
            image, mask, decision = augmentation.apply(
                image, mask, sample_id=reference.sample_id, epoch=epoch
            )
        return LoadedSample(
            sample_id=reference.sample_id,
            source_group=reference.group_id or reference.sample_id,
            image=image,
            mask=mask,
            augmentation=decision,
        )

    def load_for_role(
        self,
        reference: SampleReference,
        *,
        role: str,
        augmentation: AugmentationPipeline | None = None,
        epoch: int = 0,
    ) -> LoadedSample:
        """Load one split member while preventing validation/test augmentation."""

        if role not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split role: {role}")
        if role != "train" and augmentation is not None:
            raise ValueError(f"augmentation is forbidden for the {role} split")
        return self.load(
            reference,
            augmentation=augmentation if role == "train" else None,
            epoch=epoch,
        )


def validate_dataset_contents(
    config: DatasetConfig, index: DatasetIndex | None = None
) -> DatasetContentReport:
    selected_index = index or discover_dataset(config)
    loader = ImageMaskLoader(config)
    image_shapes: set[tuple[int, ...]] = set()
    mask_shapes: set[tuple[int, ...]] = set()
    image_dtypes: set[str] = set()
    mask_values: set[int] = set()
    for reference in selected_index.samples:
        sample = loader.load(reference)
        image_shapes.add(tuple(sample.image.shape))
        mask_shapes.add(tuple(sample.mask.shape))
        image_dtypes.add(str(sample.image.dtype))
        mask_values.update(int(value) for value in np.unique(sample.mask))
    return DatasetContentReport(
        dataset_id=config.id,
        sample_count=len(selected_index.samples),
        fingerprint=selected_index.fingerprint,
        image_shapes=tuple(sorted(image_shapes)),
        mask_shapes=tuple(sorted(mask_shapes)),
        image_dtypes=tuple(sorted(image_dtypes)),
        mask_values=tuple(sorted(mask_values)),
    )
