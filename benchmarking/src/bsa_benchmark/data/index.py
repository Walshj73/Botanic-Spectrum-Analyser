"""Deterministic image/mask discovery from dataset configuration."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from bsa_benchmark.config.schemas import DatasetConfig, PairingStrategy
from bsa_benchmark.core.interfaces import SampleReference


class DatasetValidationError(ValueError):
    """Raised when configured dataset content is missing or inconsistent."""


@dataclass(frozen=True)
class DatasetIndex:
    dataset_id: str
    root: Path
    samples: tuple[SampleReference, ...]
    fingerprint: str

    def by_id(self) -> dict[str, SampleReference]:
        return {sample.sample_id: sample for sample in self.samples}


def _files_by_stem(directory: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    allowed = {extension.casefold() for extension in extensions}
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in allowed:
            continue
        if path.stem in result:
            raise DatasetValidationError(
                f"duplicate stem {path.stem!r} in {directory}; extensions are ambiguous"
            )
        result[path.stem] = path.resolve()
    return result


def _source_group(sample_id: str, config: DatasetConfig) -> str:
    pattern = config.splits.source_group_regex
    if not config.splits.group_by_source:
        return sample_id
    if pattern is None:
        return sample_id
    match = re.search(pattern, sample_id)
    if match is None:
        raise DatasetValidationError(
            f"sample {sample_id!r} does not match source_group_regex {pattern!r}"
        )
    if match.lastindex:
        return match.group(1)
    return match.group(0)


def _content_fingerprint(dataset_id: str, samples: tuple[SampleReference, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(f"dataset:{dataset_id}\n".encode())
    content_cache: dict[Path, str] = {}
    for sample in samples:
        digest.update(f"sample:{sample.sample_id}|group:{sample.group_id}\n".encode())
        for role, path in (("image", sample.image_path), ("mask", sample.mask_path)):
            if path not in content_cache:
                file_digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        file_digest.update(chunk)
                content_cache[path] = file_digest.hexdigest()
            digest.update(
                f"{role}:{path.name}:{path.stat().st_size}:{content_cache[path]}\n".encode()
            )
    return digest.hexdigest()


def discover_dataset(config: DatasetConfig) -> DatasetIndex:
    image_directory = config.root / config.images.directory
    mask_directory = config.root / config.masks.directory
    if not image_directory.is_dir():
        raise DatasetValidationError(f"image directory does not exist: {image_directory}")
    if not mask_directory.is_dir():
        raise DatasetValidationError(f"mask directory does not exist: {mask_directory}")

    images = _files_by_stem(image_directory, config.images.extensions)
    masks = _files_by_stem(mask_directory, config.masks.extensions)
    mask_lookup = {
        (stem if config.pairing.case_sensitive else stem.casefold()): path
        for stem, path in masks.items()
    }
    used_masks: set[Path] = set()
    samples: list[SampleReference] = []
    for image_stem, image_path in images.items():
        mask_stem = image_stem
        if config.pairing.strategy == PairingStrategy.TOKEN_REPLACEMENT:
            image_token = config.pairing.image_token or ""
            if image_token not in image_stem:
                raise DatasetValidationError(
                    f"image stem {image_stem!r} does not contain pairing token {image_token!r}"
                )
            mask_stem = image_stem.replace(image_token, config.pairing.mask_token or "")
        lookup_stem = mask_stem if config.pairing.case_sensitive else mask_stem.casefold()
        mask_path = mask_lookup.get(lookup_stem)
        if mask_path is None:
            raise DatasetValidationError(
                f"no mask matching image {image_path.name}; expected stem {mask_stem!r}"
            )
        if mask_path in used_masks:
            raise DatasetValidationError(f"mask paired more than once: {mask_path}")
        used_masks.add(mask_path)
        samples.append(
            SampleReference(
                sample_id=image_stem,
                image_path=image_path,
                mask_path=mask_path,
                group_id=_source_group(image_stem, config),
                metadata={
                    "image_filename": image_path.name,
                    "mask_filename": mask_path.name,
                },
            )
        )

    unused_masks = sorted(path.name for path in set(masks.values()) - used_masks)
    if unused_masks:
        preview = ", ".join(unused_masks[:5])
        raise DatasetValidationError(f"{len(unused_masks)} unpaired masks: {preview}")
    if not samples:
        raise DatasetValidationError(f"no configured images found in {image_directory}")

    frozen_samples = tuple(samples)
    return DatasetIndex(
        dataset_id=config.id,
        root=config.root,
        samples=frozen_samples,
        fingerprint=_content_fingerprint(config.id, frozen_samples),
    )
