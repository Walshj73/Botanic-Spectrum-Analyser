"""YAML loading, path resolution, and lightweight dataset checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from bsa_benchmark.config.schemas import (
    DatasetConfig,
    ExperimentConfig,
    MethodConfig,
    PairingStrategy,
    ResolvedExperimentConfig,
)


class ConfigLoadError(ValueError):
    """Raised when a configuration cannot be read or composed."""


ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class DatasetPathReport:
    image_directory: Path
    mask_directory: Path
    image_count: int
    mask_count: int
    paired_count: int
    unmatched_images: tuple[str, ...]
    unmatched_masks: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return (
            self.image_count > 0
            and self.paired_count == self.image_count == self.mask_count
            and not self.unmatched_images
            and not self.unmatched_masks
        )


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigLoadError(f"could not locate pyproject.toml above {current}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _load(path: Path, model_type: type[ModelT]) -> ModelT:
    path = path.resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigLoadError(f"could not read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigLoadError(f"{path} must contain a YAML mapping")
    try:
        return model_type.model_validate(_expand_environment(raw))
    except ValidationError as exc:
        raise ConfigLoadError(f"invalid {model_type.__name__} in {path}:\n{exc}") from exc


def _absolute(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path).resolve()


def load_dataset_config(path: Path, project_root: Path | None = None) -> DatasetConfig:
    config = _load(path, DatasetConfig)
    root = _absolute(config.root, project_root or find_project_root(path))
    return config.model_copy(update={"root": root})


def load_method_config(path: Path) -> MethodConfig:
    return _load(path, MethodConfig)


def load_experiment_config(path: Path, project_root: Path | None = None) -> ExperimentConfig:
    config = _load(path, ExperimentConfig)
    root = project_root or find_project_root(path)
    updates: dict[str, Path] = {
        "dataset_config": _absolute(config.dataset_config, root),
        "method_config": _absolute(config.method_config, root),
        "output_root": _absolute(config.output_root, root),
    }
    return config.model_copy(update=updates)


def resolve_experiment(path: Path, project_root: Path | None = None) -> ResolvedExperimentConfig:
    root = project_root or find_project_root(path)
    experiment = load_experiment_config(path, root)
    dataset = load_dataset_config(experiment.dataset_config, root)
    method = load_method_config(experiment.method_config)
    try:
        return ResolvedExperimentConfig(
            experiment=experiment,
            dataset=dataset,
            method=method,
        )
    except ValidationError as exc:
        raise ConfigLoadError(f"incompatible experiment composition:\n{exc}") from exc


def _files_by_stem(directory: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    normalized = {extension.lower() for extension in extensions}
    files = [path for path in directory.iterdir() if path.is_file()]
    return {path.stem: path for path in files if path.suffix.lower() in normalized}


def check_dataset_paths(config: DatasetConfig) -> DatasetPathReport:
    image_directory = config.root / config.images.directory
    mask_directory = config.root / config.masks.directory
    if not image_directory.is_dir():
        raise ConfigLoadError(f"image directory does not exist: {image_directory}")
    if not mask_directory.is_dir():
        raise ConfigLoadError(f"mask directory does not exist: {mask_directory}")

    images = _files_by_stem(image_directory, config.images.extensions)
    masks = _files_by_stem(mask_directory, config.masks.extensions)
    case_sensitive = config.pairing.case_sensitive

    def normalized(value: str) -> str:
        return value if case_sensitive else value.casefold()

    expected_to_image: dict[str, str] = {}
    for image_stem in images:
        expected = image_stem
        if config.pairing.strategy == PairingStrategy.TOKEN_REPLACEMENT:
            expected = expected.replace(
                config.pairing.image_token or "", config.pairing.mask_token or ""
            )
        expected_to_image[normalized(expected)] = image_stem

    normalized_masks = {normalized(stem): stem for stem in masks}
    paired = set(expected_to_image).intersection(normalized_masks)
    unmatched_images = tuple(
        sorted(expected_to_image[key] for key in set(expected_to_image) - paired)
    )
    unmatched_masks = tuple(sorted(normalized_masks[key] for key in set(normalized_masks) - paired))
    return DatasetPathReport(
        image_directory=image_directory,
        mask_directory=mask_directory,
        image_count=len(images),
        mask_count=len(masks),
        paired_count=len(paired),
        unmatched_images=unmatched_images,
        unmatched_masks=unmatched_masks,
    )
