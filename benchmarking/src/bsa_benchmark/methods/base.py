"""Shared contracts and loading for configured segmentation methods."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bsa_benchmark.config.schemas import MethodConfig
from bsa_benchmark.core.interfaces import MethodContext
from bsa_benchmark.data.loading import LoadedSample


class OptionalDependencyError(RuntimeError):
    """Raised with an actionable extras-install instruction."""


class SegmentationMethod(ABC):
    def __init__(self, config: MethodConfig) -> None:
        self.config = config

    @abstractmethod
    def fit(
        self,
        train_samples: Sequence[LoadedSample],
        validation_samples: Sequence[LoadedSample],
        context: MethodContext,
    ) -> tuple[Any, Mapping[str, Sequence[float]]]:
        """Fit a model and return it with a serializable training history."""

    @abstractmethod
    def predict(
        self,
        model: Any,
        samples: Sequence[LoadedSample],
        context: MethodContext,
    ) -> Mapping[str, np.ndarray]:
        """Return one integer class-index mask per sample ID."""

    @abstractmethod
    def save_model(self, model: Any, destination: Path) -> Path:
        """Persist a fitted model without overwriting an existing file."""


def load_method(config: MethodConfig) -> SegmentationMethod:
    """Instantiate only the implementation named in trusted local configuration."""

    try:
        module_name, object_name = config.implementation.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"method implementation must be 'module:object': {config.implementation!r}"
        ) from exc
    module = importlib.import_module(module_name)
    implementation = getattr(module, object_name)
    method = implementation(config)
    if not isinstance(method, SegmentationMethod):
        raise TypeError(f"{config.implementation} is not a SegmentationMethod")
    return method
