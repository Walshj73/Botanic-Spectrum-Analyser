"""Algorithm-neutral extension points for datasets and segmentation methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from bsa_benchmark.config.schemas import DatasetConfig

SampleT = TypeVar("SampleT")
ModelT = TypeVar("ModelT")


@dataclass(frozen=True)
class SampleReference:
    sample_id: str
    image_path: Path
    mask_path: Path
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Maps one dataset's storage conventions onto shared sample references."""

    @abstractmethod
    def discover(self, config: DatasetConfig) -> Sequence[SampleReference]:
        """Return deterministic, paired sample references."""

    @abstractmethod
    def load(self, reference: SampleReference, config: DatasetConfig) -> SampleT:
        """Load and preprocess one image/mask pair without augmentation."""


@dataclass(frozen=True)
class MethodContext:
    run_directory: Path
    seed: int
    deterministic_operations: bool
    fold_index: int | None = None


@dataclass(frozen=True)
class FitResult(Generic[ModelT]):
    model: ModelT
    history: Mapping[str, Sequence[float]] = field(default_factory=dict)
    artifacts: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    output_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Method(ABC, Generic[SampleT, ModelT]):
    """Common lifecycle for classical and deep segmentation methods."""

    @property
    @abstractmethod
    def method_id(self) -> str:
        """Stable registry identifier."""

    @property
    @abstractmethod
    def implementation_version(self) -> str:
        """Version recorded with every result."""

    @abstractmethod
    def fit(
        self,
        train_samples: Sequence[SampleT],
        validation_samples: Sequence[SampleT],
        hyperparameters: Mapping[str, Any],
        context: MethodContext,
    ) -> FitResult[ModelT]:
        """Fit using training data and select only against validation data."""

    @abstractmethod
    def predict(
        self,
        model: ModelT,
        samples: Sequence[SampleT],
        context: MethodContext,
    ) -> Iterable[Prediction]:
        """Generate predictions for an explicitly supplied split."""

    @abstractmethod
    def save_model(self, model: ModelT, destination: Path) -> Path:
        """Persist a fitted model to a new path."""

    @abstractmethod
    def load_model(self, source: Path) -> ModelT:
        """Load a model produced by the same implementation version."""
