"""Strict schemas for BSA datasets, methods, and benchmark experiments."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Dimensions(StrictModel):
    height: int = Field(gt=0)
    width: int = Field(gt=0)


class ColourMode(StrEnum):
    RGB = "rgb"
    RGBA = "rgba"
    GRAYSCALE = "grayscale"
    RENDERED_RGB = "rendered_rgb"


class AlphaHandling(StrEnum):
    ABSENT = "absent"
    DROP = "drop"
    PRESERVE = "preserve"
    ERROR = "error"


class ImageConfig(StrictModel):
    directory: Path
    extensions: tuple[str, ...] = (".png",)
    dimensions: Dimensions
    source_channels: int = Field(gt=0)
    model_channels: int = Field(gt=0)
    colour_mode: ColourMode
    alpha_handling: AlphaHandling = AlphaHandling.ABSENT

    @model_validator(mode="after")
    def validate_channels(self) -> ImageConfig:
        if (
            self.source_channels == 4
            and self.model_channels == 3
            and self.alpha_handling != AlphaHandling.DROP
        ):
            raise ValueError("RGBA-to-RGB input requires alpha_handling='drop'")
        if self.alpha_handling == AlphaHandling.DROP and self.source_channels != 4:
            raise ValueError("alpha_handling='drop' requires four source channels")
        return self


class MaskFormat(StrEnum):
    BINARY_GRAYSCALE = "binary_grayscale"
    BINARY_RGB = "binary_rgb"
    CLASS_INDEX = "class_index"
    PALETTE = "palette"


class MaskConfig(StrictModel):
    directory: Path
    extensions: tuple[str, ...] = (".png",)
    dimensions: Dimensions
    format: MaskFormat
    classes: tuple[str, ...]
    foreground_values: tuple[int, ...] = (255,)
    threshold: float | None = Field(default=None, ge=0, le=255)
    ignore_index: int | None = None

    @model_validator(mode="after")
    def validate_classes(self) -> MaskConfig:
        if len(self.classes) < 2:
            raise ValueError("segmentation masks require at least two classes")
        return self


class PairingStrategy(StrEnum):
    IDENTICAL_STEM = "identical_stem"
    TOKEN_REPLACEMENT = "token_replacement"


class PairingConfig(StrictModel):
    strategy: PairingStrategy
    image_token: str | None = None
    mask_token: str | None = None
    case_sensitive: bool = True

    @model_validator(mode="after")
    def validate_tokens(self) -> PairingConfig:
        token_values = (self.image_token, self.mask_token)
        if self.strategy == PairingStrategy.TOKEN_REPLACEMENT and not all(token_values):
            raise ValueError("token_replacement pairing requires image_token and mask_token")
        if self.strategy == PairingStrategy.IDENTICAL_STEM and any(token_values):
            raise ValueError("identical_stem pairing must not define replacement tokens")
        return self


class Normalization(StrEnum):
    NONE = "none"
    ZERO_ONE = "zero_one"
    MINUS_ONE_ONE = "minus_one_one"
    IMAGENET = "imagenet"


class AuxiliaryChannelConfig(StrictModel):
    name: str = Field(min_length=1)
    transform: str = Field(min_length=1)
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)


class PreprocessingConfig(StrictModel):
    normalization: Normalization = Normalization.ZERO_ONE
    resize: Dimensions | None = None
    interpolation: str = "bilinear"
    mask_interpolation: str = "nearest"
    adapter: str | None = None
    auxiliary_channels: tuple[AuxiliaryChannelConfig, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)


class AugmentationConfig(StrictModel):
    enabled: bool = True
    execution: str = "runtime"
    horizontal_flip_probability: float = Field(default=0.5, ge=0, le=1)
    vertical_flip_probability: float = Field(default=0.5, ge=0, le=1)
    rotations_degrees: tuple[int, ...] = (0, 90, 180, 270)
    brightness_delta: float = Field(default=0.0, ge=0, le=1)
    contrast_range: tuple[float, float] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contrast(self) -> AugmentationConfig:
        if not self.rotations_degrees:
            raise ValueError("rotations_degrees must not be empty")
        if any(rotation % 90 != 0 for rotation in self.rotations_degrees):
            raise ValueError("rotations_degrees must contain multiples of 90")
        if self.contrast_range is not None:
            low, high = self.contrast_range
            if low <= 0 or high < low:
                raise ValueError("contrast_range must be positive and ordered")
        return self


class SplitDefaultsConfig(StrictModel):
    folds: int = Field(default=5, ge=2)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    seed: int = Field(default=42, ge=0)
    shuffle: bool = True
    stratify: bool = False
    group_by_source: bool = True
    source_group_regex: str | None = None


class DatasetConfig(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str = Field(min_length=1)
    root: Path
    images: ImageConfig
    masks: MaskConfig
    pairing: PairingConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    splits: SplitDefaultsConfig = Field(default_factory=SplitDefaultsConfig)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_dimensions(self) -> DatasetConfig:
        if self.images.dimensions != self.masks.dimensions:
            raise ValueError("image and mask dimensions must match before pairing")
        enabled_auxiliary = sum(
            1 for channel in self.preprocessing.auxiliary_channels if channel.enabled
        )
        expected_channels = (
            self.images.source_channels
            - (1 if self.images.alpha_handling == AlphaHandling.DROP else 0)
            + enabled_auxiliary
        )
        if self.images.model_channels != expected_channels:
            raise ValueError(
                "images.model_channels must equal the post-alpha image channels plus "
                f"enabled auxiliary channels (expected {expected_channels})"
            )
        return self


class MethodFamily(StrEnum):
    CLASSICAL = "classical"
    DEEP_LEARNING = "deep_learning"


class MethodConfig(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str = Field(min_length=1)
    family: MethodFamily
    implementation: str = Field(min_length=3)
    implementation_version: str = Field(min_length=1)
    accepted_input_channels: tuple[int, ...] = Field(min_length=1)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()


class EvaluationProtocol(StrEnum):
    DEVELOPMENT = "development"
    CROSS_VALIDATION = "cross_validation"
    HELD_OUT_TEST = "held_out_test"


class ProtocolConfig(StrictModel):
    type: EvaluationProtocol
    folds: int | None = Field(default=None, ge=2)
    validation_fraction: float | None = Field(default=None, gt=0, lt=1)
    test_fraction: float | None = Field(default=None, gt=0, lt=1)
    split_seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_protocol(self) -> ProtocolConfig:
        if self.type == EvaluationProtocol.DEVELOPMENT and (
            self.validation_fraction is None or self.test_fraction is None
        ):
            raise ValueError("development protocol requires validation_fraction and test_fraction")
        if self.type == EvaluationProtocol.CROSS_VALIDATION and (
            self.folds is None or self.validation_fraction is None
        ):
            raise ValueError("cross-validation requires folds and validation_fraction")
        return self


class ExperimentConfig(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str = Field(min_length=1)
    dataset_config: Path
    method_config: Path
    protocol: ProtocolConfig
    seed: int = Field(default=42, ge=0)
    deterministic_operations: bool = True
    output_root: Path = Path("runs")
    tags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class ResolvedExperimentConfig(StrictModel):
    experiment: ExperimentConfig
    dataset: DatasetConfig
    method: MethodConfig

    @model_validator(mode="after")
    def validate_compatibility(self) -> ResolvedExperimentConfig:
        channels = self.dataset.images.model_channels
        if channels not in self.method.accepted_input_channels:
            raise ValueError(f"method {self.method.id!r} does not accept {channels} input channels")
        return self
