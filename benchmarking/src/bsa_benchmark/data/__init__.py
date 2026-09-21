"""Shared dataset discovery, loading, augmentation, and splitting."""

from bsa_benchmark.data.adapters import ConfiguredDatasetAdapter
from bsa_benchmark.data.augmentation import AugmentationDecision, AugmentationPipeline
from bsa_benchmark.data.index import (
    DatasetIndex,
    DatasetValidationError,
    discover_dataset,
)
from bsa_benchmark.data.loading import (
    DatasetContentReport,
    ImageMaskLoader,
    LoadedSample,
    validate_dataset_contents,
)
from bsa_benchmark.data.splitting import (
    SplitManifest,
    assert_no_source_leakage,
    generate_cross_validation,
    generate_development_split,
    generate_protocol_splits,
    read_split_manifest,
    validate_manifest_against_index,
    write_split_manifest,
)

__all__ = [
    "AugmentationDecision",
    "AugmentationPipeline",
    "ConfiguredDatasetAdapter",
    "DatasetContentReport",
    "DatasetIndex",
    "DatasetValidationError",
    "ImageMaskLoader",
    "LoadedSample",
    "SplitManifest",
    "assert_no_source_leakage",
    "discover_dataset",
    "generate_cross_validation",
    "generate_development_split",
    "generate_protocol_splits",
    "read_split_manifest",
    "validate_dataset_contents",
    "validate_manifest_against_index",
    "write_split_manifest",
]
