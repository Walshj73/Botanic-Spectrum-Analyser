"""Default declarative adapter and extension point for exceptional datasets."""

from __future__ import annotations

from collections.abc import Sequence

from bsa_benchmark.config.schemas import DatasetConfig
from bsa_benchmark.core.interfaces import DatasetAdapter, SampleReference
from bsa_benchmark.data.index import discover_dataset
from bsa_benchmark.data.loading import ImageMaskLoader, LoadedSample


class ConfiguredDatasetAdapter(DatasetAdapter):
    """Use configuration alone for discovery, mask handling, and preprocessing."""

    def discover(self, config: DatasetConfig) -> Sequence[SampleReference]:
        return discover_dataset(config).samples

    def load(self, reference: SampleReference, config: DatasetConfig) -> LoadedSample:
        return ImageMaskLoader(config).load(reference)
