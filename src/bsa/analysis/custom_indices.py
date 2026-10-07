"""Orchestration around BSA's existing custom-index formula interpreter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


CustomCalculator = Callable[
    [np.ndarray, str, Mapping[str, Any], str],
    tuple[Any, str],
]

_BUILTIN_INDEX_NAMES = frozenset(
    {"ndvi", "pri", "psri", "sipi", "ndre", "wp1", "ndwi"}
)


class CustomIndexNameError(ValueError):
    """A custom index name is empty or conflicts with a built-in index."""


def validate_custom_index_name(index_name: str) -> None:
    """Reject names that would make the existing result columns ambiguous."""

    if not isinstance(index_name, str) or not index_name.strip():
        raise CustomIndexNameError("Custom index name must contain visible text.")
    if index_name.strip().casefold() in _BUILTIN_INDEX_NAMES:
        raise CustomIndexNameError(
            f"Custom index name {index_name!r} conflicts with a built-in index."
        )


@dataclass(frozen=True)
class CustomIndexResult:
    values: Any
    name: str
    statistics: dict[str, float] | None


def calculate_custom_statistics(
    data_masked: np.ndarray,
    formula: str,
    bands: Mapping[str, Any],
    index_name: str,
    calculator: CustomCalculator,
) -> CustomIndexResult:
    """Execute the supplied legacy calculator and summarize successful output."""

    validate_custom_index_name(index_name)
    values, calculated_name = calculator(data_masked, formula, bands, index_name)
    if values is None:
        return CustomIndexResult(values=None, name=calculated_name, statistics=None)
    statistics = {
        f"{calculated_name} mean": np.nanmean(values),
        f"{calculated_name} min": np.nanmin(values),
        f"{calculated_name} max": np.nanmax(values),
        f"{calculated_name} median": np.nanmedian(values),
        f"{calculated_name} std": np.nanstd(values),
    }
    return CustomIndexResult(
        values=values,
        name=calculated_name,
        statistics=statistics,
    )
