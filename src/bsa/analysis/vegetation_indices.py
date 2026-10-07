"""Built-in vegetation-index calculations from the authoritative BSA GUI."""

from __future__ import annotations

from typing import Mapping

import numpy as np


VNIR_MODE = 0
SWIR_MODE = 1
COMBINED_MODE = 2
GENERAL_GRID_MODE = 3


def calculate_ndvi(data_masked: np.ndarray) -> np.ndarray:
    nir = data_masked[:, :, 395]
    red = data_masked[:, :, 281]
    return np.divide(nir - red, nir + red)


def calculate_pri(data_masked: np.ndarray) -> np.ndarray:
    first = data_masked[:, :, 159]
    second = data_masked[:, :, 193]
    return np.divide(first - second, first + second)


def calculate_psri(data_masked: np.ndarray) -> np.ndarray:
    first = data_masked[:, :, 290]
    second = data_masked[:, :, 131]
    third = data_masked[:, :, 351]
    return np.divide(first - second, third)


def calculate_sipi(data_masked: np.ndarray) -> np.ndarray:
    first = data_masked[:, :, 386]
    second = data_masked[:, :, 88]
    third = data_masked[:, :, 263]
    return np.divide(first - second, first + third)


def calculate_ndre(data_masked: np.ndarray) -> np.ndarray:
    nir = data_masked[:, :, 386]
    red_edge = data_masked[:, :, 325]
    return np.divide(nir - red_edge, nir + red_edge)


def calculate_wp1(data_masked: np.ndarray) -> np.ndarray:
    first = data_masked[:, :, 276]
    second = data_masked[:, :, 320]
    return np.divide(first - second, second)


def calculate_swir_ndwi(data_masked: np.ndarray) -> np.ndarray:
    first = data_masked[:, :, 276]
    second = data_masked[:, :, 216]
    return np.divide(first, second)


def index_statistics(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Return the existing five NaN-aware statistics in column order."""

    return {
        f"{prefix} Mean": np.nanmean(values),
        f"{prefix} Min": np.nanmin(values),
        f"{prefix} Max": np.nanmax(values),
        f"{prefix} Median": np.nanmedian(values),
        f"{prefix} Std": np.nanstd(values),
    }


def calculate_vnir_statistics(
    data_masked: np.ndarray,
    file_name: str,
    label: str | None,
    custom_statistics: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Calculate all six existing VNIR indices and their output columns."""

    statistics: dict[str, object] = {"File Name": file_name, "Label": label}
    for prefix, calculator in (
        ("NDVI", calculate_ndvi),
        ("PRI", calculate_pri),
        ("PSRI", calculate_psri),
        ("SIPI", calculate_sipi),
        ("NDRE", calculate_ndre),
        ("WP1", calculate_wp1),
    ):
        statistics.update(index_statistics(calculator(data_masked), prefix))
    if custom_statistics is not None:
        statistics.update(custom_statistics)
    return statistics


def calculate_swir_statistics(
    data_masked: np.ndarray,
    file_name: str,
    label: str | None,
    custom_statistics: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Calculate the existing SWIR ratio exported under NDWI columns."""

    statistics: dict[str, object] = {"File Name": file_name, "Label": label}
    statistics.update(index_statistics(calculate_swir_ndwi(data_masked), "NDWI"))
    if custom_statistics is not None:
        statistics.update(custom_statistics)
    return statistics


def calculate_mode_statistics(
    data_masked: np.ndarray,
    sensor_mode: int,
    file_name: str,
    label: str | None,
    custom_statistics: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Return the selected profile's result dictionary."""

    if sensor_mode == VNIR_MODE:
        return calculate_vnir_statistics(
            data_masked, file_name, label, custom_statistics
        )
    if sensor_mode == SWIR_MODE:
        return calculate_swir_statistics(
            data_masked, file_name, label, custom_statistics
        )
    if sensor_mode == GENERAL_GRID_MODE:
        statistics: dict[str, object] = {"File Name": file_name, "Label": label}
        if custom_statistics is not None:
            statistics.update(custom_statistics)
        return statistics
    if sensor_mode == COMBINED_MODE:
        combined = calculate_vnir_statistics(
            data_masked, file_name, label, custom_statistics
        )
        combined.update(
            calculate_swir_statistics(
                data_masked, file_name, label, custom_statistics
            )
        )
        if custom_statistics is not None:
            combined.update(custom_statistics)
        return combined
    return {}


def update_combined_statistics(
    existing: dict[str, object],
    current: Mapping[str, object],
) -> dict[str, object]:
    """Update and return the GUI's persistent combined-mode dictionary."""

    existing.update(current)
    return existing
