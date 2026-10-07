"""Spectral aggregation and plot/export data preparation for BSA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def calculate_mean_spectrum(data_masked: np.ndarray) -> np.ndarray:
    """Average spatial dimensions with the existing NaN-aware reducer."""

    return np.nanmean(data_masked, axis=(0, 1))


def initialize_spectra_by_name(
    labels: Iterable[str | None],
) -> SpectraGroups:
    """Keep biological classes separate from generated image display names."""

    return SpectraGroups(
        class_spectra={label: [] for label in labels if label is not None},
        image_spectra={},
    )


@dataclass
class SpectraGroups:
    class_spectra: dict[str, list[np.ndarray]]
    image_spectra: dict[str, list[np.ndarray]]


def record_mean_spectrum_for_class(
    mean_spectra: list[np.ndarray],
    spectra_groups: SpectraGroups,
    spectrum: np.ndarray,
    set_index: int,
    class_name: str | None,
) -> str | None:
    """Record per-sample output and optional explicit class membership."""

    mean_spectra.append(spectrum)
    image_name = f"Image {set_index + 1}"
    spectra_groups.image_spectra[image_name] = [spectrum]
    if class_name is not None:
        if class_name not in spectra_groups.class_spectra:
            spectra_groups.class_spectra[class_name] = []
        spectra_groups.class_spectra[class_name].append(spectrum)
    return class_name


def record_mean_spectrum(
    mean_spectra: list[np.ndarray],
    spectra_groups: SpectraGroups,
    spectrum: np.ndarray,
    set_index: int,
    labels: Sequence[str | None],
    labels_exist: bool,
) -> str | None:
    """Record one image spectrum and, when applicable, its class membership."""

    label = labels[set_index] if labels_exist and set_index < len(labels) else None
    return record_mean_spectrum_for_class(
        mean_spectra,
        spectra_groups,
        spectrum,
        set_index,
        label,
    )


@dataclass(frozen=True)
class PlotSpectrum:
    values: np.ndarray
    label: str | None = None


@dataclass(frozen=True)
class SpectralPlotData:
    individual: list[PlotSpectrum]
    class_means: list[PlotSpectrum]
    labelled: bool


def prepare_plot_data(
    spectra_groups: SpectraGroups,
    labels: Sequence[str | None],
    labels_exist: bool,
) -> SpectralPlotData:
    """Prepare the same individual and class-mean curves used by the GUI."""

    individual: list[PlotSpectrum] = []
    class_means: list[PlotSpectrum] = []
    for spectra in spectra_groups.image_spectra.values():
        individual.extend(PlotSpectrum(values=spectrum) for spectrum in spectra)
    if labels_exist:
        for label in sorted(value for value in set(labels) if value is not None):
            spectra = spectra_groups.class_spectra.get(label, [])
            if spectra:
                class_means.append(
                    PlotSpectrum(values=np.mean(spectra, axis=0), label=label)
                )
    return SpectralPlotData(
        individual=individual,
        class_means=class_means,
        labelled=labels_exist,
    )


def prepare_image_spectra_rows(
    wavelengths: Sequence[float],
    spectra_groups: SpectraGroups,
) -> tuple[list[object], list[list[object]]]:
    """Prepare the automatic row-oriented ``mean_spectra.csv`` data."""

    rows: list[list[object]] = []
    for image_name in sorted(spectra_groups.image_spectra):
        rows.append([image_name] + list(spectra_groups.image_spectra[image_name][0]))
    return ["Image"] + list(wavelengths), rows


def prepare_class_spectra_rows(
    wavelengths: Sequence[float],
    labels: Sequence[str | None],
    mean_spectra: Sequence[np.ndarray],
) -> dict[str, tuple[list[object], list[list[object]]]]:
    """Prepare automatic row-oriented per-class spectral exports."""

    class_labels = sorted({label for label in labels if label is not None})
    class_data: dict[str, list[list[object]]] = {label: [] for label in class_labels}
    for label in class_labels:
        for index, image_label in enumerate(labels):
            if label == image_label:
                spectrum = mean_spectra[index]
                class_data[label].append([f"Image {index + 1}"] + list(spectrum))
    header = ["Image"] + list(wavelengths)
    return {label: (header, rows) for label, rows in class_data.items()}


def prepare_column_spectra_rows(
    wavelengths: Sequence[float],
    spectra: Sequence[np.ndarray],
    column_prefix: str = "Mean_Spectrum",
) -> tuple[list[object], list[list[object]]]:
    """Prepare the manual export's wavelength-first table."""

    header: list[object] = ["Wavelength"] + [
        f"{column_prefix}_{index + 1}" for index in range(len(spectra))
    ]
    rows = [
        [wavelengths[index]]
        + [spectra[spectrum_index][index] for spectrum_index in range(len(spectra))]
        for index in range(len(wavelengths))
    ]
    return header, rows
