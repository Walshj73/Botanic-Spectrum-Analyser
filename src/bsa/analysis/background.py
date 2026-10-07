"""Tk-free hyperspectral analysis with cooperative cancellation and staged output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import re
import tempfile
import threading
from typing import Callable

import numpy as np
import pandas as pd

from bsa.analysis import custom_indices, spectral, vegetation_indices
from bsa.analysis.custom_metric_definitions import CustomMetricDefinition, CustomMetricQueue
from bsa.exporting import analysis as analysis_export
from bsa.exporting.output_safety import PlannedOutput, available_output_plan, publish_staged_outputs, validate_output_plan
from bsa.exporting.spreadsheet_safety import write_dataframe_csv, write_dataframe_xlsx
from bsa.hyperspectral import dataset, envi_validation, io, processing
from bsa.labeling import manifest
from bsa.utils.misc import calculate_custom_index


class AnalysisCancelled(Exception):
    """The user closed or cancelled the analysis at a safe boundary."""


@dataclass(frozen=True)
class AnalysisSnapshot:
    dark_header: str
    data_header: str
    white_header: str
    image_directory: str
    mask_directory: str
    label_manifest: str
    labels_enabled: bool
    sensor_type: int
    metrics: tuple[CustomMetricDefinition, ...]
    grid_wavelengths: tuple[float, ...]
    output_directory: Path
    started_at: datetime
    output_format: str = "csv"


@dataclass(frozen=True)
class AnalysisResult:
    snapshot: AnalysisSnapshot
    frame: pd.DataFrame
    filename: str
    wavelengths: np.ndarray
    mean_spectra: list[np.ndarray]
    spectra_groups: spectral.SpectraGroups
    labels: list[str | None]
    labels_exist: bool
    excluded_unlabelled_count: int


def _check_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise AnalysisCancelled()


def run_analysis(
    snapshot: AnalysisSnapshot,
    cancelled: threading.Event,
    report: Callable[[str, object], None],
) -> AnalysisResult:
    """Run the existing scientific calls in acquisition order, without Tk access."""

    _check_cancelled(cancelled)
    validation_queue = CustomMetricQueue()
    for metric in snapshot.metrics:
        validation_queue.add_or_update(metric.name, metric.formula, metric.band_mapping())
    matched = dataset.discover_dataset(snapshot.image_directory, snapshot.mask_directory)
    validated = (
        manifest.load_and_validate_label_manifest(snapshot.label_manifest, matched)
        if snapshot.labels_enabled and snapshot.label_manifest else None
    )
    excluded = validated.excluded_unlabelled_count if validated is not None else 0
    labels_exist = bool(validated is not None and validated.labelled_count > 0)
    class_names = sorted(set(validated.classes_by_acquisition_id.values())) if validated is not None else []
    analysis_export.validate_spectrum_identity_names(class_names, matched.number_of_sets)
    filename = analysis_export.results_filename(snapshot.started_at, snapshot.output_format)
    output_directory = snapshot.output_directory
    spectra_plan = analysis_export.automatic_spectra_plan(
        output_directory, class_names, labels_exist,
    )
    result_plan = available_output_plan((
        PlannedOutput("analysis run", "Results report", output_directory / filename),
    ))
    filename = result_plan[0].path.name
    validate_output_plan((
        *spectra_plan,
        *result_plan,
    ))
    extracted = io.extract_wavelengths(snapshot.data_header)
    if not extracted:
        raise ValueError("No wavelengths were extracted. Please check the data header file.")
    if snapshot.grid_wavelengths and tuple(extracted) != snapshot.grid_wavelengths:
        raise ValueError("The selected spectral grid changed after analysis was started.")
    required_bands = {0: 396, 1: 277}
    if snapshot.sensor_type not in (0, 1, 3):
        raise ValueError("Unsupported analysis profile.")
    if snapshot.sensor_type in required_bands and len(extracted) < required_bands[snapshot.sensor_type]:
        mode = {0: "VNIR", 1: "SWIR"}[snapshot.sensor_type]
        raise ValueError(
            f"{mode} built-in indices require at least {required_bands[snapshot.sensor_type]} "
            f"bands; the selected Data HDR has {len(extracted)}."
        )
    for set_index in range(matched.number_of_sets):
        _check_cancelled(cancelled)
        paths = dataset.paths_for_set(matched, set_index)
        dark = envi_validation.validate_envi_binary(snapshot.dark_header, paths.dark)
        data = envi_validation.validate_envi_binary(snapshot.data_header, paths.data)
        white = envi_validation.validate_envi_binary(snapshot.white_header, paths.white)
        envi_validation.validate_reference_triplet(dark, data, white)
    wavelengths = np.array(extracted)
    spectra_groups = spectral.initialize_spectra_by_name(class_names)
    mean_spectra: list[np.ndarray] = []
    labels: list[str | None] = []
    rows: list[dict[str, object]] = []
    total = matched.number_of_sets
    report("total", total)
    report("progress", 0)

    for set_index in range(total):
        _check_cancelled(cancelled)
        paths = dataset.paths_for_set(matched, set_index)
        loaded = io.load_reference_cubes(
            snapshot.dark_header, snapshot.data_header, snapshot.white_header,
            paths.dark, paths.data, paths.white,
        )
        calibrated = processing.calibrate_cube(loaded.data, loaded.dark, loaded.white)
        _check_cancelled(cancelled)
        data_masked = processing.load_and_apply_mask(calibrated, paths.mask)
        label = validated.class_for(paths) if validated is not None else None
        labels.append(label)
        mean_spectrum = spectral.calculate_mean_spectrum(data_masked)
        label = spectral.record_mean_spectrum_for_class(
            mean_spectra, spectra_groups, mean_spectrum, set_index, label,
        )
        _check_cancelled(cancelled)
        custom_stats: dict[str, float] = {}
        for metric in snapshot.metrics:
            _check_cancelled(cancelled)
            calculated = custom_indices.calculate_custom_statistics(
                data_masked, metric.formula, metric.band_mapping(),
                metric.name, calculate_custom_index,
            )
            if calculated.statistics is not None:
                custom_stats.update(calculated.statistics)
        file_name = re.sub(r"'", "", os.path.splitext(matched.data_files[set_index])[0])
        calculated_stats = vegetation_indices.calculate_mode_statistics(
            data_masked, snapshot.sensor_type, file_name, label, custom_stats or None,
        )
        # Each acquisition owns its result row.
        rows.append(dict(calculated_stats))
        report("progress", set_index + 1)

    _check_cancelled(cancelled)
    frame = pd.DataFrame(rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    report("exporting", total)
    with tempfile.TemporaryDirectory(prefix=".bsa-analysis-", dir=output_directory) as temporary:
        stage = Path(temporary)
        result_path = stage / filename
        if snapshot.output_format == "csv":
            write_dataframe_csv(frame, result_path)
        elif snapshot.output_format == "xlsx":
            write_dataframe_xlsx(frame, result_path)
        else:
            raise ValueError("Unsupported analysis output format.")
        spectra_paths = analysis_export.write_automatic_spectra(
            wavelengths, spectra_groups, labels, mean_spectra, labels_exist,
            output_directory=stage,
        )
        _check_cancelled(cancelled)
        # Results publish last, after every staged output is complete.
        _check_cancelled(cancelled)
        publish_staged_outputs([
            (item, staged)
            for item, staged in zip((*spectra_plan, *result_plan), (*spectra_paths, result_path))
        ], check_cancelled=lambda: _check_cancelled(cancelled))

    return AnalysisResult(
        snapshot, frame, filename, wavelengths, mean_spectra, spectra_groups,
        labels, labels_exist, excluded,
    )
