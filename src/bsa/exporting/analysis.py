"""Portable file export helpers for BSA analysis and spectra outputs."""

from __future__ import annotations

import csv
import datetime as _datetime
import os
from pathlib import Path
import re
from typing import Any, Sequence

from bsa.analysis import spectral as spectral_analysis
from bsa.exporting.output_safety import (
    PlannedOutput, available_output_plan, validate_output_plan, write_output, write_output_bundle,
)
from bsa.exporting.spreadsheet_safety import (
    csv_safe_cell, write_dataframe_xlsx,
)


_UNSAFE_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename_component(value: object) -> str:
    """Make a user label safe as one portable filename component."""

    component = _UNSAFE_FILENAME_CHARACTERS.sub("_", str(value)).strip(" .")
    if not component or component in {".", ".."}:
        return "label"
    if component.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }:
        return f"_{component}"
    return component


def _class_spectra_stem(index: int, label: str) -> str:
    return f"mean_spectra_Class_{index:03d}_{safe_filename_component(label)}"


def _image_spectra_stem(index: int) -> str:
    return f"mean_spectra_Image_{index:03d}"


def results_filename(now: _datetime.datetime, output_format: str) -> str:
    extension = ".csv" if output_format == "csv" else ".xlsx"
    return "Results-" + now.strftime("%Y%m%d_%H%M%S") + extension


def write_rows(
    output_path: os.PathLike[str] | str,
    header: Sequence[object],
    rows: Sequence[Sequence[object]],
) -> Path:
    """Write spreadsheet-facing CSV rows with literal, quoted text cells."""

    path = Path(output_path)
    return write_output(
        PlannedOutput(str(path), "spectral CSV", path),
        lambda staged: _write_rows_unchecked(staged, header, rows),
    )


def _write_rows_unchecked(
    path: Path, header: Sequence[object], rows: Sequence[Sequence[object]],
) -> None:
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(csv_safe_cell(cell) for cell in header)
        writer.writerows(
            (csv_safe_cell(cell) for cell in row) for row in rows
        )


def _write_spectral_table_unchecked(
    path: Path, header: Sequence[object], rows: Sequence[Sequence[object]],
    extension: str,
) -> None:
    if extension == "csv":
        _write_rows_unchecked(path, header, rows)
    elif extension == "xlsx":
        import pandas as pd
        write_dataframe_xlsx(pd.DataFrame(rows, columns=header), path)
    else:
        raise ValueError("Unsupported file format. Use csv or xlsx.")


def write_spectral_table(
    output_path: os.PathLike[str] | str,
    header: Sequence[object],
    rows: Sequence[Sequence[object]],
    extension: str,
) -> Path:
    """Write a spectral table as CSV or as a genuine Excel workbook."""

    if extension not in {"csv", "xlsx"}:
        raise ValueError("Unsupported file format. Use csv or xlsx.")
    path = Path(output_path)
    return write_output(
        PlannedOutput(str(path), "spectral table", path),
        lambda staged: _write_spectral_table_unchecked(staged, header, rows, extension),
    )


def automatic_spectra_plan(
    output_directory: os.PathLike[str] | str,
    class_names: Sequence[str],
    labels_exist: bool,
) -> tuple[PlannedOutput, ...]:
    directory = Path(output_directory)
    plan = [PlannedOutput("all acquisitions", "mean spectra", directory / "mean_spectra.csv")]
    if labels_exist:
        plan.extend(
            PlannedOutput(label, "class mean spectra",
                          directory / f"{_class_spectra_stem(index, label)}.csv")
            for index, label in enumerate(sorted(class_names), 1)
        )
    return available_output_plan(plan)


def validate_spectrum_identity_names(
    class_names: Sequence[str],
    acquisition_count: int,
) -> None:
    """Preflight distinct image and class export namespaces."""

    virtual = Path(".bsa-spectrum-name-plan")
    outputs = [
        PlannedOutput(f"acquisition {index}", "acquisition spectrum",
                      virtual / f"{_image_spectra_stem(index)}.csv")
        for index in range(1, acquisition_count + 1)
    ]
    outputs.extend(
        PlannedOutput(label, "class spectrum",
                      virtual / f"{_class_spectra_stem(index, label)}.csv")
        for index, label in enumerate(sorted(class_names), 1)
    )
    validate_output_plan(outputs, check_existing=False)


def write_automatic_spectra(
    wavelengths: Sequence[float],
    spectra_groups: spectral_analysis.SpectraGroups,
    labels: Sequence[str | None],
    mean_spectra: Sequence[Any],
    labels_exist: bool,
    output_directory: os.PathLike[str] | str = ".",
) -> list[Path]:
    """Write the analyzer's automatic row-oriented CSV outputs."""

    directory = Path(output_directory)
    header, rows = spectral_analysis.prepare_image_spectra_rows(
        wavelengths, spectra_groups
    )
    class_tables = (
        spectral_analysis.prepare_class_spectra_rows(wavelengths, labels, mean_spectra)
        if labels_exist else {}
    )
    plan = automatic_spectra_plan(directory, tuple(class_tables), labels_exist)
    tables: list[tuple[Sequence[object], Sequence[Sequence[object]]]] = [(header, rows)]
    if labels_exist:
        tables.extend(class_tables.values())
    return write_output_bundle([
        (item, lambda staged, h=h, r=r: _write_rows_unchecked(staged, h, r))
        for item, (h, r) in zip(plan, tables)
    ])


def write_timestamped_mean_spectra(
    output_directory: os.PathLike[str] | str,
    wavelengths: Sequence[float],
    mean_spectra: Sequence[Any],
    spectra_groups: spectral_analysis.SpectraGroups,
    include_named_spectra: bool,
    now: _datetime.datetime,
    extension: str,
) -> list[Path]:
    """Write manual spectral exports as CSV or genuine XLSX workbooks."""

    directory = Path(output_directory)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    header, rows = spectral_analysis.prepare_column_spectra_rows(
        wavelengths, mean_spectra
    )
    plan = [PlannedOutput(
        "all acquisitions", "manual mean spectra",
        directory / f"mean_spectra-{timestamp}.{extension}",
    )]
    tables: list[tuple[Sequence[object], Sequence[Sequence[object]]]] = [(header, rows)]
    if include_named_spectra:
        for index, (image_name, spectra) in enumerate(spectra_groups.image_spectra.items(), 1):
            label_header, label_rows = spectral_analysis.prepare_column_spectra_rows(
                wavelengths,
                spectra,
                column_prefix=f"Mean_Spectrum_{image_name}",
            )
            plan.append(
                PlannedOutput(
                    image_name, "manual image spectra",
                    directory / f"{_image_spectra_stem(index)}-{timestamp}.{extension}",
                )
            )
            tables.append((label_header, label_rows))
        for index, label in enumerate(sorted(spectra_groups.class_spectra), 1):
            spectra = spectra_groups.class_spectra[label]
            label_header, label_rows = spectral_analysis.prepare_column_spectra_rows(
                wavelengths, spectra, column_prefix=f"Mean_Spectrum_{label}",
            )
            plan.append(
                PlannedOutput(
                    label, "manual class spectra",
                    directory / f"{_class_spectra_stem(index, label)}-{timestamp}.{extension}",
                )
            )
            tables.append((label_header, label_rows))
    if extension not in {"csv", "xlsx"}:
        raise ValueError("Unsupported file format. Use csv or xlsx.")
    validate_output_plan(plan)
    return write_output_bundle([
        (item, lambda staged, h=h, r=r: _write_spectral_table_unchecked(staged, h, r, extension))
        for item, (h, r) in zip(plan, tables)
    ])


def save_png(
    image: Any,
    output_directory: os.PathLike[str] | str,
    now: _datetime.datetime,
) -> Path:
    path = Path(output_directory) / (
        "ResultsImage-" + now.strftime("%Y%m%d_%H%M%S") + ".png"
    )
    return write_output(
        PlannedOutput("current plot", "PNG plot", path),
        lambda staged: image.save(str(staged), "PNG"),
    )


def save_jpeg(
    image: Any,
    output_directory: os.PathLike[str] | str,
    now: _datetime.datetime,
) -> Path:
    path = Path(output_directory) / (
        "ResultsImage-" + now.strftime("%Y%m%d_%H%M%S") + ".jpg"
    )
    return write_output(
        PlannedOutput("current plot", "JPEG plot", path),
        lambda staged: image.convert("RGB").save(str(staged), "JPEG"),
    )


def save_plot_pair(
    image: Any,
    output_directory: os.PathLike[str] | str,
    now: _datetime.datetime,
) -> list[Path]:
    """Publish the GUI's PNG and JPEG plot together."""

    directory = Path(output_directory)
    stem = "ResultsImage-" + now.strftime("%Y%m%d_%H%M%S")
    return write_output_bundle((
        (
            PlannedOutput("current plot", "PNG plot", directory / f"{stem}.png"),
            lambda staged: image.save(str(staged), "PNG"),
        ),
        (
            PlannedOutput("current plot", "JPEG plot", directory / f"{stem}.jpg"),
            lambda staged: image.convert("RGB").save(str(staged), "JPEG"),
        ),
    ))
