import os
import re
import numpy as np

from bsa.exporting.spreadsheet_safety import write_dataframe_csv, write_dataframe_xlsx
from bsa.exporting.output_safety import PlannedOutput, write_output
from pathlib import Path
from bsa.utils.expressions import evaluate_arithmetic_expression


class CustomIndexError(ValueError):
    """A requested custom index could not be calculated."""

def export_results(df, filename, format, set_index):
    """
    Export the DataFrame to the specified format.

    Args:
    df: DataFrame to export.
    filename: Name of the file to export to.
    format: Format of the output file ('csv' or 'xlsx').
    set_index: Index of the current set being processed.
    """
    if format == 'csv':
        if set_index == 0:
            path = Path(filename)
            write_output(
                PlannedOutput("manual results", "CSV report", path),
                lambda staged: write_dataframe_csv(df, staged),
            )
        else:
            write_dataframe_csv(df, filename, header=False, mode='a')
    elif format == 'xlsx':
        if set_index == 0:
            path = Path(filename)
            write_output(
                PlannedOutput("manual results", "XLSX report", path),
                lambda staged: write_dataframe_xlsx(df, staged),
            )
        else:
            # An explicit continuation appends to the prior report.
            write_dataframe_xlsx(df, filename, append=os.path.exists(filename))

def export_spectra_data(df, filename, format):
    """
    Export the spectra data to the specified format.

    Args:
    df: DataFrame containing the spectra data.
    filename: Name of the file to export to.
    format: Format of the output file ('csv' or 'xlsx').
    """
    if format == 'csv':
        path = Path(filename)
        write_output(
            PlannedOutput("manual spectra", "CSV spectra", path),
            lambda staged: write_dataframe_csv(df, staged),
        )
    elif format == 'xlsx':
        path = Path(filename)
        write_output(
            PlannedOutput("manual spectra", "XLSX spectra", path),
            lambda staged: write_dataframe_xlsx(df, staged),
        )

def calculate_custom_index(data_masked, formula, bands, index_name):
    """
    Calculate a custom index based on the user-defined formula.

    Args:
    data_masked: The masked data array (numpy array).
    formula: A string representing the formula to evaluate.
    bands: A dictionary mapping band names to their indices.
    index_name: A string representing the name of the custom index.

    Returns:
    A tuple containing the calculated custom index as a numpy array and its name.
    """
    try:
        band_values = {}
        band_count = data_masked.shape[2]
        for band_name, band_index in bands.items():
            index = int(band_index)
            if not 0 <= index < band_count:
                raise ValueError(
                    f"{band_name} band index {index} is outside 0..{band_count - 1}."
                )
            band_values[band_name] = data_masked[:, :, index]
        custom_index = evaluate_arithmetic_expression(formula, band_values)
        finite_inputs = (
            np.logical_and.reduce([np.isfinite(values) for values in band_values.values()])
            if band_values else True
        )
        if np.any(finite_inputs & ~np.isfinite(custom_index)):
            raise ValueError("The formula produced a non-finite value at a valid pixel.")
        return custom_index, index_name
    except (ValueError, TypeError, IndexError, KeyError, ZeroDivisionError) as e:
        raise CustomIndexError(f"Cannot calculate custom index {index_name!r}: {e}") from e
    
def categorize_file(fileName):
    # Patterns for Dark, Data, and White for BIL files
    dark_bil_pattern = r"-VNIR-DarkCalibration"
    data_bil_pattern = r"-VNIR-Data"
    white_bil_pattern = r"-VNIR-WhiteCalibration"

    # Patterns for Dark, Data, and White for SWIR files
    dark_swir_pattern = r"-SWIR-DarkCalibration"
    data_swir_pattern = r"-SWIR-Data"
    white_swir_pattern = r"-SWIR-WhiteCalibration"

    # Patterns for Dark, Data, and White for RAW files
    dark_raw_pattern = r"DARKREF_vnir"
    data_raw_pattern = r"vnir_\d+_\d+_\d+"  # This pattern needs to be refined
    white_raw_pattern = r"WHITEREF_vnir"

    if re.search(dark_bil_pattern, fileName, re.IGNORECASE) or re.search(dark_swir_pattern, fileName, re.IGNORECASE) or re.search(dark_raw_pattern, fileName, re.IGNORECASE):
        return 'dark'
    elif re.search(data_bil_pattern, fileName, re.IGNORECASE) or re.search(data_swir_pattern, fileName, re.IGNORECASE) or (re.search(data_raw_pattern, fileName, re.IGNORECASE) and not re.search(white_raw_pattern, fileName, re.IGNORECASE)):
        return 'data'
    elif re.search(white_bil_pattern, fileName, re.IGNORECASE) or re.search(white_swir_pattern, fileName, re.IGNORECASE) or re.search(white_raw_pattern, fileName, re.IGNORECASE):
        return 'white'
    return 'unknown'
